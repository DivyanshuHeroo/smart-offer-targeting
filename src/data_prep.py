"""
data_prep.py
============
Turn the three raw Starbucks JSON files into one clean, leakage-free modelling
table at the (customer x offer-instance) grain.

Each row = one "offer received" event. The label is whether the customer engaged
with that specific offer *within its validity window*:

    clicked  = 1 if the customer VIEWED the offer within its duration
    accepted = 1 if the customer COMPLETED the offer within its duration
               (only meaningful for bogo/discount; informational can't complete)

Crucial point for interviews -> NO LEAKAGE.
All "history" features (past spend, past offer-response rates, recency, ...) are
computed using ONLY events that happened strictly BEFORE the offer was received.
We do this by walking every customer's events in time order and snapshotting their
running state at each "offer received".

Why "viewed" as the main target?
- It applies to every offer (informational offers can be viewed but never completed).
- It is the cleanest signal of "this offer was relevant enough to engage with",
  which is exactly what an offer-targeting / recommendation system optimises.

Run:  python -m src.data_prep
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 1. Load + clean the three raw tables
# ---------------------------------------------------------------------------
def load_raw():
    portfolio = pd.read_json(config.PORTFOLIO_JSON, orient="records", lines=True)
    profile = pd.read_json(config.PROFILE_JSON, orient="records", lines=True)
    transcript = pd.read_json(config.TRANSCRIPT_JSON, orient="records", lines=True)
    return portfolio, profile, transcript


def clean_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    """One row per offer with channel flags and an attractiveness score."""
    p = portfolio.rename(columns={"id": "offer_id"}).copy()
    # Channel one-hot flags from the list column.
    for ch in ["web", "email", "mobile", "social"]:
        p[f"ch_{ch}"] = p["channels"].apply(lambda c: int(ch in c))
    p["n_channels"] = p["channels"].apply(len)

    # Offer "attractiveness": reward you get per unit you must spend.
    # Informational offers have difficulty 0 and reward 0 -> attractiveness 0.
    p["reward_per_difficulty"] = np.where(
        p["difficulty"] > 0, p["reward"] / p["difficulty"], 0.0
    )
    # Short, friendly label for dashboards.
    p["offer_label"] = (
        p["offer_type"].str.title()
        + " | reward "
        + p["reward"].astype(str)
        + " / spend "
        + p["difficulty"].astype(str)
        + " / "
        + p["duration"].astype(str)
        + "d"
    )
    return p


def clean_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Flag placeholder customers (age 118 + missing gender/income) instead of dropping."""
    pr = profile.rename(columns={"id": "customer_id"}).copy()
    pr["missing_demographics"] = (
        (pr["age"] == config.MISSING_AGE) | pr["income"].isna() | pr["gender"].isna()
    ).astype(int)

    # Age 118 is a sentinel, not a real age -> set to NaN then impute with median.
    pr.loc[pr["age"] == config.MISSING_AGE, "age"] = np.nan
    pr["age"] = pr["age"].fillna(pr["age"].median())
    pr["income"] = pr["income"].fillna(pr["income"].median())
    pr["gender"] = pr["gender"].fillna("U")  # Unknown

    # Membership tenure (days) relative to the most recent signup in the data.
    pr["became_member_on"] = pd.to_datetime(pr["became_member_on"], format="%Y%m%d")
    ref = pr["became_member_on"].max()
    pr["membership_days"] = (ref - pr["became_member_on"]).dt.days

    # Simple, interview-friendly segments.
    pr["income_segment"] = pd.cut(
        pr["income"],
        bins=[0, 40000, 60000, 80000, np.inf],
        labels=["low", "mid", "high", "premium"],
    ).astype(str)
    pr["age_group"] = pd.cut(
        pr["age"],
        bins=[17, 30, 45, 60, 120],
        labels=["18-30", "31-45", "46-60", "60+"],
    ).astype(str)
    return pr


# ---------------------------------------------------------------------------
# 2. Normalise the transcript (pull fields out of the `value` dict)
# ---------------------------------------------------------------------------
def normalise_transcript(transcript: pd.DataFrame) -> pd.DataFrame:
    t = transcript.rename(columns={"person": "customer_id"}).copy()

    def get_offer(v):
        # received/viewed use 'offer id'; completed uses 'offer_id'.
        return v.get("offer id", v.get("offer_id"))

    t["offer_id"] = t["value"].apply(get_offer)
    t["amount"] = t["value"].apply(lambda v: v.get("amount"))
    t = t.drop(columns=["value"])
    return t


# ---------------------------------------------------------------------------
# 3. Build the labelled (customer x offer-instance) table, time-aware
# ---------------------------------------------------------------------------
def build_model_table(portfolio, profile, transcript) -> pd.DataFrame:
    dur = portfolio.set_index("offer_id")["duration"].to_dict()  # days
    otype_of = portfolio.set_index("offer_id")["offer_type"].to_dict()
    rows = []

    # Process one customer at a time so history is naturally scoped to that person.
    transcript = transcript.sort_values(["customer_id", "time"]).reset_index(drop=True)

    for cust_id, g in transcript.groupby("customer_id", sort=False):
        rows_cust = list(g.itertuples(index=False))  # already time-sorted
        # Running state for THIS customer (only events seen so far).
        n_txn = 0
        spend_sum = 0.0
        last_txn_time = None
        offers_recv = 0
        offers_viewed = 0
        offers_completed = 0
        # Per offer_type response history -> "category affinity".
        type_recv = {"bogo": 0, "discount": 0, "informational": 0}
        type_viewed = {"bogo": 0, "discount": 0, "informational": 0}

        # We need to look forward for view/complete within window, so first collect
        # this customer's viewed/completed events keyed by offer_id with their times.
        viewed_times = {}     # offer_id -> sorted list of view times
        completed_times = {}  # offer_id -> sorted list of completion times
        for r in rows_cust:
            if r.event == "offer viewed":
                viewed_times.setdefault(r.offer_id, []).append(r.time)
            elif r.event == "offer completed":
                completed_times.setdefault(r.offer_id, []).append(r.time)
        for k in viewed_times:
            viewed_times[k].sort()
        for k in completed_times:
            completed_times[k].sort()

        for r in rows_cust:
            if r.event == "transaction":
                n_txn += 1
                spend_sum += float(r.amount or 0.0)
                last_txn_time = r.time

            elif r.event == "offer received":
                oid = r.offer_id
                otype = otype_of.get(oid, "unknown")
                window_end = r.time + dur.get(oid, 7) * 24

                # --- label: viewed within window? ---
                clicked = int(any(r.time <= vt <= window_end
                                  for vt in viewed_times.get(oid, [])))
                accepted = int(any(r.time <= ct <= window_end
                                   for ct in completed_times.get(oid, [])))

                # --- snapshot leakage-free history features ---
                recency = (r.time - last_txn_time) if last_txn_time is not None else -1
                row = {
                    "customer_id": cust_id,
                    "offer_id": oid,
                    "time": r.time,
                    # transaction history BEFORE this offer
                    "txn_count": n_txn,
                    "total_spend": round(spend_sum, 2),
                    "avg_spend": round(spend_sum / n_txn, 2) if n_txn else 0.0,
                    "recency": recency,
                    # offer-response history BEFORE this offer
                    "prior_offers_received": offers_recv,
                    "prior_view_rate": (offers_viewed / offers_recv) if offers_recv else 0.0,
                    "prior_completion_rate": (offers_completed / offers_recv) if offers_recv else 0.0,
                    # category (offer_type) affinity BEFORE this offer
                    "type_view_rate": (
                        type_viewed[otype] / type_recv[otype]
                        if otype in type_recv and type_recv[otype] else 0.0
                    ),
                    # labels
                    "clicked": clicked,
                    "accepted": accepted,
                }
                rows.append(row)

                # update running offer history AFTER snapshotting
                offers_recv += 1
                if otype in type_recv:
                    type_recv[otype] += 1
                # count this offer as viewed/completed for history if it was engaged
                if clicked:
                    offers_viewed += 1
                    if otype in type_viewed:
                        type_viewed[otype] += 1
                if accepted:
                    offers_completed += 1

    table = pd.DataFrame(rows)

    # Attach customer profile + offer attributes.
    table = table.merge(
        profile[["customer_id", "gender", "age", "income", "membership_days",
                 "income_segment", "age_group", "missing_demographics"]],
        on="customer_id", how="left",
    )
    table = table.merge(
        portfolio[["offer_id", "offer_type", "reward", "difficulty", "duration",
                   "n_channels", "ch_web", "ch_email", "ch_mobile", "ch_social",
                   "reward_per_difficulty", "offer_label"]],
        on="offer_id", how="left",
    )

    # A couple of simple interaction features.
    table["income_to_difficulty"] = np.where(
        table["difficulty"] > 0, table["income"] / (table["difficulty"] * 1000), 0.0
    )
    table["spend_to_difficulty"] = np.where(
        table["difficulty"] > 0, table["avg_spend"] / table["difficulty"], 0.0
    )
    return table


def main():
    print("Loading raw JSON ...")
    portfolio, profile, transcript = load_raw()
    portfolio = clean_portfolio(portfolio)
    profile = clean_profile(profile)
    transcript = normalise_transcript(transcript)

    print("Building leakage-free model table (this takes ~1-2 min) ...")
    table = build_model_table(portfolio, profile, transcript)

    table.to_csv(config.MODEL_TABLE, index=False)
    portfolio.to_csv(config.DATA_PROCESSED / "portfolio_clean.csv", index=False)
    profile.to_csv(config.DATA_PROCESSED / "profile_clean.csv", index=False)

    print(f"\nSaved model table: {config.MODEL_TABLE}")
    print(f"Rows (offer instances): {len(table):,}  |  Customers: {table.customer_id.nunique():,}")
    print(f"Click (viewed) rate : {table['clicked'].mean():.3f}")
    print(f"Accept (completed)  : {table['accepted'].mean():.3f}")
    print("\nClick rate by offer_type:")
    print(table.groupby("offer_type")["clicked"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
