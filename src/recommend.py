"""
recommend.py
============
Serving layer. For any customer, score ALL 10 offers, rank them, and return the
Top-K with a predicted click probability, an expected campaign value, and a short
human-readable reason.

How a candidate row is built
----------------------------
We take the customer's most recent leakage-free snapshot (their spend/recency/
offer-response history + profile) and cross it with every offer's attributes.
That gives 10 candidate rows with the exact feature columns the model expects.

Expected campaign value (prioritisation only, clearly an assumption)
--------------------------------------------------------------------
    expected_value = P(click) * ENGAGED_MARGIN - CONTACT_COST
ENGAGED_MARGIN and CONTACT_COST live in config.py. This is NOT claimed as real
revenue; it just lets the business trade off "likely to engage" against "cheap to
send" when picking who to target.

Run:  python -m src.recommend            # prints a sample + writes recommendations.csv
"""
from __future__ import annotations

import os
# macOS/Anaconda OpenMP safety (see train.py). Pin threads to avoid a libomp crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from . import config
from .train import FEATURES

# --- load artefacts once ---
_BUNDLE = joblib.load(config.MODELS_DIR / "best_model.pkl")
_MODEL = _BUNDLE["model"]
_MODEL_NAME = _BUNDLE["name"]
_PORTFOLIO = pd.read_csv(config.DATA_PROCESSED / "portfolio_clean.csv")
_TABLE = pd.read_csv(config.MODEL_TABLE)


def _customer_snapshot(customer_id: str) -> dict:
    """Latest history + profile features for a customer (their most recent offer row)."""
    rows = _TABLE[_TABLE.customer_id == customer_id]
    if rows.empty:
        raise KeyError(f"Unknown customer_id: {customer_id}")
    latest = rows.sort_values("time").iloc[-1]
    # observed per-offer_type engagement (used as type_view_rate for candidates)
    type_rate = rows.groupby("offer_type")[config.TARGET].mean().to_dict()
    return {
        "txn_count": latest.txn_count,
        "total_spend": latest.total_spend,
        "avg_spend": latest.avg_spend,
        "recency": latest.recency,
        "prior_offers_received": latest.prior_offers_received,
        "prior_view_rate": latest.prior_view_rate,
        "prior_completion_rate": latest.prior_completion_rate,
        "age": latest.age,
        "income": latest.income,
        "membership_days": latest.membership_days,
        "missing_demographics": latest.missing_demographics,
        "gender": latest.gender,
        "income_segment": latest.income_segment,
        "age_group": latest.age_group,
        "_type_rate": type_rate,
        "_overall_rate": float(rows[config.TARGET].mean()),
    }


def _build_candidates(snap: dict) -> pd.DataFrame:
    """Cross the customer snapshot with all offers -> a frame of FEATURES columns."""
    cand = _PORTFOLIO.copy()
    for col in ["txn_count", "total_spend", "avg_spend", "recency",
                "prior_offers_received", "prior_view_rate", "prior_completion_rate",
                "age", "income", "membership_days", "missing_demographics",
                "gender", "income_segment", "age_group"]:
        cand[col] = snap[col]
    # type affinity per candidate offer_type (fallback to overall rate)
    cand["type_view_rate"] = cand["offer_type"].map(snap["_type_rate"]).fillna(snap["_overall_rate"])
    # interactions (same formulas as data_prep)
    cand["income_to_difficulty"] = np.where(
        cand["difficulty"] > 0, cand["income"] / (cand["difficulty"] * 1000), 0.0)
    cand["spend_to_difficulty"] = np.where(
        cand["difficulty"] > 0, cand["avg_spend"] / cand["difficulty"], 0.0)
    return cand


def _reason(row, snap) -> str:
    bits = []
    best_type = max(snap["_type_rate"], key=snap["_type_rate"].get) if snap["_type_rate"] else None
    if best_type and row["offer_type"] == best_type and snap["_type_rate"][best_type] > 0.5:
        bits.append(f"history of engaging with {row['offer_type']} offers")
    if row["reward_per_difficulty"] >= 0.8 and row["difficulty"] > 0:
        bits.append("strong reward-to-spend ratio")
    if row["offer_type"] == "informational":
        bits.append("low-friction informational nudge")
    if row["ch_mobile"] == 1 or row["ch_social"] == 1:
        bits.append("reachable on mobile/social")
    if snap["recency"] >= 0 and snap["recency"] <= 72:
        bits.append("recently active spender")
    return "; ".join(bits[:2]) if bits else "broadly relevant given profile"


def recommend_for_customer(customer_id: str, k: int = config.TOP_K,
                           rank_by: str = "probability") -> pd.DataFrame:
    """Return Top-K offers for one customer.

    rank_by: 'probability' (pure affinity) or 'expected_value' (affinity x value).
    """
    snap = _customer_snapshot(customer_id)
    cand = _build_candidates(snap)
    prob = _MODEL.predict_proba(cand[FEATURES])[:, 1]
    cand = cand.assign(click_probability=prob)
    cand["expected_value"] = (cand["click_probability"] * config.ENGAGED_MARGIN
                              - config.CONTACT_COST)
    cand["recommendation_reason"] = cand.apply(lambda r: _reason(r, snap), axis=1)

    sort_col = "click_probability" if rank_by == "probability" else "expected_value"
    out = cand.sort_values(sort_col, ascending=False).head(k)
    out = out.assign(customer_id=customer_id, rank=range(1, len(out) + 1))
    cols = ["customer_id", "rank", "offer_id", "offer_type", "offer_label",
            "click_probability", "expected_value", "recommendation_reason"]
    return out[cols].reset_index(drop=True)


def recommend_batch(customer_ids, k=config.TOP_K, rank_by="probability") -> pd.DataFrame:
    return pd.concat([recommend_for_customer(c, k, rank_by) for c in customer_ids],
                     ignore_index=True)


def main():
    print(f"Serving model: {_MODEL_NAME}\n")
    sample = _TABLE["customer_id"].drop_duplicates().head(50).tolist()
    recs = recommend_batch(sample, k=config.TOP_K)
    out_path = config.REPORTS_DIR / "sample_recommendations.csv"
    recs.to_csv(out_path, index=False)

    print("Example — Top-3 offers for first 3 customers:\n")
    for c in sample[:3]:
        r = recommend_for_customer(c)
        print(f"Customer {c[:8]}...")
        print(r[["rank", "offer_type", "click_probability", "expected_value",
                 "recommendation_reason"]].to_string(index=False))
        print()
    print(f"Saved {len(recs)} rows -> {out_path}")


if __name__ == "__main__":
    main()
