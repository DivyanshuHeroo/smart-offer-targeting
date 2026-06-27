"""
Smart Offer Targeting System — Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py
(from the project root, with the venv/conda env active)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# make `src` importable when run via `streamlit run`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src import recommend  # noqa: E402

st.set_page_config(page_title="Smart Offer Targeting", page_icon="🎯", layout="wide")


@st.cache_data
def load_data():
    table = pd.read_csv(config.MODEL_TABLE)
    portfolio = pd.read_csv(config.DATA_PROCESSED / "portfolio_clean.csv")
    metrics = json.load(open(config.REPORTS_DIR / "metrics.json"))
    return table, portfolio, metrics


table, portfolio, metrics = load_data()

st.sidebar.title("🎯 Smart Offer Targeting")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Dataset", "Model performance", "Recommendations", "Business insights"],
)
st.sidebar.caption(f"Serving model: **{recommend._MODEL_NAME}**")
st.sidebar.caption("Real Starbucks Rewards offer dataset")


# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Smart Offer Targeting System")
    st.subheader("Recommendation / Financial Analytics")
    st.markdown(
        """
**Goal** — predict which card/app offers a customer is most likely to engage with,
then **rank the Top-K offers per customer** for personalised campaigns.

**Why it matters** — instead of blasting every offer to everyone, the business
shows each customer the handful of offers they are most likely to act on. That
lifts conversion, engagement and transaction growth while cutting contact cost.

**Data** — the real **Starbucks Rewards** offer dataset: 17k customers, 10 offers,
and 306k events (offers received / viewed / completed + transactions).
The label is *offer viewed within its validity window* (engagement).
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{table.customer_id.nunique():,}")
    c2.metric("Offer instances", f"{len(table):,}")
    c3.metric("Offers in catalogue", f"{portfolio.offer_id.nunique()}")
    c4.metric("Overall click rate", f"{table['clicked'].mean():.1%}")


# ---------------------------------------------------------------- Dataset
elif page == "Dataset":
    st.title("Dataset summary")
    st.markdown("**Offer catalogue (portfolio)** — the 10 real offers customers can receive:")
    st.dataframe(
        portfolio[["offer_type", "reward", "difficulty", "duration", "n_channels",
                   "reward_per_difficulty", "offer_label"]].round(2),
        use_container_width=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Click (view) rate by offer type**")
        st.image(str(config.FIGURES_DIR / "eda_click_by_offer_type.png"))
        st.markdown("**Offer volume by type**")
        st.image(str(config.FIGURES_DIR / "eda_offer_volume.png"))
    with col2:
        st.markdown("**Click rate by customer segment**")
        st.image(str(config.FIGURES_DIR / "eda_click_by_segment.png"))
        st.markdown("**Past behaviour predicts future engagement**")
        st.image(str(config.FIGURES_DIR / "eda_prior_vs_current.png"))


# ---------------------------------------------------------------- Model performance
elif page == "Model performance":
    st.title("Model performance (held-out test set)")
    mdf = pd.DataFrame(metrics).T
    show = ["roc_auc", "pr_auc", "precision", "recall", "f1",
            "precision@3", "recall@3", "map@3", "ndcg@3", "ndcg@5"]
    st.dataframe(mdf[show].round(4), use_container_width=True)
    st.image(str(config.FIGURES_DIR / "model_comparison.png"))
    st.info(
        "**Reading the table.** XGBoost / LightGBM win on classification "
        "(ROC-AUC, PR-AUC). The **LightGBM Ranker** has low global AUC (its scores "
        "are not calibrated probabilities) but wins every **ranking** metric "
        "(MAP@K, NDCG@K) — and it is the only model that clearly beats the "
        "**popularity baseline** on the per-customer ordering we actually care about."
    )


# ---------------------------------------------------------------- Recommendations
elif page == "Recommendations":
    st.title("Top-K offer recommendations")
    customers = table["customer_id"].drop_duplicates().tolist()

    colA, colB, colC = st.columns([2, 1, 1])
    cust = colA.selectbox("Select a customer", customers[:2000])
    k = colB.slider("Top-K", 1, 5, config.TOP_K)
    rank_by = colC.selectbox("Rank by", ["probability", "expected_value"])

    snap = table[table.customer_id == cust].sort_values("time")
    m1, m2, m3, m4 = st.columns(4)
    last = snap.iloc[-1]
    m1.metric("Age", int(last.age))
    m2.metric("Income", f"${int(last.income):,}")
    m3.metric("Txns so far", int(last.txn_count))
    m4.metric("Past view rate", f"{last.prior_view_rate:.0%}")

    recs = recommend.recommend_for_customer(cust, k=k, rank_by=rank_by)
    st.markdown("### Recommended offers")
    st.dataframe(
        recs.assign(click_probability=(recs.click_probability * 100).round(1).astype(str) + "%",
                    expected_value=recs.expected_value.round(2))
            [["rank", "offer_type", "offer_label", "click_probability",
              "expected_value", "recommendation_reason"]],
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(recs.set_index("offer_type")["click_probability"])


# ---------------------------------------------------------------- Business insights
elif page == "Business insights":
    st.title("Business insights")
    st.markdown(
        """
- **Offer type is the strongest lever.** BOGO offers are viewed far more than
  informational ones — the catalogue mix itself drives engagement.
- **Past behaviour is predictive.** Customers who engaged with offers before are
  much more likely to engage again (see the prior-vs-current chart) — so
  offer-response history is one of the most valuable features.
- **Ranking beats classification for this use-case.** Because the business only
  shows a few offers, optimising *ordering* (the LightGBM Ranker) lifts MAP@K /
  NDCG@K above a strong popularity baseline.
- **Targeting saves money.** Showing each customer their Top-3 instead of all
  offers focuses spend on likely engagers and cuts wasted contact cost.
        """
    )
    st.markdown("#### Average spend distribution")
    st.image(str(config.FIGURES_DIR / "eda_spend_distribution.png"))
    st.caption(
        "Expected campaign value uses simple, clearly-stated assumptions "
        f"(margin ${config.ENGAGED_MARGIN:.0f} per engagement, contact cost "
        f"${config.CONTACT_COST:.2f}) — for prioritisation, not as real revenue."
    )
