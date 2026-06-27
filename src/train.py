"""
train.py
========
Train and compare five models for offer-click prediction, then evaluate them with
both classification and ranking metrics.

Models
------
1. Logistic Regression  (interpretable baseline)
2. Random Forest        (non-linear bagging)
3. XGBoost              (gradient boosting)
4. LightGBM             (fast gradient boosting)
5. LightGBM Ranker      (learning-to-rank / LambdaRank -> optimises ordering directly)

Key design choice: we split TRAIN / VAL / TEST *by customer*. A customer's rows
never straddle two splits, so the leakage-free history features stay honest and
the per-customer ranking evaluation is fair.

Run:  python -m src.train
"""
from __future__ import annotations

import os
# macOS/Anaconda ship multiple OpenMP runtimes (one each from XGBoost & LightGBM);
# loading both can segfault. This is the standard, safe workaround. Must be set
# BEFORE importing xgboost/lightgbm.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config, metrics

warnings.filterwarnings("ignore")

NUMERIC = [
    "txn_count", "total_spend", "avg_spend", "recency",
    "prior_offers_received", "prior_view_rate", "prior_completion_rate",
    "type_view_rate", "age", "income", "membership_days", "missing_demographics",
    "reward", "difficulty", "duration", "n_channels",
    "ch_web", "ch_email", "ch_mobile", "ch_social",
    "reward_per_difficulty", "income_to_difficulty", "spend_to_difficulty",
]
CATEGORICAL = ["gender", "income_segment", "age_group", "offer_type"]
FEATURES = NUMERIC + CATEGORICAL


# ---------------------------------------------------------------------------
# Split by customer
# ---------------------------------------------------------------------------
def split_by_customer(df, target=config.TARGET):
    rng = np.random.RandomState(config.RANDOM_STATE)
    customers = df["customer_id"].unique()
    rng.shuffle(customers)
    n = len(customers)
    train_c = set(customers[: int(0.6 * n)])
    val_c = set(customers[int(0.6 * n): int(0.8 * n)])
    test_c = set(customers[int(0.8 * n):])

    tr = df[df.customer_id.isin(train_c)].copy()
    va = df[df.customer_id.isin(val_c)].copy()
    te = df[df.customer_id.isin(test_c)].copy()
    return tr, va, te


def make_preprocessor(scale_numeric: bool):
    num_tf = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", num_tf, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------
def train_all(tr, va, te):
    ytr, yva, yte = tr[config.TARGET], va[config.TARGET], te[config.TARGET]
    results = {}
    fitted = {}

    # --- Popularity baseline: rank every customer's offers by the offer's overall
    # view-rate learned on TRAIN. This answers "did the ML actually beat just
    # showing the globally most-popular offers?" Essential context for ranking gains.
    pop = tr.groupby("offer_id")[config.TARGET].mean()
    te_pop = te.copy()
    te_pop["score"] = te_pop["offer_id"].map(pop).fillna(pop.mean())
    results["PopularityBaseline"] = {
        **metrics.classification_report_dict(yte, te_pop["score"]),
        **metrics.ranking_report(te_pop, "customer_id", "score", config.TARGET, ks=(1, 3, 5)),
    }

    classifiers = {
        "LogisticRegression": (
            Pipeline([("prep", make_preprocessor(True)),
                      ("clf", LogisticRegression(max_iter=1000,
                                                 class_weight="balanced",
                                                 random_state=config.RANDOM_STATE))]),
            False,
        ),
        "RandomForest": (
            Pipeline([("prep", make_preprocessor(False)),
                      ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                                                     min_samples_leaf=20, n_jobs=-1,
                                                     class_weight="balanced",
                                                     random_state=config.RANDOM_STATE))]),
            False,
        ),
        "XGBoost": (
            Pipeline([("prep", make_preprocessor(False)),
                      ("clf", XGBClassifier(n_estimators=400, max_depth=5,
                                            learning_rate=0.05, subsample=0.8,
                                            colsample_bytree=0.8, eval_metric="logloss",
                                            n_jobs=1, random_state=config.RANDOM_STATE))]),
            False,
        ),
        "LightGBM": (
            Pipeline([("prep", make_preprocessor(False)),
                      ("clf", LGBMClassifier(n_estimators=500, num_leaves=31,
                                             learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, n_jobs=1,
                                             random_state=config.RANDOM_STATE,
                                             verbose=-1))]),
            False,
        ),
    }

    for name, (pipe, _) in classifiers.items():
        print(f"  training {name} ...")
        pipe.fit(tr[FEATURES], ytr)
        prob = pipe.predict_proba(te[FEATURES])[:, 1]
        cls = metrics.classification_report_dict(yte, prob)

        te_scored = te.copy()
        te_scored["score"] = prob
        rank = metrics.ranking_report(te_scored, "customer_id", "score",
                                      config.TARGET, ks=(1, 3, 5))
        results[name] = {**cls, **rank}
        fitted[name] = pipe

    # --- LightGBM Ranker (learning-to-rank) ---
    print("  training LightGBM Ranker (LambdaRank) ...")
    prep = make_preprocessor(False)
    tr_sorted = tr.sort_values("customer_id")
    va_sorted = va.sort_values("customer_id")
    Xtr = prep.fit_transform(tr_sorted[FEATURES])
    Xva = prep.transform(va_sorted[FEATURES])
    grp_tr = tr_sorted.groupby("customer_id").size().values
    grp_va = va_sorted.groupby("customer_id").size().values

    ranker = LGBMRanker(
        objective="lambdarank", n_estimators=500, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        n_jobs=1, random_state=config.RANDOM_STATE, verbose=-1,
    )
    ranker.fit(Xtr, tr_sorted[config.TARGET].values, group=grp_tr,
               eval_set=[(Xva, va_sorted[config.TARGET].values)], eval_group=[grp_va],
               eval_at=[3])

    Xte = prep.transform(te[FEATURES])
    te_scored = te.copy()
    te_scored["score"] = ranker.predict(Xte)
    rank = metrics.ranking_report(te_scored, "customer_id", "score",
                                  config.TARGET, ks=(1, 3, 5))
    # Ranker outputs an unbounded score (not a probability) -> classification
    # metrics that need a probability are reported as AUC/PR-AUC only.
    from sklearn.metrics import average_precision_score, roc_auc_score
    results["LightGBM_Ranker"] = {
        "roc_auc": float(roc_auc_score(te[config.TARGET], te_scored["score"])),
        "pr_auc": float(average_precision_score(te[config.TARGET], te_scored["score"])),
        "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
        **rank,
    }
    fitted["LightGBM_Ranker"] = {"prep": prep, "ranker": ranker}

    return results, fitted


# ---------------------------------------------------------------------------
# Save artefacts + comparison figure
# ---------------------------------------------------------------------------
def save_outputs(results, fitted, te):
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Pick the best classifier by PR-AUC (handles class imbalance well) for serving.
    classifier_results = {k: v for k, v in results.items()
                          if k not in ("LightGBM_Ranker", "PopularityBaseline")}
    best_name = max(classifier_results, key=lambda k: classifier_results[k]["pr_auc"])
    joblib.dump(
        {"name": best_name, "model": fitted[best_name], "features": FEATURES},
        config.MODELS_DIR / "best_model.pkl",
    )
    print(f"\nBest serving model (by PR-AUC): {best_name}")

    # Comparison table
    table = pd.DataFrame(results).T
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metric_cols = ["roc_auc", "pr_auc", "ndcg@3", "map@3", "precision@3"]
        ax = table[metric_cols].plot(kind="bar", figsize=(11, 5))
        ax.set_title("Model comparison — classification & ranking metrics")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", ncol=3, fontsize=8)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=120)
        plt.close()
        print(f"Saved figure: {config.FIGURES_DIR / 'model_comparison.png'}")
    except Exception as e:
        print(f"(skipped figure: {e})")

    return best_name, table


def main():
    print("Loading model table ...")
    df = pd.read_csv(config.MODEL_TABLE)
    tr, va, te = split_by_customer(df)
    print(f"Split by customer -> train {len(tr):,} / val {len(va):,} / test {len(te):,} rows")

    print("\nTraining models ...")
    results, fitted = train_all(tr, va, te)

    best_name, table = save_outputs(results, fitted, te)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n================  RESULTS (test set)  ================")
    show = ["roc_auc", "pr_auc", "precision", "recall", "f1",
            "precision@3", "recall@3", "map@3", "ndcg@3", "ndcg@5"]
    print(table[show].round(4).to_string())


if __name__ == "__main__":
    main()
