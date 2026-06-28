"""
tune.py
=======
Hyperparameter tuning for the two strongest classifiers (XGBoost, LightGBM) and
the LightGBM Ranker, then an honest before/after comparison against the default
models from `python -m src.train`.

Two things keep this fair and leakage-free:
1. Tuning uses **GroupKFold with customer_id as the group** — a customer never
   appears in both a CV-train and CV-validation fold, exactly like the main split.
2. Tuned models are scored on the **same held-out test set** (same seed/split) as
   the defaults, so improvements are real and comparable.

Scoring:
- classifiers -> average_precision (PR-AUC), matching how we pick the serving model.
- ranker      -> NDCG@3 on the validation customers (the metric the business cares about).

Run:  python -m src.tune
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, metrics
from .train import FEATURES, make_preprocessor, split_by_customer

N_ITER = 25          # random configs tried per model
CV_SPLITS = 3
SEED = config.RANDOM_STATE


# ---------------------------------------------------------------------------
# Classifier tuning (RandomizedSearchCV + GroupKFold)
# ---------------------------------------------------------------------------
def tune_classifier(name, estimator, param_dist, tr, te):
    ytr, yte = tr[config.TARGET], te[config.TARGET]
    pipe = Pipeline([("prep", make_preprocessor(False)), ("clf", estimator)])
    gkf = GroupKFold(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist, n_iter=N_ITER,
        scoring="average_precision", cv=gkf, random_state=SEED,
        n_jobs=1, verbose=0, refit=True,
    )
    print(f"  tuning {name} ({N_ITER} configs x {CV_SPLITS}-fold GroupKFold) ...")
    search.fit(tr[FEATURES], ytr, groups=tr["customer_id"])

    prob = search.predict_proba(te[FEATURES])[:, 1]
    te_scored = te.copy(); te_scored["score"] = prob
    result = {
        **metrics.classification_report_dict(yte, prob),
        **metrics.ranking_report(te_scored, "customer_id", "score", config.TARGET, (1, 3, 5)),
    }
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"    best CV PR-AUC={search.best_score_:.4f}  params={best}")
    return result, search.best_estimator_, best


# ---------------------------------------------------------------------------
# Ranker tuning (manual randomized search, selected by validation NDCG@3)
# ---------------------------------------------------------------------------
def tune_ranker(tr, va, te):
    prep = make_preprocessor(False)
    tr_s, va_s = tr.sort_values("customer_id"), va.sort_values("customer_id")
    Xtr = prep.fit_transform(tr_s[FEATURES]); Xva = prep.transform(va_s[FEATURES])
    Xte = prep.transform(te[FEATURES])
    gtr = tr_s.groupby("customer_id").size().values
    gva = va_s.groupby("customer_id").size().values

    grid = {
        "n_estimators": [300, 500, 700],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_samples": [20, 50, 100],
        "reg_lambda": [0.0, 1.0, 5.0],
    }
    rng = np.random.RandomState(SEED)
    print(f"  tuning LightGBM Ranker ({N_ITER} configs, selected by val NDCG@3) ...")
    best_ndcg, best_params, best_model = -1.0, None, None
    for _ in range(N_ITER):
        params = {k: rng.choice(v) for k, v in grid.items()}
        params = {k: (int(v) if k in ("n_estimators", "num_leaves", "min_child_samples")
                      else float(v)) for k, v in params.items()}
        r = LGBMRanker(objective="lambdarank", n_jobs=1, random_state=SEED,
                       verbose=-1, **params)
        r.fit(Xtr, tr_s[config.TARGET].values, group=gtr)
        va_s2 = va_s.copy(); va_s2["score"] = r.predict(Xva)
        ndcg = metrics.ranking_report(va_s2, "customer_id", "score", config.TARGET, (3,))["ndcg@3"]
        if ndcg > best_ndcg:
            best_ndcg, best_params, best_model = ndcg, params, r

    te_s = te.copy(); te_s["score"] = best_model.predict(Xte)
    from sklearn.metrics import average_precision_score, roc_auc_score
    result = {
        "roc_auc": float(roc_auc_score(te[config.TARGET], te_s["score"])),
        "pr_auc": float(average_precision_score(te[config.TARGET], te_s["score"])),
        "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
        **metrics.ranking_report(te_s, "customer_id", "score", config.TARGET, (1, 3, 5)),
    }
    print(f"    best val NDCG@3={best_ndcg:.4f}  params={best_params}")
    return result, {"prep": prep, "ranker": best_model}, best_params


# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(config.MODEL_TABLE)
    tr, va, te = split_by_customer(df)
    print(f"Split by customer -> train {len(tr):,} / val {len(va):,} / test {len(te):,}\n")

    default = json.load(open(config.REPORTS_DIR / "metrics.json"))

    xgb_dist = {
        "clf__n_estimators": [200, 300, 400, 600],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_lambda": [1.0, 2.0, 5.0],
    }
    lgbm_dist = {
        "clf__n_estimators": [300, 500, 700],
        "clf__num_leaves": [15, 31, 63],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__min_child_samples": [20, 50, 100],
        "clf__reg_lambda": [0.0, 1.0, 5.0],
    }

    tuned, tuned_models, best_params = {}, {}, {}
    print("Tuning models ...")
    tuned["XGBoost"], tuned_models["XGBoost"], best_params["XGBoost"] = tune_classifier(
        "XGBoost",
        XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=SEED),
        xgb_dist, tr, te)
    tuned["LightGBM"], tuned_models["LightGBM"], best_params["LightGBM"] = tune_classifier(
        "LightGBM",
        LGBMClassifier(n_jobs=1, random_state=SEED, verbose=-1),
        lgbm_dist, tr, te)
    tuned["LightGBM_Ranker"], tuned_models["LightGBM_Ranker"], best_params["LightGBM_Ranker"] = \
        tune_ranker(tr, va, te)

    # ---- save ----
    json.dump(tuned, open(config.REPORTS_DIR / "metrics_tuned.json", "w"), indent=2)
    json.dump(best_params, open(config.REPORTS_DIR / "best_params.json", "w"),
              indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))

    # ---- before/after comparison ----
    print("\n================  DEFAULT vs TUNED (test set)  ================")
    cols = ["roc_auc", "pr_auc", "precision@3", "map@3", "ndcg@3", "ndcg@5"]
    rows = []
    for name in ["XGBoost", "LightGBM", "LightGBM_Ranker"]:
        for tag, src in [("default", default[name]), ("tuned", tuned[name])]:
            rows.append({"model": name, "version": tag, **{c: src.get(c, float("nan")) for c in cols}})
    comp = pd.DataFrame(rows).set_index(["model", "version"])
    print(comp.round(4).to_string())

    # ---- verdict ----
    print("\n================  DID TUNING HELP?  ================")
    improved_any = False
    for name in ["XGBoost", "LightGBM", "LightGBM_Ranker"]:
        key = "ndcg@3" if name == "LightGBM_Ranker" else "pr_auc"
        d, t = default[name][key], tuned[name][key]
        delta = t - d
        verdict = "IMPROVED" if delta > 0.0005 else ("~same" if abs(delta) <= 0.0005 else "WORSE")
        improved_any |= delta > 0.0005
        print(f"  {name:18s} {key}: {d:.4f} -> {t:.4f}  (Δ {delta:+.4f})  {verdict}")

    # ---- promote tuned serving model only if it actually beats current best ----
    bundle = joblib.load(config.MODELS_DIR / "best_model.pkl")
    cur_name = bundle["name"]
    cur_prauc = default[cur_name]["pr_auc"]
    best_tuned = max(["XGBoost", "LightGBM"], key=lambda n: tuned[n]["pr_auc"])
    if tuned[best_tuned]["pr_auc"] > cur_prauc + 0.0005:
        joblib.dump({"name": f"{best_tuned}_tuned", "model": tuned_models[best_tuned],
                     "features": FEATURES}, config.MODELS_DIR / "best_model.pkl")
        print(f"\nServing model UPDATED -> {best_tuned}_tuned "
              f"(PR-AUC {cur_prauc:.4f} -> {tuned[best_tuned]['pr_auc']:.4f})")
    else:
        print(f"\nServing model kept as-is ({cur_name}); tuning did not beat it "
              f"by a meaningful margin.")

    if not improved_any:
        print("\nTakeaway: the default settings were already well-chosen; tuning gives "
              "only marginal movement here. Reported honestly rather than cherry-picked.")


if __name__ == "__main__":
    main()
