"""
tune_optuna.py
==============
Optuna (TPE + pruning) hyperparameter optimisation for XGBoost, LightGBM and the
LightGBM Ranker — a smarter, wider search than the RandomizedSearchCV in tune.py.

Same rigor as the rest of the project:
- Objective for classifiers = mean **PR-AUC across GroupKFold folds** (groups =
  customer_id), so no customer leaks between CV-train and CV-validation.
- Objective for the Ranker  = **validation NDCG@3**.
- Best models are scored on the **same held-out test set** for an honest before/after
  vs both the defaults (metrics.json) and the random search (metrics_tuned.json).
- TPE sampler proposes promising configs; MedianPruner kills weak trials early
  (we report PR-AUC fold-by-fold so a bad config can be stopped mid-CV).

Run:  python -m src.tune_optuna
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, metrics
from .train import FEATURES, make_preprocessor, split_by_customer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = config.RANDOM_STATE
CV_SPLITS = 3
N_TRIALS_CLF = 60        # classifier trials per model
N_TRIALS_RANK = 50       # ranker trials
TIMEOUT_CLF = 240        # seconds, safety cap per classifier study
TIMEOUT_RANK = 200


# ---------------------------------------------------------------------------
# Grouped-CV PR-AUC with per-fold pruning hook
# ---------------------------------------------------------------------------
def _cv_prauc(make_clf, tr, trial=None):
    gkf = GroupKFold(n_splits=CV_SPLITS)
    groups = tr["customer_id"].values
    X, y = tr[FEATURES], tr[config.TARGET].values
    scores = []
    for fold, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        prep = make_preprocessor(False)
        Xti = prep.fit_transform(X.iloc[ti]); Xvi = prep.transform(X.iloc[vi])
        clf = make_clf()
        clf.fit(Xti, y[ti])
        p = clf.predict_proba(Xvi)[:, 1]
        scores.append(average_precision_score(y[vi], p))
        if trial is not None:
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def xgb_objective(trial, tr):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    )
    return _cv_prauc(lambda: XGBClassifier(eval_metric="logloss", n_jobs=1,
                                           random_state=SEED, **params), tr, trial)


def lgbm_objective(trial, tr):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 150),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )
    return _cv_prauc(lambda: LGBMClassifier(n_jobs=1, random_state=SEED,
                                            verbose=-1, **params), tr, trial)


def run_classifier_study(name, objective, tr, te, n_trials, timeout):
    print(f"  Optuna study: {name} (<= {n_trials} trials, {timeout}s cap) ...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(lambda t: objective(t, tr), n_trials=n_trials, timeout=timeout)
    best = study.best_params
    print(f"    best CV PR-AUC={study.best_value:.4f}  ({len(study.trials)} trials run)")

    # refit best on full train as a Pipeline (so it stays compatible with
    # recommend.py / the dashboard, which call .predict_proba on the saved model).
    Maker = XGBClassifier if name == "XGBoost" else LGBMClassifier
    extra = dict(eval_metric="logloss") if name == "XGBoost" else dict(verbose=-1)
    clf = Maker(n_jobs=1, random_state=SEED, **extra, **best)
    pipe = Pipeline([("prep", make_preprocessor(False)), ("clf", clf)])
    pipe.fit(tr[FEATURES], tr[config.TARGET].values)
    prob = pipe.predict_proba(te[FEATURES])[:, 1]
    te_s = te.copy(); te_s["score"] = prob
    result = {
        **metrics.classification_report_dict(te[config.TARGET], prob),
        **metrics.ranking_report(te_s, "customer_id", "score", config.TARGET, (1, 3, 5)),
    }
    return result, pipe, best, study


def run_ranker_study(tr, va, te, n_trials, timeout):
    prep = make_preprocessor(False)
    tr_s, va_s = tr.sort_values("customer_id"), va.sort_values("customer_id")
    Xtr = prep.fit_transform(tr_s[FEATURES]); Xva = prep.transform(va_s[FEATURES])
    Xte = prep.transform(te[FEATURES])
    gtr = tr_s.groupby("customer_id").size().values
    ytr = tr_s[config.TARGET].values

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 150),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        r = LGBMRanker(objective="lambdarank", n_jobs=1, random_state=SEED,
                       verbose=-1, **params)
        r.fit(Xtr, ytr, group=gtr)
        v = va_s.copy(); v["score"] = r.predict(Xva)
        return metrics.ranking_report(v, "customer_id", "score", config.TARGET, (3,))["ndcg@3"]

    print(f"  Optuna study: LightGBM_Ranker (<= {n_trials} trials, {timeout}s cap) ...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = study.best_params
    print(f"    best val NDCG@3={study.best_value:.4f}  ({len(study.trials)} trials run)")

    r = LGBMRanker(objective="lambdarank", n_jobs=1, random_state=SEED, verbose=-1, **best)
    r.fit(Xtr, ytr, group=gtr)
    te_s = te.copy(); te_s["score"] = r.predict(Xte)
    result = {
        "roc_auc": float(roc_auc_score(te[config.TARGET], te_s["score"])),
        "pr_auc": float(average_precision_score(te[config.TARGET], te_s["score"])),
        "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
        **metrics.ranking_report(te_s, "customer_id", "score", config.TARGET, (1, 3, 5)),
    }
    return result, {"prep": prep, "ranker": r}, best, study


def _save_history_fig(study, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        ax.figure.set_size_inches(7, 4)
        ax.set_title(f"Optuna optimisation history — {name}")
        ax.figure.tight_layout()
        ax.figure.savefig(config.FIGURES_DIR / f"optuna_history_{name}.png", dpi=120)
        plt.close(ax.figure)
    except Exception as e:
        print(f"    (skipped history fig for {name}: {e})")


# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(config.MODEL_TABLE)
    tr, va, te = split_by_customer(df)
    print(f"Split by customer -> train {len(tr):,} / val {len(va):,} / test {len(te):,}\n")

    default = json.load(open(config.REPORTS_DIR / "metrics.json"))
    rand = json.load(open(config.REPORTS_DIR / "metrics_tuned.json")) \
        if (config.REPORTS_DIR / "metrics_tuned.json").exists() else {}

    opt, opt_models, opt_params = {}, {}, {}
    print("Running Optuna studies ...")
    opt["XGBoost"], opt_models["XGBoost"], opt_params["XGBoost"], s1 = \
        run_classifier_study("XGBoost", xgb_objective, tr, te, N_TRIALS_CLF, TIMEOUT_CLF)
    opt["LightGBM"], opt_models["LightGBM"], opt_params["LightGBM"], s2 = \
        run_classifier_study("LightGBM", lgbm_objective, tr, te, N_TRIALS_CLF, TIMEOUT_CLF)
    opt["LightGBM_Ranker"], opt_models["LightGBM_Ranker"], opt_params["LightGBM_Ranker"], s3 = \
        run_ranker_study(tr, va, te, N_TRIALS_RANK, TIMEOUT_RANK)

    for st, nm in [(s1, "XGBoost"), (s2, "LightGBM"), (s3, "LightGBM_Ranker")]:
        _save_history_fig(st, nm)

    json.dump(opt, open(config.REPORTS_DIR / "metrics_optuna.json", "w"), indent=2)
    json.dump(opt_params, open(config.REPORTS_DIR / "best_params_optuna.json", "w"),
              indent=2, default=float)

    # ---- before/after across all three tuning approaches ----
    print("\n============  DEFAULT vs RANDOM-SEARCH vs OPTUNA (test set)  ============")
    cols = ["roc_auc", "pr_auc", "map@3", "ndcg@3", "ndcg@5"]
    rows = []
    for name in ["XGBoost", "LightGBM", "LightGBM_Ranker"]:
        rows.append({"model": name, "tuning": "default", **{c: default[name].get(c, np.nan) for c in cols}})
        if name in rand:
            rows.append({"model": name, "tuning": "random", **{c: rand[name].get(c, np.nan) for c in cols}})
        rows.append({"model": name, "tuning": "optuna", **{c: opt[name].get(c, np.nan) for c in cols}})
    comp = pd.DataFrame(rows).set_index(["model", "tuning"])
    print(comp.round(4).to_string())

    # ---- verdict ----
    print("\n============  DID OPTUNA HELP (vs default)?  ============")
    for name in ["XGBoost", "LightGBM", "LightGBM_Ranker"]:
        key = "ndcg@3" if name == "LightGBM_Ranker" else "pr_auc"
        d, o = default[name][key], opt[name][key]
        verdict = "IMPROVED" if o - d > 0.0005 else ("~same" if abs(o - d) <= 0.0005 else "WORSE")
        print(f"  {name:18s} {key}: {d:.4f} -> {o:.4f}  (Δ {o - d:+.4f})  {verdict}")

    # ---- promote serving model if Optuna beats current best ----
    bundle = joblib.load(config.MODELS_DIR / "best_model.pkl")
    cur_prauc = max(default["XGBoost"]["pr_auc"], default["LightGBM"]["pr_auc"],
                    (rand.get("XGBoost", {}).get("pr_auc", 0)),
                    (rand.get("LightGBM", {}).get("pr_auc", 0)))
    best_opt = max(["XGBoost", "LightGBM"], key=lambda n: opt[n]["pr_auc"])
    if opt[best_opt]["pr_auc"] > cur_prauc + 0.0005:
        joblib.dump({"name": f"{best_opt}_optuna", "model": opt_models[best_opt],
                     "features": FEATURES}, config.MODELS_DIR / "best_model.pkl")
        print(f"\nServing model UPDATED -> {best_opt}_optuna "
              f"(best PR-AUC so far {cur_prauc:.4f} -> {opt[best_opt]['pr_auc']:.4f})")
    else:
        print(f"\nServing model kept ({bundle['name']}); Optuna did not beat the "
              f"current best PR-AUC ({cur_prauc:.4f}) by a meaningful margin.")


if __name__ == "__main__":
    main()
