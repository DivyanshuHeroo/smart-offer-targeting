"""
metrics.py
==========
Ranking metrics for offer recommendation, plus thin wrappers around sklearn
classification metrics.

Why ranking metrics (and not just accuracy/AUC)?
--------------------------------------------------
The business does NOT show every offer to every customer. It shows the TOP FEW.
So what actually matters is: "of the handful of offers we rank highest for a
customer, how many are ones they really engage with, and are the good ones near
the top?" That is exactly what Precision@K, Recall@K, MAP@K and NDCG@K measure.

Grouping
--------
A "query" here = one customer. The candidate items = the offers that customer
received. The relevance label = `clicked` (did they engage). We rank the
candidates by the model's predicted probability and score each customer, then
average across customers.

Plain-language definitions
--------------------------
- Precision@K : of the top-K offers we recommended, what fraction were relevant.
- Recall@K    : of all the relevant offers, what fraction appeared in the top-K.
- MAP@K       : Mean Average Precision. Rewards putting relevant offers higher up
                (an average of precision values measured at each relevant hit).
- NDCG@K      : Normalised Discounted Cumulative Gain. Like MAP but uses a smooth
                log discount for position; 1.0 = perfect ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Per-query ranking metrics
# ---------------------------------------------------------------------------
def _precision_at_k(rel_sorted: np.ndarray, k: int) -> float:
    topk = rel_sorted[:k]
    return topk.sum() / k if k > 0 else 0.0


def _recall_at_k(rel_sorted: np.ndarray, k: int) -> float:
    total_rel = rel_sorted.sum()
    if total_rel == 0:
        return np.nan  # undefined -> ignored in the average
    return rel_sorted[:k].sum() / total_rel


def _average_precision_at_k(rel_sorted: np.ndarray, k: int) -> float:
    rel_k = rel_sorted[:k]
    total_rel = rel_sorted.sum()
    if total_rel == 0:
        return np.nan
    hits, score = 0, 0.0
    for i, r in enumerate(rel_k, start=1):
        if r:
            hits += 1
            score += hits / i           # precision at this hit position
    return score / min(total_rel, k)


def _dcg(rel: np.ndarray) -> float:
    # binary gains with log2 position discount
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float((rel * discounts).sum())


def _ndcg_at_k(rel_sorted: np.ndarray, k: int) -> float:
    if rel_sorted.sum() == 0:
        return np.nan
    actual = _dcg(rel_sorted[:k])
    ideal = _dcg(np.sort(rel_sorted)[::-1][:k])
    return actual / ideal if ideal > 0 else np.nan


def ranking_report(df: pd.DataFrame, group_col: str, score_col: str,
                   label_col: str, ks=(1, 3, 5)) -> dict:
    """Average the per-customer ranking metrics across all customers.

    Only customers with >= 2 candidate offers are scored (ranking a single item
    is meaningless).
    """
    out = {f"{m}@{k}": [] for k in ks for m in ("precision", "recall", "map", "ndcg")}

    for _, g in df.groupby(group_col):
        if len(g) < 2:
            continue
        order = np.argsort(-g[score_col].values)        # high score first
        rel_sorted = g[label_col].values[order].astype(float)
        for k in ks:
            out[f"precision@{k}"].append(_precision_at_k(rel_sorted, k))
            out[f"recall@{k}"].append(_recall_at_k(rel_sorted, k))
            out[f"map@{k}"].append(_average_precision_at_k(rel_sorted, k))
            out[f"ndcg@{k}"].append(_ndcg_at_k(rel_sorted, k))

    return {key: float(np.nanmean(vals)) if len(vals) else float("nan")
            for key, vals in out.items()}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def classification_report_dict(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
