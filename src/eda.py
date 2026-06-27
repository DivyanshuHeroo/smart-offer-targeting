"""
eda.py
======
Exploratory Data Analysis on the modelling table. Saves a handful of clean,
self-explanatory figures into reports/figures/ that are reused in the README and
the dashboard.

Run:  python -m src.eda
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path.name}")


def main():
    df = pd.read_csv(config.MODEL_TABLE)
    print(f"Loaded {len(df):,} offer instances for {df.customer_id.nunique():,} customers")
    print(f"Overall click (viewed) rate: {df['clicked'].mean():.3f}\n")

    # 1. Click rate by offer type
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("offer_type")["clicked"].mean().sort_values().plot(
        kind="barh", ax=ax, color="#2a9d8f")
    ax.set_title("Click (view) rate by offer type")
    ax.set_xlabel("click rate")
    _save(fig, "eda_click_by_offer_type.png")

    # 2. Click rate by income segment & age group
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.groupby("income_segment")["clicked"].mean().reindex(
        ["low", "mid", "high", "premium"]).plot(kind="bar", ax=axes[0], color="#e76f51")
    axes[0].set_title("Click rate by income segment"); axes[0].set_ylabel("click rate")
    df.groupby("age_group")["clicked"].mean().reindex(
        ["18-30", "31-45", "46-60", "60+"]).plot(kind="bar", ax=axes[1], color="#f4a261")
    axes[1].set_title("Click rate by age group")
    for a in axes:
        a.tick_params(axis="x", rotation=0)
    _save(fig, "eda_click_by_segment.png")

    # 3. Offer popularity (how often each offer is received)
    fig, ax = plt.subplots(figsize=(7, 4))
    df["offer_type"].value_counts().plot(kind="bar", ax=ax, color="#264653")
    ax.set_title("Volume of offers received, by type")
    ax.set_ylabel("# offer instances"); ax.tick_params(axis="x", rotation=0)
    _save(fig, "eda_offer_volume.png")

    # 4. Spend distribution (clipped for readability)
    fig, ax = plt.subplots(figsize=(7, 4))
    df["avg_spend"].clip(upper=df["avg_spend"].quantile(0.99)).hist(
        bins=40, ax=ax, color="#8ab17d")
    ax.set_title("Customer average spend before offer (99th-pct clipped)")
    ax.set_xlabel("avg spend"); ax.set_ylabel("# customers")
    _save(fig, "eda_spend_distribution.png")

    # 5. Click rate vs prior view rate (does past behaviour predict future?)
    fig, ax = plt.subplots(figsize=(7, 4))
    bucket = pd.cut(df["prior_view_rate"], bins=[-0.01, 0.0001, 0.5, 0.99, 1.01],
                    labels=["0", "0-0.5", "0.5-1", "1.0"])
    df.groupby(bucket)["clicked"].mean().plot(kind="bar", ax=ax, color="#287271")
    ax.set_title("Current click rate vs prior offer-view rate")
    ax.set_xlabel("prior view rate bucket"); ax.set_ylabel("click rate")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, "eda_prior_vs_current.png")

    print("\nEDA figures written to reports/figures/")


if __name__ == "__main__":
    main()
