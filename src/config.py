"""Central configuration: paths, constants, and a couple of business assumptions.

Keeping these in one place makes the project easy to reason about and tweak.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw files (real Starbucks Rewards offer dataset)
PORTFOLIO_JSON = DATA_RAW / "portfolio.json"
PROFILE_JSON = DATA_RAW / "profile.json"
TRANSCRIPT_JSON = DATA_RAW / "transcript.json"

# Main modelling table produced by data_prep.py
MODEL_TABLE = DATA_PROCESSED / "model_table.csv"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET = "clicked"            # offer viewed within its validity window
SECONDARY_TARGET = "accepted"  # offer completed within its validity window
TOP_K = 3                    # we typically surface the top-3 offers to a customer

# Placeholder demographic values in the real data (missing customers)
MISSING_AGE = 118

# ---------------------------------------------------------------------------
# Business assumptions used only for *expected campaign value* (clearly synthetic,
# used for prioritisation, not claimed as ground truth). See README.
# ---------------------------------------------------------------------------
# Rough cost of sending one offer / contact (notification, processing).
CONTACT_COST = 0.15
# Assumed incremental margin if a customer engages with (views & acts on) an offer.
ENGAGED_MARGIN = 4.0
