"""
download_data.py
================
Download the three real Starbucks Rewards offer JSON files into data/raw/.

The dataset is the Udacity / Starbucks "Rewards Offer" capstone dataset, mirrored
on several public GitHub repos. We try a few mirrors for resilience.

Run:  python -m src.download_data
"""
from __future__ import annotations

import urllib.request

from . import config

FILES = ["portfolio.json", "profile.json", "transcript.json"]
MIRRORS = [
    "https://raw.githubusercontent.com/reachanihere/Starbucks-Capstone/master/data",
    "https://raw.githubusercontent.com/susmithagudapati/Starbucks-Capstone-Challenge/master/data",
    "https://raw.githubusercontent.com/dkhundley/starbucks-ml-capstone/master/data",
]


def _download_one(fname: str) -> bool:
    dest = config.DATA_RAW / fname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  {fname} already present, skipping")
        return True
    for base in MIRRORS:
        url = f"{base}/{fname}"
        try:
            print(f"  fetching {fname} from {base.split('/')[3]} ...")
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 0:
                print(f"    -> saved {dest.stat().st_size:,} bytes")
                return True
        except Exception as e:
            print(f"    failed: {e}")
    return False


def main():
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading real Starbucks Rewards offer dataset ...")
    ok = all(_download_one(f) for f in FILES)
    if ok:
        print("\nAll files ready in data/raw/. Next: python -m src.data_prep")
    else:
        raise SystemExit("Could not download all files. Check your connection or "
                         "see the README for manual download instructions.")


if __name__ == "__main__":
    main()
