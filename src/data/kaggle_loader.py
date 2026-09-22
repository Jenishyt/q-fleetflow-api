"""
Real Kaggle data loader (plan Sec 5): ship fuel consumption / CO2 dataset.

IMPORTANT - honest limitation: this was written and syntax-checked in a
sandboxed environment with NO network access to kaggle.com, so it has
NEVER been run against the real file. Column names below are based on
the commonly-published schema for this dataset family as of the plan's
writing, but Kaggle dataset schemas do change - if load_kaggle_csv()
throws a KeyError, open the CSV, check its actual column headers, and
fix COLUMN_MAP below. That's a 5-minute fix, not a redesign.

Usage (once you've downloaded the CSV via Kaggle locally):
    python3 src/data/kaggle_loader.py --csv data/raw/ship_fuel_efficiency.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import VESSEL_CLASSES

# Map from the dataset's likely raw column names -> our canonical schema.
# VERIFY THIS against the actual downloaded CSV's header row before trusting
# any numbers downstream - this is the single most likely thing to need a
# fix, and it's a cheap fix once you can see the real column names.
COLUMN_MAP = {
    "ship_type": "vessel_class_raw",
    "distance": "distance_nm",
    "speed": "speed_kn",
    "fuel_type": "fuel_type_raw",
    "fuel_consumption": "fuel_t_per_day_raw",  # VERIFY UNITS - could be total for the voyage, not per-day
    "CO2_emissions": "co2_t",
    "weather_conditions": "weather_raw",
}

# Loose mapping from likely raw category strings to our 4-class taxonomy.
# The real dataset's ship_type values are unknown until we can see the
# file - this is a best-effort starting point, not a verified mapping.
VESSEL_CLASS_MAP = {
    "container ship": "container", "container": "container",
    "bulk carrier": "bulk_carrier", "bulk": "bulk_carrier",
    "tanker": "tanker", "oil tanker": "tanker",
    "cargo ship": "general_cargo", "general cargo": "general_cargo",
}


def load_kaggle_csv(path: str) -> pd.DataFrame:
    """Load the raw CSV and rename to canonical-ish column names. Does NOT
    yet produce the full canonical schema (draft_ratio, wind_kn, wave_hs_m
    aren't in this dataset family and would need a secondary weather join,
    per the plan's optional stretch goal) - that's the next step once this
    function is verified against the real file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from Kaggle "
            f"('Ship Fuel Consumption & CO2 Emissions' or similar) and place "
            f"the CSV there. This loader cannot fetch it for you - it was "
            f"written without network access to kaggle.com."
        )

    df = pd.read_csv(path)
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns not found in {path}: {missing}. "
            f"Actual columns are: {df.columns.tolist()}. "
            f"Update COLUMN_MAP in this file to match - the real schema "
            f"could not be verified without network access."
        )

    df = df.rename(columns=COLUMN_MAP)
    return df


def to_canonical_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort mapping to the canonical schema used throughout the
    rest of the pipeline (synth_generator.py, gbm.py). UNVERIFIED against
    real data - run this, inspect the output, and expect to iterate."""
    out = pd.DataFrame()
    out["voyage_id"] = [f"KAG{i:06d}" for i in range(len(df))]

    out["vessel_class"] = (
        df["vessel_class_raw"].astype(str).str.lower().str.strip()
        .map(VESSEL_CLASS_MAP)
    )
    unmapped = out["vessel_class"].isna().sum()
    if unmapped > 0:
        print(f"WARNING: {unmapped}/{len(df)} rows have a vessel_class_raw value not in "
              f"VESSEL_CLASS_MAP - inspect df['vessel_class_raw'].unique() and extend the "
              f"map. Dropping unmapped rows for now.")
        out = out[out["vessel_class"].notna()]
        df = df.loc[out.index]

    out["distance_nm"] = df["distance_nm"]
    out["speed_kn"] = df["speed_kn"]
    out["fuel_type"] = df["fuel_type_raw"]
    out["fuel_t_per_day"] = df["fuel_t_per_day_raw"]  # VERIFY: units/period assumption above
    out["co2_t"] = df["co2_t"]

    # NOT in this dataset family - filled with class-average placeholders
    # until a real draft/weather source is joined in (plan's optional
    # weather stretch goal). Flagged loudly, not silently defaulted.
    out["draft_ratio"] = 0.75
    out["wind_kn"] = 0.0
    out["wave_hs_m"] = 0.0
    out["temp_c"] = 25.0
    print("NOTE: draft_ratio, wind_kn, wave_hs_m, temp_c are NOT present in this "
          "dataset family and are filled with fixed placeholders. Training on this "
          "as-is means the GBM residual layer has no real weather signal to learn "
          "from - re-run synth_generator.py's weather grid to compensate, or find a "
          "weather join source, before trusting the quantile heads' coverage.")

    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning rules (plan Sec 5.3): drop implausible rows, 3xIQR outlier
    filter (not 1.5x - maritime data is heavy-tailed), dedupe."""
    before = len(df)
    df = df[(df["fuel_t_per_day"] > 0) & (df["fuel_t_per_day"] <= 100)]

    q1, q3 = df["fuel_t_per_day"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    df = df[(df["fuel_t_per_day"] >= lower) & (df["fuel_t_per_day"] <= upper)]

    df = df.drop_duplicates(subset=["voyage_id"])
    after = len(df)
    print(f"Cleaning: {before} -> {after} rows ({before - after} dropped)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to the downloaded Kaggle CSV")
    parser.add_argument("--out", default="data/processed/kaggle_clean.parquet")
    args = parser.parse_args()

    raw = load_kaggle_csv(args.csv)
    canonical = to_canonical_schema(raw)
    cleaned = clean(canonical)

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cleaned.to_parquet(out_path, index=False)
    print(f"\nWrote {len(cleaned)} rows -> {out_path}")
    print(f"Vessel classes: {cleaned['vessel_class'].unique().tolist()}")
    print(f"\nNext: re-run src/models/gbm.py pointed at this file instead of "
          f"synthetic.parquet, and check where the real MAPE lands.")
