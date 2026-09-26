"""
Sanity check (NOT training/validation): this dataset has no speed column,
so it can't calibrate or validate our speed-fuel relationship directly.
What it CAN do: check whether our physics-anchored model's fuel-per-
nautical-mile is in a plausible real-world ballpark for each vessel class,
and whether real fuel-per-distance actually increases with worse weather
(the direction our weather_penalty() assumes).

This is a plausibility check, not a training set. Units of `distance` and
`fuel_consumption` in the source file are UNKNOWN (nm? km? tons? liters?)
- this script prints raw ranges so a human can judge plausibility rather
than silently assuming units that might be wrong.

Usage: python3 src/data/kaggle_sanity_check.py --csv data/raw/ship_fuel_efficiency.csv
"""
from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import VESSEL_CLASSES, DESIGN_SPEED_KN
from src.models.predict_api import FuelPredictor

# Best-effort mapping - EXTEND THIS once you see the real ship_type values
# printed below. Anything unmapped is dropped and reported, never silently
# guessed.
SHIP_TYPE_MAP = {
    "container ship": "container", "container": "container",
    "bulk carrier": "bulk_carrier", "bulk": "bulk_carrier",
    "tanker": "tanker", "oil tanker": "tanker", "crude tanker": "tanker", "tanker ship": "tanker",
    "cargo ship": "general_cargo", "general cargo": "general_cargo", "cargo": "general_cargo",
}


def run(csv_path: str):
    df = pd.read_csv(csv_path)

    print("=== Raw value inventory (so unmapped categories can be added above) ===")
    print(f"ship_type values: {sorted(df['ship_type'].astype(str).unique().tolist())}")
    print(f"weather_conditions values: {sorted(df['weather_conditions'].astype(str).unique().tolist())}")
    print(f"fuel_type values: {sorted(df['fuel_type'].astype(str).unique().tolist())}")
    print(f"\ndistance range: {df['distance'].min():.1f} - {df['distance'].max():.1f} "
          f"(units unknown - could be nm, km, or miles)")
    print(f"fuel_consumption range: {df['fuel_consumption'].min():.2f} - {df['fuel_consumption'].max():.2f} "
          f"(units unknown - could be tons, liters, or something else)")

    df["vessel_class"] = df["ship_type"].astype(str).str.lower().str.strip().map(SHIP_TYPE_MAP)
    unmapped = df["vessel_class"].isna().sum()
    if unmapped > 0:
        print(f"\nWARNING: {unmapped}/{len(df)} rows have an unmapped ship_type - "
              f"dropped from this check. Extend SHIP_TYPE_MAP if these should count.")
    df = df[df["vessel_class"].notna()]

    df["fuel_per_distance"] = df["fuel_consumption"] / df["distance"].replace(0, np.nan)

    print("\n=== Real fuel-per-distance by vessel class ===")
    predictor = FuelPredictor.from_synthetic()
    for vc in VESSEL_CLASSES:
        sub = df[df["vessel_class"] == vc]
        if len(sub) == 0:
            print(f"{vc:15s}  no rows in this dataset")
            continue
        real_mean = sub["fuel_per_distance"].mean()

        # our model's implied fuel-per-distance at this class's DESIGN speed
        # (the closest sane comparison point available, since real speed is unknown)
        pred = predictor.predict_fuel(vc, speed_kn=DESIGN_SPEED_KN[vc], draft_ratio=0.85)
        model_per_nm = pred.fuel_t_per_day / (DESIGN_SPEED_KN[vc] * 24)

        ratio = real_mean / model_per_nm if model_per_nm else float("nan")
        print(f"{vc:15s}  real={real_mean:.4f}/unit  model(at design speed)={model_per_nm:.4f} t/nm  "
              f"ratio={ratio:.2f}x  (n={len(sub)})")

    print("\n=== Real fuel-per-distance by weather condition (checking direction only) ===")
    weather_means = df.groupby("weather_conditions")["fuel_per_distance"].agg(["mean", "count"])
    print(weather_means.sort_values("mean"))
    print("\nExpectation: worse weather categories should show HIGHER mean fuel-per-distance, "
          "matching weather_penalty()'s assumption in physics.py. Eyeball whether that holds "
          "given the category labels above.")

    print("\n=== Interpretation ===")
    print("A ratio far from 1.0x (say, >3x or <0.3x) likely means a UNIT mismatch (e.g. distance "
          "in km vs our nm assumption), not that the physics model is wrong by that factor - "
          "this is a plausibility check, not a calibration fit. Report the ratios back and we "
          "can figure out which unit conversion (if any) reconciles them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    run(args.csv)
