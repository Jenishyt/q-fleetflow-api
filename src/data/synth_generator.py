"""
Deterministic synthetic fuel-consumption generator (seeded, reproducible).

This is the insurance policy from the plan's Sec 5.4: if the Kaggle dataset
turns out too dirty, too small, or missing key fields, we train on this
instead. It's built from the SAME physics family the prediction model uses
(admiralty prior + weather penalty), so it will pass the MAPE gate by
construction - and that fact is declared openly, never presented as real
measurements.

Grid: 4 vessel classes x speeds 8-22kn x draft 0.3-1.0 x wind 0-40kn x
wave 0-4m, with a small amount of injected noise so it's not a perfectly
learnable deterministic function (a real GBM needs some residual signal
to actually demonstrate its value over physics-only).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import f_phys, weather_penalty, VESSEL_CLASSES, DEFAULT_C_CLASS

SEED = 42
FUEL_TYPES = ["MDO", "VLSFO", "HFO", "LNG"]


def generate(
    n_speed=8,
    n_draft=5,
    n_wind=5,
    n_wave=4,
    noise_std=0.04,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate the canonical schema dataframe over a grid of operating
    conditions, one row per (vessel_class, speed, draft, wind, wave) combo.
    """
    rng = np.random.default_rng(seed)

    speeds = np.linspace(8, 22, n_speed)
    drafts = np.linspace(0.3, 1.0, n_draft)
    winds = np.linspace(0, 40, n_wind)
    waves = np.linspace(0, 4, n_wave)

    rows = []
    voyage_id = 0
    for vc in VESSEL_CLASSES:
        for s in speeds:
            for d in drafts:
                for w in winds:
                    for h in waves:
                        base = f_phys(s, d, vc, DEFAULT_C_CLASS)
                        penalty = weather_penalty(w, h)
                        # multiplicative log-normal noise -> gives the GBM
                        # residual layer something real to learn later
                        noise = np.exp(rng.normal(0, noise_std))
                        fuel = base * penalty * noise

                        rows.append({
                            "voyage_id": f"SYN{voyage_id:06d}",
                            "vessel_class": vc,
                            "distance_nm": float(rng.uniform(200, 2000)),
                            "speed_kn": float(s),
                            "draft_ratio": float(d),
                            "fuel_type": rng.choice(FUEL_TYPES),
                            "fuel_t_per_day": float(fuel),
                            "co2_t": float(fuel * 3.15),  # rough MDO CO2 factor, sanity-check only
                            "wind_kn": float(w),
                            "wave_hs_m": float(h),
                            "temp_c": float(rng.uniform(10, 32)),
                        })
                        voyage_id += 1

    df = pd.DataFrame(rows)
    return df


def save(df: pd.DataFrame, path: str = "data/processed/synthetic.parquet") -> None:
    out = Path(__file__).resolve().parents[2] / path
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    print(f"Vessel classes: {df['vessel_class'].unique().tolist()}")
    print(f"Fuel range: {df['fuel_t_per_day'].min():.2f} - {df['fuel_t_per_day'].max():.2f} t/day")
    save(df)
