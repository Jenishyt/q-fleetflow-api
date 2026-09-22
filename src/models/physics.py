"""
Physics-anchored fuel prediction prior (admiralty cubic law).

F_phys = C_class * displacement^(2/3) * speed^3 / 1e6

This is a century-old naval architecture approximation: fuel burn scales
roughly with speed cubed and displacement to the 2/3 power. It guarantees
sane extrapolation (a 30kn query never returns negative or wild fuel
numbers) - the GBM residual layer (gbm.py, built next) only has to learn
the DEVIATION from this physics floor, which is a much easier target than
raw fuel consumption.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Deadweight tonnage lookup per vessel class (rough class-average, tons).
# These are placeholders - swap for class-society reference tables when
# real fleet data is available; documented here as an explicit assumption.
DWT_LOOKUP = {
    "container": 40000.0,
    "bulk_carrier": 55000.0,
    "tanker": 60000.0,
    "general_cargo": 15000.0,
}

VESSEL_CLASSES = list(DWT_LOOKUP.keys())

# Design (100%) speed per vessel class, knots - used to derive the
# optimizer's eco/design/full speed buckets. Placeholder class-average
# figures, documented as an assumption pending real fleet specs.
DESIGN_SPEED_KN = {
    "container": 22.0,
    "bulk_carrier": 14.5,
    "tanker": 15.0,
    "general_cargo": 16.0,
}

# Starting calibration constants (C_class) before fitting on real/synthetic
# data. Order of magnitude only - fit_calibration() below replaces these.
DEFAULT_C_CLASS = {
    "container": 1.9,
    "bulk_carrier": 1.6,
    "tanker": 1.7,
    "general_cargo": 2.2,
}


def displacement(draft_ratio: np.ndarray | float, vessel_class: str) -> np.ndarray | float:
    """Displacement proxy: loaded fraction x class deadweight tonnage."""
    dwt = DWT_LOOKUP[vessel_class]
    return draft_ratio * dwt


def f_phys(
    speed_kn: np.ndarray | float,
    draft_ratio: np.ndarray | float,
    vessel_class: str,
    c_class: dict[str, float] | None = None,
) -> np.ndarray | float:
    """Admiralty cubic law fuel prior, tons/day.

    F_phys = C_class * Delta^(2/3) * s^3 / 1e6
    """
    c_map = c_class or DEFAULT_C_CLASS
    c = c_map[vessel_class]
    delta = displacement(draft_ratio, vessel_class)
    return c * np.power(delta, 2.0 / 3.0) * np.power(speed_kn, 3) / 1e6


def weather_penalty(wind_kn: np.ndarray | float, wave_hs_m: np.ndarray | float) -> np.ndarray | float:
    """Multiplicative penalty for wind/wave resistance, used by the synthetic
    generator to inject realistic non-physics-only variation. Simple,
    monotonic, and bounded - NOT a substitute for a real hydrodynamic model.
    """
    wind_term = 1.0 + 0.004 * np.asarray(wind_kn)
    wave_term = 1.0 + 0.06 * np.asarray(wave_hs_m)
    return wind_term * wave_term


def fit_calibration(df: pd.DataFrame) -> dict[str, float]:
    """Fit one C_class constant per vessel_class by least squares on
    observed fuel consumption, using the closed-form solution for a
    single-parameter linear model: C = sum(y*x) / sum(x*x), where
    x = Delta^(2/3) * s^3 / 1e6 and y = observed fuel_t_per_day.

    Expects df with columns: vessel_class, speed_kn, draft_ratio, fuel_t_per_day.
    """
    c_class = {}
    for vc in df["vessel_class"].unique():
        sub = df[df["vessel_class"] == vc]
        delta = displacement(sub["draft_ratio"].to_numpy(), vc)
        x = np.power(delta, 2.0 / 3.0) * np.power(sub["speed_kn"].to_numpy(), 3) / 1e6
        y = sub["fuel_t_per_day"].to_numpy()
        denom = np.sum(x * x)
        c_class[vc] = float(np.sum(y * x) / denom) if denom > 0 else DEFAULT_C_CLASS[vc]
    return c_class


def physics_only_mape(df: pd.DataFrame, c_class: dict[str, float]) -> float:
    """Baseline MAPE using ONLY the physics prior (no GBM residual) - this
    is the number the plan expects to land at 25-40%, recorded for
    comparison against the full physics+GBM model (target <=15%)."""
    preds = np.array([
        f_phys(row.speed_kn, row.draft_ratio, row.vessel_class, c_class)
        for row in df.itertuples()
    ])
    actual = df["fuel_t_per_day"].to_numpy()
    return float(np.mean(np.abs((actual - preds) / actual)) * 100)


if __name__ == "__main__":
    # quick sanity check: fuel should increase with speed^3
    for s in [10, 15, 20]:
        f = f_phys(s, draft_ratio=0.7, vessel_class="container")
        print(f"container @ {s}kn, draft 0.7: {f:.2f} t/day")
