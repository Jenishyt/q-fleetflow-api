"""
FuelEU Maritime compliance: well-to-wake carbon intensity vs a declining
regulatory limit.

    Intensity_plan = sum(EF_f * E_f) / sum(E_f)      [gCO2e/MJ]
    Limit(year)    = 91.16 * (1 - reduction(year))
"""
from __future__ import annotations
import yaml
import os
from functools import lru_cache
from dataclasses import dataclass

_FACTORS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "factors.yaml")


@lru_cache(maxsize=4)
def _load_factors_cached(path: str) -> dict:
    """Cached so the QIEA's thousands of fitness evaluations per run don't
    re-parse the same YAML file from disk every single time - this was
    the actual bottleneck behind the 116s runtime, not the QIEA logic."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_factors(path: str = _FACTORS_PATH) -> dict:
    return _load_factors_cached(path)


@dataclass
class FuelEUResult:
    intensity: float          # gCO2e/MJ, this plan's weighted-average
    limit: float              # gCO2e/MJ, the regulatory ceiling for this year
    excess: float             # positive = over the limit (non-compliant)
    compliant: bool
    year: int


def _ef_for_fuel(fuel: str, factors: dict) -> float:
    """Look up the WtW emission factor for a fuel, resolving the
    percent-vs-VLSFO entries into an absolute gCO2e/MJ value."""
    fuels = factors["fuel_ef_wtw"]
    vlsfo = fuels["VLSFO"]["value"]
    entry = fuels[fuel]
    if "value" in entry:
        return entry["value"]
    if "value_vs_vlsfo_pct" in entry:
        return vlsfo * (1 + entry["value_vs_vlsfo_pct"] / 100)
    raise ValueError(f"No emission factor resolvable for fuel '{fuel}' - flagged as assumption, "
                      f"needs a scenario.yaml override before use in a real run")


def fueleu_limit(year: int, factors: dict | None = None) -> float:
    factors = factors or load_factors()
    baseline = factors["fueleu"]["baseline_gco2e_per_mj"]
    schedule = factors["fueleu"]["reduction_schedule"]
    # find the most recent schedule year <= requested year
    applicable_years = sorted(y for y in schedule if y <= year)
    reduction = schedule[applicable_years[-1]] if applicable_years else 0.0
    return baseline * (1 - reduction)


def fueleu_check(
    energy_by_fuel: dict[str, float],  # {fuel_name: energy_MJ}
    year: int,
    factors: dict | None = None,
) -> FuelEUResult:
    factors = factors or load_factors()
    total_energy = sum(energy_by_fuel.values())
    if total_energy <= 0:
        raise ValueError("Total energy must be positive")

    weighted_ef = sum(
        _ef_for_fuel(fuel, factors) * energy for fuel, energy in energy_by_fuel.items()
    ) / total_energy

    limit = fueleu_limit(year, factors)
    excess = weighted_ef - limit

    return FuelEUResult(
        intensity=weighted_ef,
        limit=limit,
        excess=excess,
        compliant=excess <= 0,
        year=year,
    )
