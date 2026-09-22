"""
EU ETS cost: this feeds directly into the optimizer's J1 cost objective,
not just a side compliance report - it's a real cost line.

    ETS_cost = EUA_price * ets_share(year) * sum(CO2e_ttw)
"""
from __future__ import annotations
from dataclasses import dataclass
from src.compliance.fueleu import load_factors


@dataclass
class ETSResult:
    cost_usd: float
    ets_share: float
    year: int


def ets_share(year: int, factors: dict | None = None) -> float:
    factors = factors or load_factors()
    schedule = factors["ets"]["phase_in_schedule"]
    applicable_years = sorted(y for y in schedule if y <= year)
    if not applicable_years:
        return 0.0
    return schedule[applicable_years[-1]]


def ets_cost(
    co2e_ttw_tons: float,
    eua_price_usd_per_ton: float,
    year: int,
    factors: dict | None = None,
) -> ETSResult:
    """co2e_ttw_tons: tank-to-wake CO2e for the plan, tons.
    eua_price_usd_per_ton: scenario-set EUA price - ALWAYS a user-editable
    assumption, never presented as fact (plan Sec 16)."""
    factors = factors or load_factors()
    share = ets_share(year, factors)
    cost = eua_price_usd_per_ton * share * co2e_ttw_tons
    return ETSResult(cost_usd=cost, ets_share=share, year=year)
