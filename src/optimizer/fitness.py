"""
Fitness function (plan Sec 7.3): scores one candidate plan on three
objectives, all minimized:

    J1_cost = fuel cost + EU ETS cost + unrepaired-demand penalty
    J2_ghg  = FuelEU well-to-wake carbon intensity (gCO2e/MJ)
    J3_risk = schedule delay proxy (hours over the weekly sailing window)

This is the module that finally connects everything built so far:
predict_fuel() from the prediction engine, and fueleu/ets/cii from the
compliance engine.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.optimizer.encoding import (
    Scenario, decode_allele, SPEED_BUCKETS, VESSEL_CLASSES, DWT_LOOKUP,
)
from src.models.physics import DESIGN_SPEED_KN
from src.models.predict_api import FuelPredictor
from src.compliance.fueleu import fueleu_check, load_factors
from src.compliance.ets import ets_cost

DESIGN_DRAFT_RATIO = 0.85  # fixed assumption for MVP - not a decision variable
CII_TTW_FACTOR_T_PER_T_FUEL = 3.15  # rough combustion CO2 factor, documented assumption
DEMAND_PENALTY_PER_TON = 1e4 / 60000  # scaled so a full route's worth of deficit is meaningful


@dataclass
class PlanLedgerRow:
    gene: int
    route: str
    week: int
    vessel_class: str
    fuel: str
    speed_bucket: str
    speed_kn: float
    distance_nm: float
    duration_hours: float
    fuel_t_leg: float
    energy_mj_leg: float
    co2_ttw_t_leg: float


def _build_leg_rows(plan: np.ndarray, scenario: Scenario) -> tuple[list[dict], list[tuple]]:
    """Pure per-plan construction, no prediction call - split out so
    evaluate_population() can batch this across an entire generation's
    worth of plans into ONE predict_fuel_batch() call instead of one
    call per plan (the actual fix for the runtime gate, see qiea.py)."""
    route_map = scenario.gene_route_map()
    week_map = scenario.gene_week_map()

    rows_in = []
    meta = []
    for g, allele in enumerate(plan):
        vessel_class, fuel, speed_bucket = decode_allele(allele)
        route = scenario.routes[route_map[g]]
        speed_kn = SPEED_BUCKETS[speed_bucket] * DESIGN_SPEED_KN[vessel_class]
        rows_in.append({
            "vessel_class": vessel_class, "speed_kn": speed_kn,
            "draft_ratio": DESIGN_DRAFT_RATIO, "wind_kn": 0.0,
            "wave_hs_m": 0.0, "temp_c": 25.0, "fuel_type": "VLSFO",  # baseline; fuel-transfer head below
        })
        meta.append((g, route, vessel_class, fuel, speed_bucket, speed_kn, week_map[g]))
    return rows_in, meta


def _legs_from_predictions(meta: list[tuple], preds: list) -> list[PlanLedgerRow]:
    """Combine per-gene metadata with already-computed FuelPred results
    (the fuel-transfer head: energy-equivalent swap via LHV ratios)."""
    factors = load_factors()
    lhv = factors["lhv_mj_per_kg"]
    lhv_vlsfo = lhv["VLSFO"]

    legs = []
    for (g, route, vessel_class, fuel, speed_bucket, speed_kn, week), pred in zip(meta, preds):
        f_base = pred.fuel_t_per_day  # t/day, on VLSFO-equivalent baseline
        duration_hours = (route.distance_nm / speed_kn)
        duration_days = duration_hours / 24.0

        fuel_t_baseline = f_base * duration_days
        energy_mj = fuel_t_baseline * lhv_vlsfo * 1000
        fuel_t_leg = energy_mj / (lhv.get(fuel, lhv_vlsfo) * 1000)
        co2_ttw_t = fuel_t_leg * CII_TTW_FACTOR_T_PER_T_FUEL

        legs.append(PlanLedgerRow(
            gene=g, route=route.name, week=week, vessel_class=vessel_class,
            fuel=fuel, speed_bucket=speed_bucket, speed_kn=speed_kn,
            distance_nm=route.distance_nm, duration_hours=duration_hours,
            fuel_t_leg=fuel_t_leg, energy_mj_leg=energy_mj, co2_ttw_t_leg=co2_ttw_t,
        ))
    return legs


def _leg_details(plan: np.ndarray, scenario: Scenario, predictor: FuelPredictor) -> list[PlanLedgerRow]:
    """Single-plan convenience path (used by tests and baselines.py, where
    batching across a whole population doesn't apply). qiea.py's main loop
    uses evaluate_population() instead for the batched version."""
    rows_in, meta = _build_leg_rows(plan, scenario)
    preds = predictor.predict_fuel_batch(pd.DataFrame(rows_in))
    return _legs_from_predictions(meta, preds)


def _score_legs(
    legs: list[PlanLedgerRow], scenario: Scenario, demand_deficits: dict[str, float],
) -> tuple[float, float, float, dict]:
    """Aggregate a plan's leg rows into the 3 objectives - shared by both
    the single-plan and batched evaluation paths."""
    fuel_prices = scenario.fuel_price_usd_per_ton
    fuel_cost = sum(leg.fuel_t_leg * fuel_prices[leg.fuel] for leg in legs)
    total_co2_ttw = sum(leg.co2_ttw_t_leg for leg in legs)
    ets_result = ets_cost(total_co2_ttw, scenario.eua_price_usd_per_ton, scenario.year)
    demand_penalty = sum(demand_deficits.values()) * DEMAND_PENALTY_PER_TON * 1e4
    J1 = fuel_cost + ets_result.cost_usd + demand_penalty

    energy_by_fuel: dict[str, float] = {}
    for leg in legs:
        energy_by_fuel[leg.fuel] = energy_by_fuel.get(leg.fuel, 0.0) + leg.energy_mj_leg
    fueleu_result = fueleu_check(energy_by_fuel, scenario.year)
    J2 = fueleu_result.intensity

    risk_hours = 0.0
    for leg in legs:
        overage = max(0.0, leg.duration_hours - scenario.available_hours_per_week)
        risk_hours += overage
    J3 = risk_hours * scenario.w_delay_usd_per_hour

    ledger = {
        "fuel_cost_usd": fuel_cost,
        "ets_cost_usd": ets_result.cost_usd,
        "demand_penalty_usd": demand_penalty,
        "fueleu_intensity": fueleu_result.intensity,
        "fueleu_limit": fueleu_result.limit,
        "fueleu_compliant": fueleu_result.compliant,
        "total_co2_ttw_t": total_co2_ttw,
        "risk_hours": risk_hours,
        "energy_by_fuel_mj": energy_by_fuel,
        "n_legs": len(legs),
    }
    return J1, J2, J3, ledger


def evaluate(
    plan: np.ndarray,
    scenario: Scenario,
    predictor: FuelPredictor,
    demand_deficits: dict[str, float] | None = None,
) -> tuple[float, float, float, dict]:
    """Single-plan path: returns (J1_cost, J2_ghg, J3_risk, ledger_dict).
    All three MINIMIZED. Call repair.repair() on the plan BEFORE this.
    For scoring a whole population at once (what qiea.py's main loop
    actually needs for speed), use evaluate_population() below instead.
    """
    legs = _leg_details(plan, scenario, predictor)
    return _score_legs(legs, scenario, demand_deficits or {})


def evaluate_population(
    plans: list[np.ndarray],
    scenario: Scenario,
    predictor: FuelPredictor,
    demand_deficits_list: list[dict[str, float]] | None = None,
) -> list[tuple[float, float, float, dict]]:
    """Batched version: builds every plan's leg rows, concatenates them
    into ONE dataframe, and calls predict_fuel_batch ONCE for the entire
    population instead of once per plan. This is the actual fix for the
    QIEA runtime gate - LightGBM's per-call overhead, not the row count,
    was the dominant cost (confirmed by profiling), so cutting the number
    of predict() calls from n_pop to 1 per generation is what matters.
    """
    demand_deficits_list = demand_deficits_list or [{}] * len(plans)

    all_rows = []
    all_meta = []
    plan_boundaries = []  # (start_idx, end_idx) into all_rows for each plan
    for plan in plans:
        rows_in, meta = _build_leg_rows(plan, scenario)
        start = len(all_rows)
        all_rows.extend(rows_in)
        all_meta.extend(meta)
        plan_boundaries.append((start, len(all_rows)))

    preds = predictor.predict_fuel_batch(pd.DataFrame(all_rows))

    results = []
    for (start, end), deficits in zip(plan_boundaries, demand_deficits_list):
        legs = _legs_from_predictions(all_meta[start:end], preds[start:end])
        results.append(_score_legs(legs, scenario, deficits))
    return results
