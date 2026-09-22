"""
Repair operators (plan Sec 7.3): fix infeasible plans BEFORE scoring them,
rather than just penalizing. This is why the QIEA should converge faster
than a naive GA - generations aren't wasted evaluating garbage candidates.

Two repair operators implemented for the MVP:
  1. Fuel availability - always fully repairable (remap to nearest
     available fuel for that route, same vessel/speed).
  2. Cargo demand - loop-upgrade vessel class on cheapest slot until
     demand is met, or flag remaining deficit for a J1 penalty.

CII repair (increase eco-speed share on offending vessel-weeks) is left
as a penalty-only path for the MVP - flagged honestly as a simplification,
not hidden.
"""
from __future__ import annotations
import numpy as np

from src.optimizer.encoding import (
    Scenario, decode_allele, with_fuel, with_vessel_class,
    VESSEL_CLASSES, DWT_LOOKUP,
)

# Upgrade order: smallest capacity -> largest, used when a route-week is
# short on demand and needs more capacity from its existing slots.
_CAPACITY_ORDER = sorted(VESSEL_CLASSES, key=lambda vc: DWT_LOOKUP[vc])


def repair_fuel_availability(plan: np.ndarray, scenario: Scenario) -> np.ndarray:
    """For every gene, if the assigned fuel isn't available on that route,
    remap to the first available fuel for that route (same vessel class
    and speed bucket preserved)."""
    plan = plan.copy()
    route_map = scenario.gene_route_map()
    for g, allele in enumerate(plan):
        route = scenario.routes[route_map[g]]
        _, fuel, _ = decode_allele(allele)
        if fuel not in route.fuel_availability:
            plan[g] = with_fuel(allele, route.fuel_availability[0])
    return plan


def repair_demand(plan: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, dict[str, float]]:
    """For every (route, week), if total assigned DWT capacity is below
    demand, upgrade slots to bigger vessel classes (cheapest capacity
    increase first) until demand is met or all slots are maxed out.

    Returns (repaired_plan, remaining_deficit_by_route_week) - any
    non-zero deficit could not be repaired and must be penalized in
    fitness.py.
    """
    plan = plan.copy()
    route_map = scenario.gene_route_map()
    week_map = scenario.gene_week_map()

    deficits: dict[str, float] = {}

    for r_idx, route in enumerate(scenario.routes):
        for w in range(route.weeks):
            gene_idxs = [
                g for g in range(len(plan))
                if route_map[g] == r_idx and week_map[g] == w
            ]
            if not gene_idxs:
                continue

            def total_capacity() -> float:
                return sum(DWT_LOOKUP[decode_allele(plan[g])[0]] for g in gene_idxs)

            demand = route.demand_dwt_per_week
            # loop: while short, upgrade the smallest-capacity slot that
            # isn't already at the top of the capacity order
            guard = 0
            while total_capacity() < demand and guard < 50:
                guard += 1
                # find the slot with the smallest current capacity that can
                # still be upgraded
                candidates = sorted(
                    gene_idxs,
                    key=lambda g: DWT_LOOKUP[decode_allele(plan[g])[0]],
                )
                upgraded = False
                for g in candidates:
                    current_vc = decode_allele(plan[g])[0]
                    pos = _CAPACITY_ORDER.index(current_vc)
                    if pos < len(_CAPACITY_ORDER) - 1:
                        plan[g] = with_vessel_class(plan[g], _CAPACITY_ORDER[pos + 1])
                        upgraded = True
                        break
                if not upgraded:
                    break  # every slot already at max capacity - can't repair further

            deficit = max(0.0, demand - total_capacity())
            if deficit > 0:
                deficits[f"{route.name}_week{w}"] = deficit

    return plan, deficits


def repair(plan: np.ndarray, scenario: Scenario) -> tuple[np.ndarray, dict[str, float]]:
    """Full repair pipeline: fuel availability first (always succeeds),
    then demand (may leave a residual deficit for fitness.py to penalize)."""
    plan = repair_fuel_availability(plan, scenario)
    plan, deficits = repair_demand(plan, scenario)
    return plan, deficits
