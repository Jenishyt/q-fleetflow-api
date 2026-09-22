"""
Baseline algorithms (plan Sec 7.6 / 10.2), all using the EXACT SAME
encoding, repair(), and evaluate() functions as qiea.py - same problem,
same fitness, only the search strategy differs. This is what makes the
benchmark honest instead of a rigged comparison.

  1. random_search   - lower bound: proves QIEA does something
  2. greedy_heuristic - realism anchor: what a human dispatcher would do
  3. nsga2_pymoo      - the serious comparator, same population/generation
                        budget as QIEA
"""
from __future__ import annotations
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.optimizer.encoding import (
    Scenario, N_ALLELES, encode_allele, decode_allele,
    VESSEL_CLASSES, FUEL_OPTIONS, SPEED_BUCKET_NAMES,
)
from src.optimizer.repair import repair
from src.optimizer.fitness import evaluate_population
from src.optimizer.qiea import update_archive
from src.models.predict_api import FuelPredictor


def random_search(
    scenario: Scenario, predictor: FuelPredictor, n_evals: int = 6000, seed: int = 42,
    batch_size: int = 200,
) -> list[dict]:
    """Sample random plans, apply the SAME repair, keep the non-dominated
    set. This is the lower bound - if QIEA doesn't beat this, nothing
    else in the plan matters.

    Batched in chunks of `batch_size` plans per predict_fuel_batch() call,
    same fix as qiea.py - LightGBM's per-call overhead dominates, not row
    count, so fewer/larger calls is what actually matters for speed.
    """
    rng = np.random.default_rng(seed)
    archive: list[dict] = []
    n_batches = (n_evals + batch_size - 1) // batch_size

    for _ in range(n_batches):
        plans, deficits_list = [], []
        for _ in range(batch_size):
            plan = rng.integers(0, N_ALLELES, size=scenario.n_genes)
            repaired, deficits = repair(plan, scenario)
            plans.append(repaired)
            deficits_list.append(deficits)

        results = evaluate_population(plans, scenario, predictor, deficits_list)
        for repaired, (J1, J2, J3, ledger) in zip(plans, results):
            entry = {"plan": repaired.copy(), "obj": (J1, J2, J3), "ledger": ledger}
            archive = update_archive(archive, entry)
    return archive


def greedy_heuristic(scenario: Scenario, predictor: FuelPredictor) -> list[dict]:
    """What a human dispatcher would do with no optimization at all:
    cheapest feasible option per route-week slot - smallest vessel class
    that can plausibly carry the load, cheapest available fuel, eco speed
    (minimize cost, ignore GHG/risk entirely). Single plan, wrapped in a
    one-element archive for a uniform comparison interface."""
    route_map = scenario.gene_route_map()
    plan = np.zeros(scenario.n_genes, dtype=np.int64)

    for g in range(scenario.n_genes):
        route = scenario.routes[route_map[g]]
        # cheapest available fuel by scenario price
        cheapest_fuel = min(route.fuel_availability, key=lambda f: scenario.fuel_price_usd_per_ton[f])
        # smallest vessel class (cheapest to run) - repair_demand will
        # upgrade it afterward if capacity is actually insufficient
        vessel_idx = 0  # VESSEL_CLASSES[0] = smallest by DWT ordering isn't
        # guaranteed here, so just start at "container" and let repair fix it
        fuel_idx = FUEL_OPTIONS.index(cheapest_fuel)
        speed_idx = SPEED_BUCKET_NAMES.index("eco")
        plan[g] = encode_allele(vessel_idx, fuel_idx, speed_idx)

    repaired, deficits = repair(plan, scenario)
    (J1, J2, J3, ledger), = evaluate_population([repaired], scenario, predictor, [deficits])
    return [{"plan": repaired, "obj": (J1, J2, J3), "ledger": ledger}]


def nsga2_pymoo(
    scenario: Scenario, predictor: FuelPredictor,
    n_pop: int = 40, n_generations: int = 150, seed: int = 42,
) -> list[dict]:
    """Same chromosome encoding, same population/generation budget as
    QIEA. Genes are treated as continuous in [0, N_ALLELES) and rounded to
    int at evaluation time - a standard practical trick for discrete NSGA-II
    encodings in pymoo, avoiding version-specific mixed-variable APIs.

    _evaluate batches the WHOLE generation's population into one
    predict_fuel_batch() call, same fix as qiea.py and random_search
    above - pymoo already hands us the full population matrix per call,
    we just weren't taking advantage of it before.
    """
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    n_genes = scenario.n_genes

    class FleetProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_genes, n_obj=3, n_constr=0,
                              xl=0.0, xu=float(N_ALLELES) - 1e-6)

        def _evaluate(self, X, out, *args, **kwargs):
            plans, deficits_list = [], []
            for row in X:
                alleles = np.clip(row.astype(int), 0, N_ALLELES - 1)
                repaired, deficits = repair(alleles, scenario)
                plans.append(repaired)
                deficits_list.append(deficits)

            results = evaluate_population(plans, scenario, predictor, deficits_list)
            out["F"] = np.array([[J1, J2, J3] for J1, J2, J3, _ in results])

    problem = FleetProblem()
    algorithm = NSGA2(pop_size=n_pop)
    res = minimize(problem, algorithm, ("n_gen", n_generations), seed=seed, verbose=False)

    # recompute ledgers on the final non-dominated set for reporting
    # (res.F already holds the objective values NSGA2 evaluated internally,
    # this second pass is just to attach the compliance ledger for display)
    X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
    plans, deficits_list = [], []
    for row in X:
        alleles = np.clip(row.astype(int), 0, N_ALLELES - 1)
        repaired, deficits = repair(alleles, scenario)
        plans.append(repaired)
        deficits_list.append(deficits)
    results = evaluate_population(plans, scenario, predictor, deficits_list)

    archive = [
        {"plan": plan, "obj": (J1, J2, J3), "ledger": ledger}
        for plan, (J1, J2, J3, ledger) in zip(plans, results)
    ]
    return archive


if __name__ == "__main__":
    from src.optimizer.encoding import Scenario
    import os

    scenario_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "scenario.yaml")
    scenario = Scenario.from_yaml(scenario_path)
    predictor = FuelPredictor.from_synthetic()

    print("=== Greedy heuristic ===")
    start = time.time()
    greedy_archive = greedy_heuristic(scenario, predictor)
    print(f"Elapsed: {time.time()-start:.2f}s")
    for e in greedy_archive:
        j1, j2, j3 = e["obj"]
        print(f"  J1=${j1:,.0f}  J2={j2:.2f}  J3={j3:.1f}  compliant={e['ledger']['fueleu_compliant']}")

    print("\n=== Random search (same eval budget as QIEA: 40x150=6000) ===")
    start = time.time()
    random_archive = random_search(scenario, predictor, n_evals=6000, seed=42)
    print(f"Elapsed: {time.time()-start:.1f}s, archive size: {len(random_archive)}")
    best_random_j1 = min(e["obj"][0] for e in random_archive)
    print(f"  Best J1 found: ${best_random_j1:,.0f}")

    print("\n=== NSGA-II (pymoo), same budget as QIEA (40 pop x 150 gen) ===")
    start = time.time()
    nsga2_archive = nsga2_pymoo(scenario, predictor, n_pop=40, n_generations=150, seed=42)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.1f}s, archive size: {len(nsga2_archive)}")
    best_nsga2_j1 = min(e["obj"][0] for e in nsga2_archive)
    print(f"  Best J1 found: ${best_nsga2_j1:,.0f}")
