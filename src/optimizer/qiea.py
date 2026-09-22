"""
The real QIEA main loop (plan Sec 7.5), multi-objective this time - unlike
Session 1's knapsack toy test, which was deliberately single-objective to
isolate the register/rotation-gate mechanics.

    init 40 registers -> for N generations:
      measure -> repair -> predict_fuel_batch (via fitness.evaluate) ->
      score on 3 objectives -> update Pareto archive ->
      rotate probabilities toward a guide -> check stagnation
    -> return Pareto front + ledgers

Guide selection: since there's no single "best" plan once you have 3
competing objectives, each register draws random Dirichlet weights every
generation and picks the archive member that scalarizes best under those
weights (MOEA/D-style decomposition). This gives spread across the front
instead of collapsing to one point.
"""
from __future__ import annotations
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.optimizer.register import init_population, QRegister
from src.optimizer.gates import rotation_update, adaptive_theta, epsilon_jump
from src.optimizer.repair import repair
from src.optimizer.fitness import evaluate_population
from src.optimizer.encoding import Scenario, N_ALLELES
from src.models.predict_api import FuelPredictor


def dominates(a: tuple, b: tuple) -> bool:
    """Minimization dominance: a dominates b if a is no worse in every
    objective and strictly better in at least one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def update_archive(archive: list[dict], candidate: dict) -> list[dict]:
    """Insert candidate if non-dominated; remove any archive members it
    newly dominates."""
    obj = candidate["obj"]
    if any(dominates(e["obj"], obj) for e in archive):
        return archive  # candidate is dominated by something already in the archive
    archive = [e for e in archive if not dominates(obj, e["obj"])]
    archive.append(candidate)
    return archive


def crowding_distance(archive: list[dict]) -> np.ndarray:
    """Standard NSGA-II crowding distance, used only to decide which
    archive members to drop when it overflows max_archive."""
    n = len(archive)
    if n == 0:
        return np.array([])
    dist = np.zeros(n)
    n_obj = len(archive[0]["obj"])
    for m in range(n_obj):
        vals = np.array([e["obj"][m] for e in archive])
        order = np.argsort(vals)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        vmin, vmax = vals[order[0]], vals[order[-1]]
        spread = vmax - vmin
        if spread <= 1e-12:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / spread
    return dist


def truncate_archive(archive: list[dict], max_size: int) -> list[dict]:
    if len(archive) <= max_size:
        return archive
    dist = crowding_distance(archive)
    order = np.argsort(-dist)  # descending: keep the most spread-out points
    keep = sorted(order[:max_size])
    return [archive[i] for i in keep]


def _scalarize(obj: tuple, weights: np.ndarray, bounds: dict) -> float:
    norm = [
        (obj[i] - bounds["min"][i]) / (bounds["max"][i] - bounds["min"][i] + 1e-9)
        for i in range(len(obj))
    ]
    return float(np.dot(weights, norm))


def run_qiea(
    scenario: Scenario,
    predictor: FuelPredictor,
    n_pop: int = 40,
    n_generations: int = 150,
    seed: int = 42,
    max_archive: int = 60,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    n_genes = scenario.n_genes

    pop = init_population(n_pop, n_genes, N_ALLELES, seed=seed)
    archive: list[dict] = []
    bounds = {"min": [np.inf] * 3, "max": [-np.inf] * 3}
    history = []
    last_archive_size = 0
    stagnation_gens = 0

    for gen in range(n_generations):
        theta1 = adaptive_theta(gen, n_generations)
        theta2 = 0.05 * np.pi

        evaluated = []  # (register, measured_allele, repaired_plan, obj, ledger)
        measured_list = []
        repaired_list = []
        deficits_list = []
        for reg in pop:
            measured = reg.measure(rng)
            repaired_plan, deficits = repair(measured, scenario)
            measured_list.append(measured)
            repaired_list.append(repaired_plan)
            deficits_list.append(deficits)

        # ONE batched prediction call for the entire population this
        # generation, instead of one call per register - this is the fix
        # for the runtime gate (LightGBM per-call overhead dominates, not
        # row count; see fitness.evaluate_population's docstring).
        results = evaluate_population(repaired_list, scenario, predictor, deficits_list)

        for reg, measured, repaired_plan, (J1, J2, J3, ledger) in zip(pop, measured_list, repaired_list, results):
            obj = (J1, J2, J3)
            for i, v in enumerate(obj):
                bounds["min"][i] = min(bounds["min"][i], v)
                bounds["max"][i] = max(bounds["max"][i], v)

            entry = {"plan": repaired_plan.copy(), "obj": obj, "ledger": ledger}
            archive = update_archive(archive, entry)
            evaluated.append((reg, measured, repaired_plan, obj, ledger))

        if len(archive) > max_archive:
            archive = truncate_archive(archive, max_archive)

        # rotate every register toward a scalarized guide from the archive
        for reg, measured, repaired_plan, obj, ledger in evaluated:
            weights = rng.dirichlet(np.ones(3))
            guide = min(archive, key=lambda e: _scalarize(e["obj"], weights, bounds))
            rotation_update(reg, measured, guide["plan"], theta1, theta2, rng)

        # stagnation handling, checked every 10 generations
        if gen % 10 == 0 and gen > 0:
            if len(archive) == last_archive_size:
                stagnation_gens += 10
            else:
                stagnation_gens = 0
            last_archive_size = len(archive)

            if stagnation_gens >= 10:
                for reg in pop[: max(1, n_pop // 8)]:
                    epsilon_jump(reg, epsilon=0.15, rng=rng)
            if stagnation_gens >= 25:
                pop = init_population(n_pop, n_genes, N_ALLELES, seed=int(rng.integers(0, 1_000_000)))
                stagnation_gens = 0

        history.append({"gen": gen, "archive_size": len(archive)})
        if verbose and gen % 25 == 0:
            print(f"  gen {gen:3d}  archive_size={len(archive):3d}  "
                  f"J1_range=[{bounds['min'][0]:.0f}, {bounds['max'][0]:.0f}]")

    return archive, history


if __name__ == "__main__":
    scenario_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "scenario.yaml")
    scenario = Scenario.from_yaml(scenario_path)
    predictor = FuelPredictor.from_synthetic()

    print(f"Scenario: {len(scenario.routes)} routes, {scenario.n_genes} genes\n")
    start = time.time()
    archive, history = run_qiea(scenario, predictor, n_pop=40, n_generations=150, seed=42)
    elapsed = time.time() - start

    print(f"\n=== QIEA run complete ===")
    print(f"Elapsed:            {elapsed:.1f}s  (gate: <120s)")
    print(f"Final archive size: {len(archive)}")
    print(f"\nPareto front (sorted by cost J1):")
    print(f"{'J1 cost ($)':>14} {'J2 GHG intensity':>18} {'J3 risk':>10} {'compliant':>10}")
    for e in sorted(archive, key=lambda x: x["obj"][0])[:10]:
        j1, j2, j3 = e["obj"]
        print(f"{j1:14,.0f} {j2:18.2f} {j3:10.1f} {str(e['ledger']['fueleu_compliant']):>10}")

    gate = elapsed < 120
    print(f"\nGate <2min: {'PASS' if gate else 'FAIL'}")
