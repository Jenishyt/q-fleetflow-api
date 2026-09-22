"""
Full benchmark protocol (plan Sec 7.6 / 10.2): QIEA vs random search vs
NSGA-II across 10 seeds, scored by hypervolume, with a Wilcoxon
signed-rank test on the paired per-seed results. Greedy heuristic is
deterministic (no seed dependence) so it's run once as a reference line,
not included in the statistical test.

Writes results/benchmark_table.csv and prints the honest summary,
including the Wilcoxon p-value either way - if there's no significant
difference, the script says so, per the plan's own honesty clause.
"""
from __future__ import annotations
import time
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.optimizer.encoding import Scenario
from src.optimizer.qiea import run_qiea
from src.optimizer.baselines import random_search, greedy_heuristic, nsga2_pymoo
from src.models.predict_api import FuelPredictor

N_SEEDS = 10
N_POP = 40
N_GEN = 150


def hypervolume(archive: list[dict], ref_point: np.ndarray) -> float:
    """Dominated hypervolume via pymoo's HV indicator. Objectives are
    normalized against a SHARED reference point across all algorithms
    (computed once from the worst point seen across every run) so the
    comparison is apples-to-apples."""
    from pymoo.indicators.hv import HV
    if not archive:
        return 0.0
    F = np.array([e["obj"] for e in archive])
    hv = HV(ref_point=ref_point)
    return float(hv(F))


def feasibility_rate(archive: list[dict]) -> float:
    if not archive:
        return 0.0
    return sum(1 for e in archive if e["ledger"]["fueleu_compliant"]) / len(archive)


def run_protocol(scenario_path: str, out_csv: str):
    scenario = Scenario.from_yaml(scenario_path)
    predictor = FuelPredictor.from_synthetic()

    print(f"Scenario: {scenario.n_genes} genes. Running {N_SEEDS} seeds x 3 stochastic "
          f"algorithms (QIEA, random search, NSGA-II) + 1 deterministic greedy baseline.\n")

    # --- Pass 1: run everything, collect raw archives ---
    all_runs = {"qiea": [], "random": [], "nsga2": []}
    timings = {"qiea": [], "random": [], "nsga2": []}

    for seed in range(N_SEEDS):
        print(f"seed {seed}:", end=" ", flush=True)

        t0 = time.time()
        qiea_archive, _ = run_qiea(scenario, predictor, n_pop=N_POP, n_generations=N_GEN,
                                    seed=seed, verbose=False)
        t_qiea = time.time() - t0
        all_runs["qiea"].append(qiea_archive)
        timings["qiea"].append(t_qiea)
        print(f"QIEA={t_qiea:.1f}s", end="  ", flush=True)

        t0 = time.time()
        random_archive = random_search(scenario, predictor, n_evals=N_POP * N_GEN, seed=seed)
        t_random = time.time() - t0
        all_runs["random"].append(random_archive)
        timings["random"].append(t_random)
        print(f"random={t_random:.1f}s", end="  ", flush=True)

        t0 = time.time()
        nsga2_archive = nsga2_pymoo(scenario, predictor, n_pop=N_POP, n_generations=N_GEN, seed=seed)
        t_nsga2 = time.time() - t0
        all_runs["nsga2"].append(nsga2_archive)
        timings["nsga2"].append(t_nsga2)
        print(f"nsga2={t_nsga2:.1f}s")

    greedy_archive = greedy_heuristic(scenario, predictor)

    # --- shared reference point for hypervolume, computed AFTER all runs ---
    all_objs = []
    for algo_runs in all_runs.values():
        for archive in algo_runs:
            all_objs.extend(e["obj"] for e in archive)
    all_objs.extend(e["obj"] for e in greedy_archive)
    all_objs = np.array(all_objs)
    ref_point = all_objs.max(axis=0) * 1.1
    # guard against a degenerate objective (max == 0) collapsing the whole
    # HV to zero - here J3 (schedule risk) is 0 for every point because this
    # scenario's routes never actually trigger a delay violation, which is
    # itself worth flagging honestly rather than silently patching around
    degenerate_dims = np.where(ref_point <= 0)[0]
    if len(degenerate_dims) > 0:
        print(f"NOTE: objective dimension(s) {degenerate_dims.tolist()} are constant across "
              f"every run (max=0) - scenario.yaml's available_hours_per_week is never actually "
              f"exceeded, so J3 (schedule risk) has no discriminating power in this scenario. "
              f"Using a small positive floor so hypervolume still reflects J1/J2 spread.")
    ref_point = np.where(ref_point <= 0, 1e-3, ref_point)

    print(f"\nShared reference point (for HV): {ref_point}")

    # --- Pass 2: compute metrics per seed per algorithm ---
    rows = []
    hv_by_algo = {"qiea": [], "random": [], "nsga2": []}
    for algo, runs in all_runs.items():
        for seed, archive in enumerate(runs):
            hv = hypervolume(archive, ref_point)
            feas = feasibility_rate(archive)
            best_cost = min((e["obj"][0] for e in archive), default=float("nan"))
            hv_by_algo[algo].append(hv)
            rows.append({
                "algorithm": algo, "seed": seed, "hypervolume": hv,
                "feasibility_rate": feas, "best_cost_usd": best_cost,
                "archive_size": len(archive), "wall_clock_s": timings[algo][seed],
            })

    greedy_hv = hypervolume(greedy_archive, ref_point)
    rows.append({
        "algorithm": "greedy", "seed": "n/a (deterministic)", "hypervolume": greedy_hv,
        "feasibility_rate": feasibility_rate(greedy_archive),
        "best_cost_usd": greedy_archive[0]["obj"][0],
        "archive_size": 1, "wall_clock_s": 0.0,
    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    # --- Wilcoxon signed-rank test on paired per-seed hypervolume ---
    print(f"\n{'='*70}")
    print("SUMMARY (mean +/- std across seeds)")
    print(f"{'='*70}")
    for algo in ["qiea", "random", "nsga2"]:
        hvs = np.array(hv_by_algo[algo])
        feas = df[df.algorithm == algo]["feasibility_rate"].to_numpy()
        costs = df[df.algorithm == algo]["best_cost_usd"].to_numpy()
        print(f"{algo:10s}  HV={hvs.mean():,.0f} +/- {hvs.std():,.0f}   "
              f"feasibility={feas.mean()*100:.1f}%   best_cost=${costs.min():,.0f}")
    print(f"{'greedy':10s}  HV={greedy_hv:,.0f}   "
          f"feasibility={feasibility_rate(greedy_archive)*100:.0f}%   "
          f"cost=${greedy_archive[0]['obj'][0]:,.0f}")

    print(f"\n{'='*70}")
    print("WILCOXON SIGNED-RANK TEST (paired by seed, on hypervolume)")
    print(f"{'='*70}")
    for other in ["random", "nsga2"]:
        a = np.array(hv_by_algo["qiea"])
        b = np.array(hv_by_algo[other])
        try:
            stat, p = wilcoxon(a, b)
            sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "NOT significant (p>=0.05)"
            direction = "QIEA higher" if a.mean() > b.mean() else f"{other} higher"
            print(f"QIEA vs {other:8s}: statistic={stat:.2f}  p={p:.4f}  {sig}  ({direction} on mean HV)")
        except ValueError as ex:
            print(f"QIEA vs {other:8s}: could not compute ({ex})")

    print(f"\nWrote {out_csv}")
    return df


if __name__ == "__main__":
    scenario_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "scenario.yaml")
    out_csv = os.path.join(os.path.dirname(__file__), "..", "..", "results", "benchmark_table.csv")
    run_protocol(scenario_path, out_csv)
