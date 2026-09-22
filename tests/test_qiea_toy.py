"""
Day-1 gate check: does the QIEA core (register + rotation gates) actually
converge on ANYTHING before we trust it on the real 3-objective fleet
problem?

We use a 0/1 knapsack: N items, each with a value and weight, pick a subset
maximizing value under a weight budget. Every item is one gene with 2
alleles: {0 = leave out, 1 = take}. This is a single-objective sanity check,
not the real multi-objective fleet problem - it exists purely to prove the
Q-bit register + rotation-gate mechanics are wired correctly.

Gate: must find the known-optimal value within 5 seconds.
"""
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimizer.register import init_population, QRegister
from src.optimizer.gates import rotation_update, adaptive_theta, epsilon_jump


# --- Toy knapsack instance with a KNOWN optimal solution ---------------
VALUES = np.array([60, 100, 120, 80, 30, 50, 90, 40])
WEIGHTS = np.array([10, 20, 30, 15, 5, 12, 22, 8])
CAPACITY = 60

# brute-force the true optimum (small enough: 2^8 = 256 combos)
def brute_force_optimum():
    n = len(VALUES)
    best_val = 0
    for mask in range(2 ** n):
        bits = np.array([(mask >> i) & 1 for i in range(n)])
        w = np.sum(bits * WEIGHTS)
        if w <= CAPACITY:
            v = np.sum(bits * VALUES)
            best_val = max(best_val, v)
    return best_val


def fitness(bits: np.ndarray) -> float:
    """Higher is better. Penalize overweight solutions instead of rejecting
    them outright (repair-first philosophy, simplified for the toy case)."""
    w = np.sum(bits * WEIGHTS)
    v = np.sum(bits * VALUES)
    if w > CAPACITY:
        v -= 5 * (w - CAPACITY)  # penalty, same spirit as fitness.py later
    return v


def run_qiea(n_pop=40, n_generations=150, seed=42):
    n_genes = len(VALUES)
    n_alleles = 2  # {0, 1}
    rng = np.random.default_rng(seed)

    pop = init_population(n_pop, n_genes, n_alleles, seed=seed)
    elite = np.zeros(n_genes, dtype=np.int64)
    elite_fitness = -np.inf

    for gen in range(n_generations):
        theta1 = adaptive_theta(gen, n_generations)
        theta2 = 0.05 * np.pi

        for reg in pop:
            measured = reg.measure(rng)
            f = fitness(measured)
            if f > elite_fitness:
                elite_fitness = f
                elite = measured.copy()

        for reg in pop:
            measured = reg.measure(rng)
            rotation_update(reg, measured, elite, theta1, theta2, rng)

        # stagnation escape hatch (simplified: fixed check every 10 gens)
        if gen % 10 == 0 and gen > 0:
            for reg in pop[: n_pop // 8]:  # jump a slice of the population
                epsilon_jump(reg, epsilon=0.15, rng=rng)

    return elite, elite_fitness


def test_qiea_converges_to_known_optimum():
    true_optimum = brute_force_optimum()
    start = time.time()
    best_solution, best_value = run_qiea()
    elapsed = time.time() - start

    print(f"\nTrue optimum:        {true_optimum}")
    print(f"QIEA found:          {best_value}")
    print(f"Best solution bits:  {best_solution}")
    print(f"Elapsed:             {elapsed:.2f}s")

    assert elapsed < 5.0, f"QIEA too slow: {elapsed:.2f}s (gate: <5s)"
    assert best_value >= true_optimum, (
        f"QIEA did not reach the known optimum: got {best_value}, "
        f"expected >= {true_optimum}"
    )


if __name__ == "__main__":
    test_qiea_converges_to_known_optimum()
    print("\nPASSED - QIEA core mechanics (register + rotation gates) are wired correctly.")
