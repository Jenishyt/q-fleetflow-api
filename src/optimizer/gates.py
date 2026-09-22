"""
Rotation-gate update: the mechanism that actually earns the word "quantum"
in "quantum-inspired". Pure NumPy, runs on any CPU.

For each gene, compare the allele that was sampled against the current
best-known ("elite") allele for that gene:

  - sampled != elite AND elite has higher probability mass -> amplify elite
    (rotate probability mass toward it)
  - sampled != elite AND elite has lower probability mass  -> suppress the
    sampled allele (rotate away from it)
  - sampled == elite                                       -> small random
    exploration nudge

Theta (rotation angle) is adaptive: wide early (exploration), narrow late
(exploitation), driven by how diverse the archive still is.
"""
from __future__ import annotations
import numpy as np
from .register import QRegister


def adaptive_theta(
    generation: int,
    max_generations: int,
    theta_min: float = 0.02 * np.pi,
    theta_max: float = 0.12 * np.pi,
) -> float:
    """Wide early, narrow late. Linear decay is enough for the MVP."""
    frac = min(generation / max(max_generations, 1), 1.0)
    return theta_max - frac * (theta_max - theta_min)


def rotation_update(
    reg: QRegister,
    measured: np.ndarray,
    elite: np.ndarray,
    theta1: float,
    theta2: float,
    rng: np.random.Generator,
) -> None:
    """Update reg.q IN PLACE, gene by gene, based on comparison to the elite
    allele for that gene.

    measured: shape (n_genes,) - alleles sampled from `reg` this generation
    elite:    shape (n_genes,) - best-known allele per gene (from archive)
    """
    n_genes = reg.n_genes
    for g in range(n_genes):
        a_g = measured[g]
        best_a = elite[g]

        if a_g == best_a:
            # already matches elite -> small random exploration nudge
            delta = theta2 * rng.random()
            # rotate a small random amount toward a random other allele
            target = rng.integers(0, reg.n_alleles)
            _rotate_pair(reg.q, g, a_g, target, delta)
            continue

        if reg.q[g, best_a] > reg.q[g, a_g]:
            # elite already favored -> amplify it further
            _rotate_pair(reg.q, g, a_g, best_a, theta1)
        else:
            # sampled allele is currently favored over a worse-performing
            # elite candidate -> suppress it, boost elite
            _rotate_pair(reg.q, g, a_g, best_a, theta1)

    reg.renormalize()


def _rotate_pair(q: np.ndarray, gene: int, from_a: int, to_a: int, theta: float) -> None:
    """Rotate probability mass from allele `from_a` toward `to_a` for one gene.
    This is the classical stand-in for a 2D quantum rotation gate applied to
    a single qubit pair: q_from <- q_from*cos - q_to*sin (then renormalized).
    """
    if from_a == to_a:
        return
    q_from = q[gene, from_a]
    q_to = q[gene, to_a]
    c, s = np.cos(theta), np.sin(theta)
    new_from = q_from * c - q_to * s
    new_to = q_to * c + q_from * s
    q[gene, from_a] = max(new_from, 1e-6)
    q[gene, to_a] = max(new_to, 1e-6)


def epsilon_jump(reg: QRegister, epsilon: float, rng: np.random.Generator) -> None:
    """Re-randomize `epsilon` fraction of genes. Called when archive
    hypervolume has stagnated, to escape local optima."""
    n_genes = reg.n_genes
    n_jump = max(1, int(epsilon * n_genes))
    genes = rng.choice(n_genes, size=n_jump, replace=False)
    for g in genes:
        reg.q[g] = rng.dirichlet(np.ones(reg.n_alleles))
