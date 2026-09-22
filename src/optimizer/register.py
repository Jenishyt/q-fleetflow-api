"""
Q-bit register for the QIEA (Quantum-Inspired Evolutionary Algorithm).

Instead of a chromosome holding ONE fixed allele per gene, each gene holds a
probability distribution over all possible alleles. This is the classical
simulation of a quantum superposition: the population doesn't store single
solutions, it stores *distributions* over solutions.

A "measurement" samples one concrete allele per gene from that distribution,
producing an actual candidate plan that can be scored.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class QRegister:
    """One individual in the population.

    q: shape (n_genes, n_alleles) - each row is a probability distribution
       over alleles for that gene. Rows sum to 1.
    """
    q: np.ndarray

    @property
    def n_genes(self) -> int:
        return self.q.shape[0]

    @property
    def n_alleles(self) -> int:
        return self.q.shape[1]

    def measure(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one allele index per gene from its probability distribution.

        Returns: shape (n_genes,) int array of allele indices.
        """
        n_genes, n_alleles = self.q.shape
        out = np.empty(n_genes, dtype=np.int64)
        # vectorized categorical sampling via cumulative sum + uniform draw
        cum = np.cumsum(self.q, axis=1)
        cum[:, -1] = 1.0  # guard against floating point drift
        u = rng.random(n_genes)
        for g in range(n_genes):
            out[g] = np.searchsorted(cum[g], u[g], side="right")
        return out

    def renormalize(self) -> None:
        """Clip to a valid probability simplex after rotation updates."""
        np.clip(self.q, 1e-6, 1.0, out=self.q)
        self.q /= self.q.sum(axis=1, keepdims=True)


def init_population(
    n_pop: int,
    n_genes: int,
    n_alleles: int,
    seed: int,
) -> list[QRegister]:
    """Initialize a population of QRegisters with Dirichlet(1) priors per gene
    (i.e. uniform over the probability simplex - no allele favored at start).
    """
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_pop):
        q = rng.dirichlet(np.ones(n_alleles), size=n_genes)
        pop.append(QRegister(q=q))
    return pop


def measure_population(
    pop: list[QRegister], seed: int
) -> list[np.ndarray]:
    """Measure every register in the population with independent draws
    (same seed -> same run, for reproducibility)."""
    rng = np.random.default_rng(seed)
    return [reg.measure(rng) for reg in pop]
