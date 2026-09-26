"""
Decision encoding for the fleet plan (plan Sec 7.1) plus the scenario
definition it operates over.

One chromosome = one multi-week fleet plan. Genes are (route, week, slot).
Each gene's allele is a flattened triple:

    (vessel_class: 4) x (fuel: 5) x (speed_bucket: 3) -> 60 discrete alleles

This module owns the encode/decode logic and the Scenario dataclass so
repair.py and fitness.py both work off the same ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import yaml
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import VESSEL_CLASSES, DWT_LOOKUP, DESIGN_SPEED_KN

FUEL_OPTIONS = ["MDO", "VLSFO", "LNG", "MeOH", "NH3"]
SPEED_BUCKETS = {"eco": 0.60, "design": 0.85, "full": 1.00}
SPEED_BUCKET_NAMES = list(SPEED_BUCKETS.keys())

N_VESSEL = len(VESSEL_CLASSES)
N_FUEL = len(FUEL_OPTIONS)
N_SPEED = len(SPEED_BUCKETS)
N_ALLELES = N_VESSEL * N_FUEL * N_SPEED  # 4 * 5 * 3 = 60


def encode_allele(vessel_idx: int, fuel_idx: int, speed_idx: int) -> int:
    return vessel_idx * (N_FUEL * N_SPEED) + fuel_idx * N_SPEED + speed_idx


def decode_allele(allele: int) -> tuple[str, str, str]:
    """Returns (vessel_class, fuel, speed_bucket_name)."""
    speed_idx = allele % N_SPEED
    rem = allele // N_SPEED
    fuel_idx = rem % N_FUEL
    vessel_idx = rem // N_FUEL
    return VESSEL_CLASSES[vessel_idx], FUEL_OPTIONS[fuel_idx], SPEED_BUCKET_NAMES[speed_idx]


def alleles_for_fuel(fuel: str) -> list[int]:
    """All allele indices that use a given fuel - used by the fuel-
    availability repair operator to remap without touching vessel/speed."""
    fuel_idx = FUEL_OPTIONS.index(fuel)
    return [
        encode_allele(v, fuel_idx, s)
        for v in range(N_VESSEL) for s in range(N_SPEED)
    ]


def with_fuel(allele: int, new_fuel: str) -> int:
    """Same vessel_class + speed_bucket, different fuel."""
    _, _, speed_name = decode_allele(allele)
    speed_idx = SPEED_BUCKET_NAMES.index(speed_name)
    vessel_idx = allele // (N_FUEL * N_SPEED)
    fuel_idx = FUEL_OPTIONS.index(new_fuel)
    return encode_allele(vessel_idx, fuel_idx, speed_idx)


def with_vessel_class(allele: int, new_vessel_class: str) -> int:
    """Same fuel + speed_bucket, different (usually bigger) vessel."""
    rem = allele % (N_FUEL * N_SPEED)
    vessel_idx = VESSEL_CLASSES.index(new_vessel_class)
    return encode_allele(vessel_idx, rem // N_SPEED, rem % N_SPEED)


@dataclass
class Route:
    name: str
    distance_nm: float
    weeks: int
    slots_per_week: int
    demand_dwt_per_week: float
    fuel_availability: list[str]
    origin_port: str | None = None
    destination_port: str | None = None

    @property
    def n_genes(self) -> int:
        return self.weeks * self.slots_per_week


@dataclass
class Scenario:
    routes: list[Route]
    year: int
    eua_price_usd_per_ton: float
    fuel_price_usd_per_ton: dict[str, float]
    w_delay_usd_per_hour: float
    available_hours_per_week: float

    @property
    def n_genes(self) -> int:
        return sum(r.n_genes for r in self.routes)

    def gene_route_map(self) -> list[int]:
        """gene index -> index into self.routes, so fitness/repair know
        which route (and therefore which demand/availability) each gene
        belongs to."""
        mapping = []
        for i, r in enumerate(self.routes):
            mapping.extend([i] * r.n_genes)
        return mapping

    def gene_week_map(self) -> list[int]:
        """gene index -> week number within its route (0-indexed) - genes
        for a route are laid out slot-major within each week."""
        mapping = []
        for r in self.routes:
            for w in range(r.weeks):
                mapping.extend([w] * r.slots_per_week)
        return mapping

    @classmethod
    def from_yaml(cls, path: str) -> "Scenario":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        routes = [Route(**r) for r in cfg["routes"]]
        return cls(
            routes=routes,
            year=cfg["year"],
            eua_price_usd_per_ton=cfg["eua_price_usd_per_ton"],
            fuel_price_usd_per_ton=cfg["fuel_price_usd_per_ton"],
            w_delay_usd_per_hour=cfg["w_delay_usd_per_hour"],
            available_hours_per_week=cfg["available_hours_per_week"],
        )
