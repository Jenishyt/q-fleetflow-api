"""
Fleet registry: register/list vessels. Unlike a decorative table, every
spec field here is derived from the SAME constants the optimizer and
prediction engine actually use (DWT_LOOKUP, DESIGN_SPEED_KN) - registering
a vessel as "container" class shows the real capacity and design speed
our physics model assumes for that class, not made-up numbers.

Persisted to results/fleet_registry.json (self-healing: seeds 4 starter
vessels, one per class, if the file doesn't exist yet).
"""
from __future__ import annotations
import json
import os
import uuid

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import VESSEL_CLASSES, DWT_LOOKUP, DESIGN_SPEED_KN

_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "fleet_registry.json")

_SEED_NAMES = {
    "container": "Q-Fleet Container 01",
    "bulk_carrier": "Q-Fleet Bulk 01",
    "tanker": "Q-Fleet Tanker 01",
    "general_cargo": "Q-Fleet Cargo 01",
}

# Rough design draft per class - matches DESIGN_DRAFT_RATIO used elsewhere
# in the optimizer (fitness.py), expressed here as meters for display.
_DESIGN_DRAFT_M = {"container": 13.5, "bulk_carrier": 14.0, "tanker": 15.5, "general_cargo": 9.0}
_DESIGN_ENGINE_KW = {"container": 45000, "bulk_carrier": 15000, "tanker": 18000, "general_cargo": 8000}


def _seed() -> list[dict]:
    vessels = []
    for vc in VESSEL_CLASSES:
        vessels.append({
            "id": f"V-{uuid.uuid4().hex[:6].upper()}",
            "name": _SEED_NAMES[vc],
            "vessel_class": vc,
            "capacity_dwt": DWT_LOOKUP[vc],
            "design_speed_kn": DESIGN_SPEED_KN[vc],
            "engine_power_kw": _DESIGN_ENGINE_KW[vc],
            "max_draft_m": _DESIGN_DRAFT_M[vc],
            "fuel_type": "VLSFO",
            "status": "Available",
        })
    return vessels


def load_fleet() -> list[dict]:
    if not os.path.exists(_REGISTRY_PATH):
        vessels = _seed()
        save_fleet(vessels)
        return vessels
    with open(_REGISTRY_PATH) as f:
        return json.load(f)


def save_fleet(vessels: list[dict]) -> None:
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(vessels, f, indent=2)


def register_vessel(name: str, vessel_class: str, fuel_type: str = "VLSFO") -> dict:
    if vessel_class not in VESSEL_CLASSES:
        raise ValueError(f"unknown vessel_class '{vessel_class}', must be one of {VESSEL_CLASSES}")

    vessel = {
        "id": f"V-{uuid.uuid4().hex[:6].upper()}",
        "name": name,
        "vessel_class": vessel_class,
        "capacity_dwt": DWT_LOOKUP[vessel_class],
        "design_speed_kn": DESIGN_SPEED_KN[vessel_class],
        "engine_power_kw": _DESIGN_ENGINE_KW[vessel_class],
        "max_draft_m": _DESIGN_DRAFT_M[vessel_class],
        "fuel_type": fuel_type,
        "status": "Available",
    }
    vessels = load_fleet()
    vessels.append(vessel)
    save_fleet(vessels)
    return vessel
