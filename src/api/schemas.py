"""
Typed request/response shapes for the API (plan Sec 9). Keeping these
separate from main.py so they're reusable by the dashboard's offline mode
and any test client.
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    vessel_class: str = Field(..., examples=["container"])
    speed_kn: float = Field(..., gt=0, le=30)
    draft_ratio: float = Field(..., gt=0, le=1.0)
    wind_kn: float = 0.0
    wave_hs_m: float = 0.0
    temp_c: float = 25.0
    fuel_type: str = "VLSFO"


class PredictResponse(BaseModel):
    fuel_t_per_day: float
    q10: float
    q90: float


class OptimizeRequest(BaseModel):
    scenario_path: str = Field(default="configs/scenario.yaml")
    n_pop: int = Field(default=40, ge=4, le=200)
    n_generations: int = Field(default=150, ge=10, le=1000)
    seed: int = 42


class ParetoPoint(BaseModel):
    plan_id: str
    J1_cost_usd: float
    J2_ghg_intensity: float
    J3_schedule_risk: float
    fueleu_compliant: bool


class OptimizeResponse(BaseModel):
    run_id: str
    n_pop: int
    n_generations: int
    elapsed_s: float
    pareto_front: list[ParetoPoint]


class LedgerResponse(BaseModel):
    plan_id: str
    fuel_cost_usd: float
    ets_cost_usd: float
    demand_penalty_usd: float
    fueleu_intensity: float
    fueleu_limit: float
    fueleu_compliant: bool
    total_co2_ttw_t: float
    risk_hours: float
    n_legs: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    data_hash: str
