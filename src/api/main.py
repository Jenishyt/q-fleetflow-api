"""
FastAPI backend (plan Sec 9): 4 endpoints, nothing more.

  POST /predict            -> single fuel prediction
  POST /optimize            -> run the QIEA optimizer, return Pareto front
  GET  /plan/{run_id}/{plan_id}/ledger -> compliance breakdown for one plan
  GET  /health              -> model version + data hash sanity check

Optimizer results are explicitly cast out of numpy types (float64, int64)
into plain Python types before returning - numpy types aren't
JSON-serializable by default and this is exactly the kind of bug that
eats an integration-day hour if not handled upfront (flagged in the
plan itself).
"""
from __future__ import annotations
import os
import sys
import time
import uuid
import json
import hashlib
import asyncio
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import (
    PredictRequest, PredictResponse, OptimizeRequest, OptimizeResponse,
    ParetoPoint, LedgerResponse, HealthResponse,
)
from src.models.predict_api import FuelPredictor
from src.optimizer.encoding import Scenario
from src.optimizer.qiea import run_qiea

_predictor: FuelPredictor | None = None
_predictor_lock = threading.Lock()


def get_predictor() -> FuelPredictor:
    """Thread-safe lazy load: if two requests (or a request and the
    background warmup task) hit this at the same moment, only one
    actually trains - the other blocks on the lock and reuses the result
    instead of training a second, wasted copy."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:  # re-check inside the lock
                _predictor = FuelPredictor.from_synthetic()
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Kick off predictor training in a background thread WITHOUT blocking
    startup. An earlier version awaited this directly in lifespan and it
    broke deployment entirely: Render's port scanner needs the port open
    within a couple of minutes, training took longer than that on Render's
    free-tier CPU, so the port never opened and the deploy timed out and
    failed. Fire-and-forget here instead - the port opens immediately,
    Render's health check passes right away, and get_predictor()'s lock
    means whichever request (background warmup or a real user) gets there
    first does the training; the other just waits on the same result."""
    asyncio.create_task(asyncio.to_thread(get_predictor))
    yield


app = FastAPI(title="Q-FleetFlow API", version="0.1.0", lifespan=lifespan)

# Permissive CORS for the hackathon demo - the Next.js frontend (Vercel)
# and this API (Render) are on different origins. Tighten allow_origins
# to the real deployed frontend URL before anything more than a demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_RUNS_DIR = os.path.join(_REPO_ROOT, "results", "runs")
os.makedirs(_RUNS_DIR, exist_ok=True)


def _to_py(x):
    """Cast numpy scalar types to plain Python types for JSON serialization."""
    if hasattr(x, "item"):
        return x.item()
    return x


@app.get("/health", response_model=HealthResponse)
def health():
    data_path = os.path.join(_REPO_ROOT, "data", "processed", "synthetic.parquet")
    data_hash = "unknown"
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data_hash = hashlib.sha256(f.read()).hexdigest()[:12]
    return HealthResponse(status="ok", model_version="synthetic-v0.1", data_hash=data_hash)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor = get_predictor()
    pred = predictor.predict_fuel(
        vessel_class=req.vessel_class, speed_kn=req.speed_kn, draft_ratio=req.draft_ratio,
        wind_kn=req.wind_kn, wave_hs_m=req.wave_hs_m, temp_c=req.temp_c, fuel_type=req.fuel_type,
    )
    return PredictResponse(
        fuel_t_per_day=_to_py(pred.fuel_t_per_day),
        q10=_to_py(pred.q10),
        q90=_to_py(pred.q90),
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    scenario_path = os.path.join(_REPO_ROOT, req.scenario_path)
    if not os.path.exists(scenario_path):
        raise HTTPException(404, f"scenario not found: {req.scenario_path}")

    scenario = Scenario.from_yaml(scenario_path)
    predictor = get_predictor()

    start = time.time()
    archive, _ = run_qiea(scenario, predictor, n_pop=req.n_pop,
                           n_generations=req.n_generations, seed=req.seed, verbose=False)
    elapsed = time.time() - start

    run_id = str(uuid.uuid4())[:8]
    pareto_points = []
    plans_out = {}
    for i, e in enumerate(archive):
        plan_id = f"plan_{i:03d}"
        J1, J2, J3 = e["obj"]
        pareto_points.append(ParetoPoint(
            plan_id=plan_id, J1_cost_usd=_to_py(J1), J2_ghg_intensity=_to_py(J2),
            J3_schedule_risk=_to_py(J3), fueleu_compliant=bool(e["ledger"]["fueleu_compliant"]),
        ))
        plans_out[plan_id] = {
            "plan": [int(a) for a in e["plan"]],
            "obj": [float(J1), float(J2), float(J3)],
            "ledger": {k: (_to_py(v) if not isinstance(v, dict) else v) for k, v in e["ledger"].items()},
        }

    # persist so the dashboard's offline mode can read this WITHOUT a live
    # API round-trip - the single most important demo-day reliability
    # decision in the whole plan (Sec 9, risk R4)
    run_record = {
        "run_id": run_id, "scenario_path": req.scenario_path, "n_pop": req.n_pop,
        "n_generations": req.n_generations, "seed": req.seed, "elapsed_s": elapsed,
        "plans": plans_out,
    }
    with open(os.path.join(_RUNS_DIR, f"{run_id}.json"), "w") as f:
        json.dump(run_record, f, indent=2)

    return OptimizeResponse(
        run_id=run_id, n_pop=req.n_pop, n_generations=req.n_generations,
        elapsed_s=elapsed, pareto_front=pareto_points,
    )


@app.get("/plan/{run_id}/{plan_id}/ledger", response_model=LedgerResponse)
def plan_ledger(run_id: str, plan_id: str):
    run_path = os.path.join(_RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(run_path):
        raise HTTPException(404, f"run not found: {run_id}")
    with open(run_path) as f:
        run_record = json.load(f)
    if plan_id not in run_record["plans"]:
        raise HTTPException(404, f"plan not found: {plan_id} in run {run_id}")

    ledger = run_record["plans"][plan_id]["ledger"]
    return LedgerResponse(
        plan_id=plan_id,
        fuel_cost_usd=ledger["fuel_cost_usd"], ets_cost_usd=ledger["ets_cost_usd"],
        demand_penalty_usd=ledger["demand_penalty_usd"], fueleu_intensity=ledger["fueleu_intensity"],
        fueleu_limit=ledger["fueleu_limit"], fueleu_compliant=ledger["fueleu_compliant"],
        total_co2_ttw_t=ledger["total_co2_ttw_t"], risk_hours=ledger["risk_hours"],
        n_legs=ledger["n_legs"],
    )


@app.get("/run/{run_id}")
def get_run(run_id: str):
    """Re-fetch a full run's Pareto front after the fact - needed so the
    frontend's explorer page can be revisited/shared without re-running
    the optimizer. Reads from the same results/runs/*.json the offline
    dashboard mode uses."""
    run_path = os.path.join(_RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(run_path):
        raise HTTPException(404, f"run not found: {run_id}")
    with open(run_path) as f:
        run_record = json.load(f)
    return {
        "run_id": run_record["run_id"],
        "elapsed_s": run_record["elapsed_s"],
        "pareto_front": [
            {
                "plan_id": pid,
                "J1_cost_usd": p["obj"][0],
                "J2_ghg_intensity": p["obj"][1],
                "J3_schedule_risk": p["obj"][2],
                "fueleu_compliant": p["ledger"]["fueleu_compliant"],
            }
            for pid, p in run_record["plans"].items()
        ],
    }
