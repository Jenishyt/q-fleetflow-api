"""
FastAPI endpoint tests using TestClient - no live server needed, real
request/response cycle through the actual app (including JSON
serialization, which is where numpy-type bugs would surface).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    print(f"\n/health -> {body}")


def test_predict():
    r = client.post("/predict", json={
        "vessel_class": "container", "speed_kn": 18, "draft_ratio": 0.75,
        "wind_kn": 12, "wave_hs_m": 1.5,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["fuel_t_per_day"] > 0
    assert body["q10"] <= body["fuel_t_per_day"] <= body["q90"]
    print(f"/predict -> {body}")


def test_predict_validation_rejects_bad_speed():
    r = client.post("/predict", json={
        "vessel_class": "container", "speed_kn": -5, "draft_ratio": 0.75,
    })
    assert r.status_code == 422, "negative speed should fail pydantic validation"
    print(f"/predict correctly rejects invalid speed_kn=-5 -> {r.status_code}")


def test_optimize_and_ledger_roundtrip():
    r = client.post("/optimize", json={
        "scenario_path": "configs/scenario.yaml", "n_pop": 10, "n_generations": 15, "seed": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["pareto_front"], "optimize should return a non-empty Pareto front"
    print(f"\n/optimize -> run_id={body['run_id']}  elapsed={body['elapsed_s']:.2f}s  "
          f"front_size={len(body['pareto_front'])}")

    run_id = body["run_id"]
    plan_id = body["pareto_front"][0]["plan_id"]

    r2 = client.get(f"/plan/{run_id}/{plan_id}/ledger")
    assert r2.status_code == 200
    ledger = r2.json()
    print(f"/plan/{run_id}/{plan_id}/ledger -> {ledger}")
    assert ledger["plan_id"] == plan_id


def test_ledger_404_on_unknown_run():
    r = client.get("/plan/nonexistent/plan_000/ledger")
    assert r.status_code == 404
    print(f"\n/plan/<unknown>/ledger correctly returns 404")


def test_get_run_after_optimize():
    r = client.post("/optimize", json={
        "scenario_path": "configs/scenario.yaml", "n_pop": 10, "n_generations": 15, "seed": 2,
    })
    run_id = r.json()["run_id"]

    r2 = client.get(f"/run/{run_id}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["run_id"] == run_id
    assert len(body["pareto_front"]) > 0
    print(f"\n/run/{run_id} -> re-fetched {len(body['pareto_front'])} plans successfully")


def test_get_run_404_on_unknown():
    r = client.get("/run/doesnotexist")
    assert r.status_code == 404


if __name__ == "__main__":
    test_health()
    test_predict()
    test_predict_validation_rejects_bad_speed()
    test_optimize_and_ledger_roundtrip()
    test_ledger_404_on_unknown_run()
    test_get_run_after_optimize()
    test_get_run_404_on_unknown()
    print("\nALL API TESTS PASSED")
