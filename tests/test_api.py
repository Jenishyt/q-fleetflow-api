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


def test_memo_pdf_generation():
    r = client.post("/optimize", json={
        "scenario_path": "configs/scenario.yaml", "n_pop": 10, "n_generations": 15, "seed": 3,
    })
    run_id = r.json()["run_id"]
    plan_id = r.json()["pareto_front"][0]["plan_id"]

    r2 = client.get(f"/plan/{run_id}/{plan_id}/memo.pdf")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "application/pdf"
    assert r2.content[:5] == b"%PDF-", "response should be a real PDF, not just PDF-flavored bytes"
    assert len(r2.content) > 500, "PDF suspiciously small - likely missing content"
    print(f"\n/plan/{run_id}/{plan_id}/memo.pdf -> valid PDF, {len(r2.content)} bytes")


def test_memo_pdf_404_on_unknown_plan():
    r = client.get("/plan/nonexistent/plan_000/memo.pdf")
    assert r.status_code == 404


def test_ports_list():
    r = client.get("/ports")
    assert r.status_code == 200
    ports = r.json()["ports"]
    assert len(ports) > 0
    assert "lat" in ports[0] and "lon" in ports[0] and "locode" in ports[0]


def test_fleet_get_seeded():
    r = client.get("/fleet")
    assert r.status_code == 200
    vessels = r.json()["vessels"]
    assert len(vessels) >= 4  # one seed per vessel class
    classes = {v["vessel_class"] for v in vessels}
    assert "container" in classes


def test_fleet_register_uses_real_class_constants():
    r = client.post("/fleet", json={"name": "Test Vessel", "vessel_class": "tanker", "fuel_type": "LNG"})
    assert r.status_code == 200
    v = r.json()
    assert v["vessel_class"] == "tanker"
    assert v["capacity_dwt"] == 60000.0  # matches physics.py's DWT_LOOKUP["tanker"], not a fabricated number
    assert v["status"] == "Available"


def test_fleet_register_rejects_unknown_class():
    r = client.post("/fleet", json={"name": "Bad Ship", "vessel_class": "spaceship"})
    assert r.status_code == 422


def test_routes_geometry_resolves_real_ports():
    r = client.get("/routes")
    assert r.status_code == 200
    routes = r.json()["routes"]
    assert len(routes) == 3
    for route in routes:
        assert route["origin"] is not None, f"{route['name']} origin port should resolve"
        assert route["destination"] is not None, f"{route['name']} destination port should resolve"
        assert -90 <= route["origin"]["lat"] <= 90


if __name__ == "__main__":
    test_health()
    test_predict()
    test_predict_validation_rejects_bad_speed()
    test_optimize_and_ledger_roundtrip()
    test_ledger_404_on_unknown_run()
    test_get_run_after_optimize()
    test_get_run_404_on_unknown()
    test_memo_pdf_generation()
    test_memo_pdf_404_on_unknown_plan()
    test_ports_list()
    test_fleet_get_seeded()
    test_fleet_register_uses_real_class_constants()
    test_fleet_register_rejects_unknown_class()
    test_routes_geometry_resolves_real_ports()
    print("\nALL API TESTS PASSED")
