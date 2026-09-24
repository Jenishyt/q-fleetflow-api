---
title: Q-FleetFlow API
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Q-FleetFlow — Quantum-Inspired Fuel Prediction & Green Fleet Optimization

SIH 2026 prototype. This README reflects the ACTUAL state of the code, not
the aspirational plan — see "Honest status" below.

## Setup

```bash
pip install -r requirements.txt
```

## What's built and tested (12/12 tests passing)

Run everything:
```bash
python3 -m pytest tests/ -v
```

Run individual pieces:
```bash
python3 src/data/synth_generator.py      # generates data/processed/synthetic.parquet
python3 src/models/gbm.py                # trains prediction engine, prints MAPE gate
python3 src/models/predict_api.py        # tests the predict_fuel() interface + latency gate
python3 tests/test_qiea_toy.py           # QIEA core convergence proof (knapsack toy)
python3 tests/test_compliance.py         # LNG-trap worked example + compliance tests
python3 tests/test_fitness_repair.py     # end-to-end repair -> fitness smoke test
python3 src/optimizer/qiea.py            # full QIEA run, prints Pareto front + timing gate
python3 src/optimizer/baselines.py       # greedy / random / NSGA-II comparison
```

## Layer status

| Layer | Files | Status |
|---|---|---|
| 1. Data | `src/data/synth_generator.py`, `src/data/kaggle_loader.py`, `src/models/physics.py` | Synthetic fallback done and used everywhere. Kaggle loader written but **NEVER RUN against real data** - no network access to kaggle.com from the build environment. See "Honest status". |
| 2. Prediction | `src/models/gbm.py`, `src/models/predict_api.py` | Done on synthetic data. MAPE 3.33%, R²=0.996, 12.7ms/100-row batch. Quantile coverage 73.4% (target 75-90%, slightly under). |
| 3. Optimizer | `src/optimizer/{register,gates,encoding,repair,fitness,qiea}.py` | Done. Batched fitness evaluation: full QIEA run ~5-9s (gate <120s, large margin). |
| 4. Compliance | `src/compliance/{fueleu,ets,cii}.py`, `configs/factors.yaml` | Done. 7/7 tests pass incl. LNG-trap worked example. |
| 5. Benchmarking | `src/optimizer/baselines.py`, `src/optimizer/benchmark.py` | Done. Full 10-seed x 3-algorithm protocol with Wilcoxon test, `results/benchmark_table.csv`. |
| 6. API + dashboard | `src/api/{main,schemas}.py`, `src/dashboard/app.py` | Done. FastAPI: 4 endpoints, 5/5 tests pass. Streamlit: 4 tabs, offline-JSON-read mode, headless-tested end to end (Run button -> Pareto explorer -> ledger). |

## Full benchmark result (10 seeds, real numbers - see results/benchmark_table.csv)

| Algorithm | Hypervolume (mean +/- std) | Feasibility rate | Best cost found |
|---|---|---|---|
| Greedy | 10.7B | 0% | $142,551 |
| Random search | 21.3B +/- 0.3B | 67.1% | $109,018 |
| QIEA | 22.7B +/- 0.2B | **86.9%** | $109,755 |
| NSGA-II | **27.7B +/- 0.4B** | 73.0% | **$61,263** |

QIEA vs random: p=0.002 (significant, QIEA wins). QIEA vs NSGA-II: p=0.002 (significant, NSGA-II wins on hypervolume). QIEA's feasibility-rate advantage is the honest, defensible story - not "QIEA wins on everything." (Re-run after widening `scenario.yaml`'s available_hours_per_week so J3/schedule-risk actually triggers on some plans - previously it was a degenerate always-zero objective.)

## Honest status — read this before putting anything in a slide

- **All data is synthetic.** Real Kaggle data was never downloaded or integrated - the sandbox this was built in has no network access to kaggle.com. `src/data/kaggle_loader.py` exists and its own internal logic is tested (against a fake CSV matching the assumed schema), but it has never seen the real file. Expect to spend 15-30 minutes fixing `COLUMN_MAP` once you run it against the actual download. Every MAPE/R² number in this repo is measured on data generated FROM the same physics model the predictor uses - it will look better than real data ever will. Say this out loud to judges.
- **NSGA-II beats QIEA on raw hypervolume**, statistically significantly, across 10 seeds. QIEA has a statistically significant, meaningfully higher feasibility rate. Both directions are real, not cherry-picked.
- **The dashboard's Run tab launches the optimizer synchronously** (blocks the UI until done) - fine at ~5-9s per run, but if you increase n_generations a lot, add a background-thread + progress-bar pattern (plan Sec 9 mentions this; not yet implemented).
- The scenario used throughout (`configs/scenario.yaml`) is a small illustrative example: 3 routes, 4 weeks, 2 slots/week = 24 genes. J3 (schedule risk) never triggers in this scenario - worth widening before a real demo if you want all 3 objectives to visibly matter.
- Not built: SHAP top-3 explanation wiring into the API response (physics.py/gbm.py support it, predict_api.py's `shap_top3` field is currently always empty), weighted single-objective GA baseline (plan mentions it as a 4th comparator, not built), PDF export (dashboard exports CSV only, not the one-page PDF the plan mentions).

## Running the dashboard + API together

```bash
# terminal 1
uvicorn src.api.main:app --reload --port 8000

# terminal 2
streamlit run src/dashboard/app.py
```

The dashboard does NOT currently call the API - it runs the optimizer directly in-process (the offline-safe path). The API exists as a separate, independently-tested interface for anything else that wants to talk to the optimizer (e.g. a future front-end, or judges poking at `/docs`).

## Repo layout

```
configs/
  factors.yaml      - emission factors, sourced or flagged assumption:true
  scenario.yaml      - example fleet scenario (routes, demand, prices)
data/
  processed/synthetic.parquet   - generated by synth_generator.py
src/
  data/
    synth_generator.py       - deterministic synthetic fuel dataset
    kaggle_loader.py           - real Kaggle CSV loader (UNTESTED against real data - see Honest status)
  models/
    physics.py       - admiralty cubic law prior
    gbm.py            - LightGBM residual + quantile heads, training script
    predict_api.py    - FuelPredictor, the one public predict_fuel() interface
  compliance/
    fueleu.py         - FuelEU Maritime WtW intensity check
    ets.py            - EU ETS cost
    cii.py            - IMO CII rating bands
  optimizer/
    register.py       - Q-bit probability register
    gates.py           - rotation-gate update (the "quantum" mechanism)
    encoding.py         - gene/allele encoding, Scenario/Route dataclasses
    repair.py            - fuel-availability + demand repair operators
    fitness.py            - 3-objective scoring, batched evaluate_population()
    qiea.py                - full multi-objective main loop + Pareto archive
    baselines.py            - random search, greedy heuristic, NSGA-II (pymoo)
    benchmark.py              - 10-seed protocol + Wilcoxon signed-rank test
  api/
    schemas.py         - pydantic request/response models
    main.py              - FastAPI app: /predict /optimize /plan/.../ledger /health
  dashboard/
    app.py                - Streamlit: scenario builder, run, Pareto explorer, ledger
tests/
  test_qiea_toy.py    - Day-1 gate: knapsack convergence proof
  test_compliance.py  - LNG-trap worked example + 6 other compliance tests
  test_fitness_repair.py - end-to-end repair->fitness pipeline test
  test_api.py            - FastAPI endpoint tests via TestClient (5 tests)
```

## Running the dashboard + API together

```bash
# terminal 1
uvicorn src.api.main:app --reload --port 8000

# terminal 2
streamlit run src/dashboard/app.py
```

The dashboard does NOT call the API - it runs the optimizer directly
in-process (the offline-safe path, plan Sec 9 risk R4). The API exists as
a separate, independently-tested interface for anything else that wants
to talk to the optimizer.

## Next steps (not yet built)

1. Run `kaggle_loader.py` against the REAL downloaded CSV, fix `COLUMN_MAP` to match its actual headers, re-check the MAPE gate on real data (expect worse than 3.33% - that's normal)
2. Weighted single-objective GA baseline (plan's 4th comparator - not built)
3. SHAP top-3 wiring into `/predict`'s response (currently always empty)
4. Background-thread + progress bar for longer optimizer runs in the dashboard
5. PDF export for the decision memo (CSV export works; PDF does not exist yet)
6. Widen `scenario.yaml` so J3 (schedule risk) actually triggers sometimes
7. Possible QIEA tuning to close the gap with NSGA-II on raw hypervolume
