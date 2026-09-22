"""
Streamlit dashboard (plan Sec 9): 4 tabs. Reads results/runs/*.json
directly rather than depending on a live API round-trip - this is the
plan's own R4 risk mitigation, and the single most important engineering
decision for demo-day reliability. If the FastAPI service crashes or
isn't running, this dashboard still works off whatever's already in
results/runs/.

Run with: streamlit run src/dashboard/app.py
"""
from __future__ import annotations
import os
import sys
import json
import glob
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Q-FleetFlow", layout="wide")

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_RUNS_DIR = os.path.join(_REPO_ROOT, "results", "runs")
_CONFIGS_DIR = os.path.join(_REPO_ROOT, "configs")


def list_runs() -> list[str]:
    if not os.path.isdir(_RUNS_DIR):
        return []
    return sorted(glob.glob(os.path.join(_RUNS_DIR, "*.json")), reverse=True)


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Training prediction engine (first load only)...")
def get_cached_predictor():
    """Without caching, FuelPredictor.from_synthetic() retrains LightGBM
    from scratch on every cold start / rerun - fine locally, but makes a
    deployed app feel slow. st.cache_resource keeps one trained instance
    alive across reruns and across users of the same deployment."""
    from src.models.predict_api import FuelPredictor
    return FuelPredictor.from_synthetic()


st.title("Q-FleetFlow")
st.caption("Quantum-inspired fuel prediction & green fleet optimization")

tab1, tab2, tab3, tab4 = st.tabs(["Scenario builder", "Run", "Pareto explorer", "Compliance ledger"])

# ============================================================ TAB 1 =====
with tab1:
    st.subheader("Scenario builder")
    st.caption("Edit a scenario.yaml, or upload one, then run it from the Run tab.")

    scenario_files = glob.glob(os.path.join(_CONFIGS_DIR, "*.yaml"))
    chosen = st.selectbox("Scenario file", scenario_files, index=0 if scenario_files else None)

    year = st.select_slider("Target year", options=[2026, 2030, 2035], value=2026)
    st.caption(f"Changing the year live-updates which FuelEU/CII limits apply for that run "
               f"(currently: {year}) - this is set in the run request, not saved back to the YAML.")

    if chosen:
        with open(chosen) as f:
            content = f.read()
        st.text_area("Scenario contents (read-only preview)", content, height=280)

    uploaded = st.file_uploader("Or upload a scenario.yaml", type=["yaml", "yml"])
    if uploaded:
        st.info("Uploaded scenarios aren't wired to the Run tab yet in this MVP - "
                "save it into configs/ and select it above instead.")

# ============================================================ TAB 2 =====
with tab2:
    st.subheader("Run the optimizer")
    st.caption("Runs the QIEA optimizer directly in this process (no API server required).")

    col1, col2, col3 = st.columns(3)
    n_pop = col1.number_input("Population", min_value=4, max_value=200, value=40)
    n_gen = col2.number_input("Generations", min_value=10, max_value=1000, value=150)
    seed = col3.number_input("Seed", min_value=0, value=42)

    scenario_path_rel = st.selectbox(
        "Scenario", [os.path.relpath(p, _REPO_ROOT) for p in scenario_files],
        key="run_scenario",
    ) if scenario_files else None

    if st.button("Run QIEA", type="primary"):
        from src.optimizer.encoding import Scenario
        from src.optimizer.qiea import run_qiea
        import uuid

        progress = st.progress(0, text="Loading scenario and predictor...")
        scenario = Scenario.from_yaml(os.path.join(_REPO_ROOT, scenario_path_rel))
        predictor = get_cached_predictor()

        progress.progress(30, text=f"Running QIEA ({n_pop} pop x {n_gen} gen)...")
        start = time.time()
        archive, _ = run_qiea(scenario, predictor, n_pop=int(n_pop),
                               n_generations=int(n_gen), seed=int(seed), verbose=False)
        elapsed = time.time() - start
        progress.progress(90, text="Saving results...")

        run_id = str(uuid.uuid4())[:8]
        plans_out = {}
        for i, e in enumerate(archive):
            plan_id = f"plan_{i:03d}"
            J1, J2, J3 = e["obj"]
            plans_out[plan_id] = {
                "plan": [int(a) for a in e["plan"]],
                "obj": [float(J1), float(J2), float(J3)],
                "ledger": {k: (v.item() if hasattr(v, "item") else v) for k, v in e["ledger"].items()},
            }
        os.makedirs(_RUNS_DIR, exist_ok=True)
        with open(os.path.join(_RUNS_DIR, f"{run_id}.json"), "w") as f:
            json.dump({"run_id": run_id, "scenario_path": scenario_path_rel,
                       "n_pop": int(n_pop), "n_generations": int(n_gen), "seed": int(seed),
                       "elapsed_s": elapsed, "plans": plans_out}, f, indent=2)

        progress.progress(100, text="Done")
        st.success(f"Run {run_id} complete in {elapsed:.1f}s - {len(archive)} Pareto-optimal plans found. "
                   f"See the Pareto explorer tab.")

# ============================================================ TAB 3 =====
with tab3:
    st.subheader("Pareto explorer")
    runs = list_runs()

    if not runs:
        st.info("No runs yet. Use the Run tab to generate one, or drop a JSON file into "
                 "results/runs/ manually.")
    else:
        run_labels = [os.path.basename(r) for r in runs]
        selected_label = st.selectbox("Run", run_labels)
        run_data = load_run(runs[run_labels.index(selected_label)])

        plans = run_data["plans"]
        rows = [
            {"plan_id": pid, "J1_cost": p["obj"][0], "J2_ghg": p["obj"][1],
             "J3_risk": p["obj"][2], "compliant": p["ledger"]["fueleu_compliant"]}
            for pid, p in plans.items()
        ]
        df = pd.DataFrame(rows)

        fig = go.Figure(data=[go.Scatter3d(
            x=df["J1_cost"], y=df["J2_ghg"], z=df["J3_risk"],
            mode="markers",
            marker=dict(
                size=8,
                color=df["compliant"].map({True: "#1fb583", False: "#e2673f"}),
            ),
            text=df["plan_id"],
            hovertemplate="%{text}<br>Cost: $%{x:,.0f}<br>GHG: %{y:.2f}<br>Risk: %{z:.1f}<extra></extra>",
        )])
        fig.update_layout(
            scene=dict(xaxis_title="J1 Cost ($)", yaxis_title="J2 GHG intensity", zaxis_title="J3 Risk"),
            height=550, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Green = FuelEU compliant, orange = non-compliant. Click a point's row below to "
                   "inspect it in the Compliance ledger tab.")

        selected_plan = st.selectbox("Inspect plan", df["plan_id"])
        st.session_state["_selected_run"] = selected_label
        st.session_state["_selected_plan"] = selected_plan

        plan_detail = plans[selected_plan]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Cost", f"${plan_detail['obj'][0]:,.0f}")
            st.metric("GHG intensity", f"{plan_detail['obj'][1]:.2f} gCO2e/MJ")
        with c2:
            st.metric("Schedule risk", f"{plan_detail['obj'][2]:.1f}")
            st.metric("FuelEU compliant", "Yes" if plan_detail["ledger"]["fueleu_compliant"] else "No")

        # fuel-mix donut
        from src.optimizer.encoding import decode_allele
        fuel_counts: dict[str, int] = {}
        for allele in plan_detail["plan"]:
            _, fuel, _ = decode_allele(allele)
            fuel_counts[fuel] = fuel_counts.get(fuel, 0) + 1
        donut = go.Figure(data=[go.Pie(labels=list(fuel_counts.keys()), values=list(fuel_counts.values()), hole=0.5)])
        donut.update_layout(title="Fuel mix (by number of legs)", height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(donut, width='stretch')

# ============================================================ TAB 4 =====
with tab4:
    st.subheader("Compliance ledger")
    runs = list_runs()

    if not runs:
        st.info("No runs yet - see the Run tab.")
    else:
        run_label = st.session_state.get("_selected_run", os.path.basename(runs[0]))
        run_labels = [os.path.basename(r) for r in runs]
        run_label = st.selectbox("Run", run_labels,
                                  index=run_labels.index(run_label) if run_label in run_labels else 0,
                                  key="ledger_run")
        run_data = load_run(runs[run_labels.index(run_label)])
        plans = run_data["plans"]

        plan_id = st.session_state.get("_selected_plan", list(plans.keys())[0])
        if plan_id not in plans:
            plan_id = list(plans.keys())[0]
        plan_id = st.selectbox("Plan", list(plans.keys()),
                                index=list(plans.keys()).index(plan_id), key="ledger_plan")

        ledger = plans[plan_id]["ledger"]

        st.markdown("#### Cost breakdown")
        cost_df = pd.DataFrame([
            {"line": "Fuel cost", "usd": ledger["fuel_cost_usd"], "assumption": False},
            {"line": "EU ETS cost", "usd": ledger["ets_cost_usd"], "assumption": True},
            {"line": "Demand penalty", "usd": ledger["demand_penalty_usd"], "assumption": False},
        ])
        for _, row in cost_df.iterrows():
            badge = " :orange[(assumption-based price)]" if row["assumption"] else ""
            st.write(f"**{row['line']}**: ${row['usd']:,.2f}{badge}")

        st.markdown("#### FuelEU Maritime")
        compliant = ledger["fueleu_compliant"]
        color = "green" if compliant else "red"
        st.markdown(f"Intensity: **{ledger['fueleu_intensity']:.2f}** gCO2e/MJ vs limit "
                    f"**{ledger['fueleu_limit']:.2f}** -> :{color}[{'COMPLIANT' if compliant else 'NON-COMPLIANT'}]")

        st.markdown("#### Other")
        st.write(f"Total CO2 (tank-to-wake): {ledger['total_co2_ttw_t']:.1f} t")
        st.write(f"Schedule risk hours: {ledger['risk_hours']:.1f}")
        st.write(f"Legs in plan: {ledger['n_legs']}")

        export_df = pd.DataFrame([ledger])
        st.download_button("Export decision memo (CSV)", export_df.to_csv(index=False),
                            file_name=f"{run_data['run_id']}_{plan_id}_ledger.csv")
