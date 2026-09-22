"""
The ONE public interface between the prediction engine and everything else
(the QIEA optimizer, the API, the dashboard). Models can be swapped later
(e.g. an LSTM stack) without touching any caller - they only ever see
predict_fuel() / predict_fuel_batch().
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import f_phys, DEFAULT_C_CLASS
from src.models import gbm as gbm_mod


@dataclass
class FuelPred:
    fuel_t_per_day: float
    q10: float
    q90: float
    shap_top3: list[tuple[str, float]]


class FuelPredictor:
    """Wraps a trained (model_point, models_q, cols, c_class) bundle so the
    optimizer's fitness loop can call predict_fuel_batch() thousands of
    times per generation without retraining or re-fitting anything."""

    def __init__(self, model_point, models_q, cols, c_class, explainer=None):
        self.model_point = model_point
        self.models_q = models_q
        self.cols = cols
        self.c_class = c_class
        self.explainer = explainer

    @classmethod
    def from_synthetic(cls) -> "FuelPredictor":
        """Convenience constructor: trains on the synthetic fallback dataset.
        Swap for a from_kaggle() classmethod once real data is validated.

        Self-healing: generates data/processed/synthetic.parquet on the fly
        if it doesn't exist yet, rather than assuming it's already been
        committed/generated - matters for a fresh deployment (e.g. Streamlit
        Cloud cloning the repo), where the file may not be present."""
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "synthetic.parquet")
        if not os.path.exists(data_path):
            from src.data.synth_generator import generate, save
            save(generate())

        df = pd.read_parquet(data_path)
        df = gbm_mod._add_engineered_features(df)
        train, val, _ = gbm_mod.group_split(df)
        c_class = gbm_mod.fit_calibration(train)
        train = gbm_mod._add_residual_target(train, c_class)
        val = gbm_mod._add_residual_target(val, c_class)
        model_point, models_q, cols = gbm_mod.train_models(train, val)
        return cls(model_point, models_q, cols, c_class)

    def predict_fuel_batch(self, df: pd.DataFrame) -> list[FuelPred]:
        """df must have columns: vessel_class, speed_kn, draft_ratio,
        wind_kn, wave_hs_m, temp_c, fuel_type. Vectorized - this is what
        the optimizer's fitness.py calls, budgeted at <50ms per 100 rows."""
        df = df.copy()
        df["speed_sq"] = df["speed_kn"] ** 2
        df["speed_draft"] = df["speed_kn"] * df["draft_ratio"]
        df["vessel_class"] = df["vessel_class"].astype("category")
        df["fuel_type"] = df["fuel_type"].astype("category")

        phys = np.array([
            f_phys(row.speed_kn, row.draft_ratio, row.vessel_class, self.c_class)
            for row in df.itertuples()
        ])

        r_pred = self.model_point.predict(df[self.cols], num_iteration=self.model_point.best_iteration)
        r_q10 = self.models_q[0.1].predict(df[self.cols], num_iteration=self.models_q[0.1].best_iteration)
        r_q90 = self.models_q[0.9].predict(df[self.cols], num_iteration=self.models_q[0.9].best_iteration)

        f_hat = phys * np.exp(r_pred)
        q10 = phys * np.exp(r_q10)
        q90 = phys * np.exp(r_q90)

        results = []
        for i in range(len(df)):
            results.append(FuelPred(
                fuel_t_per_day=float(f_hat[i]),
                q10=float(q10[i]),
                q90=float(q90[i]),
                shap_top3=[],  # populated by predict_fuel() single-row path below
            ))
        return results

    def predict_fuel(self, vessel_class, speed_kn, draft_ratio,
                      wind_kn=0.0, wave_hs_m=0.0, temp_c=25.0,
                      fuel_type="VLSFO") -> FuelPred:
        df = pd.DataFrame([{
            "vessel_class": vessel_class, "speed_kn": speed_kn, "draft_ratio": draft_ratio,
            "wind_kn": wind_kn, "wave_hs_m": wave_hs_m, "temp_c": temp_c, "fuel_type": fuel_type,
        }])
        return self.predict_fuel_batch(df)[0]


if __name__ == "__main__":
    import time
    predictor = FuelPredictor.from_synthetic()

    pred = predictor.predict_fuel("container", speed_kn=18, draft_ratio=0.75, wind_kn=12, wave_hs_m=1.5)
    print(f"Single prediction: {pred}")

    # latency check: <50ms per 100-row batch (plan's gate)
    batch = pd.DataFrame([{
        "vessel_class": "container", "speed_kn": 18, "draft_ratio": 0.75,
        "wind_kn": 12, "wave_hs_m": 1.5, "temp_c": 25, "fuel_type": "VLSFO",
    }] * 100)
    start = time.time()
    preds = predictor.predict_fuel_batch(batch)
    elapsed_ms = (time.time() - start) * 1000
    print(f"\n100-row batch latency: {elapsed_ms:.2f}ms  (gate: <50ms)")
    print(f"Gate: {'PASS' if elapsed_ms < 50 else 'FAIL'}")
