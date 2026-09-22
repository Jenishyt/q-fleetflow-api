"""
GBM residual layer: learns the log-residual between observed fuel and the
physics prior, i.e. everything the admiralty cubic law doesn't capture
(hull fouling, real weather interaction, engine efficiency curves, etc).

    F_hat(x) = F_phys(x) * exp(r_theta(x))
    r_theta  = LightGBM regression on log(F_obs / F_phys)

Plus twin quantile models (alpha=0.1, 0.9) on the SAME residual target for
uncertainty bounds. This file only trains + evaluates; predict_api.py
(next) exposes the single predict_fuel() function the optimizer calls.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.physics import f_phys, fit_calibration, physics_only_mape, VESSEL_CLASSES

FEATURE_COLS = [
    "speed_kn", "draft_ratio", "wind_kn", "wave_hs_m", "temp_c",
    "speed_sq", "speed_draft",
]


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speed_sq"] = df["speed_kn"] ** 2
    df["speed_draft"] = df["speed_kn"] * df["draft_ratio"]
    return df


def _add_residual_target(df: pd.DataFrame, c_class: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    phys = np.array([
        f_phys(row.speed_kn, row.draft_ratio, row.vessel_class, c_class)
        for row in df.itertuples()
    ])
    df["f_phys"] = phys
    df["log_residual"] = np.log(df["fuel_t_per_day"] / df["f_phys"])
    return df


def group_split(df: pd.DataFrame, seed: int = 42):
    """Split by voyage_id (never by row) so the same voyage never leaks
    across train/val/test - this is the detail most teams get wrong."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=df["voyage_id"]))
    trainval = df.iloc[trainval_idx]
    test = df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)  # 0.125*0.8=0.1
    train_idx, val_idx = next(gss2.split(trainval, groups=trainval["voyage_id"]))
    train = trainval.iloc[train_idx]
    val = trainval.iloc[val_idx]
    return train, val, test


def _prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vessel_class"] = df["vessel_class"].astype("category")
    df["fuel_type"] = df["fuel_type"].astype("category")
    return df


def train_models(train: pd.DataFrame, val: pd.DataFrame):
    cat_cols = ["vessel_class", "fuel_type"]
    cols = FEATURE_COLS + cat_cols

    train = _prep_categoricals(train)
    val = _prep_categoricals(val)

    lgb_train = lgb.Dataset(train[cols], label=train["log_residual"], categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(val[cols], label=val["log_residual"], reference=lgb_train, categorical_feature=cat_cols)

    params_point = dict(objective="regression", metric="mae", verbosity=-1, seed=42,
                         num_leaves=15, learning_rate=0.05, min_data_in_leaf=10)
    model_point = lgb.train(params_point, lgb_train, num_boost_round=300,
                             valid_sets=[lgb_val],
                             callbacks=[lgb.early_stopping(20, verbose=False)])

    models_q = {}
    for alpha in (0.1, 0.9):
        params_q = dict(objective="quantile", alpha=alpha, metric="quantile",
                         verbosity=-1, seed=42, num_leaves=15, learning_rate=0.05,
                         min_data_in_leaf=10)
        models_q[alpha] = lgb.train(params_q, lgb_train, num_boost_round=300,
                                     valid_sets=[lgb_val],
                                     callbacks=[lgb.early_stopping(20, verbose=False)])

    return model_point, models_q, cols


def evaluate(df_test: pd.DataFrame, model_point, models_q, cols: list[str], c_class: dict[str, float]) -> dict:
    df_test = _prep_categoricals(df_test)
    r_pred = model_point.predict(df_test[cols], num_iteration=model_point.best_iteration)
    f_hat = df_test["f_phys"].to_numpy() * np.exp(r_pred)
    actual = df_test["fuel_t_per_day"].to_numpy()

    mape = float(np.mean(np.abs((actual - f_hat) / actual)) * 100)
    rmse = float(np.sqrt(np.mean((actual - f_hat) ** 2)))
    ss_res = np.sum((actual - f_hat) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    r_q10 = models_q[0.1].predict(df_test[cols], num_iteration=models_q[0.1].best_iteration)
    r_q90 = models_q[0.9].predict(df_test[cols], num_iteration=models_q[0.9].best_iteration)
    q10 = df_test["f_phys"].to_numpy() * np.exp(r_q10)
    q90 = df_test["f_phys"].to_numpy() * np.exp(r_q90)
    coverage = float(np.mean((actual >= q10) & (actual <= q90)) * 100)

    phys_mape = physics_only_mape(df_test, c_class)

    return dict(mape=mape, rmse=rmse, r2=r2, coverage=coverage, physics_only_mape=phys_mape)


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "synthetic.parquet"))
    df = _add_engineered_features(df)

    train, val, test = group_split(df)
    c_class = fit_calibration(train)

    train = _add_residual_target(train, c_class)
    val = _add_residual_target(val, c_class)
    test = _add_residual_target(test, c_class)

    model_point, models_q, cols = train_models(train, val)
    metrics = evaluate(test, model_point, models_q, cols, c_class)

    print("\n=== Day 2 acceptance gate ===")
    print(f"MAPE (physics+GBM):     {metrics['mape']:.2f}%   (gate: <=15%)")
    print(f"R^2:                    {metrics['r2']:.3f}   (gate: >=0.85)")
    print(f"Physics-only MAPE:      {metrics['physics_only_mape']:.2f}%  (baseline, expect higher)")
    print(f"RMSE:                   {metrics['rmse']:.3f} t/day")
    print(f"Quantile coverage:      {metrics['coverage']:.1f}%   (gate: 75-90%)")

    gate_mape = metrics['mape'] <= 15
    gate_r2 = metrics['r2'] >= 0.85
    print(f"\nGate MAPE<=15%:  {'PASS' if gate_mape else 'FAIL'}")
    print(f"Gate R2>=0.85:   {'PASS' if gate_r2 else 'FAIL'}")
