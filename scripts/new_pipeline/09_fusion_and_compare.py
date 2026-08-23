#!/usr/bin/env python3
"""Leakage-free fusion and Stage-3 model comparison.

Fusion weights are selected using *validation* predictions only, then frozen
before the final test forecast.  Every model must use the time-eligible
stratified cohort built by ``_subset.py``; comparison therefore never mixes
different series counts.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(BASE, "data", "processed_new", "stage3")
os.makedirs(PROC, exist_ok=True)
FIG = os.path.join(BASE, "figures_new")


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}


def overall_wmape(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    return np.sum(np.abs(a - p)) / np.sum(np.abs(a)) * 100 if np.sum(np.abs(a)) > 0 else np.nan


# per-month prediction files (series|date|actual|pred) for volume-weighted WMAPE
PREDS = {
    "ARIMA": "arima_preds.csv",
    "Prophet": "prophet_preds.csv",
    "Prophet+exog": "prophet_exog_preds.csv",
    "XGBoost": "xgboost_preds.csv",
    "LSTM": "lstm_preds.csv",
}


def agg_results(path, model, cohort):
    r = pd.read_csv(path)
    ok = r[r["status"] == "ok"] if "status" in r.columns else r
    ok = ok[ok["series_name"].astype(str).isin(cohort)]
    row = {
        "model": model,
        "WMAPE_mean": ok["WMAPE"].mean(),     # mean of per-series WMAPE (outlier-sensitive)
        "WMAPE_med": ok["WMAPE"].median(),    # median of per-series WMAPE (robust)
        "WMAPE_vol": np.nan,                  # volume-weighted aggregate WMAPE (robust, headline)
        "MAPE": ok["MAPE"].mean(),
        "RMSE": ok["RMSE"].mean(),
        "MAE": ok["MAE"].mean(),
        "n_series": len(ok),
    }
    pp = os.path.join(PROC, PREDS.get(model, ""))
    if os.path.exists(pp):
        p = pd.read_csv(pp)
        p = p[p["series_name"].astype(str).isin(cohort)]
        a = p["actual"].values.astype(float)
        pr = p["pred"].values.astype(float)
        row["WMAPE_vol"] = np.sum(np.abs(a - pr)) / np.sum(np.abs(a)) * 100 if np.sum(np.abs(a)) > 0 else np.nan
    return row


def main():
    # Select fusion weights on validation, never on the test actuals.
    vp = pd.read_csv(os.path.join(PROC, "prophet_val_preds.csv"))
    vx = pd.read_csv(os.path.join(PROC, "xgboost_val_preds.csv"))
    vm = vp.merge(vx, on=["series_name", "date"], suffixes=("_prophet", "_xgboost"))
    if vm.empty:
        raise RuntimeError("No overlapping validation predictions for fusion.")
    vactual = vm["actual_prophet"].values
    vwp = overall_wmape(vactual, vm["pred_prophet"].values)
    vwx = overall_wmape(vactual, vm["pred_xgboost"].values)
    w_p = (1.0 / vwp) / ((1.0 / vwp) + (1.0 / vwx))
    w_x = 1.0 - w_p

    # Apply the frozen weights once on the held-out test predictions.
    pp = pd.read_csv(os.path.join(PROC, "prophet_preds.csv"))
    xp = pd.read_csv(os.path.join(PROC, "xgboost_preds.csv"))
    m = pp.merge(xp, on=["series_name", "date"], suffixes=("_prophet", "_xgboost"))
    if m.empty:
        raise RuntimeError("No overlapping test predictions for fusion.")
    m["actual"] = m["actual_prophet"]
    print(f"[Fusion] validation-selected weights: Prophet={w_p:.3f}, XGBoost={w_x:.3f} "
          f"(val WMAPE: Prophet={vwp:.1f}%, XGBoost={vwx:.1f}%)")
    m["pred_fusion"] = w_p * m["pred_prophet"] + w_x * m["pred_xgboost"]

    # A comparison is meaningful only on the exact same series.  Preserve a
    # machine-readable audit of that intersection instead of comparing each
    # model's partially successful rows.
    pred_sets = {}
    for model, filename in PREDS.items():
        d = pd.read_csv(os.path.join(PROC, filename))
        pred_sets[model] = set(d["series_name"].astype(str))
    common_cohort = set.intersection(*pred_sets.values())
    if not common_cohort:
        raise RuntimeError("Model predictions have no common test cohort.")
    pd.DataFrame({"series_name": sorted(common_cohort)}).to_csv(
        os.path.join(PROC, "comparison_cohort.csv"), index=False
    )
    print(f"[Compare] common test cohort: {len(common_cohort)} series "
          f"(saved to comparison_cohort.csv)")
    m = m[m["series_name"].astype(str).isin(common_cohort)].copy()

    frows = []
    for name, g in m.groupby("series_name"):
        met = metrics(g["actual"].values, g["pred_fusion"].values)
        met["series_name"] = name
        frows.append(met)
    fusion_res = pd.DataFrame(frows)
    fusion_res.to_csv(os.path.join(PROC, "fusion_results.csv"), index=False)
    m[["series_name", "date", "actual", "pred_fusion"]].to_csv(
        os.path.join(PROC, "fusion_preds.csv"), index=False)

# comparison table
    fp = pd.read_csv(os.path.join(PROC, "fusion_preds.csv"))
    fa, fpv = fp["actual"].values.astype(float), fp["pred_fusion"].values.astype(float)
    fusion_vol = np.sum(np.abs(fa - fpv)) / np.sum(np.abs(fa)) * 100 if np.sum(np.abs(fa)) > 0 else np.nan
    rows = [
        agg_results(os.path.join(PROC, "arima_results.csv"), "ARIMA", common_cohort),
        agg_results(os.path.join(PROC, "prophet_results.csv"), "Prophet", common_cohort),
        agg_results(os.path.join(PROC, "prophet_exog_results.csv"), "Prophet+exog", common_cohort),
        agg_results(os.path.join(PROC, "xgboost_results.csv"), "XGBoost", common_cohort),
        agg_results(os.path.join(PROC, "lstm_results.csv"), "LSTM", common_cohort),
        {
            "model": "Prophet+XGBoost",
            "WMAPE_mean": fusion_res["WMAPE"].mean(),
            "WMAPE_med": fusion_res["WMAPE"].median(),
            "WMAPE_vol": fusion_vol,
            "MAPE": fusion_res["MAPE"].mean(),
            "RMSE": fusion_res["RMSE"].mean(),
            "MAE": fusion_res["MAE"].mean(),
            "n_series": len(fusion_res),
        },
    ]
    comp = pd.DataFrame(rows).sort_values("WMAPE_vol")
    comp.to_csv(os.path.join(PROC, "model_comparison.csv"), index=False)
    print("\n===== Stage 3 multi-model comparison (common cohort, 6-month test) =====")
    print(comp.to_string(index=False))

# bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    c = comp.sort_values("WMAPE_vol")
    colors = ["#54A24B" if i == 0 else "#4C78A8" for i in range(len(c))]
    axes[0].bar(c["model"], c["WMAPE_vol"], color=colors)
    axes[0].set_ylabel("WMAPE_vol (%)  lower=better")
    axes[0].set_title("Volume-weighted WMAPE by model")
    axes[0].tick_params(labelsize=8)
    for i, v in enumerate(c["WMAPE_vol"].values):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    c2 = comp.sort_values("MAE")
    axes[1].bar(c2["model"], c2["MAE"], color="#F58518")
    axes[1].set_ylabel("MAE (units)  lower=better")
    axes[1].set_title("MAE by model")
    axes[1].tick_params(labelsize=8)
    for i, v in enumerate(c2["MAE"].values):
        axes[1].text(i, v + 80, f"{v:.0f}", ha="center", fontsize=8)

    fig.suptitle("Stage 3 — model comparison (common time-eligible cohort, 6-month test)", fontsize=12)
    fig.savefig(os.path.join(FIG, "model_comparison.png"), dpi=130)
    print("\n[Compare] figure saved -> figures_new/model_comparison.png")
    print("[Compare] table saved -> data/processed_new/stage3/model_comparison.csv")
    print("[Compare] done.")


if __name__ == "__main__":
    main()
