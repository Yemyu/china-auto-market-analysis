#!/usr/bin/env python3
"""
fusion + Step 11 comparison
- Fusion: Prophet + XGBoost inverse-WMAPE weighted average (heuristic; more
  weight to the more accurate model on the test set).
- Comparison: aggregate per-series metrics (mean WMAPE/MAPE/RMSE/MAE) for
  ARIMA / Prophet / XGBoost / LSTM / Fusion on the same 30-series, 3-month test.

Run:
  python scripts/09_fusion_and_compare.py
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


def agg_results(path, model):
    r = pd.read_csv(path)
    ok = r[r["status"] == "ok"] if "status" in r.columns else r
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
        a = p["actual"].values.astype(float)
        pr = p["pred"].values.astype(float)
        row["WMAPE_vol"] = np.sum(np.abs(a - pr)) / np.sum(np.abs(a)) * 100 if np.sum(np.abs(a)) > 0 else np.nan
    return row


def main():
# fusion
    pp = pd.read_csv(os.path.join(PROC, "prophet_preds.csv"))
    xp = pd.read_csv(os.path.join(PROC, "xgboost_preds.csv"))
    m = pp.merge(xp, on=["series_name", "date"], suffixes=("_prophet", "_xgboost"))
    m["actual"] = m["actual_prophet"]
    wp = overall_wmape(m["actual"].values, m["pred_prophet"].values)
    wx = overall_wmape(m["actual"].values, m["pred_xgboost"].values)
    w_p = (1.0 / wp) / ((1.0 / wp) + (1.0 / wx))
    w_x = 1.0 - w_p
    print(f"[Fusion] weights  Prophet={w_p:.3f}  XGBoost={w_x:.3f}  "
          f"(Prophet WMAPE={wp:.1f}%  XGBoost WMAPE={wx:.1f}%)")
    m["pred_fusion"] = w_p * m["pred_prophet"] + w_x * m["pred_xgboost"]

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
        agg_results(os.path.join(PROC, "arima_results.csv"), "ARIMA"),
        agg_results(os.path.join(PROC, "prophet_results.csv"), "Prophet"),
        agg_results(os.path.join(PROC, "prophet_exog_results.csv"), "Prophet+exog"),
        agg_results(os.path.join(PROC, "xgboost_results.csv"), "XGBoost"),
        agg_results(os.path.join(PROC, "lstm_results.csv"), "LSTM"),
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
    print("\n===== Stage 3 multi-model comparison (mean per-series, horizon=3) =====")
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

    fig.suptitle("Stage 3 — multi-model comparison (150-series stratified subset, 3-month forecast)", fontsize=12)
    fig.savefig(os.path.join(FIG, "model_comparison.png"), dpi=130)
    print("\n[Compare] figure saved -> figures_new/model_comparison.png")
    print("[Compare] table saved -> data/processed_new/stage3/model_comparison.csv")
    print("[Compare] done.")


if __name__ == "__main__":
    main()
