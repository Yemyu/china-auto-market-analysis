#!/usr/bin/env python3
"""Measure ARIMA and XGBoost error growth across forecast horizons."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

import _model_utils as mu
import _subset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "assets/analysis")
PROC = os.path.join(BASE, "data", "processed", "forecast")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

HORIZONS = [3, 6, 9, 12]
SEED = 42


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 \
        if np.sum(np.abs(y_true)) > 0 else np.nan


def auto_order(train):
    best_aic, best = np.inf, (1, 1, 1)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(train, order=(p, d, q)).fit()
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic, best = fit.aic, (p, d, q)
                except Exception:
                    continue
    return best


def arima_cv(subset, tr):
    out = {}
    for h in HORIZONS:
        w = []
        for name in subset:
            s = (tr[tr["series_name"].astype(str) == name]
                 .sort_values("date")["monthly_sales"].astype(float).values)
            if len(s) <= h + 6:
                continue
            train_on, test_on = s[:-h], s[-h:]
            try:
                order = auto_order(train_on)
                fc = ARIMA(train_on, order=order).fit().forecast(h).clip(min=0)
                w.append(metrics(test_on, fc))
            except Exception:
                continue
        out[h] = (float(np.nanmean(w)), float(np.nanstd(w)), len(w)) if w else (np.nan, np.nan, 0)
    return out


def xgb_cv(subset, tr):
    tr = tr.copy()
    tr["series_name"] = tr["series_name"].astype(str)
    out = {}
    for h in HORIZONS:
        tr["_rev"] = tr.groupby("series_name").cumcount(ascending=False)
        mask = tr["_rev"] >= h
        model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                             objective="reg:squarederror", n_jobs=1)
        model.fit(tr.loc[mask, mu.FEAT_COLS], np.log1p(tr.loc[mask, mu.TARGET]))
        w = []
        for name in subset:
            g = tr[tr["series_name"] == name].sort_values("date").copy()
            if len(g) <= h + 6:
                continue
            g["split"] = "train"
            g.loc[g.index[-h:], "split"] = "test"   # 最后 h 月作为本 horizon 的 holdout
            preds = mu.recursive_forecast_tree(model, g)
            test_g = g[g["split"] == "test"]
            actual = test_g[mu.TARGET].astype(float).values
            pred = np.array([preds.get(d, np.nan) for d in test_g["date"].values], dtype=float)
            if not np.isnan(pred).any():
                w.append(metrics(actual, pred))
        out[h] = (float(np.nanmean(w)), float(np.nanstd(w)), len(w)) if w else (np.nan, np.nan, 0)
    return out


def main():
    tr, _, _ = mu.load_splits()
    subset = _subset.load_subset()
    print(f"[CV] 评估子集 {len(subset)} 系 | 仅用 train ({len(tr)} 行) 做多步长检查; horizons={HORIZONS}")

    print("[CV] ARIMA ..."); ar = arima_cv(subset, tr)
    print("[CV] XGBoost ..."); xg = xgb_cv(subset, tr)

    rows = []
    for h in HORIZONS:
        for model, d in [("ARIMA", ar), ("XGBoost", xg)]:
            m, s, n = d[h]
            rows.append({"model": model, "horizon": h, "mean_wmape": m,
                         "std_wmape": s, "n_series": n})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "cv_results.csv"), index=False)
    print("\n===== Rolling-origin CV (WMAPE % by horizon, on train split) =====")
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    colors = {"ARIMA": "#F58518", "XGBoost": "#54A24B"}
    for model in ["ARIMA", "XGBoost"]:
        sub = res[res["model"] == model]
        ax.plot(sub["horizon"], sub["mean_wmape"], marker="o", lw=2, color=colors[model], label=model)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("WMAPE (%) — lower is better")
    ax.set_title("Rolling-origin CV: error grows with horizon (train split only)")
    ax.set_xticks(HORIZONS)
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "cv_wmape_by_horizon.png"), dpi=130)
    print("[CV] figure saved -> assets/analysis/cv_wmape_by_horizon.png")
    print("[CV] done.")


if __name__ == "__main__":
    main()
