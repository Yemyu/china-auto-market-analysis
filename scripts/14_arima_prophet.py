#!/usr/bin/env python3
"""Evaluate per-series ARIMA and Prophet baselines on the fixed time split."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import _model_utils as mu
import _subset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed", "forecast")
os.makedirs(PROC, exist_ok=True)


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


def arima_run(subset, tr_by, va_by, te, te_by):
    rows, preds_rows = [], []
    for name in subset:
        s = tr_by[name]
        if len(s) <= 12:
            rows.append({"series_name": name, "status": "too_short"})
            continue
        tgt = te_by.get(name, np.array([]))
        if len(tgt) == 0:
            rows.append({"series_name": name, "status": "no_test"})
            continue
        try:
            order = auto_order(s)
            # Validation is now observed at the test origin, so refit the
            # selected ARIMA order on all pre-test history.
            train_final = np.r_[s, va_by.get(name, np.array([]))]
            res = ARIMA(train_final, order=order).fit()
            fc = res.get_forecast(len(tgt))
            mean = np.asarray(fc.predicted_mean).clip(min=0)
            met = mu.metrics(tgt, mean)
            met.update({"series_name": name, "order": str(order), "status": "ok"})
            rows.append(met)
            tdates = te[te["series_name"].astype(str) == name].sort_values("date")["date"].values
            for j, d in enumerate(tdates):
                preds_rows.append({"series_name": name, "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                   "actual": float(tgt[j]), "pred": float(mean[j])})
        except Exception as e:
            rows.append({"series_name": name, "status": f"error: {type(e).__name__}"})
    return rows, preds_rows


def _fit_prophet(values, dates, forecast_dates):
    """Fit on real calendar dates and forecast the supplied future dates."""
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": np.asarray(values, float)})
    m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                yearly_seasonality=True, seasonality_mode="additive")
    m.fit(df)
    future = pd.DataFrame({"ds": pd.to_datetime(forecast_dates)})
    return m.predict(future)["yhat"].clip(lower=0).values


def prophet_run(subset, tr_by, tr_dates_by, va_by, va_dates_by, te, te_by, te_dates_by):
    rows, preds_rows, val_preds_rows = [], [], []
    for name in subset:
        s = tr_by[name]
        if len(s) <= 12:
            rows.append({"series_name": name, "status": "too_short"})
            continue
        tgt = te_by.get(name, np.array([]))
        if len(tgt) == 0:
            rows.append({"series_name": name, "status": "no_test"})
            continue
        try:
            val_actual = va_by.get(name, np.array([]))
            val_dates = va_dates_by.get(name, np.array([]))
            if len(val_actual) != 6:
                rows.append({"series_name": name, "status": "no_val"})
                continue
            # Save a true train -> validation forecast for fusion weighting.
            val_fc = _fit_prophet(s, tr_dates_by[name], val_dates)
            for j, d in enumerate(val_dates):
                val_preds_rows.append({"series_name": name,
                                       "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                       "actual": float(val_actual[j]), "pred": float(val_fc[j])})

            # Refit on all data known before the test origin.
            final_values = np.r_[s, val_actual]
            final_dates = np.r_[tr_dates_by[name], val_dates]
            fc = _fit_prophet(final_values, final_dates, te_dates_by[name])
            met = mu.metrics(tgt, fc)
            met.update({"series_name": name, "status": "ok"})
            rows.append(met)
            tdates = te[te["series_name"].astype(str) == name].sort_values("date")["date"].values
            for j, d in enumerate(tdates):
                preds_rows.append({"series_name": name, "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                   "actual": float(tgt[j]), "pred": float(fc[j])})
        except Exception as e:
            rows.append({"series_name": name, "status": f"error: {type(e).__name__}"})
    return rows, preds_rows, val_preds_rows


def main():
    tr, va, te = mu.load_splits()
    subset = _subset.load_subset()
    tr_by = {n: tr[tr["series_name"].astype(str) == n].sort_values("date")["monthly_sales"]
             .astype(float).values for n in subset}
    te_by = {n: te[te["series_name"].astype(str) == n].sort_values("date")["monthly_sales"]
             .astype(float).values for n in subset}
    va_by = {n: va[va["series_name"].astype(str) == n].sort_values("date")["monthly_sales"]
             .astype(float).values for n in subset}
    tr_dates_by = {n: tr[tr["series_name"].astype(str) == n].sort_values("date")["date"].values
                   for n in subset}
    va_dates_by = {n: va[va["series_name"].astype(str) == n].sort_values("date")["date"].values
                   for n in subset}
    te_dates_by = {n: te[te["series_name"].astype(str) == n].sort_values("date")["date"].values
                   for n in subset}
    print(f"[14] data/processed/splits | 子集 {len(subset)} 系 | "
          f"train 拟合, test(6月) 评估, 无泄漏")

    print("[14] ARIMA ...")
    ar, ap = arima_run(subset, tr_by, va_by, te, te_by)
    print("[14] Prophet ...")
    pr, pp, pvp = prophet_run(subset, tr_by, tr_dates_by, va_by, va_dates_by, te, te_by, te_dates_by)

    ar_df, ap_df = pd.DataFrame(ar), pd.DataFrame(ap)
    pr_df, pp_df = pd.DataFrame(pr), pd.DataFrame(pp)
    ar_df.to_csv(os.path.join(PROC, "arima_results.csv"), index=False)
    if len(ap_df):
        ap_df.to_csv(os.path.join(PROC, "arima_preds.csv"), index=False)
    pr_df.to_csv(os.path.join(PROC, "prophet_results.csv"), index=False)
    if len(pp_df):
        pp_df.to_csv(os.path.join(PROC, "prophet_preds.csv"), index=False)
    if pvp:
        pd.DataFrame(pvp).to_csv(os.path.join(PROC, "prophet_val_preds.csv"), index=False)

    ok_ar = ar_df[ar_df["status"] == "ok"] if "status" in ar_df else pd.DataFrame()
    ok_pr = pr_df[pr_df["status"] == "ok"] if "status" in pr_df else pd.DataFrame()
    if len(ok_ar):
        a = ap_df["actual"].values.astype(float); p = ap_df["pred"].values.astype(float)
        print(f"[14] ARIMA    ok={len(ok_ar)}  WMAPE_vol={mu.wmape_vol(a, p):.1f}%  "
              f"per-series mean={ok_ar['WMAPE'].mean():.1f}%  median={ok_ar['WMAPE'].median():.1f}%")
    if len(ok_pr):
        a = pp_df["actual"].values.astype(float); p = pp_df["pred"].values.astype(float)
        print(f"[14] Prophet  ok={len(ok_pr)}  WMAPE_vol={mu.wmape_vol(a, p):.1f}%  "
              f"per-series mean={ok_pr['WMAPE'].mean():.1f}%  median={ok_pr['WMAPE'].median():.1f}%")
    print("[14] written -> data/processed/forecast/{arima,prophet}_preds.csv (+_results.csv)")
    print("[14] done.")


if __name__ == "__main__":
    main()
