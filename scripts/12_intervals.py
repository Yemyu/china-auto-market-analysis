#!/usr/bin/env python3
"""Evaluate 90% forecast intervals for ARIMA, Prophet, and XGBoost."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from prophet import Prophet

import _model_utils as mu
import _subset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "assets/analysis")
PROC = os.path.join(BASE, "data", "processed", "forecast")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

NOMINAL = 0.90
Z = 1.6448536269514722
SEED = 42


def picp_mpiw(actual, lower, upper):
    actual = np.asarray(actual, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    inside = np.mean((actual >= lower) & (actual <= upper))
    width = np.mean(upper - lower)
    width_pct = width / actual.mean() * 100 if actual.mean() != 0 else np.nan
    return inside, width, width_pct


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


def arima_pi(subset, tr, va, te):
    aa, ap, al, au = [], [], [], []
    for name in subset:
        train_on = pd.concat([
            tr[tr["series_name"].astype(str) == name],
            va[va["series_name"].astype(str) == name],
        ]).sort_values("date")["monthly_sales"].astype(float).values
        test_on = te[te["series_name"].astype(str) == name].sort_values("date")["monthly_sales"].astype(float).values
        if len(train_on) <= 12 or len(test_on) != 6:
            continue
        try:
            order = auto_order(train_on)
            fc = ARIMA(train_on, order=order).fit().get_forecast(6)
            mean = np.asarray(fc.predicted_mean).clip(min=0)
            ci = np.asarray(fc.conf_int(alpha=1 - NOMINAL))
            aa.extend(test_on); ap.extend(mean)
            al.extend(ci[:, 0].clip(min=0))
            au.extend(ci[:, 1].clip(min=0))
        except Exception:
            continue
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


def prophet_pi(subset, tr, va, te):
    aa, ap, al, au = [], [], [], []
    for name in subset:
        fit = pd.concat([
            tr[tr["series_name"].astype(str) == name],
            va[va["series_name"].astype(str) == name],
        ]).sort_values("date")
        test = te[te["series_name"].astype(str) == name].sort_values("date")
        if len(fit) <= 12 or len(test) != 6:
            continue
        try:
            df = pd.DataFrame({"ds": pd.to_datetime(fit["date"]),
                               "y": fit["monthly_sales"].astype(float)})
            m = Prophet(interval_width=NOMINAL, yearly_seasonality=True,
                        weekly_seasonality=False, daily_seasonality=False)
            m.fit(df)
            future = pd.DataFrame({"ds": pd.to_datetime(test["date"])})
            fc = m.predict(future).iloc[-6:]
            point = fc["yhat"].clip(lower=0).values
            lower = fc["yhat_lower"].clip(lower=0).values
            upper = fc["yhat_upper"].clip(lower=0).values
            aa.extend(test["monthly_sales"].astype(float).values)
            ap.extend(point); al.extend(lower); au.extend(upper)
        except Exception:
            continue
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


def xgb_pi(subset, panel):
    from _feature_join import CFG_COLS
    tr, va, _ = mu.load_splits()
    models = {}
    trva = pd.concat([tr, va], ignore_index=True)
    for a in (0.05, 0.50, 0.95):
        selector = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                                objective="reg:quantileerror", quantile_alpha=a, early_stopping_rounds=50)
        selector.fit(tr[mu.FEAT_COLS], np.log1p(tr[mu.TARGET]),
                     eval_set=[(va[mu.FEAT_COLS], np.log1p(va[mu.TARGET]))], verbose=False)
        best = getattr(selector, "best_iteration", None)
        final = XGBRegressor(n_estimators=(int(best) + 1) if best is not None else 1000,
                             max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=1,
                             objective="reg:quantileerror", quantile_alpha=a)
        final.fit(trva[mu.FEAT_COLS], np.log1p(trva[mu.TARGET]), verbose=False)
        models[a] = final

    aa, ap, al, au = [], [], [], []
    for name, g in panel.groupby("series_name"):
        g = g.sort_values("date").reset_index(drop=True)
        cfg = {d: {c: r[c] for c in CFG_COLS} for d, r in g.set_index("date").iterrows()}
        train_part = g[g["split"].isin(["train", "val"])]
        if len(train_part) == 0:
            continue
        history = train_part[mu.TARGET].astype(float).tolist()
        lo, pt, hi = [], [], []
        for _, r in g[g["split"] == "test"].iterrows():
            d = r["date"]
            h = np.asarray(history, float)
            row = {
                "lag_1": h[-1] if len(h) >= 1 else 0.0, "lag_2": h[-2] if len(h) >= 2 else 0.0,
                "lag_3": h[-3] if len(h) >= 3 else 0.0,
                "roll_mean_3": float(np.mean(h[-3:])) if len(h) >= 1 else 0.0,
                "roll_mean_6": float(np.mean(h[-6:])) if len(h) >= 1 else 0.0,
                "month_sin": np.sin(2 * np.pi * d.month / 12),
                "month_cos": np.cos(2 * np.pi * d.month / 12), "year": d.year,
            }
            for c in CFG_COLS:
                row[c] = cfg[d][c]
            X = pd.DataFrame([row], columns=mu.FEAT_COLS)
            p05 = max(float(np.expm1(models[0.05].predict(X)[0])), 0.0)
            p50 = max(float(np.expm1(models[0.50].predict(X)[0])), 0.0)
            p95 = max(float(np.expm1(models[0.95].predict(X)[0])), 0.0)
            lo.append(p05); pt.append(p50); hi.append(p95)
            history.append(p50)   # 用 median 回填 lag
        test_g = g[g["split"] == "test"]
        if len(test_g) == 6 and len(pt) == 6:
            aa.extend(test_g[mu.TARGET].astype(float).values)
            ap.extend(pt); al.extend(lo); au.extend(hi)
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


def summarize(name, aa, ap, al, au):
    picp, width, width_pct = picp_mpiw(aa, al, au)
    wmape = mu.wmape_vol(aa, ap)
    return {"model": name, "PICP": round(float(picp), 3), "MPIW": round(float(width), 1),
            "MPIW_pct": round(float(width_pct), 1), "WMAPE": round(float(wmape), 1),
            "n_points": len(aa)}


def main():
    tr, va, te = mu.load_splits()
    subset = _subset.load_subset()
    panel = mu.load_panel_for_subset(subset)
    print(f"[PI] 评估子集 {len(subset)} 系 | 区间在 test (6 月) 上评估, 名义水平 {NOMINAL}")

    print("[PI] ARIMA ..."); a = arima_pi(subset, tr, va, te)
    print("[PI] Prophet ..."); p = prophet_pi(subset, tr, va, te)
    print("[PI] XGBoost (quantile) ..."); x = xgb_pi(subset, panel)

    rows = [summarize("ARIMA", *a), summarize("Prophet", *p), summarize("XGBoost", *x)]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "interval_results.csv"), index=False)
    print("\n===== Prediction-interval coverage (90% nominal, temporal TEST) =====")
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    ax.bar(res["model"], res["PICP"], color=colors)
    ax.axhline(NOMINAL, color="red", ls="--", lw=1.2, label=f"target {NOMINAL:.0%}")
    for i, v in enumerate(res["PICP"]):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("PICP (coverage)")
    ax.set_title("Prediction-interval coverage (higher≈target)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "intervals_coverage.png"), dpi=130)

    # 示例：XGBoost 区间
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    name = subset[0]
    g = panel[panel["series_name"] == name].sort_values("date")
    trg = g[g["split"] == "train"]; teg = g[g["split"] == "test"]
    ax.plot(trg["date"], trg[mu.TARGET], color="#4C78A8", lw=1.3, label="train")
    ax.plot(teg["date"], teg[mu.TARGET], color="#F58518", lw=1.8, marker="o", ms=4, label="actual")
    idx0 = list(subset).index(name) * 6
    xs = x[1][idx0:idx0 + 6]; xl = x[2][idx0:idx0 + 6]; xu = x[3][idx0:idx0 + 6]
    ax.plot(teg["date"], xs, color="#54A24B", lw=1.6, marker="s", ms=3, label="XGBoost point")
    ax.fill_between(teg["date"], xl, xu, color="#54A24B", alpha=0.2, label="XGBoost 90% PI")
    ax.set_title(f"Prediction interval example: {name}")
    ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(FIG, "intervals_example.png"), dpi=130)
    print("[PI] figures saved -> assets/analysis/intervals_coverage.png, intervals_example.png")
    print("[PI] done.")


if __name__ == "__main__":
    main()
