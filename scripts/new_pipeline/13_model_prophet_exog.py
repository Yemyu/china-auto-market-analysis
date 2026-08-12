#!/usr/bin/env python3
"""
13_model_prophet_exog.py — Prophet + 外生变量（无舆情基线 · 腿B）

数据来源（统一）：06_make_splits.py 的 train / test。
  * 每车系在 train (2022-01..2025-06) 上拟合，外生变量：
      - add_country_holidays("CN")  春节 / 国庆月效应
      - add_regressor("price_wan")   官方指导价（价格弹性水平）
      - add_regressor("promo")       大促月指示（6·18 / 双11 / 年末清库）
  * 在 **test (2026-01..06, 6 个月)** 上评估（外生变量随未来日期前推）。

输出：
  data/processed_new/stage3/prophet_exog_results.csv / prophet_exog_preds.csv

Run:
  python scripts/new_pipeline/13_model_prophet_exog.py
"""
import os
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
from prophet import Prophet

import _model_utils as mu
import _subset

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(BASE, "figures_new")
PROC = os.path.join(BASE, "data", "processed_new", "stage3")
os.makedirs(PROC, exist_ok=True)
TRAIN_STYLE = "#4C78A8"
TEST_STYLE = "#F58518"
FC_STYLE = "#9D7EBF"
PROMO_MONTHS = {6, 11, 12}


def main():
    tr, _, te = mu.load_splits()
    feat = pd.read_csv(os.path.join(BASE, "data", "raw", "feature.csv"), low_memory=False)
    feat["series_name"] = feat["series_name"].astype(str)
    feat["official_price_wan"] = pd.to_numeric(feat["official_price_wan"], errors="coerce")
    price_map = feat.groupby("series_name")["official_price_wan"].median().to_dict()
    price_median = float(pd.Series(list(price_map.values())).median())

    subset = _subset.load_subset()
    print(f"[Prophet-exog] 评估子集 {len(subset)} 系 | 拟合 train, 评估 test (6 月)")

    # 每车系 train/test 序列
    tr_by = {n: (tr[tr["series_name"].astype(str) == n].sort_values("date")["monthly_sales"]
                 .astype(float).values) for n in subset}
    te_by = {n: (te[te["series_name"].astype(str) == n].sort_values("date")["monthly_sales"]
                 .astype(float).values) for n in subset}

    rows, preds_rows, examples = [], [], []
    for name in subset:
        s = tr_by[name]
        if len(s) <= 12:
            rows.append({"series_name": name, "status": "too_short"})
            continue
        tgt = te_by.get(name, np.array([]))
        if len(tgt) == 0:
            rows.append({"series_name": name, "status": "no_test"})
            continue
        price = float(price_map.get(name, price_median))
        df = pd.DataFrame({
            "ds": pd.date_range("2018-01-01", periods=len(s), freq="MS"),
            "y": s.astype(float),
            "price_wan": price,
            "promo": [1 if d.month in PROMO_MONTHS else 0 for d in pd.date_range("2018-01-01", periods=len(s), freq="MS")],
        })
        try:
            m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                        yearly_seasonality=True, seasonality_mode="additive")
            m.add_country_holidays("CN")
            m.add_regressor("price_wan")
            m.add_regressor("promo")
            m.fit(df)
            future = m.make_future_dataframe(periods=len(tgt), freq="MS")
            future["price_wan"] = price
            future["promo"] = [1 if d.month in PROMO_MONTHS else 0 for d in future["ds"]]
            fc = m.predict(future).iloc[-len(tgt):]["yhat"].clip(lower=0).values
            met = mu.metrics(tgt, fc)
            met.update({"series_name": name, "status": "ok"})
            rows.append(met)
            tdates = te[te["series_name"].astype(str) == name].sort_values("date")["date"].values
            for j, d in enumerate(tdates):
                preds_rows.append({"series_name": name,
                                   "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                   "actual": float(tgt[j]), "pred": float(fc[j])})
            if len(examples) < 9:
                examples.append((name, pd.Series(s), pd.Series(tgt), pd.Series(fc)))
        except Exception as e:
            rows.append({"series_name": name, "status": f"error: {type(e).__name__}"})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "prophet_exog_results.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "prophet_exog_preds.csv"), index=False)
    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"\n[Prophet-exog] test 评估 ok: {len(ok)}/{len(subset)}")
    if len(ok):
        a = pd.DataFrame(preds_rows)["actual"].values.astype(float)
        p = pd.DataFrame(preds_rows)["pred"].values.astype(float)
        print(f"  WMAPE(全局volume-weighted) = {mu.wmape_vol(a, p):.1f}%")
        print(f"  WMAPE(per-series mean)     = {ok['WMAPE'].mean():.1f}%  "
              f"(median {ok['WMAPE'].median():.1f}%)")

    n_ex = min(9, len(examples))
    if n_ex:
        cols, rows_n = 3, (n_ex + 2) // 3
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 2.6), constrained_layout=True)
        axes = np.array(axes).reshape(-1)
        for i, (name, trs, tes, fcs) in enumerate(examples):
            ax = axes[i]
            ax.plot(range(len(trs)), trs.values, color=TRAIN_STYLE, lw=1.2, label="train")
            ax.plot(range(len(trs), len(trs) + len(tes)), tes.values, color=TEST_STYLE, lw=1.6, marker="o", ms=3, label="actual")
            ax.plot(range(len(trs), len(trs) + len(fcs)), fcs.values, color=FC_STYLE, lw=1.6, marker="s", ms=3, label="forecast")
            ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")
        for j in range(n_ex, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Prophet + exogenous (holidays/promo/price) — 6-month forecast on temporal TEST", fontsize=11)
        fig.savefig(os.path.join(FIG, "prophet_exog_forecast.png"), dpi=130)
        print("[Prophet-exog] figure saved -> figures_new/prophet_exog_forecast.png")
    print("[Prophet-exog] done.")


if __name__ == "__main__":
    main()
