#!/usr/bin/env python3
"""Ablate lag and configuration features from the XGBoost baseline."""
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
from xgboost import XGBRegressor

import _model_utils as mu
import _subset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "assets/analysis")
PROC = os.path.join(BASE, "data", "processed", "forecast")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

VERSIONS = {
    "FULL": mu.LAG_COLS + mu.CAL + mu.CFG_COLS,
    "NO-LAG": mu.CAL + mu.CFG_COLS,
    "NO-CONFIG": mu.LAG_COLS + mu.CAL,
}


def fit_version(cols, tr, va):
    """Select tree count on validation, then refit on train+validation."""
    m = XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     objective="reg:squarederror", n_jobs=1, early_stopping_rounds=50)
    m.fit(tr[cols], np.log1p(tr[mu.TARGET]),
          eval_set=[(va[cols], np.log1p(va[mu.TARGET]))], verbose=False)
    best_raw = getattr(m, "best_iteration", None)
    n_estimators = (int(best_raw) + 1) if best_raw is not None else 1000
    final = XGBRegressor(n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         objective="reg:squarederror", n_jobs=1)
    trva = pd.concat([tr, va], ignore_index=True)
    final.fit(trva[cols], np.log1p(trva[mu.TARGET]), verbose=False)
    return final


def main():
    tr, va, te = mu.load_splits()
    subset = _subset.load_subset()
    panel = mu.load_panel_for_subset(subset)
    print(f"[消融] 评估子集 {len(subset)} 系 | train={len(tr)} val={len(va)} test={len(te)}")

    per_series = {v: {} for v in VERSIONS}   # version -> {series: WMAPE}
    abs_err = {v: 0.0 for v in VERSIONS}
    act_sum = {v: 0.0 for v in VERSIONS}
    example_full = None

    for vname, cols in VERSIONS.items():
        print(f"[消融:{vname:10s}] 训练 {len(cols)} 维特征 ...")
        model = fit_version(cols, tr, va)
        for name, g in panel.groupby("series_name"):
            g = g.sort_values("date")
            test_g = g[g["split"] == "test"]
            if len(test_g) == 0:
                continue
            preds = mu.recursive_forecast_tree(
                model, g, feat_cols=cols,
                history_splits=("train", "val"), forecast_splits=("test",)
            )
            actual = test_g[mu.TARGET].astype(float).values
            pred = np.array([preds.get(d, np.nan) for d in test_g["date"].values], dtype=float)
            if np.isnan(pred).any():
                continue
            met = mu.metrics(actual, pred)
            per_series[vname][name] = met["WMAPE"]
            abs_err[vname] += np.abs(actual - pred).sum()
            act_sum[vname] += np.abs(actual).sum()
            if vname == "FULL" and example_full is None:
                example_full = (name, g[g["split"].isin(["train", "val"])]
                                .set_index("date")[mu.TARGET],
                                test_g.set_index("date")[mu.TARGET],
                                pd.Series(pred, index=test_g["date"]))

    # 输出逐车系 × 版本
    out_rows = []
    for vname in VERSIONS:
        for s, w in per_series[vname].items():
            out_rows.append({"version": vname, "series_name": s, "WMAPE": w})
    out = pd.DataFrame(out_rows)
    out.to_csv(os.path.join(PROC, "xgb_ablation.csv"), index=False)

    vol = {v: (abs_err[v] / act_sum[v] * 100 if act_sum[v] > 0 else np.nan) for v in VERSIONS}
    med = {v: np.nanmedian(list(per_series[v].values())) for v in VERSIONS}

    print("\n===== 消融结果 (WMAPE, 越低越好；test 窗口 2026-01..06) =====")
    print(f"{'组别':12s} {'全局volume-weighted':>20s} {'per-series中位数':>18s}")
    for k in ["FULL", "NO-LAG", "NO-CONFIG"]:
        print(f"{k:12s} {vol[k]:19.1f}% {med[k]:17.1f}%")
    print(f"\n[lag 贡献]  去掉 lag:    {vol['FULL']:.1f}% -> {vol['NO-LAG']:.1f}%  "
          f"(误差 {(vol['NO-LAG']-vol['FULL'])/vol['FULL']*100:+.1f}%)")
    print(f"[配置贡献] 去掉配置: {vol['FULL']:.1f}% -> {vol['NO-CONFIG']:.1f}%  "
          f"(误差 {(vol['NO-CONFIG']-vol['FULL'])/vol['FULL']*100:+.1f}%)")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ks = ["FULL", "NO-LAG", "NO-CONFIG"]
    axes[0].bar(ks, [vol[k] for k in ks], color=["#54A24B", "#F58518", "#E45756"])
    axes[0].set_ylabel("WMAPE (%) 全局 volume-weighted")
    axes[0].set_title("月度预测消融：lag vs 配置 (test split)")
    for i, k in enumerate(ks):
        axes[0].text(i, vol[k] + 0.6, f"{vol[k]:.1f}", ha="center", fontsize=10)
    if example_full:
        name, tr_s, te_s, fc_s = example_full
        axes[1].plot(tr_s.index[-24:], tr_s.values[-24:], color="#4C78A8", lw=1.4, label="历史")
        axes[1].plot(te_s.index, te_s.values, color="#F58518", lw=1.8, marker="o", ms=4, label="实际")
        axes[1].plot(fc_s.index, fc_s.values, color="#E45756", lw=1.8, marker="s", ms=4, label="预测")
        axes[1].set_title(f"FULL 示例: {name}")
        axes[1].legend(fontsize=8)
        axes[1].tick_params(labelsize=8)
    fig.suptitle("XGBoost 消融 — 近期销量(lag) 与 配置 各自的贡献 (temporal test)", fontsize=12)
    fig.savefig(os.path.join(FIG, "xgb_ablation.png"), dpi=130)
    print("\n[消融] 图已保存 -> assets/analysis/xgb_ablation.png")
    print("[消融] 完成。")


if __name__ == "__main__":
    main()
