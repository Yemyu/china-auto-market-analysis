#!/usr/bin/env python3
"""Train and evaluate the baseline XGBoost sales forecaster."""
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


def main():
    tr, va, te = mu.load_splits()
    subset = _subset.load_subset()
    print(f"[XGBoost] 评估子集: {len(subset)} 系 | "
          f"train={len(tr)} val={len(va)} test={len(te)} 行")

    # --- Model selection on validation; the test set remains untouched. ---
    select_model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:squarederror", n_jobs=1, early_stopping_rounds=50,
    )
    select_model.fit(
        tr[mu.FEAT_COLS], np.log1p(tr[mu.TARGET]),
        eval_set=[(va[mu.FEAT_COLS], np.log1p(va[mu.TARGET]))],
        verbose=False,
    )
    best_raw = getattr(select_model, "best_iteration", None)
    best_iteration = (int(best_raw) + 1) if best_raw is not None else 1000
    print(f"[XGBoost] validation-selected n_estimators={best_iteration}")

    # Refit once on every observation available before the test window.
    # This is the standard train/val/test protocol: validation selects the
    # capacity; its realised sales may then seed the Jan-2026 forecast.
    final_model = XGBRegressor(
        n_estimators=best_iteration, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="reg:squarederror", n_jobs=1,
    )
    trva = pd.concat([tr, va], ignore_index=True)
    final_model.fit(trva[mu.FEAT_COLS], np.log1p(trva[mu.TARGET]), verbose=False)

    # --- Validation predictions are saved for leakage-free fusion selection. ---
    panel = mu.load_panel_for_subset(subset)
    val_preds_rows, rows, preds_rows, examples = [], [], [], []
    for name, g in panel.groupby("series_name"):
        g = g.sort_values("date")
        val_preds = mu.recursive_forecast_tree(
            select_model, g, history_splits=("train",), forecast_splits=("val",)
        )
        val_g = g[g["split"] == "val"]
        for d, a in zip(val_g["date"].values, val_g[mu.TARGET].astype(float).values):
            p = val_preds.get(d, np.nan)
            if np.isfinite(p):
                val_preds_rows.append({"series_name": name,
                                       "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                       "actual": float(a), "pred": float(p)})

        preds = mu.recursive_forecast_tree(
            final_model, g, history_splits=("train", "val"), forecast_splits=("test",)
        )
        if not preds:
            rows.append({"series_name": name, "status": "no_train"})
            continue
        test_g = g[g["split"] == "test"]
        if len(test_g) == 0:
            rows.append({"series_name": name, "status": "no_test"})
            continue
        actual = test_g[mu.TARGET].astype(float).values
        pred = np.array([preds.get(d, np.nan) for d in test_g["date"].values], dtype=float)
        if np.isnan(pred).any():
            rows.append({"series_name": name, "status": "nan_pred"})
            continue
        met = mu.metrics(actual, pred)
        met.update({"series_name": name, "status": "ok"})
        rows.append(met)
        for d, a, p in zip(test_g["date"].values, actual, pred):
            preds_rows.append({"series_name": name,
                               "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                               "actual": float(a), "pred": float(p)})
        if len(examples) < 9:
            trg = g[g["split"].isin(["train", "val"])]
            examples.append((name, trg.set_index("date")[mu.TARGET],
                             test_g.set_index("date")[mu.TARGET],
                             pd.Series(pred, index=test_g["date"])))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "xgboost_results.csv"), index=False)
    if val_preds_rows:
        pd.DataFrame(val_preds_rows).to_csv(os.path.join(PROC, "xgboost_val_preds.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "xgboost_preds.csv"), index=False)

    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"\n[XGBoost] test 评估 ok: {len(ok)}/{len(subset)}")
    if len(ok):
        a = pd.DataFrame(preds_rows)["actual"].values.astype(float)
        p = pd.DataFrame(preds_rows)["pred"].values.astype(float)
        print(f"  WMAPE(全局volume-weighted) = {mu.wmape_vol(a, p):.1f}%")
        print(f"  WMAPE(per-series mean)     = {ok['WMAPE'].mean():.1f}%  "
              f"(median {ok['WMAPE'].median():.1f}%)")
        print(f"  MAPE={ok['MAPE'].mean():.1f}%  RMSE={ok['RMSE'].mean():.1f}  "
              f"MAE={ok['MAE'].mean():.1f}")

    # --- 特征重要性 + 示例图 ---
    imp = pd.Series(final_model.feature_importances_, index=mu.FEAT_COLS).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[1].barh(imp.index, imp.values, color="#4C78A8")
    axes[1].set_title("XGBoost feature importance (train+val fitted)", fontsize=10)
    axes[1].tick_params(labelsize=8)
    if examples:
        name, tr_s, te_s, fc_s = examples[0]
        axes[0].plot(tr_s.index, tr_s.values, color="#4C78A8", lw=1.4, label="train")
        axes[0].plot(te_s.index, te_s.values, color="#F58518", lw=1.8, marker="o", ms=4, label="actual")
        axes[0].plot(fc_s.index, fc_s.values, color="#E45756", lw=1.8, marker="s", ms=4, label="forecast (test)")
        axes[0].set_title(f"XGBoost example: {name}", fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].tick_params(labelsize=8)
    else:
        axes[0].axis("off")
    fig.suptitle("XGBoost — recursive forecast on temporal TEST split (2026-01..06)", fontsize=11)
    fig.savefig(os.path.join(FIG, "xgboost_forecast.png"), dpi=130)
    print("[XGBoost] figure saved -> assets/analysis/xgboost_forecast.png")
    print("[XGBoost] done.")


if __name__ == "__main__":
    main()
