#!/usr/bin/env python3
"""Shared split, metric, and recursive-forecast utilities."""
import os

import numpy as np
import pandas as pd

from _feature_join import CFG_COLS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(BASE, "data", "processed", "splits")

LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
SEASONAL_LAG_COLS = ["lag_12", "roll_mean_12"]
CAL = ["month_sin", "month_cos", "year"]
FEAT_COLS = LAG_COLS + CAL + CFG_COLS
SEASONAL_FEAT_COLS = FEAT_COLS + SEASONAL_LAG_COLS
TARGET = "monthly_sales"


def load_splits(parse_dates=True):
    """读取固定的 train、validation 和 test 文件。"""
    kw = {"parse_dates": ["date"]} if parse_dates else {}
    tr = pd.read_csv(os.path.join(SPLITS, "train.csv"), **kw)
    va = pd.read_csv(os.path.join(SPLITS, "val.csv"), **kw)
    te = pd.read_csv(os.path.join(SPLITS, "test.csv"), **kw)
    return tr, va, te


def load_panel_for_subset(subset):
    """拼接 train+val+test，仅保留 subset 车系，按 (series_name, date) 排序。

    返回的 DataFrame 含 train/val/test 全部月份 + 特征 + 配置，供
    recursive_forecast_tree 按车系、按时间递归预测。
    """
    tr, va, te = load_splits()
    all_df = pd.concat([tr, va, te], ignore_index=True)
    ss = set(map(str, subset))
    all_df = all_df[all_df["series_name"].astype(str).isin(ss)]
    return all_df.sort_values(["series_name", "date"]).reset_index(drop=True)


def wmape_vol(y_true, y_pred):
    """Return global volume-weighted WMAPE in percent."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    s = np.abs(yt).sum()
    return (np.abs(yt - yp).sum() / s * 100) if s > 0 else np.nan


def wmape_per_series(y_true, y_pred, series):
    """Return one WMAPE value per series."""
    df = pd.DataFrame({
        "s": np.asarray(series),
        "a": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_pred, dtype=float),
    })
    g = df.groupby("s").apply(
        lambda x: (np.abs(x.a - x.p).sum() / np.abs(x.a).sum() * 100)
        if np.abs(x.a).sum() > 0 else np.nan
    )
    return g


def recursive_forecast_tree(model, series_df, feat_cols=None,
                            history_splits=("train",), forecast_splits=("val", "test")):
    """用训练好的树模型在指定时期上做递归多步预测。

    ``history_splits`` contains periods whose realised sales are legitimately
    known when the forecast starts; ``forecast_splits`` are forecast
    recursively.  For example, validation uses ``train -> val`` while the
    final test forecast uses ``train+val -> test`` after model selection.

    机制（无泄漏）：
      * seed history = ``history_splits`` 中、预测起点之前的真实 monthly_sales。
      * 对每一个 forecast 月份：
          - 用「当前 running history」构造 lag_1..3 / roll_mean_3/6（只引用过去），
          - 配置列取该月自身在 splits 中已 join 好的因果配置（year<=行年），
          - ``feat_cols`` 中的其他列直接取该预测月预先准备好的外生变量；
            调用方必须保证它们在预测起点已可获得（例如按月截断的评论特征）。
          - predict -> expm1 -> clip(>=0) -> 记为该月预测，并 append 进 history。
      * 因此 test 月份用到的 lag，除了首月来自真实 2025-06，其后均为递归预测值，
        与真实推理一致；绝不偷看该月实际销量。

    返回 {Timestamp: pred}（仅非 train 月份）。
    """
    if feat_cols is None:
        feat_cols = FEAT_COLS
    series_df = series_df.sort_values("date").reset_index(drop=True)
    cfg = {d: {c: r[c] for c in CFG_COLS}
           for d, r in series_df.set_index("date").iterrows()}
    history_part = series_df[series_df["split"].isin(history_splits)]
    if len(history_part) == 0:
        return {}
    history = history_part[TARGET].astype(float).tolist()
    preds = {}
    for _, r in series_df.iterrows():
        if r["split"] not in forecast_splits:
            continue
        d = r["date"]
        h = np.asarray(history, dtype=float)
        row = {
            "lag_1": h[-1] if len(h) >= 1 else 0.0,
            "lag_2": h[-2] if len(h) >= 2 else 0.0,
            "lag_3": h[-3] if len(h) >= 3 else 0.0,
            "roll_mean_3": float(np.mean(h[-3:])) if len(h) >= 1 else 0.0,
            "roll_mean_6": float(np.mean(h[-6:])) if len(h) >= 1 else 0.0,
            "month_sin": np.sin(2 * np.pi * d.month / 12),
            "month_cos": np.cos(2 * np.pi * d.month / 12),
            "year": d.year,
        }
        for c in CFG_COLS:
            row[c] = cfg[d][c]
        # External features must already satisfy their information cutoff.
        for c in feat_cols:
            if c not in row:
                if c not in r.index:
                    raise KeyError(f"Feature '{c}' is missing from the forecasting panel")
                row[c] = r[c]
        X = pd.DataFrame([row], columns=feat_cols)
        p = float(np.expm1(model.predict(X)[0]))
        p = max(p, 0.0)
        preds[d] = p
        history.append(p)
    return preds


def metrics(y_true, y_pred):
    """Return MAE, RMSE, MAPE, and WMAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}
