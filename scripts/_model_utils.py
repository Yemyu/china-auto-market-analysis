#!/usr/bin/env python3
"""
_model_utils.py — 月度销量预测的共享工具

集中管理「时间切分」的读取与评估，保证 07~14 全部模型脚本：
  * 只从 data/processed_new/splits/{train,val,test}.csv 取数（由 06_make_splits.py 生成）
  * 训练只用 train，调参/早停用 val，最终指标只在 test 上报告
  * 不各自重新定义 holdout、不泄漏未来信息

提供：
  load_splits()            读三份切分
  load_panel_for_subset()  取某子集车系在 train+val+test 上的完整面板（按车系×时间排序）
  FEAT_COLS / CFG_COLS / TARGET
  wmape_vol() / wmape_per_series()   双口径 WMAPE（volume-weighted 与 per-series）
  recursive_forecast_tree() 用树模型在 val+test 上做递归多步预测（seed=真实 train 历史）
"""
import os

import numpy as np
import pandas as pd

from _feature_join import CFG_COLS  # 因果配置列（year<=行年）

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS = os.path.join(BASE, "data", "processed_new", "splits")

LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
CAL = ["month_sin", "month_cos", "year"]
FEAT_COLS = LAG_COLS + CAL + CFG_COLS
TARGET = "monthly_sales"


def load_splits(parse_dates=True):
    """读取 train / val / test 三份切分（GitHub 典型布局）。"""
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
    """全局 volume-weighted WMAPE（爆款主导，诚实且低）。分母为全部真实销量绝对值之和。"""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    s = np.abs(yt).sum()
    return (np.abs(yt - yp).sum() / s * 100) if s > 0 else np.nan


def wmape_per_series(y_true, y_pred, series):
    """逐车系 WMAPE，返回 Series(index=series_name)。小销量系相对误差大，均值会被拉飞。"""
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
            调用方必须保证它们在预测起点已可获得（例如按月截断的舆情）。
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
        # Support time-eligible exogenous features without changing the
        # recursive treatment of sales lags.  This keeps the shared helper
        # usable for future feature families such as monthly sentiment.
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
    """通用逐样本指标 dict（与旧模型脚本保持同名同口径，便于 09 聚合）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}
