"""Shared split, metric, and recursive-forecast utilities."""

import numpy as np
import pandas as pd

from china_auto_market.features.configuration import CFG_COLS
from china_auto_market.paths import PROJECT_ROOT

SPLITS = PROJECT_ROOT / "data" / "processed" / "splits"

LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
SEASONAL_LAG_COLS = ["lag_12", "roll_mean_12"]
CAL = ["month_sin", "month_cos", "year"]
FEAT_COLS = LAG_COLS + CAL + CFG_COLS
SEASONAL_FEAT_COLS = FEAT_COLS + SEASONAL_LAG_COLS
TARGET = "monthly_sales"


def load_splits(parse_dates=True, *, backend="csv", login_path="local-auto"):
    """Read locked splits from CSV or the parity-checked MySQL mart."""
    if backend == "mysql":
        from china_auto_market.warehouse.sources import load_forecast_feature_mart

        panel = load_forecast_feature_mart(login_path)
        # Preserve the historical split contract. Review features are attached
        # by the evaluation module under their own point-in-time protocol.
        columns = list(dict.fromkeys([
            "series_name", "series_id", "date", "year", "month", "brand",
            "category", "category_en", TARGET, *SEASONAL_FEAT_COLS, "split",
        ]))
        panel = panel[columns]
        if not parse_dates:
            panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
        return tuple(panel.loc[panel["split"].eq(split)].copy() for split in ("train", "val", "test"))
    if backend != "csv":
        raise ValueError(f"Unsupported data backend: {backend}")
    kw = {"parse_dates": ["date"]} if parse_dates else {}
    tr = pd.read_csv(SPLITS / "train.csv", **kw)
    va = pd.read_csv(SPLITS / "val.csv", **kw)
    te = pd.read_csv(SPLITS / "test.csv", **kw)
    return tr, va, te


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
    """Forecast each requested month and feed predictions into later lags.

    Observed sales from ``history_splits`` seed the history. Once forecasting
    starts, later lag features use earlier predictions. Any extra feature in
    ``feat_cols`` must already satisfy its own information cutoff.
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
