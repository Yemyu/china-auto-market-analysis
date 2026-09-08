"""Controlled repair assessment; never writes published research outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from china_auto_market.features.configuration import load_feature_source
from china_auto_market.features.configuration_window import prepare_configuration_window
from china_auto_market.forecasting import core, rolling_origin
from china_auto_market.forecasting.reporting import prediction_metrics
from china_auto_market.paths import PROJECT_ROOT


def run_audit(output: Path, *, backend: str = "csv", login_path: str = "local-auto") -> dict:
    output = output.resolve()
    if (PROJECT_ROOT / "artifacts").resolve() not in output.parents:
        raise ValueError("Repair assessments must write below project artifacts/")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier assessments")
    train, val, test = core.load_splits(backend=backend, login_path=login_path)
    if train.attrs.get("configuration_policy") in {"raw-batch-reference-v1", "deferred-fit-window-v1"}:
        raise ValueError("Legacy replay requires archived pre-repair splits. "
                         "Use the regular evaluation and repaired reference bundle for migrated inputs.")
    panel = pd.concat([train, val, test], ignore_index=True).sort_values(["series_name", "date"])
    if backend == "mysql":
        from china_auto_market.warehouse.sources import load_raw_configuration
        source = load_raw_configuration(login_path)
    else:
        source = load_feature_source()
    if test.series_name.nunique() != 371 or len(test) != 2226:
        raise ValueError("Frozen test cohort changed")
    saved_path = rolling_origin.TEST_OUTPUT
    saved = pd.read_csv(saved_path, parse_dates=["date"])
    keys = ["series_name", "date"]
    saved = saved.sort_values(keys).reset_index(drop=True)
    # Replay the old main model before attributing differences to preprocessing.
    columns = list(core.SEASONAL_FEAT_COLS)
    old_model = rolling_origin.fit_model("SEASONAL_D5", panel.loc[panel.date.lt("2026-01-01")], columns)
    old = rolling_origin.rolling_predictions(old_model, panel, columns, "test", ("train", "val"))
    pd.testing.assert_frame_equal(old[keys], saved[keys])
    np.testing.assert_allclose(old.actual, saved.actual, rtol=0, atol=0)
    np.testing.assert_allclose(old.pred, saved.pred, rtol=0, atol=1e-8)
    print("Original SEASONAL_D5 predictions reproduced", flush=True)
    old_history = pd.read_csv(rolling_origin.VALIDATION_OUTPUT)
    rows, states = [], []
    output.mkdir(parents=True, exist_ok=True)
    for origin in (*rolling_origin.ORIGINS, pd.Timestamp("2026-01-01")):
        end = origin + pd.offsets.MonthBegin(6)
        window = panel.loc[panel.date.lt(end)].copy()
        window["split"] = np.where(window.date.lt(origin), "train", "val")
        naive = rolling_origin.naive_rolling_predictions(window, "val", ("train",))
        if origin.year == 2026:
            np.testing.assert_allclose(naive[list(rolling_origin.NAIVE_METHODS)],
                                       saved[list(rolling_origin.NAIVE_METHODS)], rtol=0, atol=1e-8)
        for version, features in (("BASE", core.FEAT_COLS), ("SEASONAL_D5", core.SEASONAL_FEAT_COLS)):
            rebuilt, state = prepare_configuration_window(window, source, origin, list(features))
            unchanged = [c for c in window.columns if c not in core.CFG_COLS]
            pd.testing.assert_frame_equal(rebuilt[unchanged], window[unchanged].reset_index(drop=True))
            old_fit_rows = int(window.loc[window.date.lt(origin), features].notna().all(axis=1).sum())
            if state["fit_rows"] != old_fit_rows:
                raise ValueError("Repair changed the effective training cohort")
            model = rolling_origin.fit_model(version, rebuilt.loc[rebuilt.date.lt(origin)], list(features))
            predicted = rolling_origin.rolling_predictions(model, rebuilt, list(features), "val", ("train",))
            predicted = predicted.merge(naive, on=[*keys, "actual"], validate="one_to_one")
            scored = prediction_metrics(predicted)
            if origin.year == 2026:
                old_wmape = prediction_metrics(saved)["wmape"] if version == "SEASONAL_D5" else None
            else:
                old_wmape = float(old_history.loc[
                    old_history.origin.eq(origin.strftime("%Y-%m-%d"))
                    & old_history.version.eq(version) & old_history["mode"].eq("ROLLING_ONE_MONTH"),
                    "global_volume_weighted_WMAPE"].item())
            rows.append({"origin": origin.strftime("%Y-%m-%d"), "version": version,
                         "fit_rows": old_fit_rows, **scored, "old_wmape": old_wmape,
                         "repair_change_pp": scored["wmape"] - old_wmape if old_wmape is not None else None,
                         "last_value_wmape": prediction_metrics(predicted, "LAST_VALUE")["wmape"]})
            states.append({"version": version, **state})
            predicted.to_csv(output / f"{origin:%Y-%m}-{version}.csv", index=False)
            print(f"{origin:%Y-%m} {version}: {scored['wmape']:.6f}% (old {old_wmape})", flush=True)
    summary = {
        "schema_version": "configuration-repair-assessment-v1",
        "backend": backend, "params": rolling_origin.MODEL_PARAMS,
        "model_selection_performed": False, "test_is_new_holdout": False,
        "original_main_predictions_reproduced": True,
        "saved_predictions_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        "scope": "Rolling forecast repair assessment; fixed reviews/cold-start and persisted marts remain legacy",
        "results": rows, "preprocessing": states,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    return summary
