import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from china_auto_market.features.configuration import CFG_CAT, CFG_COLS, CFG_NUM
from china_auto_market.features.configuration_window import prepare_configuration_window


def source_row(name, year, value, category="known"):
    return {"series_name": name, "year": year,
            **dict.fromkeys(CFG_NUM, value), **dict.fromkeys(CFG_CAT, category)}


def panel():
    return pd.DataFrame({
        "series_name": ["A", "B", "A", "B"],
        "date": pd.to_datetime(["2023-06-01", "2023-06-01", "2024-01-01", "2024-01-01"]),
        "lag_1": [1., 2., 3., 4.], "monthly_sales": [10., 20., 30., 40.],
        # Poisoned legacy values must be ignored, including their missingness.
        **dict.fromkeys(CFG_COLS, [999., np.nan, 999., 999.]),
    })


def prepare(frame, source):
    return prepare_configuration_window(frame, source, pd.Timestamp("2024-01-01"), ["lag_1", *CFG_COLS])


def test_future_values_categories_and_aliases_cannot_change_past_window():
    before = pd.DataFrame([source_row("A", 2022, 10), source_row("B", 2022, np.nan)])
    future = pd.DataFrame([source_row("A", 2024, 1e9, "AAA"),
                           source_row("B", 2027, -1e9, "new"),
                           source_row("A-", 2025, 999, "alias")])
    left, a = prepare(panel(), before)
    right, b = prepare(panel(), pd.concat([before, future], ignore_index=True))
    assert_frame_equal(left, right)
    assert a == b
    assert a["numeric_medians"][CFG_NUM[0]] == 10
    assert a["fit_rows"] == 2


def test_future_labels_do_not_fit_preprocessing_and_missing_history_is_retained():
    source = pd.DataFrame([source_row("A", 2022, 10)])
    frame = panel()
    result, info = prepare(frame, source)
    frame.loc[frame.date.ge("2024-01-01"), "monthly_sales"] = 1e12
    changed, other = prepare(frame, source)
    assert_frame_equal(result[CFG_COLS], changed[CFG_COLS])
    assert info == other
    assert len(result) == len(frame)
    assert info["unmatched_training_rows"] == 1
    assert result.loc[result.series_name.eq("B"), CFG_CAT[0] + "_enc"].eq(-1).all()


def test_source_fields_on_prediction_rows_do_not_enter_training_statistics():
    frame = panel()
    # B's 2023 config is available at origin but not at its 2022 training row.
    frame.loc[1, "date"] = pd.Timestamp("2022-06-01")
    source = pd.DataFrame([source_row("A", 2022, 10), source_row("B", 2023, 1000, "unseen")])
    result, info = prepare(frame, source)
    assert info["numeric_medians"][CFG_NUM[0]] == 10
    assert result.loc[3, CFG_NUM[0]] == 1000
    assert result.loc[3, CFG_CAT[0] + "_enc"] == -1
    assert "unseen" not in info["category_codes"][CFG_CAT[0]]


def test_all_missing_columns_use_constants_not_future_data():
    source = pd.DataFrame([source_row("A", 2022, np.nan, None)])
    result, info = prepare(panel(), source)
    assert info["all_missing_numeric_columns"] == CFG_NUM
    assert result[CFG_NUM].eq(0).all().all()
    assert result[[c + "_enc" for c in CFG_CAT]].eq(-1).all().all()


def test_no_available_source_keeps_entire_cohort():
    source = pd.DataFrame([source_row("A", 2027, 100)])
    result, info = prepare(panel(), source)
    assert len(result) == 4
    assert info["source_records_available"] == 0
    assert result[CFG_NUM].eq(0).all().all()


def test_missing_lags_define_actual_fit_rows():
    frame = panel()
    frame.loc[0, "lag_1"] = np.nan
    source = pd.DataFrame([source_row("A", 2022, 1000), source_row("B", 2022, 20)])
    result, info = prepare(frame, source)
    assert info["fit_rows"] == 1
    assert info["numeric_medians"][CFG_NUM[0]] == 20
    assert pd.isna(result.loc[0, "lag_1"])


def test_ambiguous_available_configuration_fails_instead_of_multiplying_rows():
    source = pd.DataFrame([source_row("A", 2022, 10), source_row("A", 2022, 20)])
    with pytest.raises(ValueError, match="duplicate"):
        prepare(panel(), source)


def test_future_source_and_test_labels_do_not_change_first_predictions():
    from china_auto_market.forecasting.rolling_origin import fit_model, rolling_predictions

    features = ["lag_1", *CFG_COLS]
    frame = panel()
    frame["split"] = np.where(frame.date.lt("2024-01-01"), "train", "val")
    source = pd.DataFrame([source_row("A", 2022, 10), source_row("B", 2022, np.nan)])
    fixed, _ = prepare(frame, source)
    model = fit_model("BASE", fixed.loc[fixed.split.eq("train")], features)
    expected = rolling_predictions(model, fixed, features, "val", ("train",))
    mutated = frame.copy()
    mutated.loc[mutated.split.eq("val"), "monthly_sales"] = 1e10
    future = pd.DataFrame([source_row("A", 2027, 1e12, "new")])
    rebuilt, _ = prepare(mutated, pd.concat([source, future], ignore_index=True))
    other_model = fit_model("BASE", rebuilt.loc[rebuilt.split.eq("train")], features)
    actual = rolling_predictions(other_model, rebuilt, features, "val", ("train",))
    np.testing.assert_array_equal(expected.pred, actual.pred)


def test_audit_cannot_write_into_published_results(tmp_path):
    from china_auto_market.forecasting.configuration_audit import run_audit

    with pytest.raises(ValueError, match="artifacts"):
        run_audit(tmp_path)


def test_audit_does_not_replay_legacy_model_on_deferred_inputs(monkeypatch):
    from china_auto_market.forecasting import core
    from china_auto_market.forecasting.configuration_audit import run_audit
    from china_auto_market.paths import PROJECT_ROOT

    frame = pd.DataFrame()
    frame.attrs["configuration_policy"] = "raw-batch-reference-v1"
    monkeypatch.setattr(core, "load_splits", lambda **kwargs: (frame, frame, frame))
    with pytest.raises(ValueError, match="archived pre-repair"):
        run_audit(PROJECT_ROOT / "artifacts" / "unit-test-legacy-replay-no-write")


def test_fixed_six_month_predictions_ignore_all_future_targets():
    from china_auto_market.forecasting.review_evaluation import new_model, recursive_predictions

    frame = pd.DataFrame({
        "series_name": ["A"] * 8,
        "date": pd.to_datetime(["2023-11-01", "2023-12-01", *pd.date_range("2024-01-01", periods=6, freq="MS")]),
        "monthly_sales": [10., 20., 0., 0., 100., 200., 300., 400.],
        "lag_1": [5., 10., 20., 0., 0., 100., 200., 300.],
        "split": ["train"] * 2 + ["test"] * 6,
        "review_available_prior": [0] * 8,
        "review_available_180d": [0] * 8,
        "optional_review": [np.nan] * 8,
    })
    source = pd.DataFrame([source_row("A", 2022, 10)])
    columns = ["lag_1", *CFG_COLS, "optional_review"]

    def predict(input_frame, specs):
        prepared, metadata = prepare_configuration_window(
            input_frame, specs, pd.Timestamp("2024-01-01"), ["lag_1", *CFG_COLS])
        training = prepared.loc[prepared.split.eq("train")]
        model = new_model(10)
        model.fit(training[columns], np.log1p(training.monthly_sales))
        return recursive_predictions(model, prepared, columns, "test", ("train",), "BASE", "test"), metadata

    expected, state = predict(frame, source)
    frame.loc[frame.split.eq("test"), ["monthly_sales", "lag_1"]] = 1e12
    source = pd.concat([source, pd.DataFrame([source_row("A", 2027, 1e12, "future")])])
    actual, other = predict(frame, source)
    np.testing.assert_array_equal(expected.pred, actual.pred)
    assert state == other
    assert state["fit_rows"] == 2  # Missing review features do not exclude training rows.


def test_validation_configuration_does_not_fit_on_validation_rows():
    from china_auto_market.forecasting import core
    from china_auto_market.forecasting.review_evaluation import prepare_fixed_panels

    frame = pd.DataFrame({
        "series_name": ["A", "B", "B"],
        "date": pd.to_datetime(["2025-06-01", "2025-07-01", "2026-01-01"]),
        "split": ["train", "val", "test"],
        **dict.fromkeys(core.LAG_COLS + core.CAL, [1., 1., 1.]),
    })
    frames = {"train_roll": frame.iloc[:1], "val_roll": frame.iloc[1:2],
              "val_fixed": frame.iloc[1:2], "test_roll": frame.iloc[2:], "test_fixed": frame.iloc[2:]}
    source = pd.DataFrame([source_row("A", 2024, 10), source_row("B", 2024, 1000, "B")])
    _, final, _, states = prepare_fixed_panels(frames, source)
    assert states[0]["numeric_medians"][CFG_NUM[0]] == 10
    assert states[1]["numeric_medians"][CFG_NUM[0]] == 505
    assert len(final) == 3
    permuted = {key: value.iloc[::-1] for key, value in frames.items()}
    _, reordered, _, other = prepare_fixed_panels(permuted, source.iloc[::-1])
    assert_frame_equal(final, reordered)
    assert states == other


def test_fixed_panels_have_backend_independent_training_order():
    from china_auto_market.forecasting import core
    from china_auto_market.forecasting.review_evaluation import prepare_fixed_panels

    frame = pd.DataFrame({
        "series_name": ["B", "A", "B", "A", "B", "A"],
        "date": pd.to_datetime(["2025-06-01"] * 2 + ["2025-07-01"] * 2 + ["2026-01-01"] * 2),
        "split": ["train"] * 2 + ["val"] * 2 + ["test"] * 2,
        **dict.fromkeys(core.LAG_COLS + core.CAL, [1.] * 6),
    })
    frames = {"train_roll": frame.iloc[:2], "val_roll": frame.iloc[2:4],
              "val_fixed": frame.iloc[2:4], "test_roll": frame.iloc[4:], "test_fixed": frame.iloc[4:]}
    source = pd.DataFrame([source_row("A", 2024, 10), source_row("B", 2024, 1000)])
    expected = prepare_fixed_panels(frames, source)
    actual = prepare_fixed_panels({key: value.iloc[::-1] for key, value in frames.items()}, source.iloc[::-1])
    for left, right in zip(expected[:3], actual[:3], strict=True):
        assert_frame_equal(left, right)
    assert expected[3] == actual[3]
