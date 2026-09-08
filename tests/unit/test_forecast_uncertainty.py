import numpy as np
import pandas as pd
import pytest

from china_auto_market.forecasting.uncertainty import paired_bootstrap, series_totals


@pytest.fixture
def predictions():
    base = pd.DataFrame({"series_name": ["a", "a", "b", "b"],
                         "date": ["2026-01", "2026-02"] * 2,
                         "actual": [10., 20., 0., 0.], "pred": [20., 30., 1., 1.]})
    candidate = base.assign(pred=[15., 25., 0., 0.])
    return pd.concat([base.assign(version="BASE"), candidate.assign(version="CANDIDATE")], ignore_index=True)


def test_paired_bootstrap_matches_manual_cluster_sampling(predictions):
    result = paired_bootstrap(predictions, [("BASE", "CANDIDATE")], replicates=100, seed=42).iloc[0]
    draws = np.random.default_rng(42).integers(0, 2, size=(100, 2))
    delta, undefined = [], 0
    for draw in draws:
        numerator = sum([10., 2.][i] for i in draw)
        denominator = sum([30., 0.][i] for i in draw)
        if denominator:
            delta.append(100 * numerator / denominator)
        else:
            undefined += 1
    assert result.point_improvement_pp == 40
    assert result.bootstrap_ci_2_5_pp == pytest.approx(np.quantile(delta, .025))
    assert result.bootstrap_ci_97_5_pp == pytest.approx(np.quantile(delta, .975))
    assert result.bootstrap_probability_candidate_better == 1
    assert result.bootstrap_zero_volume_replicates == undefined
    assert result.bootstrap_valid_replicates == len(delta)
    shuffled = paired_bootstrap(predictions.sample(frac=1, random_state=9), [("BASE", "CANDIDATE")], replicates=100)
    pd.testing.assert_series_equal(result, shuffled.iloc[0])


def test_zero_improvement_is_not_a_win(predictions):
    result = paired_bootstrap(predictions, [("BASE", "BASE")]).iloc[0]
    assert result.bootstrap_probability_candidate_better == 0
    assert result.bootstrap_ci_97_5_pp == 0


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "moved_date", "swapped_actual", "negative", "nan"])
def test_invalid_alignment_is_rejected(predictions, mutation):
    if mutation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
    elif mutation == "missing":
        predictions = predictions.drop(index=7)
    elif mutation == "moved_date":
        predictions.loc[4, "date"] = "2025-12"
    elif mutation == "swapped_actual":
        predictions.loc[[4, 5], "actual"] = [20., 10.]  # unchanged series total
    else:
        predictions.loc[4, "pred"] = -1 if mutation == "negative" else np.nan
    with pytest.raises(ValueError):
        series_totals(predictions)


def test_all_zero_volume_fails(predictions):
    with pytest.raises(ValueError, match="undefined"):
        paired_bootstrap(predictions.assign(actual=0), [("BASE", "CANDIDATE")])
