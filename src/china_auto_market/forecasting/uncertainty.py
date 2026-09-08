"""Paired series-cluster uncertainty for saved, already selected forecasts.

Resampling keeps all months of a series together. These conditional intervals
do not account for model selection, refitting or a new future time window.
"""
import numpy as np
import pandas as pd


def series_totals(predictions: pd.DataFrame):
    keys = ["series_name", "date"]
    required = ["version", *keys, "actual", "pred"]
    if predictions.empty or predictions[required].isna().any().any():
        raise ValueError("Predictions must have complete keys and values")
    if predictions.duplicated(["version", *keys]).any():
        raise ValueError("Duplicate version/series/month predictions")
    values = predictions[["actual", "pred"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Sales and predictions must be finite and nonnegative")
    names = sorted(predictions.series_name.unique())
    reference = None
    errors = {}
    volume = None
    for version, part in predictions.groupby("version", sort=True):
        ordered = part.sort_values(keys).reset_index(drop=True)
        target = ordered[[*keys, "actual"]]
        if reference is not None and not target.equals(reference):
            raise ValueError("Model versions must have identical keys and actuals")
        reference = target
        grouped = ordered.assign(error=(ordered.actual - ordered.pred).abs()).groupby("series_name")
        errors[version] = grouped.error.sum().reindex(names).to_numpy(float)
        volume = grouped.actual.sum().reindex(names).to_numpy(float)
    if volume.sum() <= 0:
        raise ValueError("WMAPE is undefined for zero total actual volume")
    return names, errors, volume


def paired_bootstrap(predictions, comparisons, *, replicates=5000, seed=42):
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    names, errors, volume = series_totals(predictions)
    indices = np.random.default_rng(seed).integers(0, len(names), size=(replicates, len(names)))
    denominator = volume[indices].sum(axis=1)
    valid = denominator > 0
    if not valid.any():
        raise ValueError("No defined bootstrap WMAPE replicates")
    rows = []
    for comparator, candidate in comparisons:
        difference = errors[comparator] - errors[candidate]
        improvement = 100 * difference[indices].sum(axis=1)[valid] / denominator[valid]
        rows.append({
            "comparator": comparator, "candidate": candidate,
            "point_comparator_WMAPE": 100 * errors[comparator].sum() / volume.sum(),
            "point_candidate_WMAPE": 100 * errors[candidate].sum() / volume.sum(),
            "point_improvement_pp": 100 * difference.sum() / volume.sum(),
            "bootstrap_mean_improvement_pp": float(improvement.mean()),
            "bootstrap_ci_2_5_pp": float(np.quantile(improvement, .025)),
            "bootstrap_ci_97_5_pp": float(np.quantile(improvement, .975)),
            # Historical column name retained for consumers: this is a fraction
            # of defined resamples, not a posterior probability of superiority.
            "bootstrap_probability_candidate_better": float((improvement > 0).mean()),
            "bootstrap_replicates": replicates,
            "bootstrap_valid_replicates": int(valid.sum()),
            "bootstrap_zero_volume_replicates": int((~valid).sum()),
            "bootstrap_seed": seed, "cluster_unit": "series_name",
        })
    return pd.DataFrame(rows)
