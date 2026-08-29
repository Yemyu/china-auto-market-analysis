#!/usr/bin/env python3
"""Assemble the dashboard payloads from audited analysis outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
ASPECT_ZH = {
    "appearance": "外观", "interior": "内饰", "space": "空间", "power": "动力",
    "control": "操控", "comfort": "舒适", "fuel_consumption": "能耗/油耗",
    "configuration": "配置", "intelligence": "智能化", "value": "性价比",
}
ASPECT_EN = {
    "appearance": "Appearance", "interior": "Interior", "space": "Space", "power": "Power",
    "control": "Control", "comfort": "Comfort", "fuel_consumption": "Energy/Fuel",
    "configuration": "Configuration", "intelligence": "Intelligence", "value": "Value",
}
PALETTE = ["#2c7be5", "#00a9ae", "#34c38f", "#f6c343", "#ee5b5b", "#a55eea", "#7783f5"]
MODEL_LABELS = {
    "BASE": ("销量基线（XGBoost）", "Sales baseline (XGBoost)"),
    "PLATFORM_RATING_FIXED": ("平台评分增强（XGBoost）", "Platform-rating enhanced (XGBoost)"),
    "LOCAL_LEXICON_FIXED": ("本地词典增强（XGBoost）", "Local-lexicon enhanced (XGBoost)"),
    "REVIEW_TEXT_FIXED": ("文本情感增强（XGBoost）", "Text-sentiment enhanced (XGBoost)"),
    "REVIEW_RICH_FIXED": ("用户口碑增强（XGBoost）", "User-review enhanced (XGBoost)"),
    "ALL_SENTIMENT_FIXED": ("全部口碑增强（XGBoost）", "Combined-review enhanced (XGBoost)"),
    "REVIEW_TEXT_ROLLING": ("滚动口碑增强（XGBoost）", "Rolling-review enhanced (XGBoost)"),
    "REVIEW_RICH_COLD_START": ("口碑增强（XGBoost）＋冷启动统计补充", "Review-enhanced XGBoost + statistical cold-start"),
}
FEATURE_FAMILY_LABELS = {
    "sales_lag_roll": ("历史销量", "Sales history"),
    "calendar": ("日历", "Calendar"),
    "configuration": ("产品配置", "Product configuration"),
    "review_observation_context": ("评论覆盖", "Review coverage"),
    "review_expanding_score": ("历史评价", "Historical review score"),
    "review_recent_score": ("近期评价", "Recent review score"),
    "review_positive_rate": ("正面比例", "Positive share"),
    "review_negative_rate": ("负面比例", "Negative share"),
    "review_mention_count": ("需求提及量", "Need mentions"),
    "review_mention_rate": ("需求提及率", "Mention share"),
    "review_overall": ("评论特征", "Review features"),
}
CONFIG_LABELS = {
    "official_price_wan": ("官方价格", "Official price"),
    "engine_max_power_kw": ("发动机最大功率", "Engine max power"),
    "engine_max_torque_nm": ("发动机最大扭矩", "Engine max torque"),
    "battery_capacity_kwh": ("电池容量", "Battery capacity"),
    "battery_range_km": ("纯电续航", "Battery range"),
    "length_mm": ("车长", "Length"), "width_mm": ("车宽", "Width"),
    "height_mm": ("车高", "Height"), "wheelbase_mm": ("轴距", "Wheelbase"),
    "curb_weight_kg": ("整备质量", "Curb weight"), "seat_count": ("座位数", "Seat count"),
    "door_count": ("车门数", "Door count"), "trunk_volume_l": ("后备箱容积", "Trunk volume"),
}


def _read(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def _wmape_summary(frame: pd.DataFrame, actual: str, prediction: str) -> tuple[float, float, int]:
    """Return pooled WMAPE, median per-series WMAPE, and valid series count."""
    valid = frame.loc[frame[actual].notna() & frame[prediction].notna()].copy()
    denominator = float(valid[actual].abs().sum())
    pooled = float((valid[actual] - valid[prediction]).abs().sum() / denominator * 100) if denominator else np.nan
    per_series = valid.groupby("series_name").apply(
        lambda group: (
            float((group[actual] - group[prediction]).abs().sum() / group[actual].abs().sum() * 100)
            if group[actual].abs().sum() else np.nan
        )
    )
    per_series = per_series.dropna()
    return pooled, float(per_series.median()) if not per_series.empty else np.nan, int(per_series.size)


def _feature_label(feature: str) -> tuple[str, str]:
    simple = {
        "lag_1": ("上月销量", "Sales lag 1"), "lag_2": ("前2月销量", "Sales lag 2"),
        "lag_3": ("前3月销量", "Sales lag 3"), "roll_mean_3": ("近3月销量均值", "3-month sales mean"),
        "roll_mean_6": ("近6月销量均值", "6-month sales mean"), "month_sin": ("月份季节性（正弦）", "Month seasonality (sin)"),
        "month_cos": ("月份季节性（余弦）", "Month seasonality (cos)"), "year": ("年份", "Year"),
        "review_count_prior_all": ("历史评论数量", "Prior review count"),
        "review_count_180d": ("近180天评论数量", "180-day review count"),
    }
    if feature in simple:
        return simple[feature]
    categorical_prefixes = {
        "engine_cylinder_arrangement_": ("气缸排列", "Cylinder layout"),
        "battery_type_": ("电池类型", "Battery type"),
        "oil_supply_": ("供油方式", "Fuel supply"),
        "cylinder_material_": ("气缸材料", "Cylinder material"),
        "engine_intake_type_": ("进气形式", "Intake type"),
        "fuel_form_": ("燃料形式", "Fuel form"),
        "steering_wheel_material_": ("方向盘材质", "Steering-wheel material"),
        "motor_type_": ("电机类型", "Motor type"),
        "center_screen_": ("中控屏", "Center screen"),
        "fuel_grade_": ("燃油标号", "Fuel grade"),
        "sound_brand_": ("音响品牌", "Audio brand"),
    }
    for prefix, labels in categorical_prefixes.items():
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            value_zh = "缺失" if value == "NA" else value
            value_en = {
                "NA": "Missing",
                "L": "Inline (L)",
                "直喷": "Direct injection",
                "混合喷射": "Combined injection",
                "铝": "Aluminum",
                "插电式混合动力": "Plug-in hybrid",
                "48V轻混系统": "48V mild hybrid",
                "真皮": "Leather",
                "永磁同步": "Permanent-magnet synchronous",
                "宝华韦健": "Bowers & Wilkins",
            }.get(value, value)
            return f"{labels[0]}：{value_zh}", f"{labels[1]}: {value_en}"
    if feature == "manufacturer_freq":
        return "制造商频次", "Manufacturer frequency"
    for aspect in ASPECTS:
        if f"review_{aspect}_" in feature:
            zh, en = ASPECT_ZH[aspect], ASPECT_EN[aspect]
            suffixes = {
                "score_prior_mean": ("长期情感", "expanding sentiment"),
                "score_180d_mean": ("近180天情感", "180-day sentiment"),
                "positive_180d_rate": ("正面率", "positive rate"),
                "negative_180d_rate": ("负面率", "negative rate"),
                "uniform_mention_180d_count": ("提及量", "mention count"),
                "uniform_mention_180d_rate": ("提及率", "mention rate"),
            }
            for suffix, labels in suffixes.items():
                if feature.endswith(suffix):
                    return f"{zh}{labels[0]}", f"{en} {labels[1]}"
    return CONFIG_LABELS.get(feature, (feature, feature))


class DashboardData:
    def __init__(self, root: str | Path, brand_en: Callable[[str], str], series_en: Callable[[str, str | None], str]):
        self.root = Path(root)
        self.processed = self.root / "data" / "processed"
        self.reviews = self.root / "data" / "reviews" / "processed"
        self.brand_en = brand_en
        self.series_en = series_en
        self._panel: pd.DataFrame | None = None
        self._aligned: pd.DataFrame | None = None

    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            parts = [
                _read(self.processed / "splits" / f"{name}.csv", parse_dates=["date"])
                for name in ("train", "val", "test")
            ]
            self._panel = pd.concat(parts, ignore_index=True).sort_values(["series_name", "date"])
        return self._panel.copy()

    def aligned(self) -> pd.DataFrame:
        if self._aligned is None:
            panel = self.panel()
            sentiment = _read(self.reviews / "review_features_by_series_month_rolling.csv", parse_dates=["date"])
            columns = ["series_name", "date", "review_count_180d", *[
                f"review_{aspect}_score_180d_mean" for aspect in ASPECTS
            ]]
            aligned = panel.merge(sentiment[columns], on=["series_name", "date"], how="left", validate="one_to_one")
            aligned = aligned.rename(columns={
                f"review_{aspect}_score_180d_mean": aspect for aspect in ASPECTS
            })
            aligned["overall"] = aligned[ASPECTS].mean(axis=1, skipna=True)
            aligned["period"] = aligned["date"].dt.to_period("M").astype(str)
            self._aligned = aligned
        return self._aligned.copy()

    def overview(self) -> dict[str, Any]:
        panel = self.panel()
        needs = _read_json(self.processed / "user_feedback" / "user_needs_alerts_summary.json")
        cold = _read_json(self.processed / "forecast" / "cold_start_launch_curve_summary.json")
        model_summary = _read(self.processed / "forecast" / "review_feature_ablation_summary.csv")
        benchmark = _read(self.processed / "forecast" / "forecast_benchmark_comparison.csv")
        rolling = _read_json(self.processed / "forecast" / "rolling_origin_summary.json")
        rolling_test = _read(self.processed / "forecast" / "rolling_origin_test_predictions.csv")
        rolling_global, rolling_median, rolling_median_series = _wmape_summary(rolling_test, "actual", "pred")
        last_global, last_median, _ = _wmape_summary(rolling_test, "actual", "LAST_VALUE")
        fixed_hybrid = float(cold["hybrid_full371_global_WMAPE"])
        fixed_hybrid_median = float(cold["hybrid_full371_median_per_series_WMAPE"])
        fixed_naive_row = benchmark.loc[benchmark["method"].eq("ROLLING_MEAN_6")].iloc[0]
        fixed_naive = float(fixed_naive_row["global_volume_weighted_WMAPE"])
        fixed_naive_median = float(fixed_naive_row["median_per_series_WMAPE"])
        review_point = float(_read_json(self.processed / "forecast" / "forecast_robustness_summary.json")[
            "selected_feedback_vs_base_improvement_pp"
        ])
        config_ablation = _read(self.processed / "product" / "config_attribution_ablation.csv")
        config_brand = float(config_ablation.loc[config_ablation["variant"].eq("+BRAND"), "R2_log_mean"].iloc[0])
        config_full = float(config_ablation.loc[config_ablation["variant"].eq("+CONFIG"), "R2_log_mean"].iloc[0])
        monthly = panel.groupby("date")["monthly_sales"].sum().sort_index()
        return {
            "kpis": {
                "coverage_series": int(needs["review_series"]),
                "forecast_wmape": round(rolling_global, 4),
                "forecast_median_wmape": round(rolling_median, 4),
                "baseline_wmape": round(rolling_global, 4),
                "baseline_median_wmape": round(rolling_median, 4),
                "core_forecast_wmape": round(fixed_hybrid, 4),
                "core_forecast_median_wmape": round(fixed_hybrid_median, 4),
                "naive_wmape": round(last_global, 4),
                "naive_median_wmape": round(last_median, 4),
                "relative_error_reduction_vs_naive_pct": round((last_global - rolling_global) / last_global * 100, 2),
                "wmape_gain_pp": round(last_global - rolling_global, 2),
                "wmape_relative_gain_pct": round((last_global - rolling_global) / last_global * 100, 2),
                "forecast_horizon": 1,
                "forecast_rows": int(len(rolling_test)),
                "forecast_median_series": rolling_median_series,
                "fixed_stress_wmape": round(fixed_hybrid, 4),
                "fixed_stress_median_wmape": round(fixed_hybrid_median, 4),
                "fixed_stress_naive_wmape": round(fixed_naive, 4),
                "fixed_stress_naive_median_wmape": round(fixed_naive_median, 4),
                "fixed_stress_horizon": 6,
                "fixed_stress_relative_error_reduction_pct": round((fixed_naive - fixed_hybrid) / fixed_naive * 100, 2),
                "review_fixed_point_gain_pp": round(review_point, 3),
                "config_brand_r2": round(config_brand, 3),
                "config_full_r2": round(config_full, 3),
                "alert_count": int(needs["latest_active_alerts"]),
                "historical_alerts": int(needs["historical_alert_events"]),
                "brand_count": int(panel["brand"].nunique()),
                "eval_series": 371,
                "sales_series": 1017,
                "config_series": 766,
                "attribution_series": 736,
                "attribution_rows": 2007,
                "review_count": int(needs["review_rows"]),
            },
            "monthly_trend": [
                {"month": date.strftime("%Y-%m"), "sales": round(float(value), 0)}
                for date, value in monthly.items()
            ],
            "stages": [
                {"id": 1, "name": "数据准备", "en": "Data Preparation", "status": "done"},
                {"id": 2, "name": "严格时间切分", "en": "Strict Time Split", "status": "done"},
                {"id": 3, "name": "销量预测与消融", "en": "Forecasting & Ablation", "status": "done"},
                {"id": 4, "name": "配置年度归因", "en": "Annual Config Attribution", "status": "done"},
                {"id": 5, "name": "需求主题与预警", "en": "Needs & Alerts", "status": "done"},
                {"id": 6, "name": "看板更新", "en": "Dashboard Refresh", "status": "current"},
            ],
            "findings": [
                {"zh": f"滚动单月XGBoost为{rolling_global:.2f}% WMAPE，比“沿用上月销量”朴素基准{last_global:.2f}%低{last_global-rolling_global:.2f}个百分点", "en": f"The rolling one-month XGBoost reaches {rolling_global:.2f}% WMAPE, {last_global-rolling_global:.2f} pp below the last-observed-value naive baseline at {last_global:.2f}%"},
                {"zh": f"固定六个月压力测试的综合方案为{fixed_hybrid:.2f}%，相对同场景最近6个月均值{fixed_naive:.2f}%减少{(fixed_naive-fixed_hybrid)/fixed_naive*100:.1f}%绝对误差；它与滚动主结果不是同一任务", "en": f"The fixed six-month stress test scores {fixed_hybrid:.2f}% versus {fixed_naive:.2f}% for its trailing-six-month naive comparator; it reduces absolute error by {(fixed_naive-fixed_hybrid)/fixed_naive*100:.1f}% and is a different task from the rolling headline"},
                {"zh": f"固定起点口碑增强点估计改善{review_point:.3f}个百分点，但Bootstrap区间跨0，因此保留为补充证据", "en": f"Fixed-origin review enhancement shows a {review_point:.3f} pp point estimate, but its bootstrap interval crosses zero, so it remains supporting evidence"},
                {"zh": f"736车系年度归因中，配置将分组交叉验证R²从{config_brand:.3f}提升到{config_full:.3f}", "en": f"Across 736 series, specifications raise grouped-CV annual-attribution R² from {config_brand:.3f} to {config_full:.3f}"},
                {"zh": "智能化与舒适性是负面反馈最集中的两个用户需求维度", "en": "Intelligence and comfort carry the highest complaint concentration"},
                {"zh": f"截至{needs['latest_completed_monitoring_month'][:7]}，当前有效预警{needs['latest_active_alerts']}条", "en": f"As of {needs['latest_completed_monitoring_month'][:7]}, {needs['latest_active_alerts']} active alert is detected"},
            ],
        }

    def forecast(self) -> dict[str, Any]:
        summary = _read(self.processed / "forecast" / "review_feature_ablation_summary.csv")
        shap = _read(self.processed / "forecast" / "review_feature_shap_importance.csv")
        rolling_test = _read(self.processed / "forecast" / "rolling_origin_test_predictions.csv", parse_dates=["date"])
        cold = _read_json(self.processed / "forecast" / "cold_start_launch_curve_summary.json")
        benchmark = _read(self.processed / "forecast" / "forecast_benchmark_comparison.csv")
        rolling_models = [
            ("LAST_VALUE", "沿用上月销量（朴素基准）", "Last observed value (naive)", "rolling_naive_last"),
            ("ROLLING_MEAN_3", "近3月均值（滚动朴素）", "Trailing 3-month mean (naive)", "rolling_naive_mean3"),
            ("ROLLING_MEAN_6", "近6月均值（滚动朴素）", "Trailing 6-month mean (naive)", "rolling_naive_mean6"),
            ("SEASONAL_LAG12", "去年同期销量（朴素基准）", "Same-month-last-year (naive)", "rolling_naive_seasonal"),
            ("pred", "滚动单月销量基线（XGBoost）", "Rolling one-month sales baseline (XGBoost)", "rolling_primary"),
        ]
        models: list[dict[str, Any]] = []
        for column, zh, en, scenario in rolling_models:
            wmape_vol, wmape_med, _ = _wmape_summary(rolling_test, "actual", column)
            models.append({
                "name": zh, "name_zh": zh, "name_en": en,
                "wmape_vol": round(wmape_vol, 4),
                "wmape_med": round(wmape_med, 4),
                "mae": None, "color": PALETTE[len(models) % len(PALETTE)],
                "scenario": scenario, "prediction_column": column,
            })
        fixed_models: list[dict[str, Any]] = []
        for _, row in summary.sort_values("global_volume_weighted_WMAPE").iterrows():
            zh, en = MODEL_LABELS.get(row["version"], (row["version"], row["version"]))
            fixed_models.append({
                "name": zh, "name_zh": zh, "name_en": en,
                "wmape_vol": round(float(row["global_volume_weighted_WMAPE"]), 4),
                "wmape_med": round(float(row["median_per_series_WMAPE"]), 4),
                "mae": None, "color": PALETTE[len(fixed_models) % len(PALETTE)],
                "scenario": "fixed_origin_stress_ablation",
                "version": str(row["version"]),
            })
        zh, en = MODEL_LABELS["REVIEW_RICH_COLD_START"]
        fixed_models.append({
            "name": zh, "name_zh": zh, "name_en": en,
            "wmape_vol": round(float(cold["hybrid_full371_global_WMAPE"]), 4),
            "wmape_med": round(float(cold["hybrid_full371_median_per_series_WMAPE"]), 4),
            "mae": None, "color": "#34c38f", "scenario": "fixed_origin_stress_cold_hybrid",
            "version": "SELECTED_FEEDBACK_COLD_START",
        })
        features = []
        for _, row in shap.sort_values("rank").head(12).iterrows():
            zh, en = _feature_label(str(row["feature"]))
            family_zh, family_en = FEATURE_FAMILY_LABELS.get(
                str(row["feature_family"]), ("其他", "Other")
            )
            features.append({
                "name": zh, "name_zh": zh, "name_en": en,
                "desc_zh": f"特征组：{family_zh}", "desc_en": f"Feature group: {family_en}",
                "importance": round(float(row["mean_abs_shap_log_sales"]), 4),
            })
        valid = rolling_test.loc[rolling_test["actual"].gt(0) & rolling_test["pred"].notna()].copy()
        meta = self.panel().drop_duplicates("series_name").set_index("series_name")["category_en"]
        valid["category_en"] = valid["series_name"].map(meta)
        class_rows = []
        for category, group in valid.groupby("category_en", dropna=True):
            denominator = group["actual"].abs().sum()
            class_rows.append({
                "category": str(category),
                "wmape": round(float((group["actual"] - group["pred"]).abs().sum() / denominator * 100), 1),
                "n_series": int(group["series_name"].nunique()),
            })
        class_rows.sort(key=lambda row: row["wmape"])
        fixed_naive = benchmark.loc[benchmark["method"].eq("ROLLING_MEAN_6")].iloc[0]
        fixed_hybrid = float(cold["hybrid_full371_global_WMAPE"])
        fixed_hybrid_median = float(cold["hybrid_full371_median_per_series_WMAPE"])
        rolling_primary = next(item for item in models if item["scenario"] == "rolling_primary")
        rolling_naive = next(item for item in models if item["scenario"] == "rolling_naive_last")
        _, _, median_series = _wmape_summary(rolling_test, "actual", "pred")
        return {
            "models": models,
            "primary_models": models,
            "fixed_models": fixed_models,
            "best_model": rolling_primary["name_zh"],
            "meta": {
                "eval_series": 371,
                "test_months": 6,
                "forecast_horizon": 1,
                "test_rows": int(len(rolling_test)),
                "valid_rows": int(len(valid)),
                "median_series": median_series,
                "train_end": "2025-06",
                "validation_period": "2025-07~2025-12",
                "test_period": "2026-01~2026-06",
                "cold_start_series": int(cold["cold_start_series"]),
                "best_naive_global_wmape": rolling_naive["wmape_vol"],
                "best_naive_median_wmape": rolling_naive["wmape_med"],
                "relative_error_reduction_vs_naive_pct": round(
                    (rolling_naive["wmape_vol"] - rolling_primary["wmape_vol"])
                    / rolling_naive["wmape_vol"] * 100, 2,
                ),
                "fixed_stress_wmape": round(fixed_hybrid, 4),
                "fixed_stress_median_wmape": round(fixed_hybrid_median, 4),
                "fixed_stress_naive_wmape": round(float(fixed_naive["global_volume_weighted_WMAPE"]), 4),
                "fixed_stress_naive_median_wmape": round(float(fixed_naive["median_per_series_WMAPE"]), 4),
            },
            "class_wmape": class_rows,
            "scatter": [[round(float(a), 1), round(float(p), 1)] for a, p in zip(valid["actual"], valid["pred"])],
            "features": features,
            "fixed_stress": {
                "name_zh": "固定六个月综合方案（口碑＋冷启动补充）",
                "name_en": "Fixed six-month combined method (reviews + cold-start supplement)",
                "wmape_vol": round(fixed_hybrid, 4),
                "wmape_med": round(fixed_hybrid_median, 4),
                "naive_wmape_vol": round(float(fixed_naive["global_volume_weighted_WMAPE"]), 4),
                "naive_wmape_med": round(float(fixed_naive["median_per_series_WMAPE"]), 4),
                "relative_error_reduction_pct": round((float(fixed_naive["global_volume_weighted_WMAPE"]) - fixed_hybrid) / float(fixed_naive["global_volume_weighted_WMAPE"]) * 100, 2),
            },
            "conclusion": {
                "zh": f"主结果采用每月更新的下月预测：滚动单月XGBoost为{rolling_primary['wmape_vol']:.2f}% WMAPE，较沿用上月销量的朴素基准{rolling_naive['wmape_vol']:.2f}%低{rolling_naive['wmape_vol']-rolling_primary['wmape_vol']:.2f}个百分点。固定六个月综合方案为{fixed_hybrid:.2f}%，作为压力测试单独报告。",
                "en": f"The headline task refreshes each month for a one-month-ahead forecast: rolling one-month XGBoost reaches {rolling_primary['wmape_vol']:.2f}% WMAPE, {rolling_naive['wmape_vol']-rolling_primary['wmape_vol']:.2f} pp below the last-observed-value naive baseline at {rolling_naive['wmape_vol']:.2f}%. The fixed six-month combined method scores {fixed_hybrid:.2f}% and is reported separately as a stress test.",
            },
            "feature_insight": {
                "zh": "特征贡献图来自固定六个月压力测试中的口碑增强模型；滚动主结果选择的是销量基线，避免把固定场景的口碑特征贡献误读成滚动主结果的因果证据。",
                "en": "The feature-contribution chart comes from the fixed six-month stress-test review model; the rolling headline selects the sales baseline, so the fixed-scenario review features are not presented as causal evidence for the rolling result.",
            },
        }

    def absa(self) -> dict[str, Any]:
        summary = _read(self.processed / "user_feedback" / "user_need_aspect_summary.csv").set_index("aspect")
        reviews = _read(self.reviews / "review_aspect_labels.csv", parse_dates=["publish_time"])
        monitoring = _read_json(self.processed / "user_feedback" / "user_needs_alerts_summary.json")
        cutoff = pd.Timestamp(monitoring["latest_completed_monitoring_month"]) + pd.Timedelta(days=1)
        reviews = reviews.loc[reviews["publish_time"].between(pd.Timestamp("2022-01-01"), cutoff, inclusive="left")].copy()
        reviews["period"] = reviews["publish_time"].dt.to_period("M").astype(str)
        aspect_payload, distribution, variance = [], [], []
        monthly: dict[str, pd.Series] = {}
        for index, aspect in enumerate(ASPECTS):
            mentioned = reviews[f"uniform_local_{aspect}_mentioned"].eq(1)
            values = pd.to_numeric(reviews[f"review_{aspect}_raw_polarity"], errors="coerce").where(mentioned)
            valid = values.isin([-1, 0, 1])
            values = values.where(valid)
            row = summary.loc[aspect]
            aspect_payload.append({
                "key": aspect, "name_zh": ASPECT_ZH[aspect], "name_en": ASPECT_EN[aspect],
                "avg": round(float(row["mean_polarity"]), 3), "color": PALETTE[index % len(PALETTE)],
                "mention_rate": round(float(row["mention_rate"]), 3),
            })
            scored = int(row["scored_mentions"])
            distribution.append({
                "key": aspect, "name_zh": ASPECT_ZH[aspect], "name_en": ASPECT_EN[aspect],
                "positive": round(float(row["positive_mentions"] / scored), 3),
                "neutral": round(float(row["neutral_mentions"] / scored), 3),
                "negative": round(float(row["negative_mentions"] / scored), 3),
            })
            variance.append({
                "key": aspect, "name_zh": ASPECT_ZH[aspect], "name_en": ASPECT_EN[aspect],
                "std": round(float(values.std()), 3),
            })
            monthly[aspect] = reviews.assign(_score=values).groupby("period")["_score"].mean()
        months = sorted(set().union(*[set(series.index) for series in monthly.values()]))
        return {
            "meta": {"review_rows": int(summary.iloc[0]["eligible_reviews"]), "review_series": 345, "aspects": len(ASPECTS)},
            "aspects": aspect_payload,
            "radar": {"indicators": [], "values": []},
            "distribution": distribution,
            "variance": variance,
            "monthly_trends": {
                "months": months,
                "series": [{
                    "key": aspect, "name_zh": ASPECT_ZH[aspect], "name_en": ASPECT_EN[aspect],
                    "data": [round(float(monthly[aspect].get(month)), 3) if pd.notna(monthly[aspect].get(month)) else None for month in months],
                } for aspect in ASPECTS],
            },
            "conclusion": {
                "zh": "24,175条评论显示：空间、能耗和动力最常被讨论；智能化负面率47.9%，舒适性负面率37.4%，是最集中的改进需求。",
                "en": "Across 24,175 reviews, space, energy use, and power are discussed most; intelligence has a 47.9% negative rate and comfort 37.4%, making them the clearest improvement needs.",
            },
        }

    def attribution(self) -> dict[str, Any]:
        ablation = _read(self.processed / "product" / "config_attribution_ablation.csv")
        baselines = _read(self.processed / "product" / "config_attribution_baselines.csv")
        importance = _read(self.processed / "product" / "config_importance_annual.csv")
        features = []
        for index, (_, row) in enumerate(importance.loc[importance["block"].eq("config")].head(15).iterrows()):
            zh, en = _feature_label(str(row["feature"]))
            features.append({
                "key": str(row["feature"]), "name_zh": zh, "name_en": en,
                "importance": round(float(row["gain"]), 4), "color": PALETTE[index % len(PALETTE)],
            })
        models = [{
            "variant": str(row["variant"]),
            "r2": round(float(row["R2_log_mean"]), 3),
            "r2_std": round(float(row["R2_log_std"]), 3),
            "wmape": round(float(row.get("WMAPE_oof_global", row["WMAPE_mean"])), 2),
            "wmape_fold_mean": round(float(row["WMAPE_mean"]), 2),
            "n_features": int(row["n_features"]),
        } for _, row in ablation.iterrows()]
        block = importance.groupby("block")["gain"].sum()
        block = block / block.sum()
        labels = {"config": ("配置", "Configuration"), "brand": ("品牌", "Brand"), "year": ("年份", "Year")}
        blocks = [{
            "block": key, "name_zh": labels[key][0], "name_en": labels[key][1],
            "share": round(float(value), 4),
        } for key, value in block.sort_values(ascending=False).items()]
        brand_r2 = float(ablation.loc[ablation["variant"].eq("+BRAND"), "R2_log_mean"].iloc[0])
        config_r2 = float(ablation.loc[ablation["variant"].eq("+CONFIG"), "R2_log_mean"].iloc[0])
        wmape_rows = {
            str(row["variant"]): round(float(row.get("WMAPE_oof_global", row["WMAPE_mean"])), 2)
            for _, row in ablation.iterrows()
        }
        return {
            "shap": features, "models": models, "blocks": blocks,
            "meta": {"series": 736, "series_year_rows": 2007, "cv_folds": 5, "wmape_is_secondary": True,
                     "wmape_note_zh": "年度截面补充误差，不与月度销量预测WMAPE直接比较。",
                     "wmape_note_en": "Supporting annual cross-sectional error; not directly comparable with monthly sales-forecast WMAPE."},
            "wmape_baselines": {
                str(row["method"]): round(float(row["WMAPE_mean"]), 2)
                for _, row in baselines.iterrows()
            },
            "wmape_by_variant": wmape_rows,
            "comparison": {"with": None, "without": None}, "top_example": None,
            "conclusion": {
                "zh": f"736车系、2,007条车系×年记录的分组交叉验证中，加入配置后R²由{brand_r2:.3f}提升至{config_r2:.3f}（+{config_r2-brand_r2:.3f}）；配置解释车系之间差异，不解释同车系短期涨跌。完整模型年度截面WMAPE为{wmape_rows.get('+CONFIG', np.nan):.2f}%，仅作补充误差，不能与月度销量预测直接比较。",
                "en": f"Across 736 series and 2,007 series-year rows, grouped CV R² rises from {brand_r2:.3f} to {config_r2:.3f} (+{config_r2-brand_r2:.3f}) after adding configuration. Configuration explains between-series differences, not short-term within-series changes. The full model's annual cross-sectional WMAPE is {wmape_rows.get('+CONFIG', np.nan):.2f}% and is supporting evidence, not directly comparable with monthly forecasting.",
            },
        }

    def forecast_evidence(self) -> dict[str, Any]:
        aligned = self.aligned()
        bootstrap = _read(self.processed / "forecast" / "forecast_robustness_bootstrap.csv")
        robustness = _read_json(self.processed / "forecast" / "forecast_robustness_summary.json")
        comparison = bootstrap.loc[
            bootstrap["comparator"].eq("BASE")
            & bootstrap["candidate"].isin(["PLATFORM_RATING_FIXED", "LOCAL_LEXICON_FIXED", "REVIEW_TEXT_FIXED", "REVIEW_RICH_FIXED"])
        ].sort_values("bootstrap_probability_candidate_better", ascending=False)
        evidence = {"aspects": [], "sig_rates": []}
        for _, row in comparison.iterrows():
            zh, en = MODEL_LABELS[str(row["candidate"])]
            evidence["aspects"].append({"zh": zh, "en": en})
            evidence["sig_rates"].append(round(float(row["bootstrap_probability_candidate_better"]), 4))
        model_summary = _read(self.processed / "forecast" / "review_feature_ablation_summary.csv")
        selected_versions = ["BASE", "PLATFORM_RATING_FIXED", "REVIEW_TEXT_FIXED", "REVIEW_RICH_FIXED"]
        fusion = []
        for version in selected_versions:
            row = model_summary.loc[model_summary["version"].eq(version)].iloc[0]
            zh, en = MODEL_LABELS[version]
            fusion.append({"version": zh, "version_en": en, "wmape_vol": round(float(row["global_volume_weighted_WMAPE"]), 3)})
        correlations = []
        for aspect in ASPECTS:
            part = aligned.loc[aligned[aspect].notna() & aligned["monthly_sales"].gt(0)]
            corr = part[aspect].corr(part["monthly_sales"]) if len(part) > 2 else np.nan
            correlations.append({
                "key": aspect, "name_zh": ASPECT_ZH[aspect], "name_en": ASPECT_EN[aspect],
                "corr": round(float(corr), 3) if pd.notna(corr) else 0.0,
            })
        market_rows = []
        for period, group in aligned.groupby("period", sort=True):
            market_rows.append({
                "period": period,
                "sales": float(group["monthly_sales"].sum()),
                "sentiment": _weighted_mean(group["overall"], group["review_count_180d"]),
            })
        return {
            "scenario": "fixed_origin_stress",
            "granger": evidence,
            "fusion": fusion,
            "correlation": correlations,
            "timeseries": {
                "brand": "全市场", "brand_en": "Market",
                "months": [row["period"] for row in market_rows],
                "sales": [round(row["sales"], 1) for row in market_rows],
                "sentiment": [round(row["sentiment"], 3) if pd.notna(row["sentiment"]) else None for row in market_rows],
            },
            "conclusion": {
                "zh": f"固定六个月压力测试中，选定的口碑增强相对销量基线改善{robustness['selected_feedback_vs_base_improvement_pp']:.3f}个百分点；5,000次车系聚类Bootstrap胜出概率{robustness['selected_feedback_vs_base_probability_better']:.2%}，95%区间为{robustness['selected_feedback_vs_base_bootstrap_95pct_ci_pp'][0]:.2f}至{robustness['selected_feedback_vs_base_bootstrap_95pct_ci_pp'][1]:.2f}个百分点，因此只支持小幅补充信号。",
                "en": f"In the fixed six-month stress test, the selected review enhancement improves on the sales baseline by {robustness['selected_feedback_vs_base_improvement_pp']:.3f} pp; it wins {robustness['selected_feedback_vs_base_probability_better']:.2%} of 5,000 series-cluster bootstrap samples, with a 95% interval from {robustness['selected_feedback_vs_base_bootstrap_95pct_ci_pp'][0]:.2f} to {robustness['selected_feedback_vs_base_bootstrap_95pct_ci_pp'][1]:.2f} pp. This supports only a modest supporting signal.",
            },
        }

    def alerts(self) -> dict[str, Any]:
        alerts = _read(self.processed / "user_feedback" / "sentiment_alerts.csv", parse_dates=["information_cutoff_inclusive"])
        summary = _read_json(self.processed / "user_feedback" / "user_needs_alerts_summary.json")
        risk_labels = {"high": "高危", "medium": "中危", "low": "低危"}
        rows = []
        for _, row in alerts.sort_values(["information_cutoff_inclusive", "score_change"], ascending=[False, True]).iterrows():
            brand = str(row["brand"])
            rows.append({
                "series_name": str(row["series_name"]), "brand": brand,
                "brand_en": self.brand_en(brand), "series_en": self.series_en(str(row["series_name"]), brand),
                "period": row["information_cutoff_inclusive"].strftime("%Y-%m"),
                "overall": round(float(row["current_overall_score"]), 3),
                "overall_drop": round(float(row["score_change"]), 3),
                "risk": risk_labels[str(row["risk_level"])],
                "risk_en": str(row["risk_level"]).title(),
                "current_reviews": int(row["current_reviews"]),
                "previous_reviews": int(row["previous_reviews"]),
                "worst_aspect": str(row["worst_aspect_zh"]),
                "worst_aspect_en": ASPECT_EN.get(str(row["worst_aspect"]), str(row["worst_aspect"])),
            })
        monthly = alerts.groupby(alerts["information_cutoff_inclusive"].dt.to_period("M")).size().sort_index()
        risk = alerts["risk_level"].map(risk_labels).value_counts()
        return {
            "meta": {
                "active_alerts": int(summary["latest_active_alerts"]),
                "historical_alerts": int(summary["historical_alert_events"]),
                "eligible_series": int(summary["latest_eligible_series"]),
                "latest_month": str(summary["latest_completed_monitoring_month"])[:7],
            },
            "rule": {
                "zh": "相邻180天窗口各≥5条评论；当前综合评价≤−0.10、较前窗下降≥0.15，且负面评论率≥35%",
                "en": "Two adjacent 180-day windows each have ≥5 reviews; current score ≤−0.10, drop ≥0.15, and negative-review rate ≥35%",
            },
            "alerts": rows,
            "monthly": [{"month": str(period), "count": int(count)} for period, count in monthly.items()],
            "risk_dist": [{"level": str(level), "count": int(count)} for level, count in risk.items()],
            "conclusion": {
                "zh": f"样本量约束后共有{summary['historical_alert_events']}次历史预警；截至{summary['latest_completed_monitoring_month'][:7]}，123个车系具备判定资格，当前有效预警{summary['latest_active_alerts']}条。预警是人工复核入口，不是故障定论。",
                "en": f"After sample-size controls, {summary['historical_alert_events']} historical events remain. As of {summary['latest_completed_monitoring_month'][:7]}, 123 series are eligible and {summary['latest_active_alerts']} alert is active. Alerts are review candidates, not fault verdicts.",
            },
        }

    def drilldown(self) -> dict[str, Any]:
        aligned = self.aligned()
        valid = aligned.groupby("series_name")["overall"].apply(lambda values: values.notna().sum() >= 3)
        aligned = aligned.loc[aligned["series_name"].isin(valid.loc[valid].index)].copy()
        out: dict[str, Any] = {"series": [], "brands": [], "data": {}, "brand_en": {}}
        metadata = aligned.drop_duplicates("series_name")[["series_name", "brand"]].sort_values(["brand", "series_name"])
        for _, row in metadata.iterrows():
            name, brand = str(row["series_name"]), str(row["brand"])
            out["series"].append({
                "id": name, "name": name, "brand": brand,
                "brand_en": self.brand_en(brand), "series_en": self.series_en(name, brand),
            })
            out["brand_en"][brand] = self.brand_en(brand)
        out["brands"] = sorted(metadata["brand"].astype(str).unique().tolist())
        for name, group in aligned.groupby("series_name", sort=True):
            group = group.sort_values("date")
            out["data"][str(name)] = {
                "months": group["period"].tolist(),
                "sales": [round(float(value), 1) for value in group["monthly_sales"]],
                "sentiment": [round(float(value), 3) if pd.notna(value) else None for value in group["overall"]],
                "aspects": {
                    aspect: [round(float(value), 3) if pd.notna(value) else None for value in group[aspect]]
                    for aspect in ASPECTS
                },
            }
        return out

    def brand_drilldown(self) -> dict[str, Any]:
        aligned = self.aligned()
        aligned = aligned.loc[aligned["overall"].notna()].copy()
        out: dict[str, Any] = {"brands": [], "brand_en": {}, "market_radar": {}, "data": {}}
        weights = aligned["review_count_180d"]
        out["market_radar"] = {
            aspect: round(_weighted_mean(aligned[aspect], weights), 3) for aspect in ASPECTS
        }
        for brand, brand_rows in aligned.groupby("brand", sort=True):
            monthly_rows = []
            for period, group in brand_rows.groupby("period", sort=True):
                monthly_rows.append({
                    "period": period,
                    "sales": float(group["monthly_sales"].sum()),
                    "sentiment": _weighted_mean(group["overall"], group["review_count_180d"]),
                })
            radar = {
                aspect: round(_weighted_mean(brand_rows[aspect], brand_rows["review_count_180d"]), 3)
                for aspect in ASPECTS
            }
            ranks = []
            for series_name, group in brand_rows.groupby("series_name"):
                active = group.loc[group["monthly_sales"].gt(0)]
                if active.empty:
                    continue
                ranks.append({
                    "id": str(series_name), "name": str(series_name),
                    "series_en": self.series_en(str(series_name), str(brand)),
                    "avg_monthly": round(float(active["monthly_sales"].mean()), 1),
                    "avg_sent": round(_weighted_mean(group["overall"], group["review_count_180d"]), 3),
                })
            ranks.sort(key=lambda row: row["avg_monthly"], reverse=True)
            out["data"][str(brand)] = {
                "months": [row["period"] for row in monthly_rows],
                "sales": [round(row["sales"], 1) for row in monthly_rows],
                "sentiment": [round(row["sentiment"], 3) if pd.notna(row["sentiment"]) else None for row in monthly_rows],
                "radar": radar, "series_rank": ranks[:15],
            }
            out["brand_en"][str(brand)] = self.brand_en(str(brand))
        out["brands"] = sorted(out["data"])
        return out

    def payloads(self) -> dict[str, dict[str, Any]]:
        return {
            "overview.json": self.overview(),
            "forecast.json": self.forecast(),
            "absa.json": self.absa(),
            "attribution.json": self.attribution(),
            "forecast_evidence.json": self.forecast_evidence(),
            "alerts.json": self.alerts(),
            "drilldown.json": self.drilldown(),
            "brand_drilldown.json": self.brand_drilldown(),
        }
