#!/usr/bin/env python3
"""New-scope dashboard payloads built from the 371/736-series pipeline.

The existing static pages and JSON filenames are preserved.  This module only
replaces their data sources and conclusions; it does not create a new app.
"""
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
    "BASE": ("销量基线", "Sales baseline"),
    "PLATFORM_RATING_FIXED": ("平台评分", "Platform ratings"),
    "LOCAL_LEXICON_FIXED": ("本地词典", "Local lexicon"),
    "DEEPSEEK_CORE_FIXED": ("文本情感特征", "Text sentiment"),
    "DEEPSEEK_RICH_FIXED": ("用户口碑增强", "User-review enhanced"),
    "ALL_SENTIMENT_FIXED": ("全部口碑特征", "Combined review features"),
    "DEEPSEEK_CORE_ROLLING": ("滚动口碑特征", "Rolling review signals"),
    "DEEPSEEK_RICH_FIXED_COLD_HYBRID": ("口碑增强＋冷启动", "Review enhanced + cold-start"),
}
FEATURE_FAMILY_LABELS = {
    "sales_lag_roll": ("历史销量", "Sales history"),
    "calendar": ("日历", "Calendar"),
    "configuration": ("产品配置", "Product configuration"),
    "deepseek_observation_context": ("评论覆盖", "Review coverage"),
    "deepseek_expanding_score": ("历史评价", "Historical review score"),
    "deepseek_recent_score": ("近期评价", "Recent review score"),
    "deepseek_positive_rate": ("正面比例", "Positive share"),
    "deepseek_negative_rate": ("负面比例", "Negative share"),
    "deepseek_mention_count": ("需求提及量", "Need mentions"),
    "deepseek_mention_rate": ("需求提及率", "Mention share"),
    "deepseek_overall": ("评论特征", "Review features"),
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


def _feature_label(feature: str) -> tuple[str, str]:
    simple = {
        "lag_1": ("上月销量", "Sales lag 1"), "lag_2": ("前2月销量", "Sales lag 2"),
        "lag_3": ("前3月销量", "Sales lag 3"), "roll_mean_3": ("近3月销量均值", "3-month sales mean"),
        "roll_mean_6": ("近6月销量均值", "6-month sales mean"), "month_sin": ("月份季节性（正弦）", "Month seasonality (sin)"),
        "month_cos": ("月份季节性（余弦）", "Month seasonality (cos)"), "year": ("年份", "Year"),
        "deepseek_review_count_prior_all": ("历史评论数量", "Prior review count"),
        "deepseek_review_count_180d": ("近180天评论数量", "180-day review count"),
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
        if f"deepseek_{aspect}_" in feature:
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


class DashboardV2:
    def __init__(self, root: str | Path, brand_en: Callable[[str], str], series_en: Callable[[str, str | None], str]):
        self.root = Path(root)
        self.new = self.root / "data" / "processed_new"
        self.sentiment = self.root / "data" / "sentiment_new" / "processed"
        self.brand_en = brand_en
        self.series_en = series_en
        self._panel: pd.DataFrame | None = None
        self._aligned: pd.DataFrame | None = None

    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            parts = [
                _read(self.new / "splits" / f"{name}.csv", parse_dates=["date"])
                for name in ("train", "val", "test")
            ]
            self._panel = pd.concat(parts, ignore_index=True).sort_values(["series_name", "date"])
        return self._panel.copy()

    def aligned(self) -> pd.DataFrame:
        if self._aligned is None:
            panel = self.panel()
            sentiment = _read(self.sentiment / "deepseek_features_by_series_month_rolling.csv", parse_dates=["date"])
            columns = ["series_name", "date", "deepseek_review_count_180d", *[
                f"deepseek_{aspect}_score_180d_mean" for aspect in ASPECTS
            ]]
            aligned = panel.merge(sentiment[columns], on=["series_name", "date"], how="left", validate="one_to_one")
            aligned = aligned.rename(columns={
                f"deepseek_{aspect}_score_180d_mean": aspect for aspect in ASPECTS
            })
            aligned["overall"] = aligned[ASPECTS].mean(axis=1, skipna=True)
            aligned["period"] = aligned["date"].dt.to_period("M").astype(str)
            self._aligned = aligned
        return self._aligned.copy()

    def overview(self) -> dict[str, Any]:
        panel = self.panel()
        needs = _read_json(self.new / "stage5" / "user_needs_alerts_summary.json")
        cold = _read_json(self.new / "stage3" / "cold_start_launch_curve_summary.json")
        model_summary = _read(self.new / "stage3" / "xgb_deepseek_full371_summary.csv")
        baseline = float(model_summary.loc[model_summary["version"].eq("BASE"), "global_volume_weighted_WMAPE"].iloc[0])
        rich = float(model_summary.loc[model_summary["version"].eq("DEEPSEEK_RICH_FIXED"), "global_volume_weighted_WMAPE"].iloc[0])
        hybrid = float(cold["hybrid_full371_global_WMAPE"])
        monthly = panel.groupby("date")["monthly_sales"].sum().sort_index()
        return {
            "kpis": {
                "coverage_series": int(needs["review_series"]),
                "forecast_wmape": round(hybrid, 2),
                "core_forecast_wmape": round(rich, 2),
                "baseline_wmape": round(baseline, 2),
                "wmape_gain_pp": round(baseline - hybrid, 2),
                "wmape_relative_gain_pct": round((baseline - hybrid) / baseline * 100, 2),
                "forecast_horizon": 6,
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
                {"zh": "371车系六个月递归预测：冷启动混合模型全局WMAPE为38.64%", "en": "371-series six-month recursive forecast: cold-start hybrid global WMAPE is 38.64%"},
                {"zh": "用户口碑增强相对销量基线改善1.735个百分点，但Bootstrap区间跨0", "en": "User-review features improve 1.735 pp over the sales baseline, but the bootstrap interval crosses zero"},
                {"zh": "736车系年度归因中，配置将交叉验证R²从0.154提升到0.303", "en": "Across 736 series, configuration raises annual-attribution CV R² from 0.154 to 0.303"},
                {"zh": "智能化与舒适性是负面反馈最集中的两个用户需求维度", "en": "Intelligence and comfort carry the highest complaint concentration"},
                {"zh": f"截至{needs['latest_completed_monitoring_month'][:7]}，当前有效预警{needs['latest_active_alerts']}条", "en": f"As of {needs['latest_completed_monitoring_month'][:7]}, {needs['latest_active_alerts']} active alert is detected"},
            ],
        }

    def forecast(self) -> dict[str, Any]:
        summary = _read(self.new / "stage3" / "xgb_deepseek_full371_summary.csv")
        shap = _read(self.new / "stage3" / "xgb_deepseek_full371_shap_importance.csv")
        hybrid = _read(self.new / "stage3" / "xgb_deepseek_cold_hybrid_preds.csv", parse_dates=["date"])
        cold = _read_json(self.new / "stage3" / "cold_start_launch_curve_summary.json")
        models: list[dict[str, Any]] = []
        for _, row in summary.sort_values("global_volume_weighted_WMAPE").iterrows():
            zh, en = MODEL_LABELS.get(row["version"], (row["version"], row["version"]))
            models.append({
                "name": zh, "name_zh": zh, "name_en": en,
                "wmape_vol": round(float(row["global_volume_weighted_WMAPE"]), 4),
                "wmape_med": round(float(row["median_per_series_WMAPE"]), 4),
                "mae": None, "color": PALETTE[len(models) % len(PALETTE)],
                "scenario": str(row["scenario"]),
            })
        zh, en = MODEL_LABELS["DEEPSEEK_RICH_FIXED_COLD_HYBRID"]
        models.append({
            "name": zh, "name_zh": zh, "name_en": en,
            "wmape_vol": round(float(cold["hybrid_full371_global_WMAPE"]), 4),
            "wmape_med": round(float(cold["hybrid_full371_median_per_series_WMAPE"]), 4),
            "mae": None, "color": "#34c38f", "scenario": "fixed_origin_primary_cold_hybrid",
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
        valid = hybrid.loc[hybrid["actual"].gt(0) & hybrid["pred"].notna()].copy()
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
        return {
            "models": models,
            "best_model": zh,
            "meta": {
                "eval_series": 371,
                "test_months": 6,
                "test_rows": int(len(valid)),
                "train_end": "2025-06",
                "validation_period": "2025-07~2025-12",
                "test_period": "2026-01~2026-06",
                "cold_start_series": int(cold["cold_start_series"]),
            },
            "class_wmape": class_rows,
            "scatter": [[round(float(a), 1), round(float(p), 1)] for a, p in zip(valid["actual"], valid["pred"])],
            "features": features,
            "conclusion": {
                "zh": "371车系固定起点六个月递归测试中，用户口碑增强将全局WMAPE从40.44%降至38.71%；冷启动兜底后为38.64%。平台评分与用户口碑增强仅相差0.113个百分点。",
                "en": "In the fixed-origin six-month test over 371 series, user-review features lower global WMAPE from 40.44% to 38.71%; the cold-start fallback reaches 38.64%. Platform ratings differ from the user-review model by only 0.113 pp.",
            },
            "feature_insight": {
                "zh": "销量滞后与滚动均值占SHAP绝对重要性的80.17%；评论特征合计约9.03%，主要提供补充信息而非替代历史销量。",
                "en": "Sales lags and rolling means contribute 80.17% of absolute SHAP importance; review features contribute about 9.03% and supplement rather than replace sales history.",
            },
        }

    def absa(self) -> dict[str, Any]:
        summary = _read(self.new / "stage5" / "user_need_aspect_summary.csv").set_index("aspect")
        reviews = _read(self.sentiment / "unified_deepseek_absa_review_features.csv", parse_dates=["publish_time"])
        monitoring = _read_json(self.new / "stage5" / "user_needs_alerts_summary.json")
        cutoff = pd.Timestamp(monitoring["latest_completed_monitoring_month"]) + pd.Timedelta(days=1)
        reviews = reviews.loc[reviews["publish_time"].between(pd.Timestamp("2022-01-01"), cutoff, inclusive="left")].copy()
        reviews["period"] = reviews["publish_time"].dt.to_period("M").astype(str)
        aspect_payload, distribution, variance = [], [], []
        monthly: dict[str, pd.Series] = {}
        for index, aspect in enumerate(ASPECTS):
            mentioned = reviews[f"uniform_local_{aspect}_mentioned"].eq(1)
            values = pd.to_numeric(reviews[f"deepseek_{aspect}_raw_polarity"], errors="coerce").where(mentioned)
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
        ablation = _read(self.new / "stage4" / "config_attribution_ablation.csv")
        importance = _read(self.new / "stage4" / "config_importance_annual.csv")
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
            "wmape": round(float(row["WMAPE_mean"]), 2),
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
        return {
            "shap": features, "models": models, "blocks": blocks,
            "meta": {"series": 736, "series_year_rows": 2007, "cv_folds": 5},
            "comparison": {"with": None, "without": None}, "top_example": None,
            "conclusion": {
                "zh": f"736车系、2,007条车系×年记录的分组交叉验证中，加入配置后R²由{brand_r2:.3f}提升至{config_r2:.3f}（+{config_r2-brand_r2:.3f}）；配置解释车系之间差异，不解释同车系短期涨跌。",
                "en": f"Across 736 series and 2,007 series-year rows, grouped CV R² rises from {brand_r2:.3f} to {config_r2:.3f} (+{config_r2-brand_r2:.3f}) after adding configuration. Configuration explains between-series differences, not short-term within-series changes.",
            },
        }

    def forecast_evidence(self) -> dict[str, Any]:
        aligned = self.aligned()
        bootstrap = _read(self.new / "stage3" / "xgb_deepseek_full371_bootstrap.csv")
        comparison = bootstrap.loc[
            bootstrap["comparator"].eq("BASE")
            & bootstrap["candidate"].isin(["PLATFORM_RATING_FIXED", "LOCAL_LEXICON_FIXED", "DEEPSEEK_CORE_FIXED", "DEEPSEEK_RICH_FIXED"])
        ].sort_values("bootstrap_probability_candidate_better", ascending=False)
        evidence = {"aspects": [], "sig_rates": []}
        for _, row in comparison.iterrows():
            zh, en = MODEL_LABELS[str(row["candidate"])]
            evidence["aspects"].append({"zh": zh, "en": en})
            evidence["sig_rates"].append(round(float(row["bootstrap_probability_candidate_better"]), 4))
        model_summary = _read(self.new / "stage3" / "xgb_deepseek_full371_summary.csv")
        selected_versions = ["BASE", "PLATFORM_RATING_FIXED", "DEEPSEEK_CORE_FIXED", "DEEPSEEK_RICH_FIXED"]
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
                "sentiment": _weighted_mean(group["overall"], group["deepseek_review_count_180d"]),
            })
        return {
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
                "zh": "用户口碑增强相对销量基线改善1.735个百分点，在5,000次车系聚类Bootstrap中胜出概率89.84%，但95%区间跨0；平台评分与用户口碑增强基本无法区分。",
                "en": "User-review features improve 1.735 pp over the sales baseline and win in 89.84% of 5,000 series-cluster bootstrap samples, but the 95% interval crosses zero; platform ratings and the user-review model are effectively indistinguishable.",
            },
        }

    def alerts(self) -> dict[str, Any]:
        alerts = _read(self.new / "stage5" / "sentiment_alerts.csv", parse_dates=["information_cutoff_inclusive"])
        summary = _read_json(self.new / "stage5" / "user_needs_alerts_summary.json")
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
        weights = aligned["deepseek_review_count_180d"]
        out["market_radar"] = {
            aspect: round(_weighted_mean(aligned[aspect], weights), 3) for aspect in ASPECTS
        }
        for brand, brand_rows in aligned.groupby("brand", sort=True):
            monthly_rows = []
            for period, group in brand_rows.groupby("period", sort=True):
                monthly_rows.append({
                    "period": period,
                    "sales": float(group["monthly_sales"].sum()),
                    "sentiment": _weighted_mean(group["overall"], group["deepseek_review_count_180d"]),
                })
            radar = {
                aspect: round(_weighted_mean(brand_rows[aspect], brand_rows["deepseek_review_count_180d"]), 3)
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
                    "avg_sent": round(_weighted_mean(group["overall"], group["deepseek_review_count_180d"]), 3),
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
