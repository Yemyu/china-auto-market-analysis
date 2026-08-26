#!/usr/bin/env python3
"""Build platform-rating and lexicon features for eligible reviews."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "reviews" / "processed"
CORPUS = OUT / "target_371_review_corpus.csv"

REVIEW_FEATURES = OUT / "local_sentiment_review_features.csv"
ASPECT_COVERAGE = OUT / "local_sentiment_aspect_coverage.csv"
SUMMARY = OUT / "local_sentiment_feature_summary.json"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]

RATING_COLUMNS = {
    "appearance": "rating_appearance",
    "interior": "rating_interiors",
    "space": "rating_space",
    "power": "rating_power",
    "control": "rating_control",
    "comfort": "rating_comfort",
    "fuel_consumption": "rating_oil_consumption",
    "configuration": "rating_config",
    "intelligence": None,
    "value": None,
}

ASPECT_TERMS = {
    "appearance": ["外观", "外形", "颜值", "造型", "前脸", "车尾", "尾灯", "大灯", "车灯", "轮毂", "车漆", "设计"],
    "interior": ["内饰", "中控", "座舱", "仪表", "仪表盘", "用料", "氛围灯", "内饰板", "车内"],
    "space": ["空间", "后排", "前排", "第三排", "后备箱", "储物", "头部", "腿部", "乘坐", "坐满"],
    "power": ["动力", "加速", "提速", "超车", "发动机", "马力", "扭矩", "起步", "变速箱", "换挡"],
    "control": ["操控", "转向", "方向盘", "底盘", "悬挂", "过弯", "刹车", "制动", "变道", "指向"],
    "comfort": ["舒适", "座椅", "隔音", "噪音", "胎噪", "风噪", "减震", "颠簸", "静谧", "空调", "异响", "气味"],
    "fuel_consumption": ["油耗", "能耗", "电耗", "续航", "费油", "省油", "加油", "充电", "亏电", "用车成本"],
    "configuration": ["配置", "功能", "天窗", "雷达", "影像", "座椅加热", "座椅通风", "按键", "充电口", "车门", "安全气囊"],
    "intelligence": ["智能", "车机", "导航", "语音", "ota", "辅助驾驶", "智驾", "自动泊车", "流量", "芯片", "卡顿"],
    "value": ["性价比", "价格", "价位", "优惠", "落地", "购车", "费用", "保值", "划算", "便宜", "贵", "值不值"],
}

POSITIVE_TERMS = [
    "非常满意", "满意", "喜欢", "漂亮", "好看", "大气", "精致", "舒服", "舒适", "宽敞", "充足", "强劲",
    "省油", "顺畅", "平顺", "稳定", "灵活", "扎实", "安静", "静谧", "实用", "丰富", "灵敏", "流畅",
    "划算", "便宜", "优惠", "给力", "无压力", "不错", "优秀", "很好", "足够", "可靠", "方便", "惊喜",
]
NEGATIVE_TERMS = [
    "最不满意", "不满意", "不喜欢", "难看", "粗糙", "异味", "狭窄", "拥挤", "不够", "肉", "顿挫", "无力",
    "费油", "耗油", "座椅硬", "颠簸", "噪音", "胎噪", "风噪", "卡顿", "死机", "不灵敏", "简陋", "缺少",
    "失望", "不值", "故障", "问题", "异响", "刺鼻", "不方便", "不好用", "漏水", "虚标", "延迟",
]
NEGATION_RE = re.compile(r"(?:不|没|无|没有|未|太)$")
NEGATION_EXCEPTIONS = {"不错", "不小", "不少", "不贵", "不差", "无压力"}
POSITIVE_SECTION_RE = re.compile(r"最满意|满意之处|优点|喜欢的地方")
NEGATIVE_SECTION_RE = re.compile(r"最不满意|不满意|缺点|不足|问题|吐槽|遗憾")
SPLIT_RE = re.compile(r"[\r\n。！？!?；;]+")


def as_bool(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def valid_rating(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.between(0.5, 5.0))


def count_term(text: str, term: str) -> int:
    return text.count(term)


def cue_counts(text: str, section_polarity: int) -> tuple[int, int]:
    """Count local polarity cues with a small negation check."""
    positive = 0
    negative = 0
    for term in POSITIVE_TERMS:
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            prefix = text[max(0, index - 3):index]
            if term not in NEGATION_EXCEPTIONS and NEGATION_RE.search(prefix):
                negative += 1
            else:
                positive += 1
            start = index + len(term)
    for term in NEGATIVE_TERMS:
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            prefix = text[max(0, index - 3):index]
            if NEGATION_RE.search(prefix):
                positive += 1
            else:
                negative += 1
            start = index + len(term)
    if not positive and not negative and section_polarity:
        positive = int(section_polarity > 0)
        negative = int(section_polarity < 0)
    return positive, negative


def text_segments(content: str) -> list[tuple[str, int]]:
    """Split review text while carrying explicit satisfied/dissatisfied headings."""
    polarity = 0
    output: list[tuple[str, int]] = []
    for segment in SPLIT_RE.split(str(content)):
        text = segment.strip().lower()
        if not text:
            continue
        if NEGATIVE_SECTION_RE.search(text):
            polarity = -1
        elif POSITIVE_SECTION_RE.search(text):
            polarity = 1
        output.append((text, polarity))
    return output


def score_text(content: str) -> dict[str, int | float]:
    segments = text_segments(content)
    result: dict[str, int | float] = {}
    global_positive = 0
    global_negative = 0
    for text, section_polarity in segments:
        positive, negative = cue_counts(text, section_polarity)
        global_positive += positive
        global_negative += negative
    result["text_global_positive_cues"] = global_positive
    result["text_global_negative_cues"] = global_negative
    result["text_global_polarity"] = int(np.sign(global_positive - global_negative))

    for aspect, terms in ASPECT_TERMS.items():
        positive = 0
        negative = 0
        mentioned = False
        for text, section_polarity in segments:
            if not any(count_term(text, term) for term in terms):
                continue
            mentioned = True
            pos_count, neg_count = cue_counts(text, section_polarity)
            positive += pos_count
            negative += neg_count
        result[f"text_{aspect}_mentioned"] = int(mentioned)
        result[f"text_{aspect}_positive_cues"] = positive
        result[f"text_{aspect}_negative_cues"] = negative
        result[f"text_{aspect}_polarity"] = (
            int(np.sign(positive - negative)) if mentioned else np.nan
        )
    return result


def platform_rating_features(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    overall = valid_rating(data["rating_overall"])
    output["platform_rating_overall"] = overall
    output["platform_rating_overall_sentiment"] = ((overall - 3.0) / 2.0).clip(-1.0, 1.0)
    output["platform_rating_overall_polarity"] = np.select(
        [overall.lt(3.5), overall.ge(4.5)], [-1, 1], default=0
    ).astype(float)
    output.loc[overall.isna(), "platform_rating_overall_polarity"] = np.nan
    for aspect, column in RATING_COLUMNS.items():
        rating = valid_rating(data[column]) if column else pd.Series(np.nan, index=data.index)
        output[f"platform_rating_{aspect}"] = rating
        output[f"platform_rating_{aspect}_sentiment"] = ((rating - 3.0) / 2.0).clip(-1.0, 1.0)
        output[f"platform_rating_{aspect}_available"] = rating.notna().astype(int)
    return output


def main() -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(f"Run 18_build_target_review_corpus.py first: {CORPUS}")
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_csv(CORPUS, low_memory=False)
    eligible = as_bool(corpus["eligible_for_temporal_model"])
    data = corpus.loc[eligible].copy()
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="coerce")
    data = data.loc[data["publish_time"].notna()].copy()
    data["review_id"] = data["review_id"].astype(str).str.strip()
    if data.duplicated("identity").any():
        raise ValueError("Temporal sentiment input has duplicate platform review identities")

    base_columns = [
        "identity", "review_id", "series_name_canonical", "publish_time", "corpus_source",
        "platform", "content_source", "rating_overall",
    ]
    features = data[base_columns].rename(columns={"series_name_canonical": "series_name"}).copy()
    ratings = platform_rating_features(data)
    text = pd.DataFrame([score_text(content) for content in data["content"]], index=data.index)
    features = pd.concat([features, ratings, text], axis=1)
    features = features.sort_values(["series_name", "publish_time", "identity"])
    features.to_csv(REVIEW_FEATURES, index=False, encoding="utf-8-sig")

    coverage_rows: list[dict] = []
    for aspect in ASPECTS:
        rating = features[f"platform_rating_{aspect}"]
        mentioned = features[f"text_{aspect}_mentioned"].eq(1)
        coverage_rows.append({
            "aspect": aspect,
            "eligible_review_rows": int(len(features)),
            "platform_rating_reviews": int(rating.notna().sum()),
            "platform_rating_coverage": round(float(rating.notna().mean()), 4),
            "text_mentioned_reviews": int(mentioned.sum()),
            "text_mention_coverage": round(float(mentioned.mean()), 4),
            "text_positive_mentions": int(features.loc[mentioned, f"text_{aspect}_polarity"].eq(1).sum()),
            "text_neutral_mentions": int(features.loc[mentioned, f"text_{aspect}_polarity"].eq(0).sum()),
            "text_negative_mentions": int(features.loc[mentioned, f"text_{aspect}_polarity"].eq(-1).sum()),
        })
    aspect_coverage = pd.DataFrame(coverage_rows)
    aspect_coverage.to_csv(ASPECT_COVERAGE, index=False, encoding="utf-8-sig")

    invalid_overall = pd.to_numeric(data["rating_overall"], errors="coerce").notna() & valid_rating(data["rating_overall"]).isna()
    summary = {
        "eligible_full_text_reviews": int(len(features)),
        "eligible_series": int(features["series_name"].nunique()),
        "platform_overall_rating_reviews": int(features["platform_rating_overall"].notna().sum()),
        "invalid_or_placeholder_overall_ratings": int(invalid_overall.sum()),
        "external_api_calls": 0,
        "method": "Reviewer-submitted platform ratings plus deterministic automotive lexicon ABSA; source ratings and text-derived polarity are not combined.",
        "time_rule": "Downstream monthly features must use only rows published before the relevant forecast origin.",
        "quality_rule": "Only corpus rows eligible_for_temporal_model are scored; Autohome list summaries without successful full-detail parsing are excluded.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
