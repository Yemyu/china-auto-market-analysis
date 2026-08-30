#!/usr/bin/env python3
"""Build review topics and sample-aware month-end alerts for 371 series."""
from __future__ import annotations

import importlib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import jieba
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
importlib.import_module("_font_setup")
local_sentiment = importlib.import_module("27_build_local_sentiment_features")

SENTIMENT = BASE / "data" / "reviews" / "processed"
CORPUS = SENTIMENT / "target_371_review_corpus.csv"
UNIFIED = SENTIMENT / "review_aspect_labels.csv"
TEST_SPLIT = BASE / "data" / "processed" / "splits" / "test.csv"
OUT = BASE / "data" / "processed" / "user_feedback"
FIG_DIR = BASE / "assets/analysis"

ASPECT_SUMMARY = OUT / "user_need_aspect_summary.csv"
SERIES_SUMMARY = OUT / "user_need_by_series.csv"
KEYWORDS = OUT / "user_need_keywords.csv"
TOPICS = OUT / "user_need_topics.csv"
WINDOWS = OUT / "sentiment_monitoring_windows.csv"
ALERTS = OUT / "sentiment_alerts.csv"
SUMMARY = OUT / "user_needs_alerts_summary.json"
FIGURE = FIG_DIR / "user_needs_and_alerts.png"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
ASPECT_ZH = {
    "appearance": "外观", "interior": "内饰", "space": "空间", "power": "动力",
    "control": "操控", "comfort": "舒适性", "fuel_consumption": "能耗/油耗",
    "configuration": "配置", "intelligence": "智能化", "value": "性价比",
}

WINDOW_DAYS = 180
MIN_REVIEWS_PER_WINDOW = 5
ALERT_SCORE_THRESHOLD = -0.10
ALERT_DROP_THRESHOLD = -0.15
ALERT_NEGATIVE_RATE_THRESHOLD = 0.35
TEXT_RETRIGGER_PROBABILITY_THRESHOLD = 0.70
RATING_DECLINE_PROBABILITY_THRESHOLD = 0.80
BOOTSTRAP_DRAWS = 3_000
TOPIC_COUNT = 4
TOPIC_WORDS = 10
TOPIC_MAX_DOCS = 6_000
RANDOM_SEED = 42

STOPWORDS = {
    "的", "了", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "还是",
    "但是", "就是", "非常", "感觉", "有点", "一下", "虽然", "因为", "所以", "如果", "时候", "不过",
    "真的", "比较", "觉得", "认为", "已经", "可以", "应该", "可能", "特别", "其实", "反正", "大概",
    "确实", "不是", "这个", "那个", "这些", "那些", "这么", "那么", "怎么", "什么", "不要", "不能",
    "不会", "不太", "不用", "很多", "一点", "一直", "一次", "一样", "一般", "总体", "整体", "大家",
    "我们", "你们", "他们", "它们", "这边", "那边", "这里", "那里", "车型", "车子", "汽车", "方面",
    "表现", "来说", "来讲", "目前", "当时", "平时", "日常", "个人", "开车", "驾驶", "公里", "左右",
}
TOKEN_CLEAN_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def vehicle_entity_stopwords() -> set[str]:
    metadata = pd.read_csv(TEST_SPLIT, usecols=["series_name", "brand"], low_memory=False)
    values = pd.concat([metadata["series_name"], metadata["brand"]]).dropna().astype(str).str.lower().unique()
    entities: set[str] = set()
    for value in values:
        entities.add(value.strip())
        entities.update(token.strip() for token in jieba.lcut(value) if len(token.strip()) > 1)
    return entities


VEHICLE_ENTITY_STOPWORDS = vehicle_entity_stopwords()


def as_bool(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in jieba.lcut(str(text).lower()):
        word = token.strip()
        if (
            len(word) <= 1
            or word in STOPWORDS
            or word in VEHICLE_ENTITY_STOPWORDS
            or word.isdigit()
            or TOKEN_CLEAN_RE.fullmatch(word)
        ):
            continue
        if word.replace(".", "", 1).isdigit():
            continue
        tokens.append(word)
    return tokens


def relevant_text(content: str, aspect: str) -> str:
    terms = local_sentiment.ASPECT_TERMS[aspect]
    segments = [
        text for text, _ in local_sentiment.text_segments(content)
        if any(term in text for term in terms)
    ]
    return "。".join(segments)


def load_review_data() -> pd.DataFrame:
    corpus = pd.read_csv(CORPUS, low_memory=False)
    unified = pd.read_csv(UNIFIED, low_memory=False, parse_dates=["publish_time"])
    corpus = corpus.loc[as_bool(corpus["eligible_for_temporal_model"]), ["identity", "content"]].copy()
    if corpus["identity"].duplicated().any() or unified["identity"].duplicated().any():
        raise ValueError("Review identity is not unique")
    data = unified.merge(corpus, on="identity", how="left", validate="one_to_one")
    if data["content"].isna().any():
        raise ValueError("Unified ABSA rows are missing strict-corpus review text")
    if len(data) != 24_175:
        raise ValueError(f"Expected 24,175 unified reviews, found {len(data):,}")
    data["series_name"] = data["series_name"].astype(str)
    data["publish_time"] = pd.to_datetime(data["publish_time"], errors="raise")
    return data.sort_values(["series_name", "publish_time", "identity"]).reset_index(drop=True)


def add_review_scores(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    score_columns: list[str] = []
    for aspect in ASPECTS:
        mentioned = output[f"uniform_local_{aspect}_mentioned"].eq(1)
        raw = pd.to_numeric(output[f"review_{aspect}_raw_polarity"], errors="coerce")
        column = f"monitor_{aspect}_score"
        output[column] = raw.where(mentioned & raw.isin([-1, 0, 1]))
        score_columns.append(column)
    output["monitor_overall_score"] = output[score_columns].mean(axis=1, skipna=True)
    output["monitor_negative_review"] = output["monitor_overall_score"].lt(0).where(
        output["monitor_overall_score"].notna()
    )
    output["platform_rating_overall"] = pd.to_numeric(output["rating_overall"], errors="coerce")
    invalid_rating = output["platform_rating_overall"].notna() & ~output["platform_rating_overall"].between(0.5, 5.0)
    if invalid_rating.any():
        raise ValueError("Platform ratings outside the audited 0.5-5.0 range")
    return output


def aspect_and_series_summaries(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_reviews = len(data)
    overall_rows: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        mentioned = data[f"uniform_local_{aspect}_mentioned"].eq(1)
        values = data[f"monitor_{aspect}_score"]
        valid = values.notna()
        overall_rows.append({
            "aspect": aspect,
            "aspect_zh": ASPECT_ZH[aspect],
            "eligible_reviews": total_reviews,
            "mentioned_reviews": int(mentioned.sum()),
            "mention_rate": float(mentioned.mean()),
            "scored_mentions": int(valid.sum()),
            "positive_mentions": int(values.eq(1).sum()),
            "neutral_mentions": int(values.eq(0).sum()),
            "negative_mentions": int(values.eq(-1).sum()),
            "positive_rate_among_scored": float(values.eq(1).sum() / valid.sum()) if valid.any() else np.nan,
            "negative_rate_among_scored": float(values.eq(-1).sum() / valid.sum()) if valid.any() else np.nan,
            "mean_polarity": float(values.mean()),
        })
        work = data.loc[mentioned, ["series_name", f"monitor_{aspect}_score"]].copy()
        work = work.rename(columns={f"monitor_{aspect}_score": "score"})
        for series_name, group in work.groupby("series_name", sort=True):
            scores = group["score"].dropna()
            series_rows.append({
                "series_name": series_name,
                "aspect": aspect,
                "aspect_zh": ASPECT_ZH[aspect],
                "mentioned_reviews": int(len(group)),
                "scored_mentions": int(len(scores)),
                "mean_polarity": float(scores.mean()) if len(scores) else np.nan,
                "negative_rate_among_scored": float(scores.eq(-1).mean()) if len(scores) else np.nan,
                "positive_rate_among_scored": float(scores.eq(1).mean()) if len(scores) else np.nan,
            })
    overall = pd.DataFrame(overall_rows)
    total_negative = overall["negative_mentions"].sum()
    overall["share_of_all_negative_mentions"] = np.where(
        total_negative > 0, overall["negative_mentions"] / total_negative, np.nan
    )
    overall["mention_rank"] = overall["mentioned_reviews"].rank(method="min", ascending=False).astype(int)
    overall["negative_burden_rank"] = overall["negative_mentions"].rank(method="min", ascending=False).astype(int)

    by_series = pd.DataFrame(series_rows)
    metadata = pd.read_csv(TEST_SPLIT, usecols=["series_name", "brand", "category"], low_memory=False)
    metadata["series_name"] = metadata["series_name"].astype(str)
    metadata = metadata.drop_duplicates("series_name")
    by_series = by_series.merge(metadata, on="series_name", how="left", validate="many_to_one")
    return overall.sort_values("mention_rank"), by_series.sort_values(["series_name", "aspect"])


def sample_documents(documents: pd.DataFrame) -> pd.DataFrame:
    if len(documents) <= TOPIC_MAX_DOCS:
        return documents
    return documents.sample(TOPIC_MAX_DOCS, random_state=RANDOM_SEED).sort_index()


def keyword_rows(aspect: str, documents: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    sampled_all = sample_documents(documents)
    if len(sampled_all) >= 20:
        min_df = max(2, min(10, int(np.ceil(len(sampled_all) * 0.001))))
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=min_df, max_df=0.92, max_features=4_000,
        )
        matrix = vectorizer.fit_transform(sampled_all["aspect_text"])
        terms = vectorizer.get_feature_names_out()
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        for rank, index in enumerate(scores.argsort()[::-1][:20], start=1):
            output.append({
                "aspect": aspect,
                "aspect_zh": ASPECT_ZH[aspect],
                "polarity_group": "all",
                "rank": rank,
                "keyword": terms[index].replace(" ", "·"),
                "keyword_score": float(scores[index]),
                "documents_available": int(len(documents)),
                "documents_sampled": int(len(sampled_all)),
                "min_document_frequency": min_df,
                "method": "mean TF-IDF",
            })

    contrast = documents.loc[documents["polarity"].isin([-1, 1])].copy()
    positive = sample_documents(contrast.loc[contrast["polarity"].eq(1)])
    negative = sample_documents(contrast.loc[contrast["polarity"].eq(-1)])
    if min(len(positive), len(negative)) < 20:
        return output
    sampled = pd.concat([positive, negative], ignore_index=True)
    min_df = max(2, min(10, int(np.ceil(len(sampled) * 0.001))))
    vectorizer = CountVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), min_df=min_df, max_df=0.98, max_features=4_000,
    )
    matrix = vectorizer.fit_transform(sampled["aspect_text"])
    terms = vectorizer.get_feature_names_out()
    positive_counts = np.asarray(matrix[:len(positive)].sum(axis=0)).ravel().astype(float)
    negative_counts = np.asarray(matrix[len(positive):].sum(axis=0)).ravel().astype(float)
    pooled = positive_counts + negative_counts
    prior_strength = 1_000.0
    alpha = prior_strength * pooled / pooled.sum()
    alpha = np.maximum(alpha, 1e-9)
    alpha_total = alpha.sum()

    def log_odds(counts: np.ndarray, total: float) -> np.ndarray:
        return np.log((counts + alpha) / (total + alpha_total - counts - alpha))

    delta = log_odds(negative_counts, negative_counts.sum()) - log_odds(
        positive_counts, positive_counts.sum()
    )
    variance = 1.0 / (negative_counts + alpha) + 1.0 / (positive_counts + alpha)
    z_scores = delta / np.sqrt(variance)
    group_specs = {
        "negative": (negative, z_scores),
        "positive": (positive, -z_scores),
    }
    for polarity_group, (group, group_scores) in group_specs.items():
        polarity = -1 if polarity_group == "negative" else 1
        for rank, index in enumerate(group_scores.argsort()[::-1][:20], start=1):
            output.append({
                "aspect": aspect,
                "aspect_zh": ASPECT_ZH[aspect],
                "polarity_group": polarity_group,
                "rank": rank,
                "keyword": terms[index].replace(" ", "·"),
                "keyword_score": float(group_scores[index]),
                "documents_available": int(documents["polarity"].eq(polarity).sum()),
                "documents_sampled": int(len(group)),
                "min_document_frequency": min_df,
                "method": "informative Dirichlet log-odds z-score",
            })
    return output


def bootstrap_signal_support(
    series_name: str,
    cutoff: pd.Timestamp,
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> tuple[float, float]:
    """Estimate text-rule reproducibility and platform-rating corroboration."""
    seed_material = f"{series_name}|{cutoff:%Y-%m-%d}|v3".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    current_text = current["monitor_overall_score"].to_numpy(dtype=float)
    previous_text = previous["monitor_overall_score"].to_numpy(dtype=float)
    current_negative = current["monitor_negative_review"].to_numpy(dtype=float)
    current_rating = current["platform_rating_overall"].dropna().to_numpy(dtype=float)
    previous_rating = previous["platform_rating_overall"].dropna().to_numpy(dtype=float)

    current_text_draws = rng.choice(
        current_text, size=(BOOTSTRAP_DRAWS, len(current_text)), replace=True
    ).mean(axis=1)
    previous_text_draws = rng.choice(
        previous_text, size=(BOOTSTRAP_DRAWS, len(previous_text)), replace=True
    ).mean(axis=1)
    negative_rate_draws = rng.choice(
        current_negative, size=(BOOTSTRAP_DRAWS, len(current_negative)), replace=True
    ).mean(axis=1)
    text_retrigger = (
        (current_text_draws <= ALERT_SCORE_THRESHOLD)
        & (current_text_draws - previous_text_draws <= ALERT_DROP_THRESHOLD)
        & (negative_rate_draws >= ALERT_NEGATIVE_RATE_THRESHOLD)
    )
    if not len(current_rating) or not len(previous_rating):
        return float(text_retrigger.mean()), np.nan
    rating_change_draws = rng.choice(
        current_rating, size=(BOOTSTRAP_DRAWS, len(current_rating)), replace=True
    ).mean(axis=1) - rng.choice(
        previous_rating, size=(BOOTSTRAP_DRAWS, len(previous_rating)), replace=True
    ).mean(axis=1)
    return float(text_retrigger.mean()), float((rating_change_draws < 0).mean())


def topic_rows(aspect: str, documents: pd.DataFrame) -> list[dict[str, Any]]:
    sampled = sample_documents(documents)
    if len(sampled) < 50:
        return []
    min_df = max(3, min(12, int(np.ceil(len(sampled) * 0.0015))))
    vectorizer = CountVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), min_df=min_df, max_df=0.90, max_features=1_500,
    )
    matrix = vectorizer.fit_transform(sampled["aspect_text"])
    if matrix.shape[1] < 20:
        return []
    model = LatentDirichletAllocation(
        n_components=TOPIC_COUNT, random_state=RANDOM_SEED, max_iter=12,
        learning_method="batch", n_jobs=1,
    )
    weights = model.fit_transform(matrix)
    primary = weights.argmax(axis=1)
    terms = vectorizer.get_feature_names_out()
    output: list[dict[str, Any]] = []
    for topic_index, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:TOPIC_WORDS]
        mask = primary == topic_index
        polarities = sampled.loc[mask, "polarity"].dropna()
        output.append({
            "aspect": aspect,
            "aspect_zh": ASPECT_ZH[aspect],
            "topic_id": topic_index + 1,
            "topic_label": " / ".join(terms[top_indices[:3]]).replace(" ", "·"),
            "top_words": " | ".join(terms[top_indices]).replace(" ", "·"),
            "assigned_documents": int(mask.sum()),
            "topic_prevalence_in_sample": float(mask.mean()),
            "mean_polarity": float(polarities.mean()) if len(polarities) else np.nan,
            "negative_rate": float(polarities.eq(-1).mean()) if len(polarities) else np.nan,
            "documents_available": int(len(documents)),
            "documents_sampled": int(len(sampled)),
            "min_document_frequency": min_df,
            "method": "aspect-filtered LDA",
        })
    return output


def build_text_insights(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keywords: list[dict[str, Any]] = []
    topics: list[dict[str, Any]] = []
    for aspect in ASPECTS:
        mentioned = data[f"uniform_local_{aspect}_mentioned"].eq(1)
        columns = ["identity", "content", f"monitor_{aspect}_score"]
        documents = data.loc[mentioned, columns].copy()
        documents["aspect_text"] = documents["content"].map(lambda value: relevant_text(value, aspect))
        documents["polarity"] = documents[f"monitor_{aspect}_score"]
        documents = documents.loc[documents["aspect_text"].str.strip().ne("")].copy()
        print(f"[topics:{aspect}] aspect snippets={len(documents):,}", flush=True)
        keywords.extend(keyword_rows(aspect, documents))
        topics.extend(topic_rows(aspect, documents))
    return pd.DataFrame(keywords), pd.DataFrame(topics)


def window_statistics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    data_max = data["publish_time"].max().normalize()
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    latest_completed_month_end = today.to_period("M").start_time - pd.Timedelta(days=1)
    latest_completed_month_end = min(latest_completed_month_end, data_max.to_period("M").end_time.normalize())
    first_end = max(pd.Timestamp("2022-01-31"), data["publish_time"].min().to_period("M").end_time.normalize())
    month_ends = pd.date_range(first_end, latest_completed_month_end, freq="ME")
    metadata = pd.read_csv(TEST_SPLIT, usecols=["series_name", "brand", "category"], low_memory=False)
    metadata["series_name"] = metadata["series_name"].astype(str)
    metadata = metadata.drop_duplicates("series_name").set_index("series_name")

    window_rows: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []
    for series_name, group in data.groupby("series_name", sort=True):
        group = group.loc[group["monitor_overall_score"].notna()].copy()
        if group.empty:
            continue
        brand = metadata.at[series_name, "brand"] if series_name in metadata.index else np.nan
        category = metadata.at[series_name, "category"] if series_name in metadata.index else np.nan
        for cutoff in month_ends:
            current_start = cutoff - pd.Timedelta(days=WINDOW_DAYS - 1)
            previous_start = current_start - pd.Timedelta(days=WINDOW_DAYS)
            cutoff_exclusive = cutoff + pd.Timedelta(days=1)
            current = group.loc[
                group["publish_time"].ge(current_start)
                & group["publish_time"].lt(cutoff_exclusive)
            ]
            previous = group.loc[
                group["publish_time"].ge(previous_start)
                & group["publish_time"].lt(current_start)
            ]
            if set(current["identity"]) & set(previous["identity"]):
                raise ValueError(f"Overlapping monitoring windows for {series_name} at {cutoff:%Y-%m-%d}")
            if len(current) and current["publish_time"].max() >= cutoff_exclusive:
                raise ValueError(f"Future review entered current window for {series_name} at {cutoff:%Y-%m-%d}")
            if len(previous) and previous["publish_time"].max() >= current_start:
                raise ValueError(f"Previous review entered current window for {series_name} at {cutoff:%Y-%m-%d}")
            if current.empty and previous.empty:
                continue
            current_score = float(current["monitor_overall_score"].mean()) if len(current) else np.nan
            previous_score = float(previous["monitor_overall_score"].mean()) if len(previous) else np.nan
            score_drop = current_score - previous_score if len(current) and len(previous) else np.nan
            negative_rate = float(current["monitor_negative_review"].mean()) if len(current) else np.nan
            current_rating = current["platform_rating_overall"].dropna()
            previous_rating = previous["platform_rating_overall"].dropna()
            current_rating_mean = float(current_rating.mean()) if len(current_rating) else np.nan
            previous_rating_mean = float(previous_rating.mean()) if len(previous_rating) else np.nan
            rating_change = (
                current_rating_mean - previous_rating_mean
                if len(current_rating) and len(previous_rating) else np.nan
            )
            eligible = len(current) >= MIN_REVIEWS_PER_WINDOW and len(previous) >= MIN_REVIEWS_PER_WINDOW
            text_alert_candidate = bool(
                eligible
                and current_score <= ALERT_SCORE_THRESHOLD
                and score_drop <= ALERT_DROP_THRESHOLD
                and negative_rate >= ALERT_NEGATIVE_RATE_THRESHOLD
            )
            text_retrigger_probability = np.nan
            rating_decline_probability = np.nan
            if text_alert_candidate:
                text_retrigger_probability, rating_decline_probability = bootstrap_signal_support(
                    series_name, cutoff, current, previous
                )
            corroborated_alert = bool(
                text_alert_candidate
                and text_retrigger_probability >= TEXT_RETRIGGER_PROBABILITY_THRESHOLD
                and rating_decline_probability >= RATING_DECLINE_PROBABILITY_THRESHOLD
            )
            alert_status = (
                "corroborated" if corroborated_alert
                else "watchlist" if text_alert_candidate
                else "none"
            )
            base_row = {
                "series_name": series_name,
                "brand": brand,
                "category": category,
                "information_cutoff_inclusive": cutoff,
                "window_days": WINDOW_DAYS,
                "current_window_start": current_start,
                "previous_window_start": previous_start,
                "current_reviews": int(len(current)),
                "previous_reviews": int(len(previous)),
                "current_overall_score": current_score,
                "previous_overall_score": previous_score,
                "score_change": score_drop,
                "current_negative_review_rate": negative_rate,
                "current_rating_reviews": int(len(current_rating)),
                "previous_rating_reviews": int(len(previous_rating)),
                "current_platform_rating": current_rating_mean,
                "previous_platform_rating": previous_rating_mean,
                "platform_rating_change": rating_change,
                "eligible_for_alert": eligible,
                "text_alert_candidate": text_alert_candidate,
                "text_rule_retrigger_probability": text_retrigger_probability,
                "rating_decline_probability": rating_decline_probability,
                "alert_status": alert_status,
                "alert": corroborated_alert,
            }
            window_rows.append(base_row)
            if not text_alert_candidate:
                continue
            aspect_changes: list[tuple[str, float, float, float, int, int]] = []
            for aspect in ASPECTS:
                column = f"monitor_{aspect}_score"
                cur_scores = current[column].dropna()
                prev_scores = previous[column].dropna()
                if len(cur_scores) < 2 or len(prev_scores) < 2:
                    continue
                change = float(cur_scores.mean() - prev_scores.mean())
                aspect_changes.append((aspect, change, float(cur_scores.mean()), float(prev_scores.mean()), len(cur_scores), len(prev_scores)))
            deteriorating_aspects = [item for item in aspect_changes if item[1] < 0]
            worst = min(deteriorating_aspects, key=lambda item: item[1]) if deteriorating_aspects else (
                "overall", score_drop, current_score, previous_score, len(current), len(previous)
            )
            if current_score <= -0.25 or score_drop <= -0.50 or negative_rate >= 0.60:
                risk_level = "high"
            elif current_score <= -0.15 or score_drop <= -0.30 or negative_rate >= 0.50:
                risk_level = "medium"
            else:
                risk_level = "low"
            alert_rows.append({
                **base_row,
                "alert_id": f"{series_name}::{cutoff:%Y-%m}",
                "risk_level": risk_level,
                "worst_aspect": worst[0],
                "worst_aspect_zh": ASPECT_ZH.get(worst[0], "综合情感"),
                "worst_aspect_change": worst[1],
                "worst_aspect_current_score": worst[2],
                "worst_aspect_previous_score": worst[3],
                "worst_aspect_current_mentions": worst[4],
                "worst_aspect_previous_mentions": worst[5],
                "rule_version": "v3_dual_signal_bootstrap",
            })
    windows = pd.DataFrame(window_rows)
    alerts = pd.DataFrame(alert_rows)
    return windows, alerts, latest_completed_month_end


def save_figure(aspects: pd.DataFrame, alerts: pd.DataFrame) -> None:
    ordered = aspects.sort_values("mentioned_reviews")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    axes[0].barh(ordered["aspect_zh"], ordered["mention_rate"] * 100, color="#4C78A8")
    axes[0].set_xlabel("Mention rate among reviews (%)")
    axes[0].set_title("What users discuss")

    risk = aspects.sort_values("negative_rate_among_scored")
    axes[1].barh(risk["aspect_zh"], risk["negative_rate_among_scored"] * 100, color="#E45756")
    axes[1].set_xlabel("Negative rate among scored mentions (%)")
    axes[1].set_title("Where complaints concentrate")

    corroborated = alerts.loc[alerts["alert"]] if len(alerts) else alerts
    if len(corroborated):
        monthly = corroborated.assign(month=pd.to_datetime(corroborated["information_cutoff_inclusive"]).dt.to_period("M").astype(str)).groupby("month").size()
        axes[2].bar(monthly.index, monthly.values, color="#F58518")
        axes[2].tick_params(axis="x", rotation=55, labelsize=7)
        axes[2].set_ylabel("Alert events")
        axes[2].set_title("Dual-signal alert history")
    else:
        axes[2].axis("off")
        axes[2].set_title("No alerts under the registered rule")
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = add_review_scores(load_review_data())
    aspects, series = aspect_and_series_summaries(data)
    keywords, topics = build_text_insights(data)
    windows, alerts, latest_completed = window_statistics(data)

    aspects.to_csv(ASPECT_SUMMARY, index=False, encoding="utf-8-sig")
    series.to_csv(SERIES_SUMMARY, index=False, encoding="utf-8-sig")
    keywords.to_csv(KEYWORDS, index=False, encoding="utf-8-sig")
    topics.to_csv(TOPICS, index=False, encoding="utf-8-sig")
    windows.to_csv(WINDOWS, index=False, encoding="utf-8-sig")
    alerts.to_csv(ALERTS, index=False, encoding="utf-8-sig")
    save_figure(aspects, alerts)

    latest_candidates = alerts.loc[
        pd.to_datetime(alerts["information_cutoff_inclusive"]).eq(latest_completed)
    ] if len(alerts) else alerts
    latest_alerts = latest_candidates.loc[latest_candidates["alert"]] if len(latest_candidates) else latest_candidates
    latest_watchlist = latest_candidates.loc[
        latest_candidates["alert_status"].eq("watchlist")
    ] if len(latest_candidates) else latest_candidates
    eligible_latest = windows.loc[
        pd.to_datetime(windows["information_cutoff_inclusive"]).eq(latest_completed)
        & windows["eligible_for_alert"]
    ] if len(windows) else windows
    summary = {
        "schema_version": "v2",
        "review_rows": int(len(data)),
        "review_series": int(data["series_name"].nunique()),
        "review_publish_time_min": data["publish_time"].min().isoformat(),
        "review_publish_time_max": data["publish_time"].max().isoformat(),
        "aspects": len(ASPECTS),
        "keyword_rows": int(len(keywords)),
        "topic_rows": int(len(topics)),
        "monitoring_window_days": WINDOW_DAYS,
        "minimum_reviews_each_window": MIN_REVIEWS_PER_WINDOW,
        "alert_rule": {
            "current_overall_score_lte": ALERT_SCORE_THRESHOLD,
            "score_change_lte": ALERT_DROP_THRESHOLD,
            "current_negative_review_rate_gte": ALERT_NEGATIVE_RATE_THRESHOLD,
            "text_rule_retrigger_probability_gte": TEXT_RETRIGGER_PROBABILITY_THRESHOLD,
            "platform_rating_decline_probability_gte": RATING_DECLINE_PROBABILITY_THRESHOLD,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "latest_completed_monitoring_month": latest_completed.strftime("%Y-%m-%d"),
        "latest_eligible_series": int(eligible_latest["series_name"].nunique()) if len(eligible_latest) else 0,
        "latest_alert_candidates": int(len(latest_candidates)),
        "latest_active_alerts": int(len(latest_alerts)),
        "latest_watchlist_events": int(len(latest_watchlist)),
        "historical_alert_candidates": int(len(alerts)),
        "historical_alert_events": int(alerts["alert"].sum()) if len(alerts) else 0,
        "historical_watchlist_events": int(alerts["alert_status"].eq("watchlist").sum()) if len(alerts) else 0,
        "historical_alert_series": int(alerts.loc[alerts["alert"], "series_name"].nunique()) if len(alerts) else 0,
        "candidate_mean_text_rule_retrigger_probability": float(alerts["text_rule_retrigger_probability"].mean()) if len(alerts) else np.nan,
        "corroborated_mean_text_rule_retrigger_probability": float(alerts.loc[alerts["alert"], "text_rule_retrigger_probability"].mean()) if len(alerts) and alerts["alert"].any() else np.nan,
        "topic_method": "aspect-filtered LDA",
        "keyword_method": "mean TF-IDF for prevalence; informative Dirichlet log-odds for positive/negative distinctiveness",
        "alert_interpretation": "Dual-signal corroboration improves contemporaneous evidence stability; it is not validated as a predictor of future incidents or persistent deterioration.",
        "current_incomplete_month_excluded_from_alerts": True,
        "external_api_calls": 0,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== User-need aspect summary =====", flush=True)
    print(aspects[[
        "aspect_zh", "mentioned_reviews", "mention_rate", "mean_polarity",
        "negative_rate_among_scored", "share_of_all_negative_mentions",
    ]].to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print("\n===== Monitoring summary =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if len(latest_candidates):
        print("\n===== Latest alert candidates =====", flush=True)
        print(latest_candidates[[
            "series_name", "risk_level", "current_reviews", "current_overall_score",
            "score_change", "current_negative_review_rate", "platform_rating_change",
            "text_rule_retrigger_probability", "rating_decline_probability",
            "alert_status", "worst_aspect_zh",
        ]].sort_values(["risk_level", "score_change"]).to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print(f"[output] {SUMMARY.relative_to(BASE)}", flush=True)
    print(f"[output] {FIGURE.relative_to(BASE)}", flush=True)


if __name__ == "__main__":
    main()
