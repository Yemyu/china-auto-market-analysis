#!/usr/bin/env python3
"""Label review aspects in compact, resumable API batches.

Each aspect uses ``null`` for not mentioned and ``-1 / 0 / 1`` for negative,
neutral, and positive mentions. The default run is a 300-review pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from config import settings  # noqa: E402


CORPUS = BASE / "data" / "reviews" / "processed" / "target_371_review_corpus.csv"
HISTORICAL = BASE / "data" / "resources" / "historical_reviews" / "review_absa_reference.csv.gz"
OUT = BASE / "data" / "reviews" / "processed"
CHECKPOINT = OUT / "api_aspect_labels.jsonl"
RESULT_CSV = OUT / "api_aspect_labels.csv"
CALL_LOG = OUT / "api_aspect_calls.jsonl"
RUN_HISTORY = OUT / "api_aspect_run_history.jsonl"

ASPECTS = [
    "appearance", "interior", "space", "power", "control", "comfort",
    "fuel_consumption", "configuration", "intelligence", "value",
]
ASPECT_DESCRIPTIONS = [
    "外观/造型/车漆/外观做工",
    "内饰设计/用料/内饰做工",
    "乘坐/储物/后备箱空间",
    "加速/动力响应/动力储备",
    "转向/制动/底盘/过弯",
    "座椅/滤振/隔音/乘坐舒适",
    "油耗/电耗/续航/补能效率",
    "非智能化装备与配置丰富度",
    "车机/语音/辅助驾驶/软件",
    "售价/优惠/成本/物有所值",
]

MODEL = settings.REVIEW_LABEL_MODEL
PROMPT_VERSION = "review_aspect_batch_v1"
SAMPLE_VERSION = "review_aspect_missing_v1"
MAX_OUTPUT_TOKENS = 2000
USD_TO_CNY = 7.20
PRICE_USD_PER_MILLION = {
    "off_peak": {"cache_hit_input": 0.007, "cache_miss_input": 0.22, "output": 0.66},
    "peak": {"cache_hit_input": 0.014, "cache_miss_input": 0.44, "output": 1.32},
}


class NonRetryableAPIError(RuntimeError):
    pass


def system_prompt() -> str:
    order = "；".join(f"{i}:{name}({desc})" for i, (name, desc) in enumerate(zip(ASPECTS, ASPECT_DESCRIPTIONS)))
    return f"""你是严谨的汽车用户评论 ABSA 标注员。输入是多条目标车系评论。
只判断评论者对目标车系本身的态度；竞品评价不得记到目标车系；厂商宣传但无用户态度按客观中性。
固定维度顺序：{order}
每条 labels 必须正好10项并遵循：null=未提及，1=正面，-1=负面，0=提及但客观中性或同维度褒贬相抵。
不得依据常识补充评论未表达的信息。只输出 JSON，不要解释或 Markdown。
JSON 示例：{{"results":[{{"id":"0","labels":[1,null,0,-1,null,null,null,1,null,1]}}]}}
"""


def load_missing_reviews() -> pd.DataFrame:
    corpus = pd.read_csv(CORPUS, low_memory=False)
    eligible = corpus["eligible_for_temporal_model"].astype(str).str.lower().eq("true")
    corpus = corpus.loc[eligible].copy()
    historical = pd.read_csv(HISTORICAL, low_memory=False)
    historical = historical.loc[historical["success"].astype(str).str.lower().eq("true")].copy()
    corpus["review_id_key"] = corpus["review_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    historical_ids = set(historical["review_id"].astype(str).str.replace(r"\.0$", "", regex=True))
    corpus["historical_label_available"] = (
        corpus["corpus_source"].eq("historical_archive")
        & corpus["review_id_key"].isin(historical_ids)
    )
    missing = corpus.loc[~corpus["historical_label_available"]].copy()
    required = ["identity", "review_id", "series_name", "publish_time", "corpus_source", "content"]
    if missing[required].isna().any().any():
        raise ValueError("Missing-label corpus contains null required values")
    if missing["identity"].duplicated().any():
        raise ValueError("Missing-label corpus has duplicate identities")
    missing["content"] = missing["content"].astype(str)
    missing["content_chars"] = missing["content"].str.len()
    missing["content_sha256"] = missing["content"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    return missing.sort_values(["corpus_source", "identity"]).reset_index(drop=True)


def stable_order(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_hash"] = ordered["identity"].map(
        lambda value: hashlib.sha256(f"{SAMPLE_VERSION}|{value}".encode()).hexdigest()
    )
    return ordered.sort_values("_hash").drop(columns="_hash")


def allocate_proportional(counts: pd.Series, total: int) -> dict[str, int]:
    raw = counts / counts.sum() * total
    allocation = raw.astype(int).to_dict()
    remaining = total - sum(allocation.values())
    order = (raw - raw.astype(int)).sort_values(ascending=False).index.tolist()
    for source in order[:remaining]:
        allocation[source] += 1
    return allocation


def take_length_stratified(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count >= len(frame):
        return stable_order(frame)
    work = frame.copy()
    work["_band"] = pd.qcut(work["content_chars"].rank(method="first"), 5, labels=False)
    base, remainder = divmod(count, 5)
    parts = []
    for band in range(5):
        quota = base + (1 if band < remainder else 0)
        parts.append(stable_order(work.loc[work["_band"].eq(band)]).head(quota))
    return pd.concat(parts, ignore_index=True).drop(columns="_band")


def select_pilot(missing: pd.DataFrame, limit: int) -> pd.DataFrame:
    counts = missing["corpus_source"].value_counts()
    allocation = allocate_proportional(counts, limit)
    parts = [
        take_length_stratified(missing.loc[missing["corpus_source"].eq(source)], count)
        for source, count in allocation.items()
    ]
    selected = stable_order(pd.concat(parts, ignore_index=True)).reset_index(drop=True)
    if len(selected) != limit or selected["identity"].duplicated().any():
        raise ValueError("Compact pilot selection is not exact and unique")
    return selected


def pricing_tier(moment: datetime) -> str:
    moment = moment.astimezone(timezone.utc)
    return "peak" if moment.weekday() < 5 and (1 <= moment.hour < 4 or 6 <= moment.hour < 10) else "off_peak"


def usage(body: dict[str, Any]) -> dict[str, int]:
    value = body.get("usage") or {}
    prompt = int(value.get("prompt_tokens") or 0)
    hit = int(value.get("prompt_cache_hit_tokens") or 0)
    miss = int(value.get("prompt_cache_miss_tokens") or max(prompt - hit, 0))
    completion = int(value.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "completion_tokens": completion,
        "total_tokens": int(value.get("total_tokens") or prompt + completion),
    }


def usage_cost(tokens: dict[str, int], tier: str) -> tuple[float, float]:
    rate = PRICE_USD_PER_MILLION[tier]
    usd = (
        tokens["cache_hit_tokens"] * rate["cache_hit_input"]
        + tokens["cache_miss_tokens"] * rate["cache_miss_input"]
        + tokens["completion_tokens"] * rate["output"]
    ) / 1_000_000
    return usd, usd * USD_TO_CNY


def validate_response(payload: dict[str, Any], expected_ids: set[str]) -> dict[str, list[Any]]:
    if set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ValueError("JSON must contain only a results list")
    parsed: dict[str, list[Any]] = {}
    for item in payload["results"]:
        if not isinstance(item, dict) or set(item) != {"id", "labels"}:
            raise ValueError("Each result must contain only id and labels")
        item_id = str(item["id"])
        labels = item["labels"]
        if item_id in parsed or not isinstance(labels, list) or len(labels) != len(ASPECTS):
            raise ValueError("Duplicate id or labels length mismatch")
        if any(isinstance(label, bool) or label not in (None, -1, 0, 1) for label in labels):
            raise ValueError("Labels must contain only null/-1/0/1")
        parsed[item_id] = labels
    if set(parsed) != expected_ids:
        raise ValueError(f"Response ids mismatch: expected {sorted(expected_ids)}, got {sorted(parsed)}")
    return parsed


def request_batch(batch: list[pd.Series], retries: int) -> tuple[dict[str, Any], dict[str, list[Any]] | None]:
    local_ids = {str(i) for i in range(len(batch))}
    items = [
        {"id": str(i), "series": str(row["series_name"]), "text": str(row["content"])}
        for i, row in enumerate(batch)
    ]
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": json.dumps(items, ensure_ascii=False, separators=(",", ":"))},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {settings.REVIEW_LABEL_API_KEY}", "Content-Type": "application/json"}
    started = datetime.now(timezone.utc)
    total_usage = {key: 0 for key in [
        "prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "completion_tokens", "total_tokens"
    ]}
    total_usd = total_cny = 0.0
    last_error = "unknown"
    attempts = 0
    api_model = MODEL
    tier = pricing_tier(started)
    finish_reason = ""

    for attempt in range(1, retries + 1):
        attempts = attempt
        tier = pricing_tier(datetime.now(timezone.utc))
        try:
            response = requests.post(
                f"{settings.REVIEW_LABEL_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers, json=body, timeout=settings.REVIEW_LABEL_TIMEOUT,
            )
            if response.status_code >= 400:
                detail = response.text.replace("\n", " ")[:300]
                message = f"HTTP {response.status_code}: {detail}"
                if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                    raise NonRetryableAPIError(message)
                raise RuntimeError(message)
            response_body = response.json()
            current = usage(response_body)
            for key in total_usage:
                total_usage[key] += current[key]
            usd, cny = usage_cost(current, tier)
            total_usd += usd; total_cny += cny
            choice = response_body["choices"][0]
            finish_reason = str(choice.get("finish_reason") or "")
            if finish_reason == "length":
                raise ValueError("compact JSON truncated")
            api_model = str(response_body.get("model") or MODEL)
            parsed = validate_response(json.loads(choice["message"]["content"]), local_ids)
            call = {
                "success": True, "error": "", "attempts": attempts,
                "batch_size": len(batch), "prompt_version": PROMPT_VERSION,
                "requested_model": MODEL, "api_model": api_model,
                "thinking_mode": "disabled", "pricing_tier": tier,
                **total_usage, "estimated_cost_usd": total_usd, "estimated_cost_cny": total_cny,
                "finish_reason": finish_reason, "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            return call, parsed
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            if isinstance(exc, NonRetryableAPIError):
                break
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    call = {
        "success": False, "error": last_error, "attempts": attempts,
        "batch_size": len(batch), "prompt_version": PROMPT_VERSION,
        "requested_model": MODEL, "api_model": api_model,
        "thinking_mode": "disabled", "pricing_tier": tier,
        **total_usage, "estimated_cost_usd": total_usd, "estimated_cost_cny": total_cny,
        "finish_reason": finish_reason, "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    return call, None


def result_records(batch: list[pd.Series], call: dict[str, Any], parsed: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    batch_id = hashlib.sha256("|".join(str(row["identity"]) for row in batch).encode()).hexdigest()[:16]
    records = []
    for i, row in enumerate(batch):
        labels = parsed.get(str(i)) if parsed else None
        record: dict[str, Any] = {
            "identity": row["identity"], "review_id": row["review_id"],
            "series_name": row["series_name"], "publish_time": row["publish_time"],
            "corpus_source": row["corpus_source"], "content_chars": int(row["content_chars"]),
            "content_sha256": row["content_sha256"], "historical_label_available": False,
            "prompt_version": PROMPT_VERSION, "requested_model": MODEL,
            "batch_id": batch_id, "batch_size": len(batch),
            "success": bool(call["success"]), "error": call["error"],
            "scored_at": call["finished_at"],
        }
        for aspect, label in zip(ASPECTS, labels or [None] * len(ASPECTS)):
            record[f"{aspect}_mentioned"] = label is not None if labels is not None else None
            record[f"{aspect}_polarity"] = label
            record[f"{aspect}_score"] = 0 if label is None and labels is not None else label
        records.append(record)
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def export_latest(records: list[dict[str, Any]]) -> pd.DataFrame:
    latest = {str(record["identity"]): record for record in records}
    frame = pd.DataFrame(latest.values()).sort_values(["series_name", "publish_time", "identity"])
    frame.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch review-aspect label backfill")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=300)
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-cost-cny", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.max_cost_cny <= 0 or args.max_retries < 1:
        raise ValueError("batch size, cost ceiling, and retries must be positive")
    if not args.dry_run and (
        not settings.REVIEW_LABEL_API_KEY
        or settings.REVIEW_LABEL_API_KEY == "xxxxx"
        or not settings.REVIEW_LABEL_MODEL
        or not settings.REVIEW_LABEL_BASE_URL
    ):
        raise RuntimeError("Review-label API credentials are not configured")
    missing = load_missing_reviews()
    selected = stable_order(missing) if args.all else select_pilot(missing, args.limit)
    prior = read_jsonl(CHECKPOINT)
    successful = {
        str(record["identity"]) for record in prior
        if record.get("success") and record.get("prompt_version") == PROMPT_VERSION
        and record.get("requested_model") == MODEL
    }
    todo = selected.loc[~selected["identity"].astype(str).isin(successful)].copy()
    print(f"[compact] missing={len(missing):,} selected={len(selected):,} successful={len(selected)-len(todo):,} todo={len(todo):,}", flush=True)
    print("[compact] selected sources:\n" + selected["corpus_source"].value_counts().to_string(), flush=True)
    print(f"[compact] chars min/median/max={selected.content_chars.min():,}/{selected.content_chars.median():,.0f}/{selected.content_chars.max():,}", flush=True)
    if args.dry_run:
        print("[compact] dry-run: API calls=0 cost=CNY 0", flush=True)
        return

    rows = [row for _, row in todo.sort_values("content_chars").iterrows()]
    run_calls = []; run_results = []; cost = 0.0; offset = 0; stopped_for_budget = False
    started = datetime.now(timezone.utc)
    while offset < len(rows):
        if cost >= args.max_cost_cny:
            stopped_for_budget = True; break
        size = 1 if offset == 0 else args.batch_size
        batch = rows[offset:offset + size]
        call, parsed = request_batch(batch, args.max_retries)
        call["batch_id"] = hashlib.sha256("|".join(str(row["identity"]) for row in batch).encode()).hexdigest()[:16]
        records = result_records(batch, call, parsed)
        append_jsonl(CALL_LOG, [call]); append_jsonl(CHECKPOINT, records)
        run_calls.append(call); run_results.extend(records)
        cost = sum(float(item["estimated_cost_cny"]) for item in run_calls)
        success = sum(bool(item["success"]) for item in run_results)
        print(f"[compact] progress={len(run_results)}/{len(rows)} success={success} failed={len(run_results)-success} calls={len(run_calls)} tokens={sum(c['total_tokens'] for c in run_calls):,} estimated_cost=CNY {cost:.4f}", flush=True)
        if offset == 0 and not call["success"]:
            print("[compact] smoke gate failed; stopping", flush=True); break
        offset += len(batch)

    latest = export_latest(prior + run_results)
    summary = {
        "run_started_at": started.isoformat(), "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION, "requested_model": MODEL, "thinking_mode": "disabled",
        "missing_population": int(len(missing)), "selected_reviews": int(len(selected)),
        "new_attempted_reviews": int(len(run_results)),
        "new_successful_reviews": int(sum(bool(item["success"]) for item in run_results)),
        "new_failed_reviews": int(sum(not bool(item["success"]) for item in run_results)),
        "api_calls": int(len(run_calls)), "prompt_tokens": int(sum(c["prompt_tokens"] for c in run_calls)),
        "completion_tokens": int(sum(c["completion_tokens"] for c in run_calls)),
        "total_tokens": int(sum(c["total_tokens"] for c in run_calls)),
        "estimated_cost_usd": float(sum(c["estimated_cost_usd"] for c in run_calls)),
        "estimated_cost_cny": float(cost), "max_cost_cny": float(args.max_cost_cny),
        "stopped_for_budget": stopped_for_budget,
        "latest_successful_compact_reviews": int(latest["success"].fillna(False).sum()),
        "price_usd_per_million_tokens": PRICE_USD_PER_MILLION, "usd_to_cny_assumption": USD_TO_CNY,
    }
    append_jsonl(RUN_HISTORY, [summary])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if run_results and not any(item["success"] for item in run_results):
        raise RuntimeError("All compact ABSA requests failed")


if __name__ == "__main__":
    main()
