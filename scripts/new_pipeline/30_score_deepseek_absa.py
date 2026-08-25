#!/usr/bin/env python3
"""Score the leakage-eligible 371-series review corpus with DeepSeek ABSA.

This is the formal LLM scoring path for Phase B.  It deliberately keeps
``mentioned`` separate from ``polarity`` so an unmentioned aspect is never
silently treated as neutral.  Results are appended to a resumable JSONL
checkpoint and exported as a flat CSV for later monthly aggregation.

Safety defaults:
  * explicit ``deepseek-v4-flash`` non-thinking model;
  * only 50 stratified pilot rows unless ``--all`` is explicitly supplied;
  * CNY 1.00 estimated-cost ceiling per invocation;
  * API key is loaded from gitignored ``config/.env`` and never printed;
  * every successful response records token usage and estimated cost.

Examples (always use the project Conda environment):
  conda run -n nlp-sentiment python scripts/new_pipeline/30_score_deepseek_absa.py --dry-run
  conda run -n nlp-sentiment python scripts/new_pipeline/30_score_deepseek_absa.py --limit 50 --max-cost-cny 1
  conda run -n nlp-sentiment python scripts/new_pipeline/30_score_deepseek_absa.py --all --max-cost-cny 60
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from config import settings  # noqa: E402


CORPUS = BASE / "data" / "sentiment_new" / "processed" / "target_371_review_corpus.csv"
OUT = BASE / "data" / "sentiment_new" / "processed"
CHECKPOINT = OUT / "deepseek_absa_results.jsonl"
RESULT_CSV = OUT / "deepseek_absa_results.csv"
RUN_HISTORY = OUT / "deepseek_absa_run_history.jsonl"

ASPECTS: dict[str, tuple[str, str]] = {
    "appearance": ("外观", "造型、设计、颜值、车漆和外观做工"),
    "interior": ("内饰", "座舱设计、内饰用料、内饰做工和质感"),
    "space": ("空间", "乘坐、储物、后备箱以及头部或腿部空间"),
    "power": ("动力", "加速、动力响应、动力储备和爬坡表现"),
    "control": ("操控", "转向、制动、底盘、过弯和驾驶稳定性"),
    "comfort": ("舒适", "座椅、悬架滤振、隔音、噪声和乘坐感受"),
    "fuel_consumption": ("能耗", "油耗、电耗、续航真实性和补能效率"),
    "configuration": ("配置", "非智能化的装备、功能、配置丰富度或缺失"),
    "intelligence": ("智能化", "车机、语音、辅助驾驶、智驾和软件体验"),
    "value": ("性价比", "售价、优惠、用车成本以及是否物有所值"),
}

# DeepSeek official pricing observed on 2026-08-24.  The script stores the
# rates with every run so future audits can distinguish estimates made under
# different pricing.  Costs are estimates; the provider balance is authoritative.
PRICE_USD_PER_MILLION = {
    "off_peak": {"cache_hit_input": 0.007, "cache_miss_input": 0.22, "output": 0.66},
    "peak": {"cache_hit_input": 0.014, "cache_miss_input": 0.44, "output": 1.32},
}
USD_TO_CNY = 7.20
DEFAULT_MODEL = settings.DEEPSEEK_ABSA_MODEL
MAX_OUTPUT_TOKENS = 900
PROMPT_VERSION = "autopulse_absa_v4_2026-08-24"
PILOT_SAMPLE_VERSION = "autopulse_pilot_sample_v1"


class NonRetryableAPIError(RuntimeError):
    """A 4xx request/authentication error that retries cannot repair."""


def system_prompt() -> str:
    definitions = "\n".join(
        f"- {key}（{label}）：{definition}" for key, (label, definition) in ASPECTS.items()
    )
    example_aspects = ",".join(
        f'"{key}":{{"mentioned":false,"polarity":null,"evidence":""}}'
        for key in ASPECTS
    )
    return f"""你是严谨的汽车用户评论 ABSA 标注员。请只判断评论者对目标车系本身的态度。
若评论拿竞品作比较，不得把对竞品的评价误记到目标车系；转述厂商宣传但没有用户态度时，按中性事实处理。

维度定义：
{definitions}

标注规则：
1. 每个维度必须输出 mentioned、polarity、evidence。
2. 没有提到该维度：mentioned=false，polarity=null，evidence=""。
3. 提到了且态度正面：polarity=1；负面：-1；真正中性、只有客观事实或同一维度褒贬相抵：0。
4. evidence 必须从评论正文逐字复制，禁止概括、改写或补字，最多30个字符；未提及时必须为空字符串。
5. 不依据常识补全评论没有表达的信息，不受平台星级影响。
6. 输出严格 JSON，不要解释、不要 Markdown。必须包含且只包含下列10个维度。

JSON 格式示例：
{{"aspects":{{{example_aspects}}}}}
"""


def user_prompt(series_name: str, content: str) -> str:
    return f"目标车系：{series_name}\n评论正文：\n{content}\n\n请输出 JSON："


def load_corpus() -> pd.DataFrame:
    if not CORPUS.exists():
        raise FileNotFoundError(f"Missing corpus: {CORPUS}")
    data = pd.read_csv(CORPUS, low_memory=False)
    eligible = data["eligible_for_temporal_model"].astype(str).str.lower().eq("true")
    data = data.loc[eligible].copy()
    required = {"identity", "series_name", "publish_time", "corpus_source", "content"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Corpus is missing columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Eligible corpus contains missing required values")
    if data["identity"].duplicated().any():
        raise ValueError("Eligible corpus has duplicate identities")
    data["content"] = data["content"].astype(str)
    data["content_chars"] = data["content"].str.len()
    data["content_sha256"] = data["content"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return data.sort_values(["corpus_source", "identity"]).reset_index(drop=True)


def deterministic_order(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.copy()
    ordered["sample_hash"] = ordered["identity"].map(
        # Sampling must remain stable when the scoring prompt changes, or two
        # prompt versions cannot be compared on an identical pilot cohort.
        lambda value: hashlib.sha256(f"{PILOT_SAMPLE_VERSION}|{value}".encode()).hexdigest()
    )
    return ordered.sort_values("sample_hash").drop(columns="sample_hash")


def take_across_lengths(data: pd.DataFrame, count: int) -> pd.DataFrame:
    """Deterministically spread a source quota over five content-length bands."""
    if count >= len(data):
        return deterministic_order(data)
    ranked = data.copy()
    ranked["length_band"] = pd.qcut(
        ranked["content_chars"].rank(method="first"), q=min(5, len(ranked)), labels=False
    )
    bands = sorted(ranked["length_band"].unique())
    base, remainder = divmod(count, len(bands))
    parts = []
    for position, band in enumerate(bands):
        quota = base + (1 if position < remainder else 0)
        parts.append(deterministic_order(ranked.loc[ranked["length_band"].eq(band)]).head(quota))
    selected = pd.concat(parts, ignore_index=True).drop(columns="length_band")
    if len(selected) < count:
        extra = deterministic_order(ranked.loc[~ranked["identity"].isin(selected["identity"])])
        selected = pd.concat([selected, extra.head(count - len(selected)).drop(columns="length_band")])
    return selected.head(count)


def select_pilot(data: pd.DataFrame, limit: int) -> pd.DataFrame:
    """For the first 50, deliberately include old and both newly collected sources."""
    if limit <= 0:
        return data.iloc[0:0]
    if limit != 50:
        return deterministic_order(data).head(limit)
    quotas = {"old_v1": 30, "autohome_incremental": 10, "dongchedi_incremental": 10}
    parts = []
    for source, quota in quotas.items():
        source_rows = data.loc[data["corpus_source"].eq(source)]
        if source_rows.empty:
            raise ValueError(f"Pilot source is absent: {source}")
        parts.append(take_across_lengths(source_rows, quota))
    pilot = pd.concat(parts, ignore_index=True)
    if len(pilot) != 50 or pilot["identity"].duplicated().any():
        raise ValueError("Pilot selection did not produce 50 unique review identities")
    return deterministic_order(pilot).reset_index(drop=True)


def parse_json_content(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("empty JSON content")
    return json.loads(text.strip())


def validate_absa(payload: dict[str, Any], content: str) -> dict[str, dict[str, Any]]:
    if set(payload) != {"aspects"} or not isinstance(payload["aspects"], dict):
        raise ValueError("top-level JSON must contain only an aspects object")
    aspects = payload["aspects"]
    if set(aspects) != set(ASPECTS):
        missing = sorted(set(ASPECTS) - set(aspects))
        extra = sorted(set(aspects) - set(ASPECTS))
        raise ValueError(f"aspect keys mismatch; missing={missing}, extra={extra}")
    clean: dict[str, dict[str, Any]] = {}
    for aspect in ASPECTS:
        value = aspects[aspect]
        if not isinstance(value, dict):
            raise ValueError(f"{aspect} must be an object")
        if set(value) != {"mentioned", "polarity", "evidence"}:
            raise ValueError(f"{aspect} must contain mentioned, polarity, evidence")
        mentioned = value["mentioned"]
        polarity = value["polarity"]
        evidence = value["evidence"]
        if not isinstance(mentioned, bool):
            raise ValueError(f"{aspect}.mentioned must be boolean")
        if mentioned:
            if isinstance(polarity, bool) or polarity not in (-1, 0, 1):
                raise ValueError(f"{aspect}.polarity must be -1, 0, or 1 when mentioned")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"{aspect}.evidence must be non-empty when mentioned")
            evidence = evidence.strip()
            normalized_evidence = "".join(evidence.split())
            normalized_content = "".join(str(content).split())
            evidence_exact = len(evidence) <= 30 and normalized_evidence in normalized_content
        else:
            if polarity is not None or evidence != "":
                raise ValueError(f"{aspect} must use null polarity and empty evidence when unmentioned")
            evidence_exact = None
        clean[aspect] = {
            "mentioned": mentioned,
            "polarity": polarity,
            # Evidence is an audit aid, not a modelling feature.  Preserve a
            # bounded copy and flag imperfect quotations instead of discarding
            # an otherwise valid polarity label or paying for blind retries.
            "evidence": evidence.strip()[:100] if isinstance(evidence, str) else "",
            "evidence_exact": evidence_exact,
        }
    return clean


def pricing_tier(moment: datetime) -> str:
    """Official peak: Mon-Fri 01:00-04:00 and 06:00-10:00 UTC."""
    hour = moment.astimezone(timezone.utc).hour
    weekday = moment.astimezone(timezone.utc).weekday()
    return "peak" if weekday < 5 and (1 <= hour < 4 or 6 <= hour < 10) else "off_peak"


def usage_from_response(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss_value = usage.get("prompt_cache_miss_tokens")
    miss = int(miss_value) if miss_value is not None else max(prompt - hit, 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or prompt + completion),
    }


def cost_from_usage(usage: dict[str, int], tier: str) -> tuple[float, float]:
    rates = PRICE_USD_PER_MILLION[tier]
    usd = (
        usage["cache_hit_tokens"] * rates["cache_hit_input"]
        + usage["cache_miss_tokens"] * rates["cache_miss_input"]
        + usage["completion_tokens"] * rates["output"]
    ) / 1_000_000
    return usd, usd * USD_TO_CNY


def add_usage(total: dict[str, int], current: dict[str, int]) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0) + int(value)


def call_api(row: pd.Series, model: str, retries: int) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(str(row["series_name"]), str(row["content"]))},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
    }
    started = datetime.now(timezone.utc)
    cumulative_usage = {key: 0 for key in (
        "prompt_tokens", "cache_hit_tokens", "cache_miss_tokens", "completion_tokens", "total_tokens"
    )}
    cumulative_usd = 0.0
    cumulative_cny = 0.0
    last_error = "unknown API error"
    last_tier = pricing_tier(started)
    attempts_used = 0

    for attempt in range(1, retries + 1):
        attempts_used = attempt
        call_time = datetime.now(timezone.utc)
        last_tier = pricing_tier(call_time)
        try:
            response = requests.post(
                f"{settings.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=settings.LLM_REQUEST_TIMEOUT,
            )
            if response.status_code >= 400:
                # Never include request headers or the API key in the error.
                detail = response.text.replace("\n", " ")[:300]
                message = f"HTTP {response.status_code}: {detail}"
                if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                    raise NonRetryableAPIError(message)
                raise RuntimeError(message)
            body = response.json()
            usage = usage_from_response(body)
            add_usage(cumulative_usage, usage)
            usd, cny = cost_from_usage(usage, last_tier)
            cumulative_usd += usd
            cumulative_cny += cny
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("JSON response truncated at max_tokens")
            content = choice["message"].get("content") or ""
            aspects = validate_absa(parse_json_content(content), str(row["content"]))
            return {
                "success": True,
                "error": "",
                "attempts": attempt,
                "aspects": aspects,
                "usage": cumulative_usage,
                "estimated_cost_usd": cumulative_usd,
                "estimated_cost_cny": cumulative_cny,
                "pricing_tier": last_tier,
                "api_model": body.get("model", model),
                "finish_reason": choice.get("finish_reason", ""),
                "started_at": started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            if isinstance(exc, NonRetryableAPIError):
                break
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    return {
        "success": False,
        "error": last_error,
        "attempts": attempts_used,
        "aspects": {},
        "usage": cumulative_usage,
        "estimated_cost_usd": cumulative_usd,
        "estimated_cost_cny": cumulative_cny,
        "pricing_tier": last_tier,
        "api_model": model,
        "finish_reason": "",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def flatten_record(row: pd.Series, result: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "identity": row["identity"],
        "review_id": row.get("review_id", ""),
        "series_name": row["series_name"],
        "publish_time": row["publish_time"],
        "corpus_source": row["corpus_source"],
        "content_chars": int(row["content_chars"]),
        "content_sha256": row["content_sha256"],
        "prompt_version": PROMPT_VERSION,
        "success": result["success"],
        "error": result["error"],
        "attempts": result["attempts"],
        "requested_model": DEFAULT_MODEL,
        "api_model": result["api_model"],
        "thinking_mode": "disabled",
        "finish_reason": result["finish_reason"],
        **result["usage"],
        "pricing_tier": result["pricing_tier"],
        "estimated_cost_usd": result["estimated_cost_usd"],
        "estimated_cost_cny": result["estimated_cost_cny"],
        "started_at": result["started_at"],
        "finished_at": result["finished_at"],
    }
    for aspect in ASPECTS:
        value = result["aspects"].get(aspect, {})
        record[f"{aspect}_mentioned"] = value.get("mentioned")
        record[f"{aspect}_polarity"] = value.get("polarity")
        record[f"{aspect}_evidence"] = value.get("evidence", "")
        record[f"{aspect}_evidence_exact"] = value.get("evidence_exact")
    return record


def read_checkpoint() -> list[dict[str, Any]]:
    if not CHECKPOINT.exists():
        return []
    records = []
    with CHECKPOINT.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid checkpoint JSON at line {number}") from exc
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def export_latest_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["identity"])] = record
    frame = pd.DataFrame(latest.values()).sort_values(["series_name", "publish_time", "identity"])
    frame.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    return frame


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Leakage-safe DeepSeek ABSA scorer")
    scope = result.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=50, help="Stratified pilot size (default: 50)")
    scope.add_argument("--all", action="store_true", help="Explicitly select the full eligible corpus")
    result.add_argument("--workers", type=int, default=4, help="Concurrent requests (default: 4)")
    result.add_argument("--max-cost-cny", type=float, default=1.0, help="Estimated per-run cost ceiling")
    result.add_argument("--max-retries", type=int, default=3)
    result.add_argument("--dry-run", action="store_true", help="Select and audit rows without API calls")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1 or args.max_retries < 1 or args.max_cost_cny <= 0:
        raise ValueError("workers/retries/cost ceiling must be positive")
    if not args.dry_run and (not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "xxxxx"):
        raise RuntimeError("DEEPSEEK_API_KEY is not configured in gitignored config/.env")

    corpus = load_corpus()
    selected = deterministic_order(corpus) if args.all else select_pilot(corpus, args.limit)
    prior = read_checkpoint()
    successful = {
        str(record["identity"])
        for record in prior
        if record.get("success")
        and record.get("prompt_version") == PROMPT_VERSION
        and record.get("requested_model") == DEFAULT_MODEL
    }
    todo = selected.loc[~selected["identity"].astype(str).isin(successful)].copy()
    print(
        f"[DeepSeek ABSA] model={DEFAULT_MODEL} thinking=disabled prompt={PROMPT_VERSION}\n"
        f"[DeepSeek ABSA] eligible={len(corpus):,} selected={len(selected):,} "
        f"already_successful={len(selected)-len(todo):,} todo={len(todo):,}"
    )
    print("[DeepSeek ABSA] selected source counts:")
    print(selected["corpus_source"].value_counts().to_string())
    print(
        f"[DeepSeek ABSA] content chars: min={selected['content_chars'].min():,} "
        f"median={selected['content_chars'].median():,.0f} max={selected['content_chars'].max():,}"
    )
    if args.dry_run:
        print("[DeepSeek ABSA] dry-run complete; API calls=0 cost=CNY 0")
        return
    if todo.empty:
        export_latest_csv(prior)
        print("[DeepSeek ABSA] nothing to do")
        return

    run_started = datetime.now(timezone.utc)
    run_records: list[dict[str, Any]] = []
    run_cost = 0.0
    stopped_for_budget = False
    rows = [row for _, row in todo.iterrows()]

    # Work in bounded batches.  The budget is checked between batches so at
    # most ``workers`` already-started requests can cross the estimate ceiling.
    offset = 0
    while offset < len(rows):
        if run_cost >= args.max_cost_cny:
            stopped_for_budget = True
            break
        # The first request is a smoke gate.  A model name, authentication, or
        # schema incompatibility therefore stops after one review instead of
        # failing an entire concurrent batch.
        batch_size = 1 if offset == 0 else args.workers
        batch = rows[offset: offset + batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(call_api, row, DEFAULT_MODEL, args.max_retries): row
                for row in batch
            }
            completed_batch = []
            for future in as_completed(futures):
                row = futures[future]
                completed_batch.append(flatten_record(row, future.result()))
        append_jsonl(CHECKPOINT, completed_batch)
        run_records.extend(completed_batch)
        run_cost = sum(float(record["estimated_cost_cny"]) for record in run_records)
        ok = sum(bool(record["success"]) for record in run_records)
        print(
            f"[DeepSeek ABSA] progress={len(run_records)}/{len(todo)} "
            f"success={ok} failed={len(run_records)-ok} "
            f"tokens={sum(int(r['total_tokens']) for r in run_records):,} "
            f"estimated_cost=CNY {run_cost:.4f}"
        )
        if offset == 0 and not completed_batch[0]["success"]:
            print("[DeepSeek ABSA] smoke request failed; stopping before the remaining pilot rows")
            break
        offset += len(batch)

    all_records = prior + run_records
    latest = export_latest_csv(all_records)
    summary = {
        "run_started_at": run_started.isoformat(),
        "run_finished_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "requested_model": DEFAULT_MODEL,
        "thinking_mode": "disabled",
        "selected_reviews": int(len(selected)),
        "new_attempted_reviews": int(len(run_records)),
        "new_successful_reviews": int(sum(bool(record["success"]) for record in run_records)),
        "new_failed_reviews": int(sum(not bool(record["success"]) for record in run_records)),
        "checkpoint_latest_successful_reviews": int(latest["success"].fillna(False).sum()),
        "prompt_tokens": int(sum(record["prompt_tokens"] for record in run_records)),
        "completion_tokens": int(sum(record["completion_tokens"] for record in run_records)),
        "total_tokens": int(sum(record["total_tokens"] for record in run_records)),
        "estimated_cost_usd": float(sum(record["estimated_cost_usd"] for record in run_records)),
        "estimated_cost_cny": float(run_cost),
        "max_cost_cny": float(args.max_cost_cny),
        "stopped_for_budget": stopped_for_budget,
        "price_usd_per_million_tokens": PRICE_USD_PER_MILLION,
        "usd_to_cny_assumption": USD_TO_CNY,
    }
    append_jsonl(RUN_HISTORY, [summary])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if run_records and not any(record["success"] for record in run_records):
        raise RuntimeError("All DeepSeek ABSA requests failed; inspect error column without retrying full scope")


if __name__ == "__main__":
    main()
