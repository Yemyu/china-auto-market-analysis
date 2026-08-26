#!/usr/bin/env python3
"""Collect a source-separated, resumable Autohome review increment for target series.

The collector intentionally begins only with IDs manually verified against an
official Autohome series page.  It never guesses an ID from a series name and
does not overwrite either the legacy review corpus or the Dongchedi increment.
The public list endpoint returns structured review summaries.  Long summaries
can be abbreviated by the source, so each row carries an explicit flag rather
than representing the text as a full review.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "processed_new" / "splits"
ID_MAP = BASE / "data" / "processed_new" / "phase_b" / "autohome_id_resolutions.csv"
AVAILABILITY = BASE / "data" / "sentiment_new" / "processed" / "review_temporal_availability_by_series.csv"
OUT = BASE / "data" / "sentiment_new" / "raw"
REVIEWS_OUT = OUT / "autohome_incremental_reviews.csv"
MANIFEST_OUT = OUT / "autohome_incremental_manifest.csv"

API_URL = "https://koubeiipv6.app.autohome.com.cn/pc/series/list"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://k.autohome.com.cn/",
}


def target_roster(mode: str) -> pd.DataFrame:
    """Return only verified IDs belonging to the new 371-series population."""
    if not ID_MAP.exists():
        raise FileNotFoundError(f"Missing verified ID register: {ID_MAP}")
    target = pd.read_csv(SPLITS / "test.csv", usecols=["series_name"]).drop_duplicates()
    ids = pd.read_csv(ID_MAP)
    ids = ids.loc[ids["resolution_status"].eq("verified")].copy()
    ids["autohome_series_id"] = pd.to_numeric(ids["autohome_series_id"], errors="raise").astype(int)
    roster = target.merge(ids, on="series_name", how="inner", validate="one_to_one")
    if mode == "missing":
        if not AVAILABILITY.exists():
            raise FileNotFoundError(f"Run 25_build_review_temporal_availability.py first: {AVAILABILITY}")
        test = pd.read_csv(SPLITS / "test.csv", usecols=["date"])
        first_test_month = pd.to_datetime(test["date"], errors="raise").min().to_period("M")
        column = f"reviews_available_before_{first_test_month.strftime('%Y_%m')}"
        availability = pd.read_csv(AVAILABILITY, usecols=["series_name", column])
        roster = roster.merge(availability, on="series_name", how="left", validate="one_to_one")
        roster = roster.loc[roster[column].fillna(0).eq(0)].drop(columns=column)
    return roster.sort_values("series_name").reset_index(drop=True)


def endpoint_url(series_id: int, page: int, page_size: int) -> str:
    query = urlencode({
        "pm": 3, "seriesId": int(series_id), "pageIndex": int(page), "pageSize": int(page_size),
        "yearid": 0, "ge": 0, "seriesSummaryKey": 0, "order": 0,
    })
    return f"{API_URL}?{query}"


def unpack(raw: dict) -> tuple[list[dict], int, bool, str]:
    if int(raw.get("returncode", -1)) != 0:
        raise ValueError(f"source returncode={raw.get('returncode')}: {raw.get('message', '')}")
    result = raw.get("result") or {}
    reviews = result.get("list") or []
    total = int(result.get("rowcount") or 0)
    page = int(result.get("pageindex") or 1)
    pages = int(result.get("pagecount") or 1)
    return reviews, total, page < pages, str(result.get("seriesname") or "")


def fetch_page(session: requests.Session, series_id: int, page: int, page_size: int) -> tuple[list[dict] | None, int, bool, str, str]:
    """Use curl first, with requests retained as a portable transport fallback."""
    url = endpoint_url(series_id, page, page_size)
    curl_error = ""
    try:
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--http1.1", "--tlsv1.2",
            "--connect-timeout", "15", "--max-time", "30", "-A", HEADERS["User-Agent"],
            "-e", HEADERS["Referer"], url,
        ], check=True, capture_output=True, text=True, timeout=35)
        reviews, total, has_more, returned_name = unpack(json.loads(result.stdout))
        return reviews, total, has_more, returned_name, ""
    except Exception as exc:
        curl_error = f"curl {type(exc).__name__}: {exc}"
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        reviews, total, has_more, returned_name = unpack(response.json())
        return reviews, total, has_more, returned_name, ""
    except Exception as exc:
        return None, 0, False, "", f"{curl_error}; requests {type(exc).__name__}: {exc}"


def content_from_sections(review: dict) -> tuple[str, bool]:
    sections = review.get("contents") or []
    text = "\n".join(
        f"{str(item.get('structuredname') or '评价').strip()}: {str(item.get('content') or '').strip()}"
        for item in sections if str(item.get("content") or "").strip()
    ).strip()
    # The list API visibly marks abbreviated passages with an ellipsis.  This
    # is intentionally a conservative flag: False means no visible marker,
    # not a guarantee that it is the original full-detail text.
    abbreviated = any(str(item.get("content") or "").rstrip().endswith(("...", "…")) for item in sections)
    return text, abbreviated


def parse_review(review: dict, series_id: int, series_name: str) -> dict:
    content, abbreviated = content_from_sections(review)
    score_map = {
        "appearance": review.get("apperance"), "interiors": review.get("internal"),
        "space": review.get("space"), "power": review.get("power"),
        "control": review.get("maneuverability"), "comfort": review.get("comfortableness"),
        "oil_consumption": review.get("oilConsumption"), "value": review.get("costEfficient"),
    }
    review_id = str(review.get("showId") or review.get("Koubeiid") or "").strip()
    out = {
        "series_id": series_id, "platform_series_id": series_id, "series_name": series_name,
        "review_id": review_id, "platform": "autohome", "user_nickname": review.get("username", ""),
        "user_id": str(review.get("userid") or ""), "publish_time": review.get("posttime", ""),
        "content": content, "content_len": len(content), "content_source": "list_structured_summary",
        "content_possibly_abbreviated": abbreviated, "rating_overall": review.get("averageScore"),
        "digg_count": review.get("helpfulcount"), "comment_count": review.get("commentcount"),
        "view_count": review.get("viewcount"), "car_model": review.get("specname", ""),
        "buy_location": review.get("buyplace", ""), "buy_price": review.get("buyprice", ""),
        "buy_time": review.get("boughtDate", ""), "fuel_type": review.get("powertype", ""),
        "consumption": review.get("actual_oil_consumption", ""), "series_name_from_source": review.get("carname", ""),
        "source_url": f"https://k.autohome.com.cn/{series_id}/",
    }
    for dimension, value in score_map.items():
        out[f"rating_{dimension}"] = value
    return out


def load_manifest() -> pd.DataFrame:
    columns = ["autohome_series_id", "series_name", "status", "review_count", "api_total", "pages_fetched", "checked_at", "error"]
    return pd.read_csv(MANIFEST_OUT) if MANIFEST_OUT.exists() else pd.DataFrame(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["missing", "verified"], default="missing")
    parser.add_argument("--max-pages", type=int, default=1, help="Safety limit per series; default is one list page.")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--limit-series", type=int, default=None, help="Pilot limit after filtering pending series.")
    parser.add_argument("--series-name", action="append", default=[], help="Restrict to one or more verified target series (repeatable).")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=2.0)
    args = parser.parse_args()
    if args.max_pages < 1 or args.page_size < 1:
        raise ValueError("--max-pages and --page-size must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    roster = target_roster(args.mode)
    if args.series_name:
        requested = set(args.series_name)
        available = set(roster["series_name"])
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Requested series are not in the verified {args.mode} roster: {unknown}")
        roster = roster.loc[roster["series_name"].isin(requested)].copy()
    manifest = load_manifest()
    done = set(pd.to_numeric(manifest.loc[manifest["status"].isin(["ok", "empty"]), "autohome_series_id"], errors="coerce").dropna().astype(int))
    if not args.retry_failed:
        done |= set(pd.to_numeric(manifest.loc[manifest["status"].eq("error"), "autohome_series_id"], errors="coerce").dropna().astype(int))
    pending = roster.loc[~roster["autohome_series_id"].isin(done)].copy()
    if args.limit_series:
        pending = pending.head(args.limit_series)
    print(f"[autohome] verified-target={len(roster)} pending={len(pending)} mode={args.mode} -> {REVIEWS_OUT}")

    existing_ids: dict[int, set[str]] = {}
    if REVIEWS_OUT.exists():
        existing = pd.read_csv(REVIEWS_OUT, usecols=["platform_series_id", "review_id"], low_memory=False)
        for sid, group in existing.groupby("platform_series_id"):
            existing_ids[int(sid)] = set(group["review_id"].astype(str))
    session = requests.Session()
    written = 0
    for position, row in pending.reset_index(drop=True).iterrows():
        sid, name = int(row.autohome_series_id), str(row.series_name)
        print(f"[{position + 1}/{len(pending)}] {name} (Autohome {sid})", flush=True)
        collected: list[dict] = []
        total, status, error, page, pages_fetched = 0, "ok", "", 1, 0
        while page <= args.max_pages:
            reviews, total, has_more, returned_name, error = fetch_page(session, sid, page, args.page_size)
            if reviews is None:
                status = "error"
                break
            pages_fetched += 1
            collected.extend(parse_review(item, sid, name) for item in reviews)
            if not has_more:
                break
            page += 1
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        if not collected and status == "ok":
            status = "empty"
        known = existing_ids.setdefault(sid, set())
        new_records = [record for record in collected if record["review_id"] and record["review_id"] not in known]
        known.update(record["review_id"] for record in new_records)
        if new_records:
            pd.DataFrame(new_records).to_csv(REVIEWS_OUT, mode="a", header=not REVIEWS_OUT.exists(), index=False, encoding="utf-8-sig")
            written += len(new_records)
        manifest_row = pd.DataFrame([{
            "autohome_series_id": sid, "series_name": name, "status": status,
            "review_count": len(known), "api_total": total, "pages_fetched": pages_fetched,
            "checked_at": datetime.now().isoformat(timespec="seconds"), "error": error,
        }])
        manifest = pd.concat([manifest, manifest_row], ignore_index=True).drop_duplicates("autohome_series_id", keep="last")
        manifest.to_csv(MANIFEST_OUT, index=False, encoding="utf-8-sig")
        if position + 1 < len(pending):
            time.sleep(random.uniform(args.delay_min, args.delay_max))
    print(f"[autohome] finished this run: {written} new rows; manifest={MANIFEST_OUT}")


if __name__ == "__main__":
    main()
