#!/usr/bin/env python3
"""Incrementally collect Dongchedi reviews for the new 371-series population.

The old crawler targets the retired V1 roster and appends directly to the old
corpus.  This script instead takes the authoritative new forecast population
from ``processed_new/splits/test.csv``, obtains its Dongchedi ``series_id``
from ``raw/feature.csv``, and writes an isolated Phase-B staging corpus.

Default mode is deliberately conservative: only series with *no* reusable old
review coverage are crawled.  Use ``--mode all`` only when a complete refresh
is desired.  A per-series manifest records successful, empty and failed calls
so it is safe to resume after interruption.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[2]
SPLITS = BASE / "data" / "processed_new" / "splits"
SERIES_INDEX = BASE / "data" / "raw" / "series_index.csv"
READINESS = BASE / "data" / "processed_new" / "phase_b" / "sentiment_readiness.csv"
OUT = BASE / "data" / "sentiment_new" / "raw"
REVIEWS_OUT = OUT / "dongchedi_incremental_reviews.csv"
MANIFEST_OUT = OUT / "dongchedi_incremental_manifest.csv"

API_URL = "https://www.dongchedi.com/motor/pc/car/series/get_review_list"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.dongchedi.com/",
}


def target_roster(mode: str) -> pd.DataFrame:
    population = pd.read_csv(SPLITS / "test.csv", usecols=["series_name"]).drop_duplicates()
    # ``feature.csv`` uses the *PCauto* ``sgXXXX`` source ID, not a Dongchedi
    # numeric ID.  The latter is what the review endpoint requires.  The
    # cross-platform index is therefore the only valid source for this join.
    if not SERIES_INDEX.exists():
        raise FileNotFoundError("Run 00_build_series_index.py before collecting reviews.")
    index = pd.read_csv(SERIES_INDEX, usecols=["series_name", "dongchedi_series_id"])
    roster = population.merge(index, on="series_name", how="left")
    roster["series_id"] = pd.to_numeric(roster.pop("dongchedi_series_id"), errors="coerce")
    roster = roster.dropna(subset=["series_id"]).copy()
    roster["series_id"] = roster["series_id"].astype(int)
    # The manifest retains a brand field for auditability; it is descriptive,
    # never used to infer or translate platform IDs.
    brand = pd.read_csv(BASE / "data" / "raw" / "feature.csv", usecols=["series_name", "brand_name", "year"])
    brand = brand.sort_values("year").drop_duplicates("series_name", keep="last")[["series_name", "brand_name"]]
    roster = roster.merge(brand, on="series_name", how="left")
    if mode == "missing":
        if not READINESS.exists():
            raise FileNotFoundError("Run 15_audit_sentiment_readiness.py before --mode missing.")
        readiness = pd.read_csv(READINESS, usecols=["series_name", "old_reviews_available"])
        roster = roster.merge(readiness, on="series_name", how="left")
        roster = roster[~roster["old_reviews_available"].fillna(False)].drop(columns="old_reviews_available")
    return roster.sort_values(["brand_name", "series_name"]).reset_index(drop=True)


def _unpack_payload(raw: dict) -> tuple[list[dict], int, bool]:
    payload = raw.get("data") or {}
    return (payload.get("review_list") or [], int(payload.get("total_count", 0)),
            bool(payload.get("has_more", False)))


def _curl_page(series_id: int, page: int, attempts: int = 3) -> tuple[list[dict] | None, int, bool, str]:
    """TLS fallback for hosts where Python/OpenSSL is rejected by the source.

    This remains a normal public GET request.  ``subprocess`` receives an
    argument list (not a shell string), so the numerical ID cannot alter the
    command.  The fallback makes the crawler portable without silently
    treating a transport error as an empty review list.
    """
    url = (f"{API_URL}?series_id={int(series_id)}&page={int(page)}&size=15"
           "&city_name=&sort_by=default")
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run([
                "curl", "--http1.1", "--tlsv1.2", "--fail", "--silent", "--show-error",
                "-A", HEADERS["User-Agent"], "-e", HEADERS["Referer"], url,
            ], check=True, capture_output=True, text=True, timeout=30)
            reviews, total, has_more = _unpack_payload(json.loads(result.stdout))
            return reviews, total, has_more, ""
        except Exception as exc:
            last_error = f"curl attempt {attempt}/{attempts} {type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(4 * attempt + random.uniform(0, 1))
    return None, 0, False, last_error


def fetch_page(session: requests.Session, series_id: int, page: int) -> tuple[list[dict] | None, int, bool, str]:
    # The target source currently accepts the system curl TLS profile but
    # closes the Python/OpenSSL handshake.  Try the working transport first;
    # retain requests as a portable fallback for other environments.
    reviews, total, has_more, curl_error = _curl_page(series_id, page)
    if reviews is not None:
        return reviews, total, has_more, ""
    try:
        resp = session.get(API_URL, headers=HEADERS, params={
            "series_id": series_id, "page": page, "size": 15,
            "city_name": "", "sort_by": "default",
        }, timeout=20)
        resp.raise_for_status()
        reviews, total, has_more = _unpack_payload(resp.json())
        return reviews, total, has_more, ""
    except Exception as exc:  # network failures are recorded in the manifest
        return None, 0, False, f"curl first: {curl_error}; requests {type(exc).__name__}: {exc}"


def parse_review(review: dict, series_id: int, series_name: str) -> dict:
    user = review.get("user_info") or {}
    buy = review.get("buy_car_info") or {}
    score = review.get("score_info") or {}
    ts = review.get("create_time")
    published = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
    score_map = {
        "appearance": "appearance_score", "space": "space_score", "interiors": "interiors_score",
        "power": "power_score", "control": "control_score", "comfort": "comfort_score",
        "oil_consumption": "oil_consumption_score", "config": "configuration_score",
    }
    out = {
        "series_id": series_id, "series_name": series_name, "review_id": review.get("gid_str", ""),
        "platform": "dongchedi", "user_nickname": user.get("name", ""),
        "user_id": str(user.get("user_id", "")), "publish_time": published,
        "content": review.get("content", ""), "content_len": len(review.get("content", "")),
        "rating_overall": score.get("score", 0) / 100 if score.get("score") is not None else None,
        "digg_count": review.get("digg_count_en"), "comment_count": review.get("comment_count_en"),
        "car_model": buy.get("car_name", ""), "buy_location": buy.get("location", ""),
        "buy_price": buy.get("price", ""), "buy_time": buy.get("bought_time", ""),
        "fuel_type": buy.get("fuel_form", ""), "consumption": buy.get("consumption", ""),
    }
    for short, key in score_map.items():
        val = score.get(key)
        out[f"rating_{short}"] = val / 100 if val is not None else None
    return out


def load_manifest() -> pd.DataFrame:
    if MANIFEST_OUT.exists():
        return pd.read_csv(MANIFEST_OUT)
    return pd.DataFrame(columns=["series_id", "series_name", "brand_name", "status", "review_count", "api_total", "checked_at", "error"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["missing", "all"], default="missing")
    parser.add_argument("--max-per-series", type=int, default=200)
    parser.add_argument("--limit-series", type=int, default=None, help="Pilot only: process first N pending series")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--delay-min", type=float, default=1.5)
    parser.add_argument("--delay-max", type=float, default=3.5)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    roster = target_roster(args.mode)
    manifest = load_manifest()
    done = set(manifest.loc[manifest["status"].isin(["ok", "empty"]), "series_id"].astype(int))
    if not args.retry_failed:
        done |= set(manifest.loc[manifest["status"] == "error", "series_id"].astype(int))
    pending = roster[~roster["series_id"].isin(done)].copy()
    if args.limit_series:
        pending = pending.head(args.limit_series)
    print(f"[reviews] target={len(roster)} pending={len(pending)} mode={args.mode} -> {REVIEWS_OUT}")

    session = requests.Session()
    records, manifest_rows = [], []
    existing_ids: dict[int, set[str]] = {}
    if REVIEWS_OUT.exists():
        existing = pd.read_csv(REVIEWS_OUT, usecols=["series_id", "review_id"], low_memory=False)
        for sid, group in existing.groupby("series_id"):
            existing_ids[int(sid)] = set(group["review_id"].astype(str))
    for i, row in pending.reset_index(drop=True).iterrows():
        sid, name = int(row.series_id), str(row.series_name)
        print(f"[{i + 1}/{len(pending)}] {name} ({sid})", flush=True)
        collected, page, total, status, error = [], 1, 0, "ok", ""
        while len(collected) < args.max_per_series:
            reviews, total, has_more, error = fetch_page(session, sid, page)
            if reviews is None:
                status = "error"
                break
            collected.extend(parse_review(r, sid, name) for r in reviews[:args.max_per_series - len(collected)])
            if not has_more or not reviews:
                break
            page += 1
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        if not collected and status == "ok":
            status = "empty"
        # Retrying a partial series starts from page 1.  Keep the raw corpus
        # idempotent by writing only unseen review IDs, while the manifest
        # reports the cumulative per-series count.
        known = existing_ids.setdefault(sid, set())
        new_records = [record for record in collected if str(record["review_id"]) not in known]
        known.update(str(record["review_id"]) for record in new_records)
        records.extend(new_records)
        manifest_rows.append({"series_id": sid, "series_name": name, "brand_name": row.brand_name,
                              "status": status, "review_count": len(known), "api_total": total,
                              "checked_at": datetime.now().isoformat(timespec="seconds"), "error": error})
        if new_records:
            pd.DataFrame(new_records).to_csv(REVIEWS_OUT, mode="a", header=not REVIEWS_OUT.exists(), index=False, encoding="utf-8-sig")
        manifest = pd.concat([manifest, pd.DataFrame(manifest_rows)], ignore_index=True)
        manifest = manifest.drop_duplicates("series_id", keep="last")
        manifest.to_csv(MANIFEST_OUT, index=False, encoding="utf-8-sig")
        manifest_rows = []
        if i + 1 < len(pending):
            time.sleep(random.uniform(args.delay_min, args.delay_max))
    print(f"[reviews] finished this run: {len(records)} reviews; manifest={MANIFEST_OUT}")


if __name__ == "__main__":
    main()
