#!/usr/bin/env python3
"""Fetch full Autohome review pages for collected list summaries."""
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
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "reviews" / "raw"
SUMMARIES = RAW / "autohome_incremental_reviews.csv"
DETAILS = RAW / "autohome_incremental_review_details.csv"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def detail_url(review_id: str) -> str:
    return f"https://k.autohome.com.cn/detail/view_{review_id}.html"


def fetch_html(session: requests.Session, review_id: str, referer: str) -> tuple[str | None, str]:
    url = detail_url(review_id)
    try:
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--http1.1", "--tlsv1.2",
            "--connect-timeout", "15", "--max-time", "30", "-A", HEADERS["User-Agent"],
            "-e", referer, url,
        ], check=True, capture_output=True, text=True, timeout=35)
        return result.stdout, ""
    except Exception as exc:
        curl_error = f"curl {type(exc).__name__}: {exc}"
    try:
        response = session.get(url, headers={**HEADERS, "Referer": referer}, timeout=30)
        response.raise_for_status()
        return response.text, ""
    except Exception as exc:
        return None, f"{curl_error}; requests {type(exc).__name__}: {exc}"


def parse_detail(html: str) -> tuple[str, int, str]:
    soup = BeautifulSoup(html, "lxml")
    sections: list[str] = []
    for item in soup.select("div.kb-item"):
        message = item.select_one("p.kb-item-msg")
        if message is None:
            continue
        text = message.get_text(" ", strip=True)
        if not text:
            continue
        heading = item.select_one("h1")
        label = heading.get_text(" ", strip=True) if heading else "评价"
        sections.append(f"{label}: {text}")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    return "\n".join(sections), len(sections), title


def load_details() -> pd.DataFrame:
    columns = [
        "series_name", "review_id", "platform", "detail_url", "detail_status",
        "detail_content", "detail_content_len", "detail_section_count", "page_title",
        "fetched_at", "error",
    ]
    return pd.read_csv(DETAILS, low_memory=False) if DETAILS.exists() else pd.DataFrame(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--one-per-series", action="store_true", help="Cross-series parser pilot.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--delay-min", type=float, default=0.8)
    parser.add_argument("--delay-max", type=float, default=1.4)
    args = parser.parse_args()
    if not SUMMARIES.exists():
        raise FileNotFoundError(f"No Autohome list corpus: {SUMMARIES}")

    summaries = pd.read_csv(SUMMARIES, low_memory=False)
    summaries["review_id"] = summaries["review_id"].astype(str).str.strip()
    if summaries["review_id"].duplicated().any():
        raise ValueError("Autohome staged review IDs must be unique before detail enrichment")
    details = load_details()
    details["review_id"] = details["review_id"].astype(str).str.strip()
    completed = set(details.loc[details["detail_status"].isin(["ok", "empty"]), "review_id"])
    if not args.retry_failed:
        completed |= set(details.loc[details["detail_status"].eq("error"), "review_id"])
    pending = summaries.loc[~summaries["review_id"].isin(completed)].copy()
    if args.one_per_series:
        pending = pending.drop_duplicates("series_name", keep="first")
    if args.limit:
        pending = pending.head(args.limit)
    print(f"[autohome-detail] staged={len(summaries)} pending={len(pending)} -> {DETAILS}")

    session = requests.Session()
    status_counts = {"ok": 0, "empty": 0, "error": 0}
    for position, row in pending.reset_index(drop=True).iterrows():
        review_id = str(row.review_id)
        referer = str(row.get("source_url") or f"https://k.autohome.com.cn/{int(row.series_id)}/")
        print(f"[{position + 1}/{len(pending)}] {row.series_name} {review_id}", flush=True)
        html, error = fetch_html(session, review_id, referer)
        content, section_count, title = "", 0, ""
        status = "error" if html is None else "ok"
        if html is not None:
            try:
                content, section_count, title = parse_detail(html)
                if not content:
                    status = "empty"
                    error = "HTTP succeeded but no div.kb-item p.kb-item-msg content was found"
            except Exception as exc:
                status = "error"
                error = f"parse {type(exc).__name__}: {exc}"
        record = pd.DataFrame([{
            "series_name": row.series_name, "review_id": review_id, "platform": "autohome",
            "detail_url": detail_url(review_id), "detail_status": status,
            "detail_content": content, "detail_content_len": len(content),
            "detail_section_count": section_count, "page_title": title,
            "fetched_at": datetime.now().isoformat(timespec="seconds"), "error": error,
        }])
        details = pd.concat([details, record], ignore_index=True).drop_duplicates("review_id", keep="last")
        details.to_csv(DETAILS, index=False, encoding="utf-8-sig")
        status_counts[status] += 1
        if position + 1 < len(pending):
            time.sleep(random.uniform(args.delay_min, args.delay_max))
    print(json.dumps({"processed_this_run": len(pending), **status_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
