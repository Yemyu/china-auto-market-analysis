#!/usr/bin/env python3
"""Backfill Autohome reviews published at or before a declared cutoff."""
from __future__ import annotations

import argparse
import importlib.util
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "reviews" / "raw"
REVIEWS_OUT = RAW / "autohome_incremental_reviews.csv"
MANIFEST = RAW / "autohome_pre_cutoff_backfill_manifest.csv"
CORPUS = BASE / "data" / "reviews" / "processed" / "target_371_review_corpus.csv"
ID_MAP = BASE / "data" / "processed" / "review_collection" / "autohome_id_resolutions.csv"
COLLECTOR = BASE / "scripts" / "20_crawl_autohome_incremental.py"


def load_collector():
    spec = importlib.util.spec_from_file_location("autohome_collector", COLLECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import collector: {COLLECTOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roster(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return verified targets without an eligible review by the cutoff."""
    if not CORPUS.exists():
        raise FileNotFoundError(f"Run 18_build_target_review_corpus.py first: {CORPUS}")
    target = pd.read_csv(BASE / "data" / "processed" / "splits" / "test.csv", usecols=["series_name"])
    target = target.drop_duplicates()
    corpus = pd.read_csv(
        CORPUS,
        usecols=["series_name_canonical", "publish_time", "eligible_for_temporal_model"],
        low_memory=False,
    )
    corpus["publish_time"] = pd.to_datetime(corpus["publish_time"], errors="coerce")
    eligible = (corpus["eligible_for_temporal_model"].fillna(False).astype(str)
                .str.strip().str.lower().isin(["true", "1", "yes"]))
    with_safe_review = set(corpus.loc[
        eligible & corpus["publish_time"].le(cutoff), "series_name_canonical"
    ])
    gap = target.loc[~target["series_name"].isin(with_safe_review), ["series_name"]]
    ids = pd.read_csv(ID_MAP)
    ids = ids.loc[ids["resolution_status"].eq("verified"), ["series_name", "autohome_series_id"]].copy()
    ids["autohome_series_id"] = pd.to_numeric(ids["autohome_series_id"], errors="raise").astype(int)
    out = gap.merge(ids, on="series_name", how="inner", validate="one_to_one")
    return out.sort_values("series_name").reset_index(drop=True)


def existing_ids() -> dict[int, set[str]]:
    if not REVIEWS_OUT.exists():
        return {}
    data = pd.read_csv(REVIEWS_OUT, usecols=["platform_series_id", "review_id"], low_memory=False)
    return {int(sid): set(group["review_id"].astype(str)) for sid, group in data.groupby("platform_series_id")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2025-12-31", help="Inclusive YYYY-MM-DD review cutoff.")
    parser.add_argument("--max-pages", type=int, default=8, help="Safety cap while traversing historical pages.")
    parser.add_argument("--target-per-series", type=int, default=10)
    parser.add_argument("--series-name", action="append", default=None, help="Restrict a resumed run to one or more exact target names.")
    parser.add_argument("--delay-min", type=float, default=0.9)
    parser.add_argument("--delay-max", type=float, default=1.5)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if args.max_pages < 1 or args.target_per_series < 1:
        raise ValueError("--max-pages and --target-per-series must be positive")

    source = load_collector()
    targets = roster(cutoff)
    if args.series_name:
        wanted = set(args.series_name)
        targets = targets.loc[targets["series_name"].isin(wanted)].copy()
        unknown = wanted - set(targets["series_name"])
        if unknown:
            raise ValueError(f"Requested series are not verified zero-coverage targets: {sorted(unknown)}")
    if targets.empty:
        print("[backfill] no verified series lack pre-test coverage")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    known = existing_ids()
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    session = source.requests.Session()
    print(f"[backfill] cutoff={cutoff.date()} targets={len(targets)} max_pages={args.max_pages}")
    for i, item in targets.reset_index(drop=True).iterrows():
        sid, name = int(item.autohome_series_id), str(item.series_name)
        print(f"[{i + 1}/{len(targets)}] {name} (Autohome {sid})", flush=True)
        before, pages, eligible_seen, source_total, status, error = len(known.get(sid, set())), 0, 0, 0, "partial", ""
        page = 1
        while page <= args.max_pages and eligible_seen < args.target_per_series:
            reviews, source_total, has_more, returned_name, error = source.fetch_page(session, sid, page, 20)
            if reviews is None:
                status = "error"
                break
            pages += 1
            parsed = [source.parse_review(review, sid, name) for review in reviews]
            for record in parsed:
                published = pd.to_datetime(record["publish_time"], errors="coerce")
                if pd.notna(published) and published <= cutoff:
                    eligible_seen += 1
                    if record["review_id"] and record["review_id"] not in known.setdefault(sid, set()):
                        known[sid].add(record["review_id"])
                        rows.append(record)
            if not has_more:
                break
            page += 1
            time.sleep(random.uniform(args.delay_min, args.delay_max))
        if status != "error":
            status = "ok" if eligible_seen >= args.target_per_series else "partial"
        manifest_rows.append({
            "series_name": name, "autohome_series_id": sid, "cutoff": cutoff.date().isoformat(),
            "status": status, "pages_fetched": pages, "source_total": source_total,
            "eligible_records_seen": eligible_seen, "new_records_added": len(known.get(sid, set())) - before,
            "checked_at": datetime.now().isoformat(timespec="seconds"), "error": error,
        })
        if i + 1 < len(targets):
            time.sleep(random.uniform(args.delay_min, args.delay_max))
    if rows:
        pd.DataFrame(rows).to_csv(REVIEWS_OUT, mode="a", header=not REVIEWS_OUT.exists(), index=False, encoding="utf-8-sig")
    latest_manifest = pd.DataFrame(manifest_rows)
    if MANIFEST.exists():
        previous_manifest = pd.read_csv(MANIFEST, low_memory=False)
        latest_manifest = pd.concat([previous_manifest, latest_manifest], ignore_index=True)
    latest_manifest = latest_manifest.drop_duplicates(["series_name", "cutoff"], keep="last")
    latest_manifest.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
    print(f"[backfill] added={len(rows)} manifest={MANIFEST}")


if __name__ == "__main__":
    main()
