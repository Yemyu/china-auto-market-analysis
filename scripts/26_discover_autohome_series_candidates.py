#!/usr/bin/env python3
"""Discover Autohome series-ID candidates through the suggestion endpoint."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
COLLECTION_DIR = BASE / "data" / "processed" / "review_collection"
SPLITS = BASE / "data" / "processed" / "splits"
AVAILABILITY = BASE / "data" / "reviews" / "processed" / "review_temporal_availability_by_series.csv"
CANDIDATES = COLLECTION_DIR / "autohome_id_candidates.csv"
RESOLUTIONS = COLLECTION_DIR / "autohome_id_resolutions.csv"
EXCEPTIONS = COLLECTION_DIR / "sentiment_resolution_exceptions.csv"
OUT = COLLECTION_DIR / "autohome_id_discovery.csv"
SUMMARY = COLLECTION_DIR / "autohome_id_discovery_summary.json"
API = "https://sou.api.autohome.com.cn/sug/_suggest?plat=pc&uid=&q={}"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def normalise(value: object) -> str:
    text = str(value or "").lower().replace("＋", "+")
    return re.sub(r"[\s\-－_（）()【】\[\]·./]", "", text)


def fetch(name: str) -> tuple[list[dict], str]:
    try:
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--http1.1", "--tlsv1.2",
            "--connect-timeout", "15", "--max-time", "30", "-A", USER_AGENT,
            API.format(quote(name)),
        ], check=True, capture_output=True, text=True, timeout=35)
        payload = json.loads(result.stdout)
        if int(payload.get("returncode", -1)) != 0:
            return [], f"returncode={payload.get('returncode')}: {payload.get('message', '')}"
        return (payload.get("result") or {}).get("data") or [], ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-name", action="append", default=[])
    parser.add_argument("--limit-series", type=int, default=None)
    return parser.parse_args()


def zero_coverage_gap() -> pd.DataFrame:
    """Return targets with no eligible review before the first test origin."""
    if not AVAILABILITY.exists():
        raise FileNotFoundError(f"Run 25_build_review_temporal_availability.py first: {AVAILABILITY}")
    test = pd.read_csv(SPLITS / "test.csv", usecols=["date"])
    first_test_month = pd.to_datetime(test["date"], errors="raise").min().to_period("M")
    column = f"reviews_available_before_{first_test_month.strftime('%Y_%m')}"
    availability = pd.read_csv(AVAILABILITY, usecols=["series_name", column])
    return availability.loc[availability[column].fillna(0).eq(0), ["series_name"]].sort_values("series_name")


def main() -> None:
    args = parse_args()
    gap = zero_coverage_gap()
    if args.series_name:
        requested = set(args.series_name)
        unknown = sorted(requested - set(gap["series_name"]))
        if unknown:
            raise ValueError(f"Requested series are not current zero-coverage targets: {unknown}")
        gap = gap.loc[gap["series_name"].isin(requested)]

    already_registered: set[str] = set()
    id_assignments: dict[int, set[str]] = {}
    if CANDIDATES.exists():
        registered = pd.read_csv(CANDIDATES)
        already_registered |= set(registered["series_name"].astype(str))
    if RESOLUTIONS.exists():
        resolved = pd.read_csv(RESOLUTIONS)
        resolved = resolved.loc[resolved["resolution_status"].eq("verified")]
        already_registered |= set(resolved["series_name"].astype(str))
        for sid, group in resolved.groupby("autohome_series_id"):
            id_assignments[int(sid)] = set(group["series_name"].astype(str))
    if EXCEPTIONS.exists():
        exceptions = pd.read_csv(EXCEPTIONS, usecols=["series_name"])
        already_registered |= set(exceptions["series_name"].astype(str))

    targets = gap.loc[~gap["series_name"].isin(already_registered)].copy()
    if args.limit_series:
        targets = targets.head(args.limit_series)

    rows: list[dict] = []
    for position, target in enumerate(targets["series_name"], start=1):
        print(f"[{position}/{len(targets)}] {target}", flush=True)
        suggestions, error = fetch(str(target))
        series = [item for item in suggestions if int(item.get("wordtype") or 0) == 3 and int(item.get("wordid") or 0) > 0]
        exact = [item for item in series if normalise(item.get("key")) == normalise(target)]
        candidate_id = int(exact[0]["wordid"]) if len(exact) == 1 else 0
        candidate_name = str(exact[0].get("key") or "") if len(exact) == 1 else ""
        conflicts = sorted(id_assignments.get(candidate_id, set()) - {str(target)}) if candidate_id else []
        if error:
            decision, reason = "error", error
        elif len(exact) > 1:
            decision, reason = "ambiguous", "multiple exact normalised-name series suggestions"
        elif len(exact) == 1 and conflicts:
            decision, reason = "conflict", f"candidate ID already assigned to: {', '.join(conflicts)}"
        elif len(exact) == 1:
            decision, reason = "ready_for_endpoint_verification", "unique official suggestion name match"
        elif series:
            decision, reason = "manual_or_missing", "no exact name match; do not auto-accept alias"
        else:
            decision, reason = "not_found", "no official series suggestion returned"
        rows.append({
            "series_name": target,
            "candidate_autohome_series_id": candidate_id or "",
            "candidate_source_name": candidate_name,
            "decision": decision,
            "decision_reason": reason,
            "series_suggestions": json.dumps(
                [{"name": item.get("key", ""), "id": int(item.get("wordid") or 0)} for item in series[:5]],
                ensure_ascii=False,
            ),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        })

    latest = pd.DataFrame(rows)
    if OUT.exists() and not latest.empty:
        previous = pd.read_csv(OUT)
        previous = previous.loc[~previous["series_name"].isin(set(latest["series_name"]))]
        persisted = pd.concat([previous, latest], ignore_index=True)
    elif OUT.exists():
        persisted = pd.read_csv(OUT)
    else:
        persisted = latest
    COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
    persisted.to_csv(OUT, index=False, encoding="utf-8-sig")
    summary = {
        "targets_this_run": int(len(latest)),
        "ready_for_endpoint_verification": int(latest["decision"].eq("ready_for_endpoint_verification").sum()) if len(latest) else 0,
        "manual_or_missing": int(latest["decision"].isin(["manual_or_missing", "ambiguous", "conflict", "not_found"]).sum()) if len(latest) else 0,
        "errors": int(latest["decision"].eq("error").sum()) if len(latest) else 0,
        "rule": "Only a unique exact normalised-name suggestion advances automatically; all candidates still require official review-endpoint verification.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
