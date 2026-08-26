#!/usr/bin/env python3
"""Verify Autohome series-ID candidates against the review endpoint."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
COLLECTION_DIR = BASE / "data" / "processed" / "review_collection"
CANDIDATES = COLLECTION_DIR / "autohome_id_candidates.csv"
RESOLUTIONS = COLLECTION_DIR / "autohome_id_resolutions.csv"
SPLITS = BASE / "data" / "processed" / "splits"
AVAILABILITY = BASE / "data" / "reviews" / "processed" / "review_temporal_availability_by_series.csv"
AUDIT = COLLECTION_DIR / "autohome_id_candidate_audit.csv"
SUMMARY = COLLECTION_DIR / "autohome_id_candidate_audit_summary.json"

API_URL = "https://koubeiipv6.app.autohome.com.cn/pc/series/list"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def normalise_name(value: object) -> str:
    text = str(value or "").lower().replace("＋", "+")
    return re.sub(r"[\s\-－_（）()【】\[\]·./]", "", text)


def name_relation(target: str, source: str) -> tuple[str, float]:
    left, right = normalise_name(target), normalise_name(source)
    score = SequenceMatcher(None, left, right).ratio() if left and right else 0.0
    if left == right and left:
        return "exact", score
    if left and right and (left in right or right in left):
        return "contains", score
    return "different", score


def endpoint_url(series_id: int) -> str:
    query = urlencode({
        "pm": 3, "seriesId": int(series_id), "pageIndex": 1, "pageSize": 20,
        "yearid": 0, "ge": 0, "seriesSummaryKey": 0, "order": 0,
    })
    return f"{API_URL}?{query}"


def verify_candidate(series_id: int) -> tuple[dict | None, str]:
    try:
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--http1.1", "--tlsv1.2",
            "--connect-timeout", "15", "--max-time", "30", "-A", USER_AGENT,
            "-e", "https://k.autohome.com.cn/", endpoint_url(series_id),
        ], check=True, capture_output=True, text=True, timeout=35)
        payload = json.loads(result.stdout)
        if int(payload.get("returncode", -1)) != 0:
            return None, f"source returncode={payload.get('returncode')}: {payload.get('message', '')}"
        source = payload.get("result") or {}
        returned_id = int(source.get("seriesid") or 0)
        if returned_id != int(series_id):
            return None, f"requested ID {series_id}, endpoint returned {returned_id}"
        return {
            "source_series_name": str(source.get("seriesname") or ""),
            "api_review_total": int(source.get("rowcount") or 0),
            "api_page_count": int(source.get("pagecount") or 0),
        }, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series-name",
        action="append",
        default=[],
        help="Only audit this target series (repeatable). Existing audit rows are preserved.",
    )
    return parser.parse_args()


def zero_coverage_targets() -> set[str]:
    """Return targets without an eligible review before the first test origin."""
    if not AVAILABILITY.exists():
        raise FileNotFoundError(f"Run 25_build_review_temporal_availability.py first: {AVAILABILITY}")
    test = pd.read_csv(SPLITS / "test.csv", usecols=["date"])
    first_test_month = pd.to_datetime(test["date"], errors="raise").min().to_period("M")
    column = f"reviews_available_before_{first_test_month.strftime('%Y_%m')}"
    availability = pd.read_csv(AVAILABILITY, usecols=["series_name", column])
    return set(availability.loc[availability[column].fillna(0).eq(0), "series_name"])


def main() -> None:
    args = parse_args()
    if not CANDIDATES.exists():
        raise FileNotFoundError(CANDIDATES)
    candidates = pd.read_csv(CANDIDATES)
    required = {"series_name", "candidate_autohome_series_id", "expected_source_name", "evidence_url", "evidence_source"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate register missing columns: {sorted(missing)}")
    candidates["candidate_autohome_series_id"] = pd.to_numeric(
        candidates["candidate_autohome_series_id"], errors="raise"
    ).astype(int)
    if candidates.duplicated(["series_name", "candidate_autohome_series_id"]).any():
        raise ValueError("Duplicate target/candidate pairs in candidate register")
    if args.series_name:
        requested = set(args.series_name)
        found = set(candidates.loc[candidates["series_name"].isin(requested), "series_name"])
        missing_targets = sorted(requested - found)
        if missing_targets:
            raise ValueError(f"Requested target series not in candidate register: {missing_targets}")
        candidates = candidates.loc[candidates["series_name"].isin(requested)].copy()

    remaining = zero_coverage_targets()
    assignments: dict[int, set[str]] = {}
    if RESOLUTIONS.exists():
        resolved = pd.read_csv(RESOLUTIONS)
        resolved = resolved.loc[resolved["resolution_status"].eq("verified")]
        for sid, group in resolved.groupby("autohome_series_id"):
            assignments[int(sid)] = set(group["series_name"].astype(str))

    rows: list[dict] = []
    for position, row in candidates.reset_index(drop=True).iterrows():
        target, sid = str(row.series_name), int(row.candidate_autohome_series_id)
        print(f"[{position + 1}/{len(candidates)}] {target} -> {sid}", flush=True)
        result, error = verify_candidate(sid)
        source_name = result["source_series_name"] if result else ""
        relation, similarity = name_relation(target, source_name)
        expected_relation, _ = name_relation(str(row.expected_source_name), source_name)
        conflicting_targets = sorted(assignments.get(sid, set()) - {target})
        in_gap = target in remaining
        if error:
            decision, reason = "error", error
        elif not in_gap:
            decision, reason = "out_of_scope", "target is no longer in the zero-coverage gap"
        elif conflicting_targets:
            decision = "manual_review"
            reason = f"candidate ID already assigned to: {', '.join(conflicting_targets)}"
        elif relation == "exact":
            decision, reason = "verified", "official endpoint ID and target name agree exactly"
        elif expected_relation == "exact":
            decision = "manual_review"
            reason = "official endpoint matches the documented alias/renamed display name"
        elif relation == "contains" and similarity >= 0.6:
            decision, reason = "manual_review", "target and source names partially contain one another"
        else:
            decision, reason = "rejected", "official endpoint series name does not match the target"
        rows.append({
            **row.to_dict(), "source_series_name": source_name,
            "name_relation": relation, "name_similarity": round(similarity, 4),
            "api_review_total": result["api_review_total"] if result else 0,
            "api_page_count": result["api_page_count"] if result else 0,
            "existing_id_assignments": "|".join(conflicting_targets),
            "target_in_zero_coverage_gap": in_gap, "decision": decision,
            "decision_reason": reason, "checked_at": datetime.now().isoformat(timespec="seconds"),
        })

    audit = pd.DataFrame(rows)
    if AUDIT.exists():
        previous = pd.read_csv(AUDIT)
        keys = set(zip(audit["series_name"], audit["candidate_autohome_series_id"]))
        previous_keys = list(zip(previous["series_name"], previous["candidate_autohome_series_id"]))
        previous = previous.loc[[key not in keys for key in previous_keys]]
        persisted_audit = pd.concat([previous, audit], ignore_index=True)
    else:
        persisted_audit = audit
    persisted_audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")
    summary = {
        "candidate_pairs_in_this_run": int(len(audit)),
        "candidate_pairs_in_persisted_audit": int(len(persisted_audit)),
        "verified_exact": int(audit["decision"].eq("verified").sum()),
        "manual_review": int(audit["decision"].eq("manual_review").sum()),
        "rejected": int(audit["decision"].eq("rejected").sum()),
        "errors": int(audit["decision"].eq("error").sum()),
        "zero_coverage_target_series": int(len(remaining)),
        "potential_review_total_for_verified_exact": int(audit.loc[audit["decision"].eq("verified"), "api_review_total"].sum()),
        "rule": "Only exact target/source name matches are auto-verified. Aliases, renames and one-ID-to-many-target conflicts require explicit review.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
