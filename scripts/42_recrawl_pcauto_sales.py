#!/usr/bin/env python3
"""Recollect PCauto sales pages and compare them with the frozen raw snapshot.

The collector is deliberately non-destructive: HTML responses are cached outside
the versioned dataset, and all comparisons are written to ``processed/data_quality``.
It uses June and December as six-month page anchors, so a complete year normally
requires two requests rather than one request per series-month.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
RAW_SALES = ROOT / "data" / "raw" / "monthly_sales.csv"
QUEUE = ROOT / "data" / "processed" / "data_quality" / "sales_manual_verification_queue.csv"
RAW_CACHE = ROOT / "data" / "raw" / "pcauto_sales_recrawl"
QUALITY = ROOT / "data" / "processed" / "data_quality"
DEFAULT_START = pd.Timestamp("2022-01-01")
DEFAULT_END = pd.Timestamp("2026-06-01")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)


@dataclass
class FetchResult:
    source_url: str
    status: str
    http_status: int | None
    fetched_at: str
    cache_path: str
    content_sha256: str
    error: str


def normalize_name(value: object) -> str:
    """Normalize only harmless typography; do not merge semantically different names."""
    return re.sub(r"[\s\-－·•]", "", str(value)).casefold()


def parse_month(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.day != 1:
        raise ValueError(f"Month must be the first calendar day: {value}")
    return parsed


def page_anchors(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return June/December anchors covering an inclusive month range."""
    if start > end:
        raise ValueError("Start month must not be after end month")
    anchors: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        for month in (6, 12):
            anchor = pd.Timestamp(year=year, month=month, day=1)
            window_start = anchor - pd.DateOffset(months=5)
            if window_start <= end and anchor >= start:
                anchors.add(min(anchor, end) if year == end.year and anchor > end else anchor)
    anchors.add(end)
    return sorted(anchor for anchor in anchors if start <= anchor <= end)


def parse_sales_page(page_html: str, source_id: str) -> tuple[str, list[dict[str, object]]]:
    """Parse the named series row from one six-month PCauto sales page."""
    soup = BeautifulSoup(page_html, "lxml")
    row = soup.find("tr", class_=lambda value: value and "tr-ser" in value)
    if row is None:
        raise ValueError("sales row with class tr-ser not found")
    table = row.find_parent("table")
    if table is None:
        raise ValueError("sales row is not inside a table")

    month_pattern = re.compile(
        rf"/salescar/{re.escape(source_id)}/y(?P<year>\d{{4}})-m(?P<month>\d{{1,2}})/"
    )
    months: list[tuple[int, int]] = []
    for link in table.find_all("a", href=True):
        match = month_pattern.search(str(link["href"]))
        if match:
            key = (int(match.group("year")), int(match.group("month")))
            if key not in months:
                months.append(key)

    cells = row.find_all("td", recursive=False)
    if not months or len(cells) < len(months) + 1:
        raise ValueError(
            f"month/value shape mismatch: months={len(months)} direct_cells={len(cells)}"
        )
    source_name = html.unescape(cells[0].get_text(" ", strip=True))
    records: list[dict[str, object]] = []
    for (year, month), cell in zip(months, cells[1 : 1 + len(months)]):
        raw_value = cell.get_text(" ", strip=True).replace(",", "")
        number_match = re.search(r"-?\d+", raw_value)
        value = int(number_match.group()) if number_match else None
        records.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=1),
                "source_sales": value,
            }
        )
    return source_name, records


def fetch_page(
    session: requests.Session,
    source_id: str,
    anchor: pd.Timestamp,
    timeout: float,
    retries: int,
    proxy: str | None,
    refresh: bool,
) -> tuple[FetchResult, str | None]:
    url = (
        f"https://price.pcauto.com.cn/salescar/{source_id}/"
        f"y{anchor.year}-m{anchor.month}/"
    )
    cache_path = RAW_CACHE / source_id / f"y{anchor.year}-m{anchor.month}.html"
    if cache_path.exists() and not refresh:
        body = cache_path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        checked_at = datetime.fromtimestamp(cache_path.stat().st_mtime).astimezone().isoformat(
            timespec="seconds"
        )
        return (
            FetchResult(
                url,
                "cached",
                200,
                checked_at,
                str(cache_path.relative_to(ROOT)),
                digest,
                "",
            ),
            body,
        )

    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    request_proxies = {"http": proxy, "https": proxy} if proxy else None
    last_error = ""
    last_status: int | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
                proxies=request_proxies,
            )
            last_status = int(response.status_code)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding or "utf-8"
            body = response.text
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(body, encoding="utf-8")
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            return (
                FetchResult(
                    url,
                    "ok",
                    last_status,
                    checked_at,
                    str(cache_path.relative_to(ROOT)),
                    digest,
                    "",
                ),
                body,
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    return (
        FetchResult(
            url,
            "error",
            last_status,
            checked_at,
            str(cache_path.relative_to(ROOT)),
            "",
            last_error,
        ),
        None,
    )


def select_roster(
    raw: pd.DataFrame,
    queue: pd.DataFrame,
    requested_names: list[str],
    limit_series: int,
) -> pd.DataFrame:
    metadata = raw[
        ["series_name", "series_id", "source_series_id", "brand"]
    ].drop_duplicates("series_name")
    queue_fields = queue[
        [
            "series_name",
            "in_model_cohort",
            "verification_priority",
            "risk_score",
            "verification_reason",
        ]
    ]
    ranked = metadata.merge(queue_fields, on="series_name", how="left", validate="one_to_one")
    ranked["in_model_cohort"] = ranked["in_model_cohort"].fillna(False).astype(bool)
    ranked["verification_priority"] = ranked["verification_priority"].fillna(
        "reviewed_or_not_queued"
    )
    ranked["risk_score"] = ranked["risk_score"].fillna(0).astype(int)
    ranked["verification_reason"] = ranked["verification_reason"].fillna(
        "not present in the current external-verification queue"
    )
    ranked["priority_order"] = ranked["verification_priority"].map(
        {"critical": 0, "high": 1, "medium": 2}
    ).fillna(9)
    ranked = ranked.sort_values(
        ["priority_order", "in_model_cohort", "risk_score", "series_name"],
        ascending=[True, False, False, True],
    )
    if requested_names:
        unknown = sorted(set(requested_names) - set(metadata["series_name"]))
        if unknown:
            raise ValueError(f"Requested series are absent from the sales snapshot: {unknown}")
        ranked = ranked[ranked["series_name"].isin(requested_names)]
        requested_order = {name: position for position, name in enumerate(requested_names)}
        ranked = ranked.assign(
            requested_order=ranked["series_name"].map(requested_order)
        ).sort_values("requested_order")
    else:
        ranked = ranked[ranked["verification_priority"].isin(["critical", "high", "medium"])]
        ranked = ranked.head(limit_series)
    if ranked["source_series_id"].isna().any():
        missing = ranked.loc[ranked["source_series_id"].isna(), "series_name"].tolist()
        raise ValueError(f"Selected series lack PCauto source IDs: {missing}")
    return ranked[
        [
            "series_name",
            "series_id",
            "source_series_id",
            "brand",
            "in_model_cohort",
            "verification_priority",
            "risk_score",
            "verification_reason",
        ]
    ].reset_index(drop=True)


def collapse_observations(observations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "series_name",
        "date",
        "source_sales",
        "source_observation_count",
        "source_value_count",
        "source_identity_match",
        "source_urls",
    ]
    if observations.empty:
        return pd.DataFrame(columns=columns)

    def collapse(group: pd.DataFrame) -> pd.Series:
        values = sorted(group["source_sales"].dropna().astype(int).unique().tolist())
        return pd.Series(
            {
                "source_sales": values[0] if len(values) == 1 else pd.NA,
                "source_observation_count": len(group),
                "source_value_count": len(values),
                "source_identity_match": bool(group["source_identity_match"].all()),
                "source_urls": " | ".join(sorted(group["source_url"].unique())),
            }
        )

    collapsed = observations.groupby(["series_name", "date"], as_index=False).apply(
        collapse, include_groups=False
    )
    return collapsed[columns]


def build_diff(
    raw: pd.DataFrame,
    roster: pd.DataFrame,
    collapsed: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    selected = raw[raw["series_name"].isin(roster["series_name"])].copy()
    selected["date"] = pd.to_datetime(
        dict(year=selected["year"], month=selected["month"], day=1)
    )
    selected = selected[selected["date"].between(start, end)]
    selected = selected[
        ["series_name", "series_id", "source_series_id", "brand", "date", "monthly_sales"]
    ].rename(columns={"monthly_sales": "raw_sales"})
    positive_span = (
        selected[selected["raw_sales"].gt(0)]
        .groupby("series_name", as_index=False)
        .agg(
            raw_first_positive_date=("date", "min"),
            raw_last_positive_date=("date", "max"),
        )
    )
    diff = selected.merge(collapsed, on=["series_name", "date"], how="left")
    diff = diff.merge(positive_span, on="series_name", how="left")
    diff["source_value_count"] = diff["source_value_count"].fillna(0).astype(int)

    def classify(row: pd.Series) -> str:
        if pd.isna(row["source_observation_count"]):
            return "not_retrieved"
        if row["source_value_count"] > 1:
            return "source_conflict"
        if row["source_value_count"] == 0 or pd.isna(row["source_sales"]):
            return "source_missing"
        if not bool(row["source_identity_match"]):
            return "identity_mismatch"
        raw_value, source_value = int(row["raw_sales"]), int(row["source_sales"])
        if raw_value == source_value:
            return "exact_match"
        if raw_value == 0 and source_value > 0:
            return "raw_zero_source_positive"
        if raw_value > 0 and source_value == 0:
            return "raw_positive_source_zero"
        if raw_value > 0 and source_value > 0:
            return "nonzero_mismatch"
        return "other_mismatch"

    diff["comparison_status"] = diff.apply(classify, axis=1)

    def locate_source_missing(row: pd.Series) -> str:
        if row["comparison_status"] != "source_missing":
            return ""
        if pd.isna(row["raw_first_positive_date"]):
            return "no_positive_reference"
        if row["date"] < row["raw_first_positive_date"]:
            return "before_first_positive"
        if row["date"] > row["raw_last_positive_date"]:
            return "after_last_positive"
        return "inside_positive_span"

    diff["source_missing_position"] = diff.apply(locate_source_missing, axis=1)
    diff["sales_delta"] = pd.to_numeric(diff["source_sales"], errors="coerce") - diff[
        "raw_sales"
    ]
    return diff.sort_values(["series_name", "date"]).reset_index(drop=True)


def write_outputs(
    roster: pd.DataFrame,
    manifests: list[dict[str, object]],
    observations: list[dict[str, object]],
    diff: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_prefix: str,
) -> dict[str, object]:
    QUALITY.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifests)
    observation_columns = [
        "series_name",
        "source_series_id",
        "source_series_name",
        "source_identity_match",
        "date",
        "source_sales",
        "source_url",
        "fetched_at",
        "content_sha256",
    ]
    observation_frame = pd.DataFrame(observations, columns=observation_columns)
    roster.to_csv(
        QUALITY / f"{output_prefix}_roster.csv", index=False, encoding="utf-8-sig"
    )
    manifest.to_csv(
        QUALITY / f"{output_prefix}_manifest.csv", index=False, encoding="utf-8-sig"
    )
    observation_frame.to_csv(
        QUALITY / f"{output_prefix}_observations.csv", index=False, encoding="utf-8-sig"
    )
    diff.to_csv(
        QUALITY / f"{output_prefix}_diff.csv", index=False, encoding="utf-8-sig"
    )
    page_counts = manifest["fetch_status"].value_counts().to_dict() if len(manifest) else {}
    comparison_counts = diff["comparison_status"].value_counts().to_dict()
    missing_position_counts = (
        diff.loc[diff["source_missing_position"].ne(""), "source_missing_position"]
        .value_counts()
        .to_dict()
    )
    successful_pages = int(sum(page_counts.get(key, 0) for key in ("ok", "cached")))
    summary = {
        "schema_version": "v1",
        "source": "PCauto sales series pages",
        "period": {"start": start.strftime("%Y-%m"), "end": end.strftime("%Y-%m")},
        "selected_series": int(roster["series_name"].nunique()),
        "requested_pages": int(len(manifest)),
        "successful_pages": successful_pages,
        "fetch_status_counts": {str(k): int(v) for k, v in page_counts.items()},
        "parsed_observations": int(len(observation_frame)),
        "comparison_status_counts": {
            str(k): int(v) for k, v in comparison_counts.items()
        },
        "source_missing_position_counts": {
            str(k): int(v) for k, v in missing_position_counts.items()
        },
        "acquisition_status": (
            "ok" if successful_pages else "blocked_no_successful_pages"
        ),
        "decision": (
            "A source dash means no numeric value was supplied; it is not a confirmed zero. "
            "Keep not-retrieved pages separate from source dashes, and review identity mismatches, "
            "source conflicts, and changed values before creating any correction overlay. Never "
            "overwrite data/raw/monthly_sales.csv."
        ),
    }
    (QUALITY / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START.strftime("%Y-%m-01"))
    parser.add_argument("--end", default=DEFAULT_END.strftime("%Y-%m-01"))
    parser.add_argument("--limit-series", type=int, default=10)
    parser.add_argument("--series-name", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay-min", type=float, default=1.0)
    parser.add_argument("--delay-max", type=float, default=1.5)
    parser.add_argument("--proxy", default=None, help="Optional explicit HTTP/SOCKS proxy URL.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="pcauto_recrawl_pilot",
        help="Filename prefix under processed/data_quality (letters, numbers, underscores).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Request only the latest anchor for the first selected series.",
    )
    args = parser.parse_args()
    if args.limit_series < 1 or args.timeout <= 0 or args.retries < 0:
        raise ValueError("Series limit and timeout must be positive; retries cannot be negative")
    if args.delay_min < 0 or args.delay_max < args.delay_min:
        raise ValueError("Invalid delay range")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.output_prefix):
        raise ValueError("--output-prefix may contain only letters, numbers, and underscores")

    start, end = parse_month(args.start), parse_month(args.end)
    raw = pd.read_csv(RAW_SALES, low_memory=False)
    queue = pd.read_csv(QUEUE)
    roster = select_roster(raw, queue, args.series_name, args.limit_series)
    anchors = page_anchors(start, end)
    if args.preflight_only:
        roster = roster.head(1)
        anchors = [anchors[-1]]

    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifests: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    total_pages = len(roster) * len(anchors)
    page_number = 0
    for _, series in roster.iterrows():
        name, source_id = str(series["series_name"]), str(series["source_series_id"])
        for anchor in anchors:
            page_number += 1
            print(
                f"[{page_number}/{total_pages}] {name} {anchor.strftime('%Y-%m')}",
                flush=True,
            )
            fetched, body = fetch_page(
                session,
                source_id,
                anchor,
                args.timeout,
                args.retries,
                args.proxy,
                args.refresh,
            )
            fetched_fields = asdict(fetched)
            fetched_fields["fetch_status"] = fetched_fields.pop("status")
            manifest = {
                "series_name": name,
                "source_series_id": source_id,
                "anchor_month": anchor.strftime("%Y-%m-01"),
                **fetched_fields,
                "source_series_name": "",
                "identity_match": False,
                "parsed_months": 0,
                "parse_error": "",
            }
            if body is not None:
                try:
                    source_name, records = parse_sales_page(body, source_id)
                    identity_match = normalize_name(source_name) == normalize_name(name)
                    manifest.update(
                        {
                            "source_series_name": source_name,
                            "identity_match": identity_match,
                            "parsed_months": len(records),
                        }
                    )
                    for record in records:
                        if start <= record["date"] <= end:
                            observations.append(
                                {
                                    "series_name": name,
                                    "source_series_id": source_id,
                                    "source_series_name": source_name,
                                    "source_identity_match": identity_match,
                                    "date": record["date"],
                                    "source_sales": record["source_sales"],
                                    "source_url": fetched.source_url,
                                    "fetched_at": fetched.fetched_at,
                                    "content_sha256": fetched.content_sha256,
                                }
                            )
                except (ValueError, TypeError) as exc:
                    manifest["parse_error"] = f"{type(exc).__name__}: {exc}"
            manifests.append(manifest)
            if page_number < total_pages:
                time.sleep(random.uniform(args.delay_min, args.delay_max))

    observation_frame = pd.DataFrame(observations)
    collapsed = collapse_observations(observation_frame)
    diff = build_diff(raw, roster, collapsed, start, end)
    summary = write_outputs(
        roster,
        manifests,
        observations,
        diff,
        start,
        end,
        args.output_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
