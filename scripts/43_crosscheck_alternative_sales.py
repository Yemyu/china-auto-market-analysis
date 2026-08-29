#!/usr/bin/env python3
"""Cross-check the repaired sales panel against a second historical-sales site.

The job is intentionally non-destructive.  It uses an explicit, reviewable
series map, caches source pages outside the versioned dataset, and writes only
comparison artifacts under ``processed/data_quality``.  A name match is not
enough: the returned page name and brand/manufacturer text must also agree with
the mapping before observations are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from _sales_repair import apply_verified_sales_corrections, load_sales_correction_register


ROOT = Path(__file__).resolve().parent.parent
RAW_SALES = ROOT / "data" / "raw" / "monthly_sales.csv"
QUALITY = ROOT / "data" / "processed" / "data_quality"
SOURCE_MAP = QUALITY / "alternative_sales_source_map.csv"
RAW_CACHE = ROOT / "data" / "raw" / "alternative_sales_recrawl" / "icauto"
OUTPUT_PREFIX = "alternative_sales_crosscheck"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)


def normalize_name(value: object) -> str:
    """Remove harmless spacing/punctuation without collapsing generations."""
    return re.sub(r"[\s\-－·•]", "", str(value)).casefold()


def parse_icauto_page(page_html: str) -> tuple[str, str, list[dict[str, object]]]:
    soup = BeautifulSoup(page_html, "lxml")
    heading = soup.find("h1")
    if heading is None:
        raise ValueError("source page has no h1")
    heading_text = heading.get_text(" ", strip=True)
    source_name = re.sub(r"\s*\d{4}年.*$", "", heading_text).strip()

    body_text = soup.get_text(" ", strip=True)
    brand_match = re.search(r"品牌国别[：:]\s*([^\s]+)", body_text)
    manufacturer_match = re.search(r"车厂[：:]\s*([^\s]+)", body_text)
    identity_text = " ".join(
        value for value in [
            brand_match.group(1) if brand_match else "",
            manufacturer_match.group(1) if manufacturer_match else "",
        ] if value
    )

    section = soup.find(
        lambda tag: tag.name in {"h2", "h3"}
        and "销量同比数据" in tag.get_text(" ", strip=True)
    )
    if section is None:
        raise ValueError("year-by-month sales section not found")
    table = section.find_next("table")
    if table is None:
        raise ValueError("year-by-month sales table not found")

    records: list[dict[str, object]] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True).replace(",", "") for cell in row.find_all("td")]
        if len(cells) != 13 or not re.fullmatch(r"\d{4}年", cells[0]):
            continue
        year = int(cells[0][:-1])
        for month, raw_value in enumerate(cells[1:], start=1):
            if not re.fullmatch(r"\d+", raw_value):
                continue
            records.append(
                {
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "source_sales": int(raw_value),
                }
            )
    if not records:
        raise ValueError("year-by-month table contains no numeric observations")
    return source_name, identity_text, records


def cache_path_for(url: str) -> Path:
    match = re.search(r"/car_(\d+)\.html", url)
    name = f"car_{match.group(1)}.html" if match else hashlib.sha256(url.encode()).hexdigest() + ".html"
    return RAW_CACHE / name


def fetch_page(
    session: requests.Session,
    url: str,
    timeout: float,
    proxy: str | None,
    refresh: bool,
) -> tuple[str, str, int | None, str, str]:
    cache_path = cache_path_for(url)
    if cache_path.exists() and not refresh:
        body = cache_path.read_text(encoding="utf-8", errors="replace")
        digest = hashlib.sha256(body.encode()).hexdigest()
        return "cached", body, 200, str(cache_path.relative_to(ROOT)), digest

    response = session.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
        proxies={"http": proxy, "https": proxy} if proxy else None,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    body = response.text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode()).hexdigest()
    return "ok", body, int(response.status_code), str(cache_path.relative_to(ROOT)), digest


def compare_observations(
    current_sales: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    panel = current_sales[["series_name", "date", "monthly_sales"]].rename(
        columns={"monthly_sales": "current_sales"}
    )
    diff = observations.merge(
        panel, on=["series_name", "date"], how="left", validate="one_to_one"
    )
    diff["difference"] = diff["source_sales"] - diff["current_sales"]
    diff["comparison"] = "exact"
    diff.loc[diff["current_sales"].isna(), "comparison"] = "outside_panel"
    diff.loc[
        diff["current_sales"].eq(0) & diff["source_sales"].gt(0), "comparison"
    ] = "current_zero_source_positive"
    diff.loc[
        diff["current_sales"].gt(0) & diff["source_sales"].eq(0), "comparison"
    ] = "source_zero_current_positive"
    diff.loc[
        diff["current_sales"].gt(0)
        & diff["source_sales"].gt(0)
        & diff["difference"].ne(0),
        "comparison",
    ] = "positive_value_conflict"
    return diff.sort_values(["series_name", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=None, help="Optional HTTP/SOCKS proxy URL")
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    source_map = pd.read_csv(SOURCE_MAP).fillna("")
    raw = pd.read_csv(RAW_SALES, low_memory=False)
    raw["date"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1))
    current, _ = apply_verified_sales_corrections(raw, load_sales_correction_register())

    session = requests.Session()
    manifests: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")

    for row in source_map.itertuples(index=False):
        base = {
            "role": row.role,
            "series_name": row.series_name,
            "expected_source_name": row.expected_source_name,
            "expected_identity_text": row.expected_identity_text,
            "source_name": row.source_name,
            "source_url": row.source_url,
            "mapping_status": row.mapping_status,
            "fetched_at": fetched_at,
        }
        if row.mapping_status != "mapped" or not row.source_url:
            manifests.append({**base, "fetch_status": "not_mapped", "identity_status": "not_checked", "http_status": "", "cache_path": "", "content_sha256": "", "error": ""})
            continue
        try:
            status, body, http_status, cache_path, digest = fetch_page(
                session, row.source_url, args.timeout, args.proxy, args.refresh
            )
            returned_name, identity_text, parsed = parse_icauto_page(body)
            name_ok = normalize_name(returned_name) == normalize_name(row.expected_source_name)
            identity_ok = normalize_name(row.expected_identity_text) in normalize_name(identity_text)
            identity_status = "verified" if name_ok and identity_ok else "mismatch"
            manifests.append(
                {
                    **base,
                    "fetch_status": status,
                    "identity_status": identity_status,
                    "returned_source_name": returned_name,
                    "returned_identity_text": identity_text,
                    "http_status": http_status,
                    "cache_path": cache_path,
                    "content_sha256": digest,
                    "error": "" if identity_status == "verified" else "returned name or brand/manufacturer did not match explicit map",
                }
            )
            if identity_status == "verified":
                records.extend(
                    {
                        **base,
                        "returned_source_name": returned_name,
                        **record,
                    }
                    for record in parsed
                )
        except (requests.RequestException, ValueError) as exc:
            manifests.append(
                {
                    **base,
                    "fetch_status": "error",
                    "identity_status": "not_checked",
                    "returned_source_name": "",
                    "returned_identity_text": "",
                    "http_status": "",
                    "cache_path": str(cache_path_for(row.source_url).relative_to(ROOT)),
                    "content_sha256": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest = pd.DataFrame(manifests)
    observations = pd.DataFrame(records)
    if len(observations):
        observations["date"] = pd.to_datetime(observations["date"])
        diff = compare_observations(current, observations)
        observations["date"] = observations["date"].dt.strftime("%Y-%m-%d")
        diff["date"] = diff["date"].dt.strftime("%Y-%m-%d")
    else:
        diff = pd.DataFrame(columns=["series_name", "date", "source_sales", "current_sales", "difference", "comparison"])

    QUALITY.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(QUALITY / f"{OUTPUT_PREFIX}_manifest.csv", index=False, encoding="utf-8-sig")
    observations.to_csv(QUALITY / f"{OUTPUT_PREFIX}_observations.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(QUALITY / f"{OUTPUT_PREFIX}_diff.csv", index=False, encoding="utf-8-sig")

    control = diff[diff.get("role", pd.Series(dtype=str)).eq("calibration")]
    comparable_control = control[control["comparison"].ne("outside_panel")]
    summary = {
        "schema_version": "v1",
        "source": "ICAUTO historical year-by-month tables",
        "generated_at": fetched_at,
        "mapped_series": int(source_map["mapping_status"].eq("mapped").sum()),
        "unmapped_series": int(source_map["mapping_status"].ne("mapped").sum()),
        "verified_identity_pages": int(manifest["identity_status"].eq("verified").sum()),
        "failed_or_mismatched_pages": int((~manifest["identity_status"].isin(["verified", "not_checked"])).sum() + manifest["fetch_status"].eq("error").sum()),
        "parsed_observations": int(len(observations)),
        "comparison_counts": diff["comparison"].value_counts().to_dict() if len(diff) else {},
        "calibration": {
            "comparable_months": int(len(comparable_control)),
            "exact_months": int(comparable_control["comparison"].eq("exact").sum()),
            "exact_rate": float(comparable_control["comparison"].eq("exact").mean()) if len(comparable_control) else None,
        },
        "decision": "Use exact name plus brand/manufacturer and overlapping monthly fingerprints. Cross-source differences remain review candidates; do not overwrite the raw snapshot.",
    }
    (QUALITY / f"{OUTPUT_PREFIX}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
