#!/usr/bin/env python3
"""Load immutable source snapshots into auto_raw with batch lineage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from china_auto_market.ingestion.raw_loaders import DEFAULT_SOURCES, LOADERS, result_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", *LOADERS],
        default="all",
        help="Source contract to load; 'all' performs a full initialization in dependency-free order.",
    )
    parser.add_argument("--mode", choices=["full", "incremental", "backfill"], default="full")
    parser.add_argument(
        "--month",
        help="Optional YYYY-MM partition. Sales/corrections use the month, config uses its year, reviews use publish month.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Override one dataset's source file; valid only when --dataset is not 'all'.",
    )
    parser.add_argument(
        "--login-path",
        default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"),
        help="Encrypted mysql_config_editor login path; no plaintext password is accepted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source and args.dataset == "all":
        raise ValueError("--source requires one explicit --dataset")
    datasets = list(LOADERS) if args.dataset == "all" else [args.dataset]
    results = []
    for dataset in datasets:
        source = args.source if args.source else DEFAULT_SOURCES[dataset]
        result = LOADERS[dataset](
            args.login_path,
            source,
            mode=args.mode,
            month=args.month,
        )
        results.append(result_dict(result))
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
