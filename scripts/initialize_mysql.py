#!/usr/bin/env python3
"""Inspect or initialize the four project MySQL databases safely."""

from __future__ import annotations

import argparse
import json
import os

from china_auto_market.warehouse.schema import (
    TARGET_DATABASES,
    apply_ddl,
    existing_target_databases,
    query,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--login-path",
        default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"),
        help="Encrypted mysql_config_editor login path; no password is accepted here.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply ordered CREATE IF NOT EXISTS DDL files. Default mode is read-only.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow idempotent re-application after confirming existing target databases are this project's.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = query(
        args.login_path,
        "SELECT VERSION(), @@character_set_server, @@collation_server, @@sql_mode;",
    ).strip().split("\t")
    existing_before = sorted(existing_target_databases(args.login_path))
    report: dict[str, object] = {
        "server": {
            "version": server[0],
            "character_set": server[1],
            "collation": server[2],
            "sql_mode": server[3],
        },
        "target_databases": list(TARGET_DATABASES),
        "existing_before": existing_before,
        "mode": "apply" if args.apply else "check_only",
    }
    if args.apply:
        applied = apply_ddl(args.login_path, allow_existing=args.allow_existing)
        report["ddl_files"] = [str(path) for path in applied]
        report["existing_after"] = sorted(existing_target_databases(args.login_path))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
