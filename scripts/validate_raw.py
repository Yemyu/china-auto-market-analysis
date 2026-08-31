#!/usr/bin/env python3
"""Validate source-file parity for completed full raw batches."""

from __future__ import annotations

import argparse
import json
import os

from china_auto_market.ingestion.validation import validate_full_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--login-path",
        default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"),
        help="Encrypted mysql_config_editor login path.",
    )
    args = parser.parse_args()
    result = validate_full_sources(args.login_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
