#!/usr/bin/env python3
"""Atomically rebuild auto_staging and enforce critical data-quality gates."""

from __future__ import annotations

import argparse
import json
import os

from china_auto_market.quality.staging import rebuild_staging, result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--login-path",
        default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"),
        help="Encrypted mysql_config_editor login path.",
    )
    args = parser.parse_args()
    result = rebuild_staging(args.login_path)
    print(json.dumps(result_dict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
