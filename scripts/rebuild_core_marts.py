#!/usr/bin/env python3
"""Rebuild the three business marts after source parity checks."""

from __future__ import annotations

import argparse
import json
import os

from china_auto_market.marts.build import rebuild_core_marts, result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--login-path",
        default=os.environ.get("MYSQL_LOGIN_PATH", "local-auto"),
        help="Encrypted mysql_config_editor login path.",
    )
    args = parser.parse_args()
    print(json.dumps(result_dict(rebuild_core_marts(args.login_path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
