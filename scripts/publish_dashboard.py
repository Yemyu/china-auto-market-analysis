#!/usr/bin/env python3
"""Build and validate static dashboard data from one approved pipeline manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from china_auto_market.publishing.release import publish_dashboard, result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace app/static/data after all checks pass.",
    )
    args = parser.parse_args()
    result = publish_dashboard(args.manifest, apply=args.apply)
    print(json.dumps(result_dict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
