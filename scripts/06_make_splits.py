#!/usr/bin/env python3
"""Compatibility CLI for the frozen chronological dataset split."""

from china_auto_market.features.splits import *  # noqa: F401,F403
from china_auto_market.features.splits import main


if __name__ == "__main__":
    main()
