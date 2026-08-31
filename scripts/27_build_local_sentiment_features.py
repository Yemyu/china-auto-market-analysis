#!/usr/bin/env python3
"""Compatibility CLI for local review sentiment feature generation."""

from china_auto_market.reviews.local_sentiment import *  # noqa: F401,F403
from china_auto_market.reviews.local_sentiment import main


if __name__ == "__main__":
    main()
