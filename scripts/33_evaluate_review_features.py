#!/usr/bin/env python3
"""Compatibility CLI for the fixed-origin review-feature evaluation."""

from china_auto_market.forecasting.review_evaluation import *  # noqa: F401,F403
from china_auto_market.forecasting.review_evaluation import main


if __name__ == "__main__":
    main()
