#!/usr/bin/env python3
"""Compatibility CLI for the operational rolling-origin forecast."""

from china_auto_market.forecasting.rolling_origin import *  # noqa: F401,F403
from china_auto_market.forecasting.rolling_origin import main


if __name__ == "__main__":
    main()
