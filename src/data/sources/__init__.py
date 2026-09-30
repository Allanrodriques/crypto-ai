"""New public data sources for V2 experiments.

Each module here is a thin, well-documented downloader.  They deliberately do
*not* do feature engineering and they do *not* align timestamps - that is
:mod:`src.alignment.availability`'s job - so the no-look-ahead rule has one
implementation.

Availability is documented per source in :data:`SOURCE_CONTRACTS`, derived from
what the endpoints actually return (see ``reports/data_availability.csv``).
"""

from __future__ import annotations

from src.data.sources.funding import FUNDING_PATH, fetch_funding_rates
from src.data.sources.futures import FUTURES_KLINES_PATH, fetch_futures_klines
from src.data.sources.sentiment import FEAR_GREED_URL, fetch_fear_greed

__all__ = [
    "FUNDING_PATH",
    "FUTURES_KLINES_PATH",
    "FEAR_GREED_URL",
    "fetch_funding_rates",
    "fetch_futures_klines",
    "fetch_fear_greed",
]
