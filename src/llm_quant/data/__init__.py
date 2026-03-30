"""Data pipeline: market data (yfinance) and macro data (FRED)."""

from llm_quant.data.fred_fetcher import FRED_SERIES, FredFetcher

__all__ = ["FredFetcher", "FRED_SERIES"]
