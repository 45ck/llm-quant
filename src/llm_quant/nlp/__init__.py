"""NLP data infrastructure for llm-quant.

Provides three components:

- :class:`TextClassifier` — Claude API classifier for 10-K / earnings call sentences
- :class:`EdgarFetcher` — SEC EDGAR 10-K downloader and MD&A extractor
- :class:`FomcFetcher` — Federal Reserve FOMC minutes fetcher and hedging scorer
"""

from llm_quant.nlp.edgar_fetcher import EdgarFetcher
from llm_quant.nlp.fomc_fetcher import FomcFetcher
from llm_quant.nlp.text_classifier import TextClassifier

__all__ = [
    "EdgarFetcher",
    "FomcFetcher",
    "TextClassifier",
]
