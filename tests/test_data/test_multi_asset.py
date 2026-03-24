"""Tests for multi-asset (equity, crypto, forex) support."""

from __future__ import annotations

from llm_quant.config import AssetEntry, ETFEntry, RiskLimits, UniverseConfig


class TestAssetEntry:
    """AssetEntry model and backward compat."""

    def test_default_asset_class_is_equity(self):
        entry = AssetEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad_market")
        assert entry.asset_class == "equity"

    def test_crypto_asset_class(self):
        entry = AssetEntry(symbol="BTC-USD", name="Bitcoin", category="crypto", sector="layer1", asset_class="crypto")
        assert entry.asset_class == "crypto"
        assert entry.symbol == "BTC-USD"

    def test_forex_asset_class(self):
        entry = AssetEntry(symbol="EURUSD=X", name="EUR/USD", category="forex", sector="major_pairs", asset_class="forex")
        assert entry.asset_class == "forex"

    def test_etfentry_is_alias(self):
        """ETFEntry is a backward-compatible alias for AssetEntry."""
        assert ETFEntry is AssetEntry
        entry = ETFEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad_market")
        assert entry.asset_class == "equity"


class TestUniverseConfig:
    """UniverseConfig with multi-asset support."""

    def test_assets_field(self):
        entries = [
            AssetEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad"),
            AssetEntry(symbol="BTC-USD", name="Bitcoin", category="crypto", sector="layer1", asset_class="crypto"),
        ]
        config = UniverseConfig(name="Test", assets=entries)
        assert len(config.assets) == 2
        assert config.assets[0].asset_class == "equity"
        assert config.assets[1].asset_class == "crypto"

    def test_etfs_backward_compat_property(self):
        entries = [
            AssetEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad"),
        ]
        config = UniverseConfig(name="Test", assets=entries)
        assert config.etfs is config.assets
        assert len(config.etfs) == 1


class TestRiskLimitsMultiAsset:
    """Per-asset-class risk parameters."""

    def test_crypto_risk_defaults(self):
        limits = RiskLimits()
        assert limits.crypto_max_position_weight == 0.05
        assert limits.crypto_default_stop_loss_pct == 0.15

    def test_forex_risk_defaults(self):
        limits = RiskLimits()
        assert limits.forex_max_position_weight == 0.08
        assert limits.forex_default_stop_loss_pct == 0.03

    def test_crypto_tighter_than_equity(self):
        limits = RiskLimits()
        assert limits.crypto_max_position_weight < limits.max_position_weight

    def test_forex_wider_than_crypto(self):
        limits = RiskLimits()
        assert limits.forex_max_position_weight > limits.crypto_max_position_weight


class TestGetTradeableSymbols:
    """get_tradeable_symbols with multi-asset universe."""

    def test_includes_all_asset_classes(self, sample_config):
        from llm_quant.data.universe import get_tradeable_symbols
        symbols = get_tradeable_symbols(sample_config)
        # sample_config from conftest has SPY, QQQ, TLT, GLD, BTC-USD, EURUSD=X
        assert "SPY" in symbols
        assert "BTC-USD" in symbols
        assert "EURUSD=X" in symbols

    def test_excludes_non_tradeable(self):
        from llm_quant.config import (
            AppConfig,
            AssetEntry,
            UniverseConfig,
        )
        from llm_quant.data.universe import get_tradeable_symbols
        config = AppConfig(
            universe=UniverseConfig(
                name="Test",
                assets=[
                    AssetEntry(symbol="SPY", name="S&P 500", category="equity", sector="broad"),
                    AssetEntry(symbol="DOGE-USD", name="Dogecoin", category="crypto", sector="meme", asset_class="crypto", tradeable=False),
                ],
            ),
        )
        symbols = get_tradeable_symbols(config)
        assert "SPY" in symbols
        assert "DOGE-USD" not in symbols


class TestSyncUniverse:
    """sync_universe_to_db with multi-asset."""

    def test_syncs_all_asset_classes(self, tmp_db, sample_config):
        from llm_quant.data.universe import sync_universe_to_db
        count = sync_universe_to_db(tmp_db, sample_config)
        assert count == len(sample_config.universe.assets)

        # Verify DB has all symbols
        rows = tmp_db.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()
        db_symbols = [r[0] for r in rows]
        assert "BTC-USD" in db_symbols
        assert "EURUSD=X" in db_symbols
        assert "SPY" in db_symbols
