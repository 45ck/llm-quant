"""Tests for pod config loading and validation."""

from llm_quant.config import load_config, load_config_for_pod, validate_pod_id


def test_validate_pod_id_valid():
    """Valid pod IDs: lowercase alphanumeric with hyphens, 2+ chars."""
    assert validate_pod_id("default") is True
    assert validate_pod_id("momentum-01") is True
    assert validate_pod_id("ab") is True


def test_validate_pod_id_invalid():
    """Invalid pod IDs: uppercase, spaces, leading/trailing hyphens, empty."""
    assert validate_pod_id("A") is False
    assert validate_pod_id("with spaces") is False
    assert validate_pod_id("-leading") is False
    assert validate_pod_id("trailing-") is False
    assert validate_pod_id("") is False


def test_load_config_for_pod_default():
    """load_config_for_pod('default') returns same config as load_config()."""
    base = load_config()
    pod_cfg = load_config_for_pod("default")
    assert base.model_dump() == pod_cfg.model_dump()


def test_load_config_for_pod_benchmark():
    """load_config_for_pod('benchmark') overlays benchmark risk limits."""
    cfg = load_config_for_pod("benchmark")
    assert cfg.risk.max_position_weight == 0.25
    assert cfg.risk.max_trade_size == 0.10
    assert cfg.risk.max_gross_exposure == 1.0
    assert cfg.risk.max_sector_concentration == 0.50
    assert cfg.risk.max_drawdown_pct == 0.30


def test_load_config_for_pod_missing(tmp_path):
    """Unknown pod returns base config (no overlay file)."""
    # Create a minimal config dir with no strategies sub-dir
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cfg = load_config_for_pod("nonexistent-pod", config_dir=config_dir)
    # Should fall back to defaults
    base = load_config(config_dir=config_dir)
    assert cfg.risk.max_position_weight == base.risk.max_position_weight
