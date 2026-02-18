from __future__ import annotations

import logging

from app.config import Settings


def test_settings_boot_with_replay_env_example_values(monkeypatch) -> None:
    monkeypatch.setenv("MODE", "paper")
    monkeypatch.setenv("RUN_MODE", "replay")
    monkeypatch.setenv("PROFILE", "profit")
    monkeypatch.setenv("STRATEGY_PROFILE", "PROP_PASS")
    monkeypatch.setenv("ENGINE_MODE", "paper")
    monkeypatch.setenv("SYMBOLS", '["ETHUSDT"]')
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "replay")
    monkeypatch.setenv("MARKET_DATA_REPLAY_PATH", "data/replay")
    monkeypatch.setenv("CANDLE_INTERVAL", "5m")
    monkeypatch.setenv("BE_MIN_SECONDS_OPEN", "300")
    monkeypatch.setenv("MOVE_TO_BREAKEVEN_TRIGGER_R", "1.3")

    settings = Settings(_env_file=None)

    assert settings.symbols == ["ETHUSDT"]
    assert settings.candle_interval == "5m"
    assert settings.move_to_breakeven_min_seconds_open == 300
    assert settings.move_to_breakeven_trigger_r == 1.3


def test_apply_mode_defaults_skips_unknown_keys(monkeypatch, caplog) -> None:
    monkeypatch.setenv("STRATEGY_PROFILE", "SCALPER_STABLE")
    original_profile_defaults = Settings._profile_defaults

    def _with_unknown_default(self: Settings) -> dict[str, object]:
        defaults = original_profile_defaults(self)
        defaults["be_min_seconds_open"] = 300
        return defaults

    monkeypatch.setattr(Settings, "_profile_defaults", _with_unknown_default)

    with caplog.at_level(logging.WARNING):
        settings = Settings(_env_file=None)

    assert settings.strategy_profile == "SCALPER_STABLE"
    assert "Config default skipped (field missing): be_min_seconds_open" in caplog.text
