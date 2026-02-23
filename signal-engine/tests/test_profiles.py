from __future__ import annotations

from app.config import Settings


def test_profile_defaults_profit(monkeypatch) -> None:
    monkeypatch.delenv("PROFILE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.PROFILE == "profit"
    assert settings.candle_interval == "1m"
    assert settings.min_signal_score == 60
    assert settings.trend_strength_min == 0.45
    assert settings.cooldown_minutes_after_loss == 10
    assert settings.max_trades_per_day == 8


def test_profile_defaults_diag(monkeypatch) -> None:
    monkeypatch.setenv("PROFILE", "diag")
    settings = Settings(_env_file=None)
    assert settings.PROFILE == "diag"
    assert settings.candle_interval == "1m"
    assert settings.min_signal_score == 35
    assert settings.trend_strength_min == 0.30
    assert settings.cooldown_minutes_after_loss == 0
    assert settings.max_trades_per_day == 50
    assert settings.engine_mode == "signal_only"


def test_profile_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PROFILE", "diag")
    monkeypatch.setenv("MIN_SIGNAL_SCORE", "92")
    monkeypatch.setenv("TREND_STRENGTH_MIN", "0.9")
    settings = Settings(_env_file=None)
    assert settings.min_signal_score == 92
    assert settings.trend_strength_min == 0.9


def test_intraday_trend_selective_profile_defaults(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_PROFILE", "INTRADAY_TREND_SELECTIVE")
    settings = Settings(_env_file=None)
    assert settings.strategy_profile == "INTRADAY_TREND_SELECTIVE"
    assert settings.min_signal_score == 65
    assert settings.trend_strength_min == 0.50
    assert settings.cooldown_minutes_after_loss == 15
    assert settings.max_trades_per_day == 6


def test_instant_profile_disables_prop_governor_by_default(monkeypatch) -> None:
    monkeypatch.setenv("PROFILE", "instant_funded")
    monkeypatch.delenv("PROP_ENABLED", raising=False)
    monkeypatch.delenv("PROP_GOVERNOR_ENABLED", raising=False)
    settings = Settings(_env_file=None)
    assert settings.strategy_profile == "INSTANT_FUNDED"
    assert settings.prop_enabled is False
    assert settings.prop_governor_enabled is False


def test_instant_profile_allows_explicit_prop_overrides(monkeypatch) -> None:
    monkeypatch.setenv("PROFILE", "instant_funded")
    monkeypatch.setenv("PROP_ENABLED", "true")
    monkeypatch.setenv("PROP_GOVERNOR_ENABLED", "true")
    settings = Settings(_env_file=None)
    assert settings.prop_enabled is True
    assert settings.prop_governor_enabled is True
