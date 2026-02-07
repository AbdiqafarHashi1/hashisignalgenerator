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
