from __future__ import annotations

from app.config import Settings


def test_settings_alias_choices(monkeypatch) -> None:
    monkeypatch.setenv("DEBUG_LOOSEN", "true")
    monkeypatch.setenv("STRATEGY", "baseline")
    settings = Settings(_env_file=None)
    assert settings.debug_loosen is True
    assert settings.strategy == "baseline"

    monkeypatch.delenv("DEBUG_LOOSEN")
    monkeypatch.delenv("STRATEGY")
    monkeypatch.setenv("debug_disable_hard_risk_gates", "true")
    settings = Settings(_env_file=None)
    assert settings.debug_disable_hard_risk_gates is True


def test_warmup_settings_alias_choices(monkeypatch) -> None:
    monkeypatch.setenv("WARMUP_MIN_BARS_5M", "333")
    monkeypatch.setenv("warmup_min_bars_1h", "123")
    settings = Settings(_env_file=None)
    assert settings.warmup_min_bars_5m == 333
    assert settings.warmup_min_bars_1h == 123
