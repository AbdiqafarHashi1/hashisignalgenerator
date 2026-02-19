from __future__ import annotations

import pytest

from app.config import Settings


def test_legacy_env_keys_fail_fast_by_default(monkeypatch) -> None:
    monkeypatch.setenv("MAX_TRADES_PER_DAY", "3")
    with pytest.raises(ValueError, match="Legacy env keys are not allowed"):
        Settings(_env_file=None)


def test_legacy_env_keys_allowed_when_adapter_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SETTINGS_ENABLE_LEGACY", "true")
    monkeypatch.setenv("MAX_TRADES_PER_DAY", "3")
    settings = Settings(_env_file=None)
    assert settings.prop_max_trades_per_day == 3
