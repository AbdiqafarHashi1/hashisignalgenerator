from __future__ import annotations

import pytest
from pydantic_settings import SettingsError

from app.config import Settings


def test_symbols_parse_json_list(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", '["btcusdt", " ethusdt "]')
    settings = Settings(_env_file=None)
    assert settings.symbols == ["BTCUSDT", "ETHUSDT"]


def test_symbols_invalid_json_raises(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    with pytest.raises(SettingsError):
        Settings(_env_file=None)
