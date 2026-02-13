from __future__ import annotations

from app.config import Settings


def test_symbols_parse_json_list(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", '["btcusdt", " ethusdt "]')
    settings = Settings(_env_file=None)
    assert settings.symbols == ["BTCUSDT", "ETHUSDT"]


def test_symbols_parse_csv_list(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "BTCUSDT,ETHUSDT")
    settings = Settings(_env_file=None)
    assert settings.symbols == ["BTCUSDT", "ETHUSDT"]


def test_bybit_defaults_mainnet() -> None:
    settings = Settings(_env_file=None)
    assert settings.bybit_testnet is False
    assert settings.bybit_rest_base == "https://api.bybit.com"
