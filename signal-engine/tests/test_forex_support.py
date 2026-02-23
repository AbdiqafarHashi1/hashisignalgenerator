from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.config import Settings
from app.providers import bybit as bybit_module


def test_provider_selection_forex_oanda(monkeypatch) -> None:
    class StubProvider:
        def __init__(self, *, api_token: str, account_id: str, environment: str) -> None:
            assert api_token == "token"
            assert account_id == "acct"
            assert environment == "practice"

        async def fetch_candles(self, *, symbol: str, interval: str, limit: int):
            candle = bybit_module.BybitCandle(
                open_time=datetime.now(timezone.utc),
                close_time=datetime.now(timezone.utc),
                open=1.1,
                high=1.2,
                low=1.0,
                close=1.15,
                volume=0.0,
            )
            return bybit_module.BybitKlineSnapshot(
                symbol=symbol,
                interval=interval,
                price=1.15,
                volume=0.0,
                kline_open_time_ms=1,
                kline_close_time_ms=2,
                kline_is_closed=True,
                candle=candle,
                candles=[candle],
                provider_name="oanda",
                provider_category="forex",
                provider_endpoint="stub",
            )

    monkeypatch.setattr("app.providers.oanda.OandaPriceProvider", StubProvider)

    async def run() -> None:
        snap = await bybit_module.fetch_symbol_klines(
            symbol="EURUSD",
            interval="5m",
            provider="oanda",
            oanda_api_token="token",
            oanda_account_id="acct",
            oanda_env="practice",
        )
        assert snap is not None
        assert snap.provider_name == "oanda"

    asyncio.run(run())


def test_provider_selection_crypto_bybit_default() -> None:
    settings = Settings.model_validate({"ASSET_CLASS": "crypto"})
    assert settings.market_provider == "bybit"


def test_provider_selection_forex_defaults() -> None:
    settings = Settings.model_validate({"ASSET_CLASS": "forex", "SYMBOLS": "EURUSD,GBPUSD"})
    assert settings.market_provider == "oanda"
    assert settings.market_data_provider == "oanda"
    assert settings.oanda_instruments == ["EUR_USD", "GBP_USD"]
