from datetime import datetime, timezone

from app.config import Settings
from app.models import Direction, Posture, Status, TradePlan
from app.services.database import Database
from app.services.paper_trader import PaperTrader
from app.state import StateStore
from app.strategy.risk import choose_risk_pct, position_size


def test_risk_bump_and_clamp() -> None:
    cfg = Settings(MODE="prop_cfd", _env_file=None)
    risk_pct = choose_risk_pct(Posture.OPPORTUNISTIC, 85, cfg)
    assert risk_pct == cfg.max_risk_pct

    risk = position_size(100.0, 99.0, risk_pct, cfg)
    assert risk.position_size_usd <= cfg.account_size * cfg.max_notional_account_multiplier


def test_position_size_usd_cap_is_optional() -> None:
    cfg = Settings(_env_file=None, account_size=1_000, max_notional_account_multiplier=1.0, position_size_usd_cap=None)
    risk = position_size(100.0, 99.8, 0.01, cfg)
    assert risk.position_size_usd == 1_000.0


def test_position_size_usd_cap_applies_when_configured(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        symbols=["ETHUSDT"],
        data_dir=str(tmp_path),
        account_size=25_000,
        base_risk_pct=0.01,
        max_notional_account_multiplier=10.0,
        position_size_usd_cap=5_000,
        fee_rate_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
    )
    db = Database(settings)
    trader = PaperTrader(settings, db)
    plan = TradePlan(
        status=Status.TRADE,
        direction=Direction.long,
        entry_zone=(2000.0, 2000.0),
        stop_loss=1990.0,
        take_profit=2010.0,
        risk_pct_used=settings.base_risk_pct,
        position_size_usd=settings.position_size_usd_cap,
        signal_score=90,
        posture=Posture.NORMAL,
        rationale=["unit-test"],
        raw_input_snapshot={},
    )

    trade_id = trader.maybe_open_trade("ETHUSDT", plan, allow_multiple=True)
    assert trade_id is not None
    sizing = trader.last_sizing_decision("ETHUSDT") or {}
    assert sizing.get("size_usd", 0.0) <= 5_000.0
    assert sizing.get("why_size_small") == ["position_size_usd_cap"]


def test_global_drawdown_limit_locks_state() -> None:
    cfg = Settings(_env_file=None, account_size=1000, max_daily_loss_pct=0.5, global_drawdown_limit_pct=0.08)
    store = StateStore()
    store.set_global_equity(1000)
    store.record_outcome("BTCUSDT", pnl_usd=-90, win=False, timestamp=datetime.now(timezone.utc))
    allowed, status, reasons = store.check_limits("BTCUSDT", cfg, datetime.now(timezone.utc))

    assert allowed is False
    assert status.value == "RISK_OFF"
    assert "global_drawdown_limit" in reasons


def test_manual_kill_switch_locks_state() -> None:
    cfg = Settings(_env_file=None, manual_kill_switch=True)
    store = StateStore()
    allowed, status, reasons = store.check_limits("BTCUSDT", cfg, datetime.now(timezone.utc))

    assert allowed is False
    assert status.value == "RISK_OFF"
    assert "manual_kill_switch" in reasons


def test_risk_env_knobs_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ACCOUNT_SIZE", "5000")
    monkeypatch.setenv("PROP_RISK_BASE_PCT", "0.01")
    cfg = Settings(_env_file=None)
    risk = position_size(100.0, 99.0, cfg.prop_risk_base_pct, cfg)
    assert risk.position_size_usd > 0
    assert cfg.account_size == 5000
    assert cfg.prop_risk_base_pct == 0.01


def test_position_size_usd_cap_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("MAX_POSITION_SIZE_USD", "5000")
    cfg = Settings(_env_file=None)
    assert cfg.position_size_usd_cap == 5000
    assert cfg.env_alias_map()["position_size_usd_cap"] == "POSITION_SIZE_USD_CAP"
