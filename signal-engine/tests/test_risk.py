from datetime import datetime, timezone

from app.config import Settings
from app.models import Posture
from app.state import StateStore
from app.strategy.risk import choose_risk_pct, position_size


def test_risk_bump_and_clamp() -> None:
    cfg = Settings(MODE="prop_cfd", _env_file=None)
    risk_pct = choose_risk_pct(Posture.OPPORTUNISTIC, 85, cfg)
    assert risk_pct == cfg.max_risk_pct

    risk = position_size(100.0, 99.0, risk_pct, cfg)
    assert risk.position_size_usd <= cfg.account_size * 3.0


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
