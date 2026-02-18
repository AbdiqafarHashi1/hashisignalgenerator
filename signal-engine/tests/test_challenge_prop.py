from datetime import datetime, timezone

from app.config import Settings
from app.services.challenge import ChallengeService
from app.services.database import Database
from app.services.prop_governor import PropRiskGovernor
from app.state import StateStore
from app.providers.replay import ReplayProvider


def _settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path}/test.db",
        data_dir=str(tmp_path),
        run_mode="replay",
        symbols=["ETHUSDT"],
        strategy_profile="PROP_PASS",
    )


def test_pass_transition_when_target_and_min_days_met(tmp_path):
    s = _settings(tmp_path)
    db = Database(s)
    svc = ChallengeService(s, db)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    state = svc.reset(now, 1000)
    state.days_traded_count = 5
    svc.save(state)
    out = svc.update(equity=1085, daily_start_equity=1000, now=now, traded_today=False)
    assert out.status == "PASSED"


def test_fail_transition_on_daily_loss(tmp_path):
    s = _settings(tmp_path)
    db = Database(s)
    svc = ChallengeService(s, db)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    svc.reset(now, 1000)
    out = svc.update(equity=940, daily_start_equity=1000, now=now, traded_today=False)
    assert out.status == "FAILED"
    assert out.violation_reason == "daily_loss_limit"


def test_fail_transition_on_global_drawdown(tmp_path):
    s = _settings(tmp_path)
    db = Database(s)
    svc = ChallengeService(s, db)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    st = svc.reset(now, 1000)
    st.peak_equity = 1200
    svc.save(st)
    out = svc.update(equity=1070, daily_start_equity=1070, now=now, traded_today=False)
    assert out.status == "FAILED"
    assert out.violation_reason == "global_drawdown_limit"


def test_governor_stepdown_stepup(tmp_path):
    s = _settings(tmp_path)
    db = Database(s)
    gov = PropRiskGovernor(s, db)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    gov.on_trade_close(net_r=-1.0, now=now)
    st = gov.load(now)
    assert st.risk_pct <= s.prop_risk_base_pct
    gov.on_trade_close(net_r=1.2, now=now)
    st2 = gov.load(now)
    assert st2.risk_pct >= st.risk_pct


def test_replay_status_metadata(tmp_path):
    rp = ReplayProvider("tests/fixtures/replay")
    status = rp.status("ETHUSDT", "3m")
    assert status["exists"] is True
    assert status["row_count"] > 0
    assert status["first_ts"]
    assert status["last_ts"]


def test_reset_storage_fast(tmp_path):
    s = _settings(tmp_path)
    db = Database(s)
    start = datetime.now(timezone.utc)
    db.reset_all()
    elapsed = (datetime.now(timezone.utc)-start).total_seconds()
    assert elapsed < 2.0


def test_replay_day_rollover_clears_daily_trade_locks(tmp_path):
    s = _settings(tmp_path)
    s.max_trades_per_day = 2
    s.prop_max_trades_per_day = 2
    db = Database(s)
    gov = PropRiskGovernor(s, db)
    state = StateStore()

    day_one = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    state.set_clock(lambda: day_one)
    state.record_trade("ETHUSDT")
    state.record_trade("ETHUSDT")
    gov.on_trade_close(net_r=1.0, now=day_one)
    gov.on_trade_close(net_r=1.0, now=day_one)

    allowed, reason = state.risk_check("ETHUSDT", s, day_one)
    assert allowed is False
    assert reason == "max_trades_exceeded"

    gov_allowed, gov_reason = gov.allow_new_trade(day_one)
    assert gov_allowed is False
    assert gov_reason == "max_trades_per_day"

    day_two = datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc)
    allowed_day_two, reason_day_two = state.risk_check("ETHUSDT", s, day_two)
    assert allowed_day_two is True
    assert reason_day_two is None

    gov_allowed_day_two, gov_reason_day_two = gov.allow_new_trade(day_two)
    assert gov_allowed_day_two is True
    assert gov_reason_day_two is None


def test_governor_reset_clears_day_key_and_counters(tmp_path):
    s = _settings(tmp_path)
    db = Database(s)
    gov = PropRiskGovernor(s, db)
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    gov.on_trade_close(net_r=-1.0, now=now)

    reset_now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    state = gov.reset(reset_now)

    assert state.day_key == "2024-01-03"
    assert state.daily_net_r == 0.0
    assert state.daily_losses == 0
    assert state.daily_trades == 0
    assert state.consecutive_losses == 0
    assert state.locked_until_ts is None
