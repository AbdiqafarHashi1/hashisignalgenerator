from datetime import datetime, timezone

from app.config import Settings
from app.services.challenge import ChallengeService
from app.services.database import Database
from app.services.prop_governor import PropRiskGovernor
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
