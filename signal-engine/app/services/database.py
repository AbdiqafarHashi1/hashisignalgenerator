from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ..config import Settings


class Base(DeclarativeBase):
    pass


class DecisionRow(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    skip_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    inputs_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class TradeRow(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    entry: Mapped[float] = mapped_column(Float, nullable=False)
    exit: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees: Mapped[float | None] = mapped_column(Float, nullable=True)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trade_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="paper")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")


class PositionRow(Base):
    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    side: Mapped[str] = mapped_column(String(16), nullable=False)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())


class EquityCurveRow(Base):
    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl_today: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class RuntimeStateRow(Base):
    __tablename__ = "runtime_state"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    value_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


@dataclass
class TradeRecord:
    id: int
    symbol: str
    entry: float
    exit: float | None
    stop: float
    take_profit: float
    size: float
    pnl_usd: float | None
    pnl_r: float | None
    side: str
    opened_at: str
    closed_at: str | None
    result: str | None
    trade_mode: str
    fees: float | None = None




class _CompatConn:
    def __init__(self, database: "Database") -> None:
        self._database = database

    def __enter__(self):
        self._conn = self._database._engine.raw_connection()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._conn.commit()
        self._conn.close()

    def execute(self, sql: str, params: tuple[object, ...] = ()):
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor

class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_engine(
            self._resolve_database_url(settings),
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        self._conn = _CompatConn(self)
        self.init_schema()

    def init_schema(self) -> None:
        Base.metadata.create_all(self._engine)
        if self._engine.url.get_backend_name() == "sqlite":
            with self._engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA busy_timeout=1500;")
                conn.commit()

    def _resolve_database_url(self, settings: Settings) -> str:
        if settings.database_url:
            return settings.database_url
        db_path = Path(settings.data_dir) / "trades.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    def add_signal(self, timestamp: datetime, symbol: str, score: int | None, status: str, rationale: str, entry: float | None, stop: float | None, take_profit: float | None, decision: str = "skip", skip_reason: str | None = None, regime: str | None = None, scores: dict[str, Any] | None = None, inputs_snapshot: dict[str, Any] | None = None) -> None:
        with self._Session() as session:
            session.add(DecisionRow(
                timestamp=_as_utc(timestamp),
                symbol=symbol,
                regime=regime,
                decision=decision,
                skip_reason=skip_reason,
                score=score,
                rationale=rationale,
                status=status,
                entry=entry,
                stop=stop,
                take_profit=take_profit,
                scores=scores,
                inputs_snapshot=inputs_snapshot,
            ))
            session.commit()

    def log_event(self, event_type: str, payload: dict[str, Any], correlation_id: str) -> None:
        with self._Session() as session:
            session.add(EventRow(event_type=event_type, payload=payload, correlation_id=correlation_id, timestamp=datetime.now(timezone.utc)))
            session.commit()

    def open_trade(self, symbol: str, entry: float, stop: float, take_profit: float, size: float, side: str, opened_at: datetime, trade_mode: str = "paper") -> int:
        with self._Session() as session:
            row = TradeRow(symbol=symbol, entry=entry, stop=stop, take_profit=take_profit, size=size, side=side, opened_at=_as_utc(opened_at), trade_mode=trade_mode, status="open")
            session.add(row)
            session.flush()
            self._upsert_position(session, symbol=symbol, side=side, qty=size, entry=entry, stop=stop, take_profit=take_profit)
            session.commit()
            return int(row.id)

    def update_trade_stop(self, trade_id: int, stop: float) -> None:
        with self._Session() as session:
            trade = session.get(TradeRow, trade_id)
            if trade is None:
                return
            trade.stop = stop
            self._upsert_position(session, symbol=trade.symbol, side=trade.side, qty=trade.size, entry=trade.entry, stop=stop, take_profit=trade.take_profit)
            session.commit()

    def close_trade(self, trade_id: int, exit_price: float, pnl_usd: float, pnl_r: float, closed_at: datetime, result: str, fees: float = 0.0) -> None:
        with self._Session() as session:
            trade = session.get(TradeRow, trade_id)
            if trade is None:
                return
            trade.exit = exit_price
            trade.pnl_usd = pnl_usd
            trade.pnl_r = pnl_r
            trade.fees = fees
            trade.closed_at = _as_utc(closed_at)
            trade.result = result
            trade.status = "closed"
            session.execute(
                select(PositionRow).where(PositionRow.symbol == trade.symbol)
            )
            pos = session.get(PositionRow, trade.symbol)
            if pos is not None:
                session.delete(pos)
            session.commit()

    def fetch_open_trades(self, symbol: str | None = None) -> list[TradeRecord]:
        with self._Session() as session:
            stmt = select(TradeRow).where(TradeRow.closed_at.is_(None)).order_by(TradeRow.opened_at.desc())
            if symbol:
                stmt = stmt.where(TradeRow.symbol == symbol)
            rows = session.execute(stmt).scalars().all()
            return [self._to_trade_record(row) for row in rows]

    def fetch_trades(self, limit: int | None = None) -> list[TradeRecord]:
        with self._Session() as session:
            stmt = select(TradeRow).order_by(TradeRow.opened_at.desc())
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [self._to_trade_record(row) for row in rows]

    def fetch_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._Session() as session:
            rows = session.execute(select(EventRow).order_by(EventRow.timestamp.desc()).limit(limit)).scalars().all()
            return [
                {
                    "id": row.id,
                    "timestamp": row.timestamp.isoformat(),
                    "event_type": row.event_type,
                    "correlation_id": row.correlation_id,
                    "payload": row.payload or {},
                }
                for row in reversed(rows)
            ]

    def record_equity(self, equity: float, realized_pnl_today: float, drawdown_pct: float) -> None:
        with self._Session() as session:
            session.add(EquityCurveRow(timestamp=datetime.now(timezone.utc), equity=equity, realized_pnl_today=realized_pnl_today, drawdown_pct=drawdown_pct))
            session.commit()

    def fetch_equity_curve(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._Session() as session:
            rows = session.execute(select(EquityCurveRow).order_by(EquityCurveRow.timestamp.desc()).limit(limit)).scalars().all()
            return [
                {
                    "timestamp": row.timestamp.isoformat(),
                    "equity": row.equity,
                    "pnl_today": row.realized_pnl_today,
                    "drawdown_pct": row.drawdown_pct,
                }
                for row in reversed(rows)
            ]

    def set_runtime_state(self, key: str, value_text: str | None = None, value_number: float | None = None, symbol: str | None = None) -> None:
        with self._Session() as session:
            row = session.get(RuntimeStateRow, key)
            if row is None:
                row = RuntimeStateRow(key=key, symbol=symbol)
                session.add(row)
            row.symbol = symbol
            row.value_text = value_text
            row.value_number = value_number
            row.updated_at = datetime.now(timezone.utc)
            session.commit()

    def get_runtime_state(self, key: str) -> RuntimeStateRow | None:
        with self._Session() as session:
            return session.get(RuntimeStateRow, key)

    def reset_trades(self) -> None:
        with self._Session() as session:
            session.query(TradeRow).delete()
            session.query(PositionRow).delete()
            session.commit()

    def reset_all(self) -> None:
        with self._Session() as session:
            session.query(TradeRow).delete()
            session.query(PositionRow).delete()
            session.query(DecisionRow).delete()
            session.query(EventRow).delete()
            session.query(EquityCurveRow).delete()
            session.query(RuntimeStateRow).delete()
            session.commit()

    def dumps_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def loads_json(self, payload: str) -> dict[str, Any]:
        return json.loads(payload)

    def _upsert_position(self, session: Session, *, symbol: str, side: str, qty: float, entry: float, stop: float | None, take_profit: float | None) -> None:
        pos = session.get(PositionRow, symbol)
        if pos is None:
            pos = PositionRow(symbol=symbol, side=side, qty=qty, entry_price=entry, stop_loss=stop, take_profit=take_profit, updated_at=datetime.now(timezone.utc))
            session.add(pos)
            return
        pos.side = side
        pos.qty = qty
        pos.entry_price = entry
        pos.stop_loss = stop
        pos.take_profit = take_profit
        pos.updated_at = datetime.now(timezone.utc)

    def _to_trade_record(self, row: TradeRow) -> TradeRecord:
        return TradeRecord(
            id=row.id,
            symbol=row.symbol,
            entry=row.entry,
            exit=row.exit,
            stop=row.stop,
            take_profit=row.take_profit,
            size=row.size,
            pnl_usd=row.pnl_usd,
            pnl_r=row.pnl_r,
            fees=row.fees,
            side=row.side,
            opened_at=_iso_utc(row.opened_at) or "",
            closed_at=_iso_utc(row.closed_at),
            result=row.result,
            trade_mode=row.trade_mode,
        )


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    dt = _as_utc(value)
    return dt.isoformat()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
