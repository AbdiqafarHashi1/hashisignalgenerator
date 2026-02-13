from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..config import Settings


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


class Database:
    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.data_dir) / "trades.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    score INTEGER,
                    status TEXT NOT NULL,
                    rationale TEXT,
                    entry REAL,
                    stop REAL,
                    take_profit REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry REAL NOT NULL,
                    exit REAL,
                    stop REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    size REAL NOT NULL,
                    pnl_usd REAL,
                    pnl_r REAL,
                    side TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    result TEXT,
                    trade_mode TEXT NOT NULL DEFAULT "paper"
                )
                """
            )

        with self._conn:
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(trades)").fetchall()}
            if "trade_mode" not in cols:
                self._conn.execute('ALTER TABLE trades ADD COLUMN trade_mode TEXT NOT NULL DEFAULT "paper"')

    def add_signal(
        self,
        timestamp: datetime,
        symbol: str,
        score: int | None,
        status: str,
        rationale: str,
        entry: float | None,
        stop: float | None,
        take_profit: float | None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO signals (timestamp, symbol, score, status, rationale, entry, stop, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp.isoformat(),
                    symbol,
                    score,
                    status,
                    rationale,
                    entry,
                    stop,
                    take_profit,
                ),
            )

    def open_trade(
        self,
        symbol: str,
        entry: float,
        stop: float,
        take_profit: float,
        size: float,
        side: str,
        opened_at: datetime,
        trade_mode: str = "paper",
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO trades (symbol, entry, stop, take_profit, size, side, opened_at, trade_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    entry,
                    stop,
                    take_profit,
                    size,
                    side,
                    opened_at.isoformat(),
                    trade_mode,
                ),
            )
        return int(cursor.lastrowid)


    def update_trade_stop(self, trade_id: int, stop: float) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE trades
                SET stop = ?
                WHERE id = ?
                """,
                (stop, trade_id),
            )

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl_usd: float,
        pnl_r: float,
        closed_at: datetime,
        result: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE trades
                SET exit = ?, pnl_usd = ?, pnl_r = ?, closed_at = ?, result = ?
                WHERE id = ?
                """,
                (
                    exit_price,
                    pnl_usd,
                    pnl_r,
                    closed_at.isoformat(),
                    result,
                    trade_id,
                ),
            )

    def fetch_open_trades(self, symbol: str | None = None) -> list[TradeRecord]:
        query = "SELECT * FROM trades WHERE closed_at IS NULL"
        params: Iterable[Any] = []
        if symbol:
            query += " AND symbol = ?"
            params = [symbol]
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_trade(row) for row in rows]

    def fetch_trades(self, limit: int | None = None) -> list[TradeRecord]:
        query = "SELECT * FROM trades ORDER BY opened_at DESC"
        params: Iterable[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params = [limit]
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_trade(row) for row in rows]

    def reset_trades(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM trades")

    def reset_all(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM trades")
            self._conn.execute("DELETE FROM signals")

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        return TradeRecord(
            id=row["id"],
            symbol=row["symbol"],
            entry=row["entry"],
            exit=row["exit"],
            stop=row["stop"],
            take_profit=row["take_profit"],
            size=row["size"],
            pnl_usd=row["pnl_usd"],
            pnl_r=row["pnl_r"],
            side=row["side"],
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
            result=row["result"],
            trade_mode=row["trade_mode"] if "trade_mode" in row.keys() else "paper",
        )
