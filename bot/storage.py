"""SQLite storage for Deriv decisions, trades, and Kronos forecasts."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    bar_ts          TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    target_candle   INTEGER,
    expiry_minutes  INTEGER,
    conviction      REAL,
    upside_prob     REAL,
    downside_prob   REAL,
    regime_dir      TEXT,
    regime_conv     REAL,
    spot            REAL,
    stake           REAL,
    taken           INTEGER NOT NULL,
    reason          TEXT,
    gates_json      TEXT,
    candidates_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_dec_ts ON decisions(ts);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER,
    contract_id     INTEGER UNIQUE,
    symbol          TEXT    NOT NULL,
    contract_type   TEXT    NOT NULL,
    opened_ts       REAL    NOT NULL,
    expiry_ts       REAL,
    closed_ts       REAL,
    target_candle   INTEGER,
    expiry_minutes  INTEGER,
    entry_spot      REAL,
    exit_spot       REAL,
    stake           REAL,
    payout          REAL,
    profit          REAL,
    profit_pct      REAL,
    status          TEXT,
    mode            TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);
CREATE INDEX IF NOT EXISTS idx_tr_opened ON trades(opened_ts);

CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    symbol        TEXT    NOT NULL,
    interval      TEXT    NOT NULL,
    bar_ts        TEXT    NOT NULL,
    summary_json  TEXT    NOT NULL,
    mean_path     TEXT,
    band_lo       TEXT,
    band_hi       TEXT,
    y_ts          TEXT
);
CREATE INDEX IF NOT EXISTS idx_fc_ts ON forecasts(ts);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        if not hasattr(self._local, "c"):
            self._local.c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self._local.c.row_factory = sqlite3.Row
            self._local.c.execute("PRAGMA journal_mode=WAL")
        c = self._local.c
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise

    # ---------------- Writes ----------------

    def log_decision(self, plan, bar_ts: str, taken: bool) -> int:
        d = plan.to_dict()
        with self.conn() as c:
            cur = c.execute(
                """INSERT INTO decisions
                   (ts, bar_ts, symbol, direction, target_candle, expiry_minutes,
                    conviction, upside_prob, downside_prob, regime_dir, regime_conv,
                    spot, stake, taken, reason, gates_json, candidates_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), bar_ts, d["symbol"], d["direction"], d["target_candle"],
                 d["expiry_minutes"], d["conviction"], d["upside_prob"], d["downside_prob"],
                 d["regime_direction"], d["regime_conviction"], d["spot"], d["stake"],
                 int(taken), d["reason"], json.dumps(d.get("gates", {})),
                 json.dumps(d.get("candidates", [])))
            )
            return int(cur.lastrowid)

    def open_trade(self, decision_id: int, contract_id: int, plan, buy_price: float,
                   payout: float, mode: str) -> int:
        with self.conn() as c:
            cur = c.execute(
                """INSERT OR REPLACE INTO trades
                   (decision_id, contract_id, symbol, contract_type, opened_ts,
                    target_candle, expiry_minutes, entry_spot, stake, payout, status, mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (decision_id, contract_id, plan.symbol, plan.direction, time.time(),
                 plan.target_candle, plan.expiry_minutes, plan.spot, buy_price, payout,
                 "open", mode)
            )
            return int(cur.lastrowid)

    def close_trade(self, contract_id: int, exit_spot: float, profit: float,
                    payout: float, status: str) -> None:
        profit_pct = 0.0
        with self.conn() as c:
            row = c.execute("SELECT stake FROM trades WHERE contract_id=?", (contract_id,)).fetchone()
            if row and row["stake"] > 0:
                profit_pct = (profit / row["stake"]) * 100.0
            c.execute(
                """UPDATE trades SET closed_ts=?, exit_spot=?, payout=?, profit=?,
                   profit_pct=?, status=? WHERE contract_id=?""",
                (time.time(), exit_spot, payout, profit, profit_pct, status, contract_id)
            )

    def log_forecast(self, fc) -> None:
        lo, hi = fc.band(10, 90)
        with self.conn() as c:
            c.execute(
                """INSERT INTO forecasts
                   (ts, symbol, interval, bar_ts, summary_json, mean_path,
                    band_lo, band_hi, y_ts)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (time.time(), fc.symbol, fc.interval, fc.last_ts.isoformat(),
                 json.dumps(fc.summary()),
                 json.dumps([round(float(x), 6) for x in fc.mean_path]),
                 json.dumps([round(float(x), 6) for x in lo]),
                 json.dumps([round(float(x), 6) for x in hi]),
                 json.dumps([t.isoformat() for t in fc.y_timestamps]))
            )

    # ---------------- Reads ----------------

    def recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))]

    def recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))]

    def stats(self) -> Dict[str, Any]:
        with self.conn() as c:
            row = c.execute(
                """SELECT COUNT(*) n, SUM(profit) pnl,
                          SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) wins,
                          SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) losses
                   FROM trades WHERE status IN ('won', 'lost')"""
            ).fetchone()
            dec = c.execute("SELECT COUNT(*) n, SUM(taken) taken FROM decisions").fetchone()

        n = row["n"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        return {
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / n, 4) if n > 0 else 0.0,
            "profit_usd": round(row["pnl"] or 0.0, 2),
            "decisions": dec["n"] or 0,
            "taken": dec["taken"] or 0,
        }
