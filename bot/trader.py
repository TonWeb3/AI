"""Main Deriv trading loop for Rise/Fall contracts.

Workflow:
- 5m bar close  -> Kronos forecasts 20 future candles -> picks peak probability candle -> buys Rise/Fall contract
- 30m bar close -> Refreshes macro regime forecast
- Position loop -> Streams active contract until expiry / settlement
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import CFG
from .deriv_client import DerivClient
from .engine import INTERVAL_MINUTES, KronosEngine
from .signals import CALL, FLAT, PUT, RiseFallTradePlan, build_plan
from .storage import Store

log = logging.getLogger(__name__)

NUM_COLS = ["open", "high", "low", "close", "volume", "amount"]


class DerivCandleStore:
    """Stores rolling closed candles for multiple timeframes."""

    def __init__(self, client: DerivClient, intervals: List[str], maxlen: int = 1200):
        self.client = client
        self.intervals = intervals
        self.maxlen = maxlen
        self._frames: Dict[str, pd.DataFrame] = {}
        self._lock = asyncio.Lock()
        self.bar_closed: Dict[str, asyncio.Event] = {i: asyncio.Event() for i in intervals}

    def _to_df(self, raw_candles: List[dict]) -> pd.DataFrame:
        if not raw_candles:
            return pd.DataFrame(columns=["timestamps", "open_time", "close_time"] + NUM_COLS)
        df = pd.DataFrame(raw_candles)
        df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in NUM_COLS:
            df[c] = df[c].astype(float)
        # Drop current forming bar if present
        now_ms = int(time.time() * 1000)
        if len(df) and df["close_time"].iloc[-1] >= now_ms:
            df = df.iloc[:-1]
        return df[["timestamps", "open_time", "close_time"] + NUM_COLS].drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    async def backfill(self) -> None:
        for iv in self.intervals:
            sec = INTERVAL_MINUTES.get(iv, 5) * 60
            raw = await self.client.fetch_candles(granularity=sec, count=self.maxlen)
            df = self._to_df(raw)
            self._frames[iv] = df
            log.info("Backfilled %s %s: %d closed bars, latest: %s",
                     self.client.symbol, iv, len(df), df["timestamps"].iloc[-1] if len(df) else "none")

    def get(self, interval: str) -> pd.DataFrame:
        return self._frames.get(interval, pd.DataFrame()).copy()

    def _append(self, interval: str, row: dict) -> bool:
        df = self._frames.get(interval)
        if df is None:
            return False
        if len(df) and row["open_time"] <= int(df["open_time"].iloc[-1]):
            return False
        self._frames[interval] = pd.concat(
            [df, pd.DataFrame([row])], ignore_index=True
        ).iloc[-self.maxlen:].reset_index(drop=True)
        return True

    async def poll(self) -> None:
        """Polls for new closed candles around bar boundaries."""
        base_sec = min(INTERVAL_MINUTES.get(iv, 5) for iv in self.intervals) * 60
        while True:
            try:
                now = time.time()
                for iv in self.intervals:
                    sec = INTERVAL_MINUTES.get(iv, 5) * 60
                    raw = await self.client.fetch_candles(granularity=sec, count=4)
                    if raw:
                        df = self._to_df(raw)
                        for _, r in df.iterrows():
                            row = {
                                "timestamps": r["timestamps"],
                                "open_time": int(r["open_time"]),
                                "close_time": int(r["close_time"]),
                                **{c: float(r[c]) for c in NUM_COLS}
                            }
                            async with self._lock:
                                fresh = self._append(iv, row)
                            if fresh:
                                log.info("%s %s closed @ %.4f", self.client.symbol, iv, row["close"])
                                self.bar_closed[iv].set()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Candle poll error: %s", e)

            since = time.time() % base_sec
            delay = 1.0 if since < 10.0 else min(10.0, base_sec - since + 0.5)
            await asyncio.sleep(max(0.5, delay))


@dataclass
class ActiveContract:
    contract_id: int
    decision_id: int
    direction: str              # CALL | PUT
    target_candle: int
    expiry_minutes: int
    entry_spot: float
    current_spot: float
    buy_price: float
    payout: float
    profit: float = 0.0
    profit_percentage: float = 0.0
    status: str = "open"
    opened_ts: float = field(default_factory=time.time)
    expiry_ts: float = 0.0
    is_sold: bool = False


class Bot:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.store = Store(cfg.db_path)
        self.client = DerivClient(cfg.app_id, cfg.token, symbol=cfg.symbol, mode=cfg.mode)
        self.candles = DerivCandleStore(self.client, [cfg.entry_tf, cfg.regime_tf],
                                        maxlen=max(600, cfg.lookback + 100))
        self.engine: Optional[KronosEngine] = None

        self.regime_fc = None
        self.entry_fc = None
        self.active_contract: Optional[ActiveContract] = None
        self.bar_index = 0
        self.last_entry_bar = -10_000

        self.day_key = time.strftime("%Y-%m-%d", time.gmtime())
        self.day_pnl = 0.0
        self.day_trades = 0
        self.trading_enabled = False
        self.halted_reason = "stopped (click Start to trade)"
        self.last_error = ""
        self.started_at = time.time()
        self.status = "starting"
        self.busy = ""
        self.running = True

    async def start(self) -> None:
        self.status = "connecting deriv"
        asyncio.create_task(self.client.start())

        # Wait for initial WS connection
        t0 = time.time()
        while not self.client.connected and (time.time() - t0) < 15:
            await asyncio.sleep(0.5)

        self.status = "backfilling candles"
        await self.candles.backfill()

        self.status = "loading model"
        log.info("Loading Kronos %s ...", self.cfg.model_id)
        self.engine = await asyncio.to_thread(
            KronosEngine, self.cfg.model_id, self.cfg.tokenizer_id,
            self.cfg.device, self.cfg.max_context
        )

        self.status = "ready"
        log.info("Kronos Deriv Bot ready. Mode=%s Symbol=%s Risk=%s %s (%s)",
                 self.cfg.mode, self.cfg.symbol, self.cfg.risk_value,
                 "%" if self.cfg.risk_type == "percent" else self.cfg.currency, self.cfg.risk_type)

    async def stop(self) -> None:
        self.running = False
        self.client.close()

    async def run(self) -> None:
        await self.start()
        tasks = [
            asyncio.create_task(self.candles.poll(), name="poll"),
            asyncio.create_task(self._entry_loop(), name="entry"),
            asyncio.create_task(self._regime_loop(), name="regime"),
            asyncio.create_task(self._manage_loop(), name="manage"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await self.stop()

    async def _forecast(self, interval: str, horizon: int, n_paths: int):
        df = self.candles.get(interval)
        if len(df) < 50:
            log.warning("Candle context insufficient for %s (%d/50 bars). Retrying backfill...", interval, len(df))
            await self.candles.backfill()
            df = self.candles.get(interval)
            if len(df) < 50:
                raise ValueError(f"insufficient candle history for {interval}: {len(df)}/50 bars")
        return await asyncio.to_thread(
            self.engine.forecast, df, interval, horizon, n_paths,
            self.cfg.lookback, self.cfg.symbol, self.cfg.temperature,
            self.cfg.top_k, self.cfg.top_p, self.cfg.max_batch
        )

    async def _regime_loop(self) -> None:
        ev = self.candles.bar_closed[self.cfg.regime_tf]
        while self.running:
            try:
                self.busy = "regime forecast"
                self.regime_fc = await self._forecast(
                    self.cfg.regime_tf, self.cfg.regime_pred_len, self.cfg.n_paths_regime)
                self.busy = ""
                self.store.log_forecast(self.regime_fc)
                log.info("Regime 30m: upside=%.1f%% confidence=%.1f%%",
                         self.regime_fc.upside_prob * 100, self.regime_fc.confidence * 100)
            except Exception as e:
                self.busy = ""
                self.last_error = f"regime: {e}"
                log.exception("Regime forecast failed")
                await asyncio.sleep(5)
                continue
            ev.clear()
            await ev.wait()

    async def _entry_loop(self) -> None:
        ev = self.candles.bar_closed[self.cfg.entry_tf]

        # Startup preview forecast: retry every 5s until Deriv candle feed is ready
        while self.running and self.entry_fc is None:
            try:
                df = self.candles.get(self.cfg.entry_tf)
                if len(df) >= 50:
                    await self._on_entry_bar(preview=True)
                    if self.entry_fc is not None:
                        break
                else:
                    self.busy = f"accumulating candle data ({len(df)}/50 bars)"
                    log.info("Awaiting sufficient closed candles (%d/50). Retrying backfill...", len(df))
                    await self.candles.backfill()
            except Exception as e:
                self.last_error = f"startup preview: {e}"
                log.warning("Startup preview pending: %s. Retrying in 5s...", e)
            await asyncio.sleep(5)

        while self.running:
            ev.clear()
            await ev.wait()
            self.bar_index += 1
            try:
                await self._on_entry_bar()
            except Exception as e:
                self.last_error = f"entry: {e}"
                log.exception("Entry bar handling failed")

    def _roll_day(self) -> None:
        k = time.strftime("%Y-%m-%d", time.gmtime())
        if k != self.day_key:
            self.day_key, self.day_pnl, self.day_trades = k, 0.0, 0
            if self.halted_reason.startswith("daily"):
                self.halted_reason = ""

    def _risk_block(self) -> str:
        self._roll_day()
        if not self.trading_enabled:
            return "bot stopped (click Start)"
        if self.halted_reason:
            return self.halted_reason
        if self.active_contract is not None:
            return "contract already open"
        if self.day_pnl <= -abs(self.cfg.max_daily_loss_usd):
            self.halted_reason = "daily loss limit"
            return self.halted_reason
        if self.day_trades >= self.cfg.max_trades_per_day:
            return "daily trade cap"
        if self.bar_index - self.last_entry_bar < self.cfg.cooldown_bars:
            return "cooldown"
        return ""

    async def _on_entry_bar(self, preview: bool = False) -> None:
        t0 = time.time()
        self.busy = "forecasting 20 candles"
        self.entry_fc = await self._forecast(
            self.cfg.entry_tf, self.cfg.entry_pred_len, self.cfg.n_paths)
        self.store.log_forecast(self.entry_fc)

        spot = self.client.last_price or self.entry_fc.last_close
        plan = build_plan(self.cfg, self.entry_fc, self.regime_fc, spot, balance=self.client.balance)

        self.busy = ""
        block = "startup preview" if preview else self._risk_block()
        if block and plan.direction != FLAT:
            plan.direction = FLAT
            plan.reason = f"blocked: {block}"

        taken = plan.tradeable and not preview
        decision_id = self.store.log_decision(plan, self.entry_fc.last_ts.isoformat(), taken)
        log.info("5m close | %s | Peak C#%d (%dm) conv=%.1f%% | %s | %.1fs",
                 plan.direction, plan.target_candle, plan.expiry_minutes,
                 plan.conviction * 100, plan.reason, time.time() - t0)

        if taken:
            await self._execute(plan, decision_id)

    async def trigger_immediate_entry(self) -> None:
        """Called when user clicks Start — if there is a fresh peak signal in the current bar, enter immediately."""
        if not self.trading_enabled or self.active_contract is not None:
            return
        if self.entry_fc is None:
            return
        age = time.time() - self.entry_fc.last_ts.timestamp()
        if age > 270:
            return
        spot = self.client.last_price or self.entry_fc.last_close
        plan = build_plan(self.cfg, self.entry_fc, self.regime_fc, spot, balance=self.client.balance)
        block = self._risk_block()
        if block:
            return
        if plan.tradeable:
            decision_id = self.store.log_decision(plan, self.entry_fc.last_ts.isoformat(), taken=True)
            log.info("Immediate start entry | %s | Peak C#%d (%dm) conv=%.1f%%",
                     plan.direction, plan.target_candle, plan.expiry_minutes, plan.conviction * 100)
            await self._execute(plan, decision_id)

    async def _execute(self, plan: RiseFallTradePlan, decision_id: int) -> None:
        if not self.client.authorized:
            log.warning("Cannot execute trade: Deriv client not authorized")
            return

        # 1. Price contract via proposal
        prop = await self.client.get_proposal(
            contract_type=plan.direction,
            amount=plan.stake,
            duration=plan.expiry_minutes,
            duration_unit="m",
            currency=plan.currency
        )
        if not prop.get("ok"):
            log.warning("Proposal rejected: %s", prop.get("error"))
            return

        proposal_id = prop["id"]
        ask_price = prop["ask_price"]
        payout = prop["payout"]

        # 2. Buy contract
        buy_res = await self.client.buy(proposal_id, ask_price)
        if not buy_res.get("ok"):
            log.warning("Buy failed: %s", buy_res.get("error"))
            return

        contract_id = int(buy_res["contract_id"])
        self.store.open_trade(decision_id, contract_id, plan, ask_price, payout, self.cfg.mode)

        self.active_contract = ActiveContract(
            contract_id=contract_id,
            decision_id=decision_id,
            direction=plan.direction,
            target_candle=plan.target_candle,
            expiry_minutes=plan.expiry_minutes,
            entry_spot=prop.get("spot") or plan.spot,
            current_spot=prop.get("spot") or plan.spot,
            buy_price=ask_price,
            payout=payout,
            opened_ts=time.time(),
            expiry_ts=time.time() + (plan.expiry_minutes * 60)
        )
        self.last_entry_bar = self.bar_index
        self.day_trades += 1
        log.info("BOUGHT %s %s | Contract #%s | Stake $%.2f Payout $%.2f Expiry %dm",
                 plan.direction, plan.symbol, contract_id, ask_price, payout, plan.expiry_minutes)

    async def _manage_loop(self) -> None:
        """Monitors active contract until settlement."""
        while self.running:
            await asyncio.sleep(1.0)
            if self.active_contract is None:
                continue
            try:
                cid = self.active_contract.contract_id
                st = await self.client.contract_status(cid)
                if not st.get("ok"):
                    continue

                self.active_contract.current_spot = st.get("current_spot") or self.active_contract.current_spot
                if st.get("entry_spot"):
                    self.active_contract.entry_spot = st["entry_spot"]
                self.active_contract.profit = st.get("profit") or 0.0
                self.active_contract.profit_percentage = st.get("profit_percentage") or 0.0
                self.active_contract.status = st.get("status") or "open"

                if st.get("is_sold"):
                    exit_spot = st.get("current_spot") or self.client.last_price or 0.0
                    profit = st.get("profit") or 0.0
                    payout = st.get("payout") or 0.0
                    status = "won" if profit > 0 else "lost"

                    self.store.close_trade(cid, exit_spot, profit, payout, status)
                    self.day_pnl += profit
                    log.info("SETTLED contract #%s: %s | Profit: $%.2f | Balance: %s",
                             cid, status.upper(), profit, self.client.balance)

                    await self.client.fetch_balance()
                    self.active_contract = None
            except Exception as e:
                self.last_error = f"manage: {e}"
                log.warning("Active contract tracking error: %s", e)

    def _next_bar_eta(self) -> float:
        step = INTERVAL_MINUTES.get(self.cfg.entry_tf, 5) * 60
        return round(step - (time.time() % step), 1)

    def state(self) -> dict:
        act = None
        if self.active_contract:
            c = self.active_contract
            rem_sec = max(0.0, c.expiry_ts - time.time()) if c.expiry_ts else 0.0
            act = {
                "contract_id": c.contract_id,
                "direction": c.direction,
                "target_candle": c.target_candle,
                "expiry_minutes": c.expiry_minutes,
                "entry_spot": c.entry_spot,
                "current_spot": c.current_spot,
                "buy_price": c.buy_price,
                "payout": c.payout,
                "profit": round(c.profit, 2),
                "profit_percentage": round(c.profit_percentage, 2),
                "status": c.status,
                "remaining_sec": round(rem_sec, 0),
            }

        return {
            "status": self.status,
            "busy": self.busy,
            "symbol": self.cfg.symbol,
            "mode": self.cfg.mode,
            "connected": self.client.connected,
            "authorized": self.client.authorized,
            "account_id": self.client.account_id or "—",
            "balance": round(self.client.balance, 2) if self.client.balance is not None else None,
            "currency": self.client.currency,
            "spot": self.client.last_price,
            "next_bar_sec": self._next_bar_eta(),
            "entry_tf": self.cfg.entry_tf,
            "regime_tf": self.cfg.regime_tf,
            "day_pnl": round(self.day_pnl, 2),
            "day_trades": self.day_trades,
            "halted": self.halted_reason,
            "trading_enabled": self.trading_enabled and not bool(self.halted_reason),
            "last_error": self.last_error,
            "active_contract": act,
            "entry_forecast": self.entry_fc.summary() if self.entry_fc else None,
            "regime_forecast": self.regime_fc.summary() if self.regime_fc else None,
        }
