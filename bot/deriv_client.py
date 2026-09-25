"""Deriv client supporting both New REST+OTP WebSocket and direct token authorization.

Handles:
- Account discovery (Demo vs Real) and balance synchronization
- Real-time tick streaming (live spot)
- Historical candle retrieval (ticks_history)
- Proposal pricing and contract execution for Rise/Fall (CALL/PUT)
- Real-time open contract tracking until settlement
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional

import aiohttp

log = logging.getLogger(__name__)

REST_BASE = "https://api.derivws.com/trading/v1"
WS_FALLBACK = "wss://ws.derivws.com/websockets/v3"

def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class DerivError(Exception):
    def __init__(self, code: Any, message: str, subcode: Optional[str] = None, code_args: Optional[list] = None):
        self.code = code
        self.message = message
        self.subcode = subcode
        self.code_args = code_args or []
        super().__init__(f"[{code}] {message}")


class DerivClient:
    def __init__(self, app_id: str, token: str, symbol: str = "R_100", mode: str = "demo",
                 rest_base: str = REST_BASE):
        self.app_id = str(app_id or "33LuGFJlOlfbqzHVZLFCi")
        self.token = token or ""
        self.symbol = symbol
        self.mode = (mode or "demo").lower()
        self.rest_base = rest_base

        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.closed = False
        self.connected = False
        self.authorized = False

        self.last_price: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.pip_size: int = 2

        self.balance: Optional[float] = None
        self.currency: str = "USD"
        self.accounts: List[Dict[str, Any]] = []
        self.account_id: Optional[str] = None
        self.account_type: Optional[str] = None
        self.last_error: Optional[str] = None

        self._req_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_virtual(self) -> bool:
        return self.account_type == "demo" if self.account_type else (self.mode == "demo")

    def _headers(self) -> Dict[str, str]:
        return {
            "Deriv-App-ID": self.app_id,
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    async def _rest(self, method: str, path: str) -> Dict[str, Any]:
        url = self.rest_base + path
        async with aiohttp.ClientSession() as s:
            async with s.request(method, url, headers=self._headers(),
                                 timeout=aiohttp.ClientTimeout(total=15)) as r:
                text = await r.text()
                try:
                    data = json.loads(text)
                except Exception:
                    data = {"_raw": text}
                if r.status >= 400:
                    msg = (data.get("message") if isinstance(data, dict) else None) or text[:120]
                    raise DerivError(r.status, msg)
                return data

    async def fetch_accounts(self) -> List[Dict[str, Any]]:
        try:
            data = await self._rest("GET", "/options/accounts")
            self.accounts = data.get("data") or []
            return self.accounts
        except Exception as e:
            log.warning("REST accounts fetch failed: %s", e)
            return []

    async def _get_otp_ws_url(self, account_id: str) -> Optional[str]:
        try:
            data = await self._rest("POST", f"/options/accounts/{account_id}/otp")
            return (data.get("data") or {}).get("url")
        except Exception as e:
            log.warning("OTP fetch failed: %s", e)
            return None

    def _select_account(self) -> Optional[Dict[str, Any]]:
        want = "demo" if self.mode == "demo" else "real"
        for a in self.accounts:
            if (a.get("account_type") or "").lower() == want:
                return a
        return self.accounts[0] if self.accounts else None

    async def start(self):
        """Main connection and message routing loop."""
        self._loop = asyncio.get_running_loop()
        while not self.closed:
            try:
                ws_url = None
                use_otp = False
                
                # 1. Try modern REST + OTP if token is present
                if self.token:
                    await self.fetch_accounts()
                    acct = self._select_account()
                    if acct:
                        self.account_id = acct.get("account_id")
                        self.account_type = (acct.get("account_type") or "").lower()
                        if acct.get("balance") is not None:
                            self.balance = float(acct["balance"])
                        self.currency = acct.get("currency", self.currency)
                        ws_url = await self._get_otp_ws_url(self.account_id)
                        if ws_url:
                            use_otp = True

                # 2. Fallback to direct WebSocket API
                if not ws_url:
                    ws_url = f"{WS_FALLBACK}?app_id={self.app_id}"

                self._session = aiohttp.ClientSession()
                async with self._session.ws_connect(ws_url, heartbeat=20) as ws:
                    self.ws = ws
                    self.connected = True
                    self.last_error = None
                    log.info("Deriv WS connected (%s, mode=%s)", "OTP" if use_otp else "direct", self.mode)

                    # Authorize if using direct WS
                    if not use_otp and self.token:
                        auth_resp = await self._send({"authorize": self.token})
                        if "authorize" in auth_resp:
                            a = auth_resp["authorize"]
                            self.authorized = True
                            self.account_id = a.get("loginid")
                            self.account_type = "demo" if a.get("is_virtual") else "real"
                            self.balance = _f(a.get("balance"))
                            self.currency = a.get("currency", self.currency)
                            log.info("Authorized as %s (%s) balance=%s %s",
                                     self.account_id, self.account_type, self.balance, self.currency)
                    else:
                        self.authorized = True if use_otp else False

                    # Subscribe to ticks, balance, and fetch symbol metadata
                    await self._send({"ticks": self.symbol, "subscribe": 1}, wait=False)
                    await self._send({"balance": 1, "subscribe": 1}, wait=False)
                    asyncio.create_task(self.fetch_symbol_info())
                    asyncio.create_task(self.fetch_balance())

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._dispatch(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("Deriv connection error: %s. Reconnecting in 3s...", e)
            finally:
                self.connected = False
                self.authorized = False
                self.ws = None
                if self._session:
                    await self._session.close()
                    self._session = None
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.set_exception(ConnectionError("Deriv WS dropped"))
                self._pending.clear()

            if not self.closed:
                await asyncio.sleep(3)

    def close(self):
        self.closed = True
        if self.ws:
            asyncio.create_task(self.ws.close())

    async def _send(self, payload: Dict[str, Any], wait: bool = True, timeout: float = 15.0) -> Dict[str, Any]:
        if self.ws is None or self.ws.closed:
            raise ConnectionError("Deriv WS not connected")
        self._req_id += 1
        rid = self._req_id
        payload = dict(payload, req_id=rid)
        if wait:
            fut = self._loop.create_future()
            self._pending[rid] = fut
        await self.ws.send_str(json.dumps(payload))
        if not wait:
            return {}
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(rid, None)

    def _dispatch(self, data: Dict[str, Any]):
        rid = data.get("req_id")
        mtype = data.get("msg_type")

        if mtype == "tick" and "tick" in data:
            t = data["tick"]
            q = t.get("quote")
            if q is not None:
                self.last_price = float(q)
                self.last_ts = float(t.get("epoch") or time.time())
                if t.get("pip_size") is not None:
                    try:
                        self.pip_size = int(t["pip_size"])
                    except Exception:
                        pass

        elif mtype == "balance" and "balance" in data:
            b = data["balance"]
            v = _f(b.get("balance"))
            if v is not None:
                self.balance = v
                self.currency = b.get("currency", self.currency)

        if "error" in data:
            err = data["error"] or {}
            self.last_error = f"{err.get('code')}: {err.get('message')}"

        fut = self._pending.get(rid) if rid is not None else None
        if fut is not None and not fut.done():
            if "error" in data:
                err = data["error"] or {}
                fut.set_exception(DerivError(err.get("code"), err.get("message"),
                                             err.get("subcode"), err.get("code_args")))
            else:
                fut.set_result(data)

    async def fetch_symbol_info(self) -> Dict[str, Any]:
        try:
            resp = await self._send({"active_symbols": "brief"})
            for x in (resp.get("active_symbols") or []):
                sym = x.get("symbol") or x.get("underlying_symbol")
                if sym == self.symbol:
                    pv = x.get("pip_size")
                    if isinstance(pv, (int, float)) and 0 < pv < 1:
                        self.pip_size = max(0, round(-math.log10(pv)))
                    elif pv is not None:
                        self.pip_size = int(pv)
                    return {"ok": True, "pip_size": self.pip_size}
            return {"ok": False, "error": "symbol not found"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def fetch_balance(self) -> Optional[float]:
        try:
            resp = await self._send({"balance": 1})
            b = resp.get("balance") or {}
            v = _f(b.get("balance"))
            if v is not None:
                self.balance = v
                self.currency = b.get("currency", self.currency)
            return v
        except Exception as e:
            log.warning("fetch_balance error: %s", e)
            return self.balance

    async def fetch_candles(self, granularity: int = 300, count: int = 360) -> List[Dict[str, Any]]:
        """Fetch historical closed candles via ticks_history.
        
        granularity: bar duration in seconds (60 = 1m, 300 = 5m, 1800 = 30m)
        """
        try:
            resp = await self._send({
                "ticks_history": self.symbol,
                "style": "candles",
                "granularity": granularity,
                "count": count,
                "end": "latest"
            })
            out = []
            span_ms = granularity * 1000
            for c in resp.get("candles", []):
                epoch_ms = int(c["epoch"]) * 1000
                out.append({
                    "open_time": epoch_ms,
                    "close_time": epoch_ms + span_ms - 1,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c.get("volume", 0.0) or 0.0),
                    "amount": float(c.get("amount", 0.0) or 0.0)
                })
            return out
        except Exception as e:
            self.last_error = f"fetch_candles: {e}"
            log.error("Failed to fetch Deriv candles (%s, g=%d): %s", self.symbol, granularity, e)
            return []

    # ---------------- Rise/Fall Trading Methods ----------------

    async def get_proposal(self, contract_type: str, amount: float,
                           duration: int, duration_unit: str = "m",
                           currency: Optional[str] = None) -> Dict[str, Any]:
        """Request a Rise/Fall contract pricing proposal."""
        req = {
            "proposal": 1,
            "amount": round(float(amount), 2),
            "basis": "stake",
            "contract_type": contract_type.upper(),
            "currency": currency or self.currency or "USD",
            "underlying_symbol": self.symbol,
            "duration": int(duration),
            "duration_unit": duration_unit
        }
        try:
            resp = await self._send(req)
            p = resp.get("proposal", {})
            return {
                "ok": True,
                "id": p.get("id"),
                "ask_price": _f(p.get("ask_price")),
                "payout": _f(p.get("payout")),
                "spot": _f(p.get("spot")),
                "spot_time": p.get("spot_time")
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def buy(self, proposal_id: str, price: float) -> Dict[str, Any]:
        """Purchase the priced contract."""
        try:
            resp = await self._send({"buy": proposal_id, "price": round(float(price), 2)})
            b = resp.get("buy", {})
            if b.get("balance_after") is not None:
                self.balance = _f(b["balance_after"])
            return {
                "ok": True,
                "contract_id": b.get("contract_id"),
                "buy_price": _f(b.get("buy_price")),
                "payout": _f(b.get("payout")),
                "shortcode": b.get("shortcode")
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def contract_status(self, contract_id: int) -> Dict[str, Any]:
        """Fetch real-time status of an open or recently closed contract."""
        try:
            resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id})
            c = resp.get("proposal_open_contract", {})
            return {
                "ok": True,
                "contract_id": c.get("contract_id"),
                "is_sold": bool(c.get("is_sold")),
                "is_expired": bool(c.get("is_expired")),
                "status": c.get("status"),          # 'open', 'won', 'lost'
                "profit": _f(c.get("profit")),
                "profit_percentage": _f(c.get("profit_percentage")),
                "payout": _f(c.get("payout")),
                "buy_price": _f(c.get("buy_price")),
                "current_spot": _f(c.get("current_spot")),
                "entry_spot": _f(c.get("entry_spot")),
                "barrier": _f(c.get("barrier")),
                "date_expiry": c.get("date_expiry"),
                "date_start": c.get("date_start"),
            }
        except DerivError as e:
            return {"ok": False, "error": f"{e.code}: {e.message}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
