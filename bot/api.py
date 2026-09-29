"""FastAPI backend serving Deriv Rise/Fall trading bot and dashboard."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import CFG, coerce, settings_schema
from .trader import Bot

log = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(ROOT, "templates")

app = FastAPI(title="Kronos Deriv Bot")
BOT: Bot | None = None
BOT_TASK: asyncio.Task | None = None
BOOT_CFG: dict = {}


def _snapshot() -> dict:
    from .config import NEEDS_RESTART
    return {k: getattr(CFG, k) for k in NEEDS_RESTART}


def _restart_pending() -> list:
    return sorted(k for k, v in BOOT_CFG.items() if getattr(CFG, k) != v)


@app.on_event("startup")
async def _startup() -> None:
    global BOT, BOT_TASK, BOOT_CFG
    BOT = Bot()
    BOOT_CFG = _snapshot()
    BOT_TASK = asyncio.create_task(BOT.run(), name="bot")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if BOT_TASK:
        BOT_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await BOT_TASK


@app.get("/api/state")
async def api_state():
    if BOT is None:
        return JSONResponse({"status": "starting"})
    return BOT.state()


@app.get("/api/predictions")
async def api_predictions():
    """Return the step-by-step probability distribution across the 20 future candles."""
    if BOT is None:
        return {"ready": False, "candidates": [], "message": "Bot initializing…"}
    if BOT.entry_fc is None:
        msg = BOT.busy or (f"Pending: {BOT.last_error}" if BOT.last_error else f"Awaiting {BOT.cfg.entry_tf} candle close & Monte Carlo forecast…")
        return {"ready": False, "candidates": [], "message": msg, "error": BOT.last_error}
    return {
        "ready": True,
        "symbol": BOT.cfg.symbol,
        "entry_tf": BOT.cfg.entry_tf,
        "last_close": BOT.entry_fc.last_close,
        "candidates": BOT.entry_fc.step_predictions(),
        "peak": BOT.entry_fc.peak_candle,
    }


@app.get("/api/config")
async def api_config():
    return {"config": CFG.redacted(), "errors": CFG.validate()}


@app.get("/api/stats")
async def api_stats():
    return BOT.store.stats() if BOT else {}


@app.get("/api/decisions")
async def api_decisions(limit: int = 40):
    return BOT.store.recent_decisions(limit) if BOT else []


@app.get("/api/trades")
async def api_trades(limit: int = 40):
    return BOT.store.recent_trades(limit) if BOT else []


@app.get("/api/chart")
async def api_chart(history: int = 120):
    """OHLC candlestick history + Kronos Monte Carlo forecast envelope for 5m."""
    if BOT is None:
        return JSONResponse({"status": "starting"})
    iv = BOT.cfg.entry_tf
    fc = BOT.entry_fc
    try:
        df = BOT.candles.get(iv)
        if len(df):
            df = df.iloc[-history:]
    except Exception:
        df = None

    if df is None or not len(df):
        return JSONResponse({"status": "no data", "forecast_pending": True})

    out = {
        "interval": iv,
        "forecast_pending": fc is None,
        "hist_ts": [t.isoformat() for t in df["timestamps"]],
        "hist_close": [round(float(x), 4) for x in df["close"]],
        "hist_ohlcv": [
            [round(float(r.open), 4), round(float(r.high), 4),
             round(float(r.low), 4), round(float(r.close), 4),
             round(float(r.volume), 2)]
            for r in df.itertuples()
        ],
        "spot": BOT.client.last_price if BOT.client and BOT.client.last_price else (round(float(df["close"].iloc[-1]), 4) if len(df) else None),
    }

    if fc is not None:
        lo, hi = fc.band(10, 90)
        lo2, hi2 = fc.band(0, 100)
        out.update({
            "fc_ts": [t.isoformat() for t in fc.y_timestamps],
            "fc_mean": [round(float(x), 4) for x in fc.mean_path],
            "fc_lo": [round(float(x), 4) for x in lo],
            "fc_hi": [round(float(x), 4) for x in hi],
            "fc_min": [round(float(x), 4) for x in lo2],
            "fc_max": [round(float(x), 4) for x in hi2],
            "summary": fc.summary(),
            "peak_candle": fc.peak_candle.get("candle_index"),
            "fc_ohlc": [
                [round(float(v), 4) for v in row]
                for row in np.median(fc.paths[:, :, :4], axis=0)
            ],
            "spot": BOT.client.last_price or fc.last_close,
        })
    return out


@app.get("/api/settings")
async def api_settings():
    return {
        "fields": settings_schema(CFG),
        "errors": CFG.validate(),
        "restart_pending": _restart_pending()
    }


@app.post("/api/settings")
async def api_settings_save(payload: dict = Body(...)):
    changes = payload.get("changes", payload) or {}
    trial = deepcopy(CFG)
    applied, rejected = {}, {}
    for k, v in changes.items():
        if not hasattr(trial, k):
            rejected[k] = "unknown setting"
            continue
        if k == "token" and v == "***set***":
            continue
        try:
            setattr(trial, k, coerce(trial, k, v))
            applied[k] = getattr(trial, k)
        except Exception as e:
            rejected[k] = str(e)

    errs = trial.validate()
    if errs or rejected:
        return {"ok": False, "errors": errs, "rejected": rejected}

    for k, v in applied.items():
        setattr(CFG, k, v)
    path = CFG.save()
    pending = _restart_pending()

    # If model_id or tokenizer_id changed, download the new weights immediately
    downloaded_model = None
    if "model_id" in applied or "tokenizer_id" in applied:
        target_model = getattr(CFG, "model_id")
        target_tok = getattr(CFG, "tokenizer_id")
        log.info("Model configuration updated. Pre-downloading '%s' and '%s'...", target_model, target_tok)
        try:
            from .engine import ensure_weights
            await asyncio.to_thread(ensure_weights, target_model, target_tok)
            downloaded_model = target_model
            log.info("New model weights downloaded successfully.")
        except Exception as e:
            log.error("Failed to download new model weights: %s", e)
            return {
                "ok": False,
                "saved": path,
                "applied": list(applied),
                "error": f"Settings saved to config.json, but model download failed: {e}",
                "restart_required": [k for k in applied if k in pending],
                "restart_pending": pending,
            }

    return {
        "ok": True,
        "saved": path,
        "applied": list(applied),
        "downloaded_model": downloaded_model,
        "restart_required": [k for k in applied if k in pending],
        "restart_pending": pending,
    }


@app.post("/api/restart")
async def api_restart(force: bool = False):
    global BOT, BOT_TASK, BOOT_CFG
    if BOT is not None and BOT.active_contract is not None and not force:
        return {"ok": False, "error": "A contract is currently active — wait for expiry or pass force=true"}
    if BOT_TASK:
        BOT_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await BOT_TASK
    if BOT:
        with contextlib.suppress(Exception):
            await BOT.stop()

    # Ensure model weights are ready for the active CFG before starting bot
    try:
        from .engine import ensure_weights
        await asyncio.to_thread(ensure_weights, CFG.model_id, CFG.tokenizer_id)
    except Exception as e:
        log.warning("Could not pre-verify weights on restart: %s", e)

    BOT = Bot()
    BOOT_CFG = _snapshot()
    BOT_TASK = asyncio.create_task(BOT.run(), name="bot")
    log.info("Bot restarted with new configuration")
    return {"ok": True}


@app.post("/api/start")
async def api_start():
    if BOT:
        BOT.trading_enabled = True
        BOT.halted_reason = ""
        asyncio.create_task(BOT.trigger_immediate_entry())
    return {"ok": True, "trading_enabled": True}


@app.post("/api/stop")
async def api_stop():
    if BOT:
        BOT.trading_enabled = False
        BOT.halted_reason = "manual stop"
    return {"ok": True, "trading_enabled": False}


@app.post("/api/halt")
async def api_halt():
    if BOT:
        BOT.trading_enabled = False
        BOT.halted_reason = "manual halt"
    return {"ok": True, "halted": BOT.halted_reason if BOT else ""}


@app.post("/api/resume")
async def api_resume():
    if BOT:
        BOT.trading_enabled = True
        BOT.halted_reason = ""
    return {"ok": True, "trading_enabled": True}


def _page(name: str) -> str:
    with open(os.path.join(TEMPLATES, name), encoding="utf-8") as fh:
        return fh.read()


@app.get("/", response_class=HTMLResponse)
async def index():
    return _page("index.html")
