"""Entry point for Kronos Deriv Rise/Fall Trading Bot.

    python main.py                  # run bot + dashboard
    python main.py --check          # validate config.json and exit
    python main.py --headless       # trading loop only, no web server
    python main.py --port 8120      # override port
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot.config import CONFIG_PATH, CFG


def banner() -> None:
    print(f"""
  ======================================================
  KRONOS DERIV RISE / FALL BOT
  ======================================================
  config    {CONFIG_PATH}
  market    {CFG.symbol} | entry {CFG.entry_tf} x{CFG.entry_pred_len} candles | regime {CFG.regime_tf}
  model     {CFG.model_id} | device={CFG.device} | paths={CFG.n_paths}
  mode      {CFG.mode.upper()} account | risk {CFG.risk_value}{'%' if CFG.risk_type == 'percent' else ' ' + CFG.currency} ({CFG.risk_type})
  strategy  peak probability candle selection + 30m regime filter
  dashboard http://{'localhost' if CFG.host in ('0.0.0.0', '') else CFG.host}:{CFG.port}
  ======================================================
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate config.json and exit")
    ap.add_argument("--headless", action="store_true", help="run trading loop with no web server")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args()

    if a.host:
        CFG.host = a.host
    if a.port:
        CFG.port = a.port

    logging.basicConfig(
        level=getattr(logging, CFG.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    banner()
    errs = CFG.validate()
    if errs:
        print("CONFIG ERRORS:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("  config OK\n")
    if a.check:
        return

    if CFG.mode == "real":
        print("  !! REAL MONEY TRADING ACTIVE. Ctrl-C within 5s to abort.\n")
        import time
        time.sleep(5)

    if a.headless:
        import asyncio
        from bot.trader import Bot
        print("  headless mode: running trading loop only\n")
        asyncio.run(Bot().run())
        return

    import uvicorn
    from bot.api import app
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level=CFG.log_level.lower())


if __name__ == "__main__":
    main()
