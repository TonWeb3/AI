"""Kronos inference engine for Deriv candlestick forecasting.

Samples N independent future trajectories via Monte Carlo simulation and evaluates
step-by-step probability distributions across all predicted future candles.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot import WEIGHTS_DIR

log = logging.getLogger(__name__)


def ensure_weights(model_id: str, tokenizer_id: str, cache_dir: str = WEIGHTS_DIR, verbose: bool = True) -> bool:
    """Pre-downloads and verifies that tokenizer and model weights exist in cache_dir.
    Uses snapshot_download for reliable, resumable downloads with progress.
    """
    from huggingface_hub import snapshot_download
    os.makedirs(cache_dir, exist_ok=True)
    if verbose:
        print(f"Ensuring model weights in {cache_dir}...")
        print(f"  Fetching tokenizer: {tokenizer_id}")
    snapshot_download(repo_id=tokenizer_id, cache_dir=cache_dir)
    if verbose:
        print(f"  Fetching model:     {model_id}")
    snapshot_download(repo_id=model_id, cache_dir=cache_dir)
    if verbose:
        print("  Model weights verified successfully.\n")
    return True


KRONOS_COLS = ["open", "high", "low", "close", "volume", "amount"]

INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}


@dataclass
class Forecast:
    """Monte Carlo forecast distribution across future steps."""

    symbol: str
    interval: str
    last_close: float
    last_ts: pd.Timestamp
    horizon_bars: int
    n_paths: int
    # (n_paths, horizon_bars, 6) in real price units, cols = KRONOS_COLS
    paths: np.ndarray
    y_timestamps: pd.DatetimeIndex
    hist_vol: float
    elapsed_sec: float = 0.0

    @property
    def terminal_close(self) -> np.ndarray:
        return self.paths[:, -1, 3]

    @property
    def upside_prob(self) -> float:
        """P(price at terminal horizon > last known price)."""
        return float((self.terminal_close > self.last_close).mean())

    @property
    def direction(self) -> str:
        if self.upside_prob > 0.5:
            return "UP"
        if self.upside_prob < 0.5:
            return "DOWN"
        return "FLAT"

    @property
    def confidence(self) -> float:
        return max(self.upside_prob, 1.0 - self.upside_prob)

    @property
    def mean_path(self) -> np.ndarray:
        return self.paths[:, :, 3].mean(axis=0)

    def band(self, lo: float = 0.0, hi: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        c = self.paths[:, :, 3]
        return np.percentile(c, lo, axis=0), np.percentile(c, hi, axis=0)

    @property
    def path_vol(self) -> np.ndarray:
        closes = np.concatenate(
            [np.full((self.n_paths, 1), self.last_close), self.paths[:, :, 3]], axis=1)
        rets = np.diff(np.log(np.maximum(closes, 1e-12)), axis=1)
        return rets.std(axis=1, ddof=1)

    @property
    def vol_amplification_prob(self) -> float:
        if self.hist_vol <= 0:
            return 0.0
        return float((self.path_vol > self.hist_vol).mean())

    def move_pct(self) -> np.ndarray:
        return (self.terminal_close / self.last_close - 1.0) * 100.0

    # ---------------- Step-by-Step Probability Distribution & Peak Detection ----------------

    def step_predictions(self) -> List[Dict[str, Any]]:
        """Calculate Rise/Fall probabilities for every single predicted future candle."""
        iv_min = INTERVAL_MINUTES.get(self.interval, 5)
        steps = []
        best_conv = -1.0
        best_idx = 0

        # Pass 1: compute step metrics
        for k in range(self.horizon_bars):
            closes = self.paths[:, k, 3]
            p_up = float((closes > self.last_close).mean())
            p_down = float((closes < self.last_close).mean())
            conv = max(p_up, p_down)
            if conv > best_conv:
                best_conv = conv
                best_idx = k

            step_time = self.y_timestamps[k] if k < len(self.y_timestamps) else (self.last_ts + pd.Timedelta(minutes=iv_min * (k + 1)))
            steps.append({
                "candle_index": k + 1,
                "expiry_minutes": (k + 1) * iv_min,
                "timestamp": step_time.isoformat(),
                "time_str": step_time.strftime("%H:%M"),
                "upside_prob": round(p_up, 4),
                "downside_prob": round(p_down, 4),
                "conviction": round(conv, 4),
                "direction": "CALL" if p_up > 0.5 else "PUT" if p_down > 0.5 else "TIE",
                "median_price": round(float(np.median(closes)), 4),
                "is_peak": False
            })

        if steps:
            steps[best_idx]["is_peak"] = True
        return steps

    @property
    def peak_candle(self) -> Dict[str, Any]:
        """Returns the specific future candle step that has the highest conviction."""
        steps = self.step_predictions()
        if not steps:
            return {"candle_index": 1, "expiry_minutes": 5, "conviction": 0.5, "direction": "TIE"}
        for s in steps:
            if s.get("is_peak"):
                return s
        return steps[0]

    def summary(self) -> dict:
        m = self.move_pct()
        peak = self.peak_candle
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "last_close": self.last_close,
            "last_ts": self.last_ts.isoformat(),
            "horizon_bars": self.horizon_bars,
            "n_paths": self.n_paths,
            "upside_prob": round(self.upside_prob, 4),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "peak_candle": peak["candle_index"],
            "peak_expiry_minutes": peak["expiry_minutes"],
            "peak_direction": peak["direction"],
            "peak_conviction": peak["conviction"],
            "median_move_pct": round(float(np.median(m)), 4),
            "median_abs_move_pct": round(float(np.median(np.abs(m))), 4),
            "hist_vol": round(self.hist_vol, 6),
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


class KronosEngine:
    """Loads a Kronos model and produces batched Monte Carlo forecasts."""

    def __init__(self, model_id: str, tokenizer_id: str, device: str = "auto",
                 max_context: int = 512, clip: int = 5):
        import torch
        from model import Kronos, KronosPredictor, KronosTokenizer

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.max_context = max_context

        t0 = time.time()
        ensure_weights(model_id, tokenizer_id, cache_dir=WEIGHTS_DIR, verbose=False)
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, cache_dir=WEIGHTS_DIR)
        model = Kronos.from_pretrained(model_id, cache_dir=WEIGHTS_DIR)
        model.eval()
        self.predictor = KronosPredictor(model, tokenizer, device=device,
                                         max_context=max_context, clip=clip)
        self.n_params = sum(p.numel() for p in model.parameters())
        self._lock = threading.Lock()
        log.info("loaded %s (%.1fM params) on %s in %.1fs",
                 model_id, self.n_params / 1e6, device, time.time() - t0)

    @staticmethod
    def _future_index(last_ts: pd.Timestamp, interval: str, n: int) -> pd.DatetimeIndex:
        step = pd.Timedelta(minutes=INTERVAL_MINUTES.get(interval, 5))
        return pd.date_range(last_ts + step, periods=n, freq=step)

    def forecast(self, df: pd.DataFrame, interval: str, horizon_bars: int,
                  n_paths: int, lookback: int, symbol: str = "",
                  T: float = 1.0, top_k: int = 0, top_p: float = 0.9,
                  max_batch: int = 64) -> Forecast:
        """Sample n_paths independent futures from the last lookback bars."""
        actual_lookback = min(lookback, len(df))
        if actual_lookback < 50:
            raise ValueError(f"need at least 50 bars of context, got {len(df)}")

        ctx = df.iloc[-actual_lookback:].reset_index(drop=True)
        # Ensure volume and amount exist even if synthetic indices have 0
        for col in ["volume", "amount"]:
            if col not in ctx.columns:
                ctx[col] = 0.0

        x_df = ctx[KRONOS_COLS]
        x_ts = ctx["timestamps"]
        last_ts = pd.Timestamp(x_ts.iloc[-1])
        last_close = float(ctx["close"].iloc[-1])
        y_ts = self._future_index(last_ts, interval, horizon_bars)
        y_ser = pd.Series(y_ts)

        recent = ctx["close"].to_numpy()[-(horizon_bars + 1):]
        hist_vol = float(np.diff(np.log(recent)).std(ddof=1)) if len(recent) > 2 else 0.0

        t0 = time.time()
        chunks = []
        with self._lock:
            remaining = n_paths
            while remaining > 0:
                b = min(remaining, max_batch)
                dfs = self.predictor.predict_batch(
                    df_list=[x_df] * b,
                    x_timestamp_list=[x_ts] * b,
                    y_timestamp_list=[y_ser] * b,
                    pred_len=horizon_bars,
                    T=T, top_k=top_k, top_p=top_p,
                    sample_count=1, verbose=False,
                )
                chunks.append(np.stack([d[KRONOS_COLS].to_numpy() for d in dfs]))
                remaining -= b
        paths = np.concatenate(chunks, axis=0).astype(np.float64)

        return Forecast(
            symbol=symbol, interval=interval, last_close=last_close,
            last_ts=last_ts, horizon_bars=horizon_bars, n_paths=int(paths.shape[0]),
            paths=paths, y_timestamps=y_ts, hist_vol=hist_vol,
            elapsed_sec=time.time() - t0,
        )
