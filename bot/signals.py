"""Deriv Rise/Fall signal generation.

Selects the future candle with the highest probability peak from the 5m forecast,
confirms alignment against the 30m macro regime, and checks risk gates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .engine import Forecast

log = logging.getLogger(__name__)

CALL, PUT, FLAT = "CALL", "PUT", "FLAT"


@dataclass
class RiseFallTradePlan:
    symbol: str
    direction: str                # CALL | PUT | FLAT
    target_candle: int            # Index of peak candle (1 to 20)
    expiry_minutes: int           # Duration in minutes (e.g. 35)
    conviction: float             # Peak conviction score (0.5 to 1.0)
    upside_prob: float            # P(Rise) for peak candle
    downside_prob: float          # P(Fall) for peak candle
    regime_direction: str         # CALL | PUT | TIE
    regime_conviction: float
    spot: float
    stake: float
    currency: str = "USD"
    reason: str = ""
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)
    gates: Dict[str, Any] = field(default_factory=dict)

    @property
    def tradeable(self) -> bool:
        return self.direction in (CALL, PUT) and self.stake > 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "target_candle": self.target_candle,
            "expiry_minutes": self.expiry_minutes,
            "conviction": round(self.conviction, 4),
            "upside_prob": round(self.upside_prob, 4),
            "downside_prob": round(self.downside_prob, 4),
            "regime_direction": self.regime_direction,
            "regime_conviction": round(self.regime_conviction, 4),
            "spot": self.spot,
            "stake": self.stake,
            "currency": self.currency,
            "tradeable": self.tradeable,
            "reason": self.reason,
            "gates": self.gates,
            "candidates": self.all_candidates,
        }


def build_plan(cfg, entry_fc: Forecast, regime_fc: Optional[Forecast],
               spot: float, balance: Optional[float] = None) -> RiseFallTradePlan:
    """Build trade plan by finding peak-probability candle and verifying regime."""
    candidates = entry_fc.step_predictions()
    peak = entry_fc.peak_candle

    peak_dir = peak.get("direction", "TIE")
    peak_conv = peak.get("conviction", 0.5)
    target_candle = peak.get("candle_index", 1)
    expiry_minutes = peak.get("expiry_minutes", 5)

    # 30m Regime Evaluation (only if require_regime_agree is enabled and regime_fc exists)
    use_regime = bool(cfg.require_regime_agree and regime_fc is not None)
    if use_regime:
        regime_up = regime_fc.upside_prob
        regime_down = 1.0 - regime_up
        regime_conf = max(regime_up, regime_down)
        regime_dir = CALL if regime_up > 0.5 else PUT if regime_up < 0.5 else "TIE"
        is_opposed = (
            (peak_dir == CALL and regime_dir == PUT and regime_conf >= cfg.min_regime_prob) or
            (peak_dir == PUT and regime_dir == CALL and regime_conf >= cfg.min_regime_prob)
        )
        regime_ok = not is_opposed
    else:
        regime_up = 0.5
        regime_conf = 0.5
        regime_dir = "N/A"
        regime_ok = True

    gates: Dict[str, Any] = {}
    gates["peak_prob"] = {
        "value": round(peak_conv, 4),
        "min": cfg.min_prob,
        "direction": peak_dir,
        "candle": target_candle,
        "pass": peak_conv >= cfg.min_prob and peak_dir in (CALL, PUT)
    }

    if use_regime:
        gates["regime_agree"] = {
            "value": round(regime_conf, 4),
            "min": cfg.min_regime_prob,
            "regime_dir": regime_dir,
            "peak_dir": peak_dir,
            "pass": regime_ok
        }

    stake = cfg.calculate_stake(balance)

    plan = RiseFallTradePlan(
        symbol=cfg.symbol,
        direction=peak_dir if peak_dir in (CALL, PUT) else FLAT,
        target_candle=target_candle,
        expiry_minutes=expiry_minutes,
        conviction=peak_conv,
        upside_prob=peak.get("upside_prob", 0.5),
        downside_prob=peak.get("downside_prob", 0.5),
        regime_direction=regime_dir,
        regime_conviction=regime_conf,
        spot=spot,
        stake=stake,
        currency=cfg.currency,
        all_candidates=candidates,
        gates=gates,
    )

    failed = [k for k, v in gates.items() if not v.get("pass")]
    if failed:
        plan.direction = FLAT
        plan.reason = "blocked: " + ", ".join(failed)
    else:
        if use_regime:
            plan.reason = (f"{peak_dir} @ C#{target_candle} ({expiry_minutes}m) "
                           f"conf={peak_conv:.1%} regime={regime_dir}({regime_conf:.1%})")
        else:
            plan.reason = (f"{peak_dir} @ C#{target_candle} ({expiry_minutes}m) "
                           f"conf={peak_conv:.1%} (single TF)")

    return plan
