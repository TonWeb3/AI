"""Configuration for Kronos Deriv Rise/Fall Trading Bot."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.getenv("KRONOS_CONFIG", os.path.join(ROOT, "config.json"))

SECTIONS = {
    "deriv": ["app_id", "token", "mode"],
    "market": ["symbol", "entry_tf", "regime_tf"],
    "kronos": ["model_id", "tokenizer_id", "device", "max_context", "lookback",
               "entry_pred_len", "regime_pred_len", "n_paths", "n_paths_regime",
               "max_batch", "temperature", "top_p", "top_k"],
    "strategy": ["min_prob", "require_regime_agree", "min_regime_prob",
                 "mid_candle_check", "check_interval_sec",
                 "risk_type", "risk_value", "currency", "max_concurrent",
                 "cooldown_bars", "max_trades_per_day", "max_daily_loss_usd"],
    "infra": ["db_path", "host", "port", "log_level"],
}


@dataclass
class Config:
    # ---------------- deriv ----------------
    app_id: str = "33LuGFJlOlfbqzHVZLFCi"
    token: str = ""
    mode: str = "demo"             # demo | real

    # ---------------- market ----------------
    symbol: str = "R_100"          # Deriv Synthetic or Forex symbol
    entry_tf: str = "5m"           # trigger timeframe
    regime_tf: str = "30m"         # macro trend filter

    # ---------------- kronos ----------------
    model_id: str = "NeoQuasar/Kronos-mini"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-2k"
    device: str = "auto"
    max_context: int = 512
    lookback: int = 360            # bars of historical context
    entry_pred_len: int = 20       # 20 bars ahead (e.g. 20 x 5m = 100m)
    regime_pred_len: int = 24      # 24 bars ahead (24 x 30m = 12h)
    n_paths: int = 48              # Monte Carlo trajectories
    n_paths_regime: int = 24
    max_batch: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 0

    # ---------------- strategy (Rise / Fall) ----------------
    min_prob: float = 0.58         # min conviction required for peak candle
    require_regime_agree: bool = True
    min_regime_prob: float = 0.50  # 30m regime direction conviction threshold
    mid_candle_check: bool = True  # evaluate signals mid-candle inside 5m & 30m bars
    check_interval_sec: int = 150  # interval between evaluations (default 150s = 2.5m)
    risk_type: str = "percent"     # percent (% of balance) | fixed (fixed dollar amount)
    risk_value: float = 10.0       # percent or dollars
    currency: str = "USD"
    max_concurrent: int = 1
    cooldown_bars: int = 1
    max_trades_per_day: int = 20
    max_daily_loss_usd: float = 50.0

    # ---------------- infra ----------------
    db_path: str = "kronos_deriv.db"
    host: str = "0.0.0.0"
    port: int = 8120
    log_level: str = "INFO"

    def calculate_stake(self, balance: Optional[float] = None) -> float:
        """Calculate stake honoring 'percent' of balance or 'fixed' dollar amount."""
        if self.risk_type == "percent":
            bal = balance if balance is not None and balance > 0 else 100.0
            stake = (self.risk_value / 100.0) * bal
        else:
            stake = float(self.risk_value)
        # Enforce Deriv minimum stake of 0.35 USD
        return max(0.35, round(stake, 2))

    # ---------------- io ----------------
    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "Config":
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        names = {f.name for f in fields(cls)}
        flat: dict = {}
        for key, val in raw.items():
            if isinstance(val, dict) and key in SECTIONS:
                flat.update(val)
            elif key in names:
                flat[key] = val
        unknown = [k for k in flat if k not in names]
        for k, v in flat.items():
            if k in names:
                setattr(cfg, k, v)
        cfg._unknown = unknown
        cfg._path = path
        return cfg

    def to_nested(self) -> dict:
        d = asdict(self)
        out: dict = {}
        for section, keys in SECTIONS.items():
            out[section] = {k: d[k] for k in keys if k in d}
        return out

    def save(self, path: str | None = None) -> str:
        path = path or getattr(self, "_path", CONFIG_PATH)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_nested(), fh, indent=2)
        return path

    def redacted(self) -> dict:
        d = self.to_nested()
        if d.get("deriv", {}).get("token"):
            d["deriv"]["token"] = "***set***"
        return d

    # ---------------- validation ----------------
    def validate(self) -> list:
        errs: list = []
        for k in getattr(self, "_unknown", []):
            errs.append(f"unknown key in config.json: {k!r}")
        if self.risk_value <= 0:
            errs.append("risk_value must be > 0")
        if self.risk_type not in ("percent", "fixed"):
            errs.append("risk_type must be 'percent' or 'fixed'")
        if self.min_prob < 0.5 or self.min_prob > 1.0:
            errs.append("min_prob must be between 0.50 and 1.00")
        if self.lookback > self.max_context:
            errs.append(f"lookback={self.lookback} exceeds max_context={self.max_context}")
        if self.entry_pred_len < 1:
            errs.append("entry_pred_len must be >= 1")
        if self.mode == "real" and not self.token:
            errs.append("real mode requires deriv.token to be set")
        return errs


CFG = Config.load()

CHOICES = {
    "mode": ["demo", "real"],
    "symbol": ["R_100", "R_75", "R_50", "R_25", "R_10", "1HZ100V", "1HZ75V", "1HZ50V", "frxEURUSD", "cryBTCUSD"],
    "entry_tf": ["1m", "5m", "15m"],
    "regime_tf": ["15m", "30m", "1h", "4h"],
    "risk_type": ["percent", "fixed"],
    "model_id": ["NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-base"],
    "tokenizer_id": ["NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-Tokenizer-base"],
    "device": ["auto", "cuda:0", "cpu"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
}

NEEDS_RESTART = {
    "symbol", "entry_tf", "regime_tf", "model_id", "tokenizer_id", "device",
    "max_context", "mode", "token", "app_id", "db_path"
}

SECRET_FIELDS = {"token"}

HELP = {
    "symbol": "Deriv market symbol (e.g. Volatility 100 Index R_100, R_75, EUR/USD).",
    "mode": "Trading account mode: demo or real.",
    "entry_tf": "Entry candle timeframe (default 5m).",
    "regime_tf": "Macro regime trend filter timeframe (default 30m).",
    "entry_pred_len": "Number of future candles to predict (e.g. 20 candles = 100 mins at 5m).",
    "min_prob": "Minimum conviction threshold max(P(Rise), P(Fall)) required for trade execution.",
    "require_regime_agree": "When enabled, evaluates the 30m macro regime filter alongside 5m entries. When disabled, uses strictly the single entry timeframe (5m) and hides the 30m regime filter from the dashboard.",
    "min_regime_prob": "How strongly the 30m regime must lean in the trade's direction to confirm entry.",
    "mid_candle_check": "Enable mid-candle signal evaluations inside the 5m and 30m bars without waiting only for candle close.",
    "check_interval_sec": "Frequency in seconds for mid-candle rechecks (default 150s = halfway through 5m bar).",
    "risk_type": "Risk per trade sizing mode: 'percent' (% of account balance) or 'fixed' (fixed dollar amount).",
    "risk_value": "Percent of balance (e.g. 10.0 = 10%) or fixed dollar amount (e.g. 10.0 = $10).",
    "n_paths": "Number of Monte Carlo future paths sampled by Kronos per forecast.",
}


def settings_schema(cfg: Config) -> list:
    from dataclasses import fields as dc_fields
    out = []
    by_key = {k: sec for sec, keys in SECTIONS.items() for k in keys}
    for f in dc_fields(cfg):
        if f.name not in by_key:
            continue
        val = getattr(cfg, f.name)
        kind = "bool" if isinstance(val, bool) else "int" if isinstance(val, int) else "float" if isinstance(val, float) else "str"
        out.append({
            "key": f.name,
            "section": by_key[f.name],
            "type": "select" if f.name in CHOICES else kind,
            "options": CHOICES.get(f.name),
            "value": ("***set***" if f.name in SECRET_FIELDS and val else val),
            "secret": f.name in SECRET_FIELDS,
            "restart": f.name in NEEDS_RESTART,
            "help": HELP.get(f.name, ""),
        })
    return out


def coerce(cfg: Config, key: str, raw: Any) -> Any:
    cur = getattr(cfg, key)
    if isinstance(cur, bool):
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(float(raw))
    if isinstance(cur, float):
        return float(raw)
    return str(raw)
