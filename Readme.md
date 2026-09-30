# Kronos Deriv — Rise & Fall Autonomous Trading Bot

An autonomous **Deriv Rise/Fall (`CALL`/`PUT`)** trading bot driven by [Kronos](https://github.com/shiyu-coder/Kronos), the foundation model for financial candlesticks.

The bot operates on **5-minute (`5m`) entries** with a **30-minute (`30m`) macro regime filter**, projecting a **20-candle forward probability horizon** (100 minutes) to dynamically identify and trade the optimal contract expiry duration.

---

## 1. How the Strategy Works

```
5m Candle Closes
   │
   ├─► Kronos Monte Carlo Engine samples 24 forward paths across the next 20 candles (100m)
   ├─► Calculates Upside P(Close_k > Spot) & Downside P(Close_k < Spot) for each candle k ∈ [1, 20]
   ├─► Identifies the Peak Conviction Candle: k* = argmax_k conviction_k
   ├─► Sets Deriv Contract Expiry Duration: Duration = k* × 5 minutes
   │
   ├─► 30m Macro Regime Filter Check:
   │      • CALL (Rise) requires 30m regime P(Up) ≥ 50%
   │      • PUT (Fall) requires 30m regime P(Up) < 50%
   │
   ├─► Minimum Conviction Filter:
   │      • Peak Conviction must clear threshold (default ≥ 58%)
   │
   └─► If Autonomous Trading is ACTIVE:
          • Requests Deriv Rise/Fall proposal for selected duration
          • Executes buy contract
          • Monitors floating P&L, spot vs entry, and countdown until settlement
```

### 20-Candle Horizon Peak Selection Engine
Instead of guessing a fixed expiry, the bot evaluates the Monte Carlo trajectory distribution across the entire 20-candle horizon:
- If Candle #3 has the highest upside conviction (e.g. 78%), the bot selects **Candle #3 (15m expiry)**.
- If a trend takes longer to unfold and Candle #12 shows the peak probability (e.g. 96%), the bot selects **Candle #12 (60m expiry)**.
- The contract direction matches the peak candle: `CALL` (Rise) or `PUT` (Fall).

### 30m Macro Regime Filter
Ensures trades are executed in the direction of the higher-timeframe trend:
- Prevents buying `CALL` options in a macro downtrend.
- Prevents buying `PUT` options in a macro uptrend.

---

## 2. Key Features

- **Deriv WebSocket API**: Direct, low-latency WebSocket connection with token authorization, live tick streaming, and automated proposal pricing.
- **Position Sizing (Percent & Fixed)**:
  - `"percent"`: Stakes a percentage of live account balance (e.g. `10.0` = 10%).
  - `"fixed"`: Stakes a fixed dollar amount (e.g. `10.0` = $10.00 USD).
  - Enforces Deriv's minimum contract stake floor ($0.35 USD).
- **Start / Stop Control**:
  - Starts in **Standby (`AUTONOMOUS: STOPPED`)** on boot.
  - Live market data, 5m candlestick charting, and AI predictions stream continuously in standby.
  - Trades are executed only after clicking the green **`▶ Start`** button.
  - Clicking red **`■ Stop`** pauses order placement immediately without interrupting feeds.
- **Cyber-Teal Real-Time Dashboard**:
  - **Locked 5M SVG Candlestick Chart**: Renders 120 historical 5m bars, median forecast candles, uncertainty envelopes, and a cyan guideline marking the optimal expiry candle (`★ C#k (Xm EXPIRY)`).
  - **20-Candle Expiry Grid**: Visual cards for every future candle displaying conviction %, direction, and target time.
  - **Strategy Expiry Decision Hero**: Prominently displays the chosen candidate, duration, and regime alignment.
  - **Active Contract Card**: Displays live open contract details (stake, potential payout, entry spot vs current spot, floating P&L, remaining timer).
  - **Decisions & Trades History**: Complete audit trail of all evaluated signals and settled Deriv contracts stored in SQLite.

---

## 3. Project Structure

```
Kronos Deriv/
├── .gitignore             # Excludes model weights cache, databases, and checkpoints
├── config.json            # Active configuration (credentials, market, strategy, risk)
├── main.py                # Main application entry point (CLI & server launcher)
├── requirements.txt       # Python dependencies
├── ReadMe.md              # Documentation
├── bot/                   # Core trading engine & backend API
│   ├── api.py             # FastAPI REST endpoints & WebSocket dashboard handlers
│   ├── config.py          # Configuration schema, validation & dynamic settings
│   ├── deriv_client.py    # Deriv WebSocket client, tick stream, proposal & buy
│   ├── engine.py          # Kronos model loader, Monte Carlo inference & peak calculation
│   ├── signals.py         # RiseFallTradePlan builder, regime checks & sizing
│   ├── storage.py         # SQLite persistence for decisions, trades & forecasts
│   └── trader.py          # Main trading loop, 5m/30m candle stores & position manager
├── model/                 # Kronos foundation model architecture
│   ├── kronos.py          # Kronos transformer implementation
│   ├── module.py          # Attention modules, quantizers & tokenizer
│   └── weights/           # Local Hugging Face cache (auto-downloaded, git-ignored)
└── templates/
    └── index.html         # Cyber-teal real-time web dashboard
```

---

## 4. Getting Started

### Prerequisites
- Python 3.10 to 3.12
- Deriv account (Demo or Real) and API token from [app.deriv.com](https://app.deriv.com/account/api-token)

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd "Kronos Deriv"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure credentials in `config.json`:
   ```json
   {
     "deriv": {
       "app_id": "33LuGFJlOlfbqzHVZLFCi",
       "token": "YOUR_DERIV_API_TOKEN",
       "mode": "demo"
     },
     "market": {
       "symbol": "R_100",
       "entry_tf": "5m",
       "regime_tf": "30m"
     },
     "strategy": {
       "min_prob": 0.58,
       "require_regime_agree": true,
       "risk_type": "percent",
       "risk_value": 10.0
     }
   }
   ```

4. Verify configuration:
   ```bash
   python main.py --check
   ```

---

## 5. Running the Bot

Start the bot and web dashboard:
```bash
python main.py
```

Open your browser to:
```
http://localhost:8120
```

- The 5m candlestick chart will load immediately.
- The 20-candle forecast envelope and peak expiry selection (`★ C#k`) will generate within ~15 seconds.
- Click the green **`▶ Start`** button in the top right to enable autonomous trade execution.
- Click the red **`■ Stop`** button at any time to pause trading.

---

## 6. Supported Symbols (Rise / Fall)

The bot supports all Deriv markets offering Rise/Fall contracts:
- **Gold & Commodities**: `frxXAUUSD` (Gold/USD), `frxXAGUSD` (Silver), `frxXPDUSD` (Palladium), `frxXPTUSD` (Platinum), `frxXBRUSD` (Brent Oil), `frxXTIUSD` (WTI Oil)
- **Synthetic Volatility Indices (24/7)**: `R_100`, `R_75`, `R_50`, `R_25`, `R_10`
- **1-Second Volatility Indices (24/7)**: `1HZ100V`, `1HZ75V`, `1HZ50V`, `1HZ30V`, `1HZ25V`, `1HZ15V`, `1HZ10V`, `1HZ90V`, `1HZ150V`, `1HZ200V`, `1HZ250V`, `1HZ300V`
- **Crash & Boom Indices**: `CRASH_1000`, `CRASH_500`, `CRASH_300`, `BOOM_1000`, `BOOM_500`, `BOOM_300`
- **Jump & Step Indices**: `JD10`, `JD25`, `JD50`, `JD75`, `JD100`, `stpRNG`
- **Forex Majors & Pairs**: `frxEURUSD`, `frxGBPUSD`, `frxUSDJPY`, `frxAUDUSD`, `frxUSDCAD`, `frxUSDCHF`, `frxEURGBP`, `frxEURJPY`, `frxGBPJPY`
- **Cryptocurrencies (24/7)**: `cryBTCUSD`, `cryETHUSD`, `cryLTCUSD`, `cryXRPUSD`

---

## 7. Model Selection (CPU vs GPU)

You can select models directly from the dashboard **Settings** tab:
- **`NeoQuasar/Kronos-mini` (4.1M parameters)**: **Default / Recommended for CPU**. Inference completes in ~15–18 seconds.
- **`NeoQuasar/Kronos-small` (24M parameters)**: Recommended for fast multi-core CPUs or entry GPUs.
- **`NeoQuasar/Kronos-base` (100M+ parameters)**: Recommended when running with an NVIDIA GPU (`cuda:0`).

*(Hugging Face will automatically download the chosen model from the Hub on first run).*
