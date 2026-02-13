# Polynance

Simulated trading bot for 15-minute crypto prediction markets (BTC, ETH, SOL, XRP).

Supports **Polymarket** and **Kalshi** exchanges via a unified abstraction layer. Collects real-time prediction prices and Binance spot prices every 30 seconds, builds 15-minute windows, and executes simulated trades using configurable contrarian/momentum strategies.

## Quick Start

```bash
# 1. Clone and install
git clone git@github.com:YOUR_USERNAME/polynance.git
cd polynance
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -e .

# 2. Create default config
polynance-trade --init-config

# 3. Run the trading bot (data collection + simulated trading + live dashboard)
polynance-trade
```

Requires **Python 3.11+**.

## Exchange Support

| Exchange | Config Value | Fee Model | Status |
|----------|-------------|-----------|--------|
| Polymarket | `"exchange": "polymarket"` | Flat (fee_rate + spread) | Dry-run ready |
| Kalshi | `"exchange": "kalshi"` | Probability-weighted | Dry-run ready |

Set `"exchange"` and `"fee_model"` in your config JSON. Kalshi uses the public API (no authentication required for market data).

Run both exchanges simultaneously with separate configs:

```bash
polynance-trade --config config/config_consensus.json          # Polymarket
polynance-trade --config config/config_consensus_kalshi.json   # Kalshi
```

## Entry Modes

All strategies use **fixed bet sizing** ($50 per trade, capped at 5% of bankroll), with optional **bet scaling** to increase bet size as bankroll grows.

Set the active strategy via `entry_mode` in `config/config.json`:

### Contrarian Family (recommended)

These strategies bet on mean-reversion after one or more strong previous windows.

| Mode | Description | Key Idea |
|------|-------------|----------|
| `contrarian` | Single contrarian | Previous window strong in one direction, bet on reversal |
| `contrarian_consensus` | Contrarian + cross-asset | Like contrarian, but N-of-4 assets must show same strong prev signal |
| `accel_dbl` | Double contrarian + neutral acceleration | **Two** consecutive strong prev windows + t0 must be near 0.50 (neutral) |
| `combo_dbl` | Double contrarian + cross-asset + stop-loss | **Two** consecutive strong prev windows + 2+ other assets also double-strong, with stop-loss |
| `triple_filter` | Double contrarian + cross-asset + PM t0 confirm | **Two** consecutive strong prev windows + N assets double-strong + PM t0 confirms direction |

#### How Contrarian Works

1. At the end of each 15-min window, record pm_yes@t12.5 (the Polymarket YES price at t=12.5 minutes)
2. If that price was "strong" (>= 0.75 for bullish, <= 0.25 for bearish), the next window is a contrarian candidate
3. **Double contrarian** (accel_dbl, combo_dbl): requires **two consecutive** strong windows in the same direction before entering
4. At the configured entry time (default t=5 min), if the current PM price confirms the reversal direction, enter the trade
5. Exit at the configured exit time (default t=12.5 min) using early-exit P&L calculation

#### ACCEL_DBL (Acceleration + Double Contrarian)

The best risk-adjusted strategy from backtesting. On top of double contrarian, it requires:
- **t0 neutral filter**: pm_yes at t=0 must be within `accel_neutral_band` of 0.50 (market hasn't already moved)
- This catches windows where the market is "loading" for a reversal move

#### COMBO_DBL (Combo + Double Contrarian)

Adds cross-asset confirmation and risk management:
- **Cross-asset filter**: at least `combo_xasset_min` other assets must also be double-strong in the same direction
- **Stop-loss**: at `combo_stop_time` (default t=7.5), if the position has moved against by >= `combo_stop_delta`, exit early

#### TRIPLE_FILTER (Triple Confirmation)

The most selective strategy. Requires all three filters to pass:
- **Double contrarian**: two consecutive strong prev windows in the same direction
- **Cross-asset consensus**: at least `triple_xasset_min` assets must be double-strong
- **PM t0 confirmation**: pm_yes at t=0 must confirm the expected direction (>= `triple_pm0_bull_min` for bull, <= `triple_pm0_bear_max` for bear)

### Legacy Modes

| Mode | Description |
|------|-------------|
| `two_stage` | Signal at t=7.5, confirm at t=10. Filters "faders" that reverse between signal and confirm |
| `single` | Single threshold check at t=7.5, enter immediately |

## Configuration

Config lives at `config/config.json`. Create a default one with:

```bash
polynance-trade --init-config
```

### Full Config Reference

```jsonc
{
  // Exchange
  "exchange": "polymarket",         // "polymarket" or "kalshi"
  "fee_model": "flat",              // "flat" (polymarket) or "probability_weighted" (kalshi)

  // Strategy selection
  "entry_mode": "accel_dbl",        // "contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter", "two_stage", "single"

  // --- Contrarian base settings (used by all contrarian-family modes) ---
  "contrarian_prev_thresh": 0.75,   // prev window pm@t12.5 must be >= this (or <= 1-this) to be "strong"
  "contrarian_bull_thresh": 0.50,   // current pm must be >= this to confirm bull reversal
  "contrarian_bear_thresh": 0.50,   // current pm must be <= this to confirm bear reversal
  "contrarian_entry_time": "t0",    // entry sample time (for plain contrarian mode)
  "contrarian_exit_time": "t12.5",  // exit sample time

  // --- Consensus settings (contrarian_consensus mode) ---
  "consensus_min_agree": 3,         // N-of-4 assets must agree on direction
  "consensus_entry_time": "t5",
  "consensus_exit_time": "t12.5",

  // --- ACCEL_DBL settings ---
  "accel_neutral_band": 0.15,       // t0 pm must be within 0.50 +/- this
  "accel_prev_thresh": 0.75,        // prev window strength threshold (double required)
  "accel_bull_thresh": 0.55,        // bull confirmation at entry time
  "accel_bear_thresh": 0.45,        // bear confirmation at entry time
  "accel_entry_time": "t5",
  "accel_exit_time": "t12.5",

  // --- COMBO_DBL settings ---
  "combo_prev_thresh": 0.75,
  "combo_bull_thresh": 0.55,
  "combo_bear_thresh": 0.45,
  "combo_entry_time": "t5",
  "combo_exit_time": "t12.5",
  "combo_stop_time": "t7.5",        // stop-loss check time
  "combo_stop_delta": 0.10,         // exit early if position moved against by this much
  "combo_xasset_min": 2,            // min other assets that must also be double-strong

  // --- TRIPLE_FILTER settings ---
  "triple_prev_thresh": 0.70,        // prev window strength (double required)
  "triple_bull_thresh": 0.55,
  "triple_bear_thresh": 0.45,
  "triple_entry_time": "t5",
  "triple_exit_time": "t12.5",
  "triple_xasset_min": 3,            // min assets that must be double-strong
  "triple_pm0_bull_min": 0.50,       // pm t0 must be >= this for bull
  "triple_pm0_bear_max": 0.50,       // pm t0 must be <= this for bear

  // --- Two-stage / Single mode settings ---
  "signal_threshold_bull": 0.70,
  "signal_threshold_bear": 0.30,
  "confirm_threshold_bull": 0.85,
  "confirm_threshold_bear": 0.15,
  "bull_threshold": 0.80,           // single mode only
  "bear_threshold": 0.20,           // single mode only

  // --- Financial settings ---
  "initial_bankroll": 1000.0,
  "base_bet": 50.0,                 // fixed bet size per trade
  "fee_rate": 0.001,                // 0.1% taker fee
  "spread_cost": 0.005,             // 0.5% spread estimate
  "max_bet_pct": 0.05,              // max 5% of bankroll per trade

  // --- Bet scaling (optional) ---
  "bet_scale_threshold": 1.0,       // scale up every 100% gain (0 = disabled)
  "bet_scale_increase": 0.20,       // +20% per threshold step

  // --- Trading mechanics ---
  "min_trajectory": 0.20,           // minimum pm move from t0 to entry (trajectory filter)
  "pause_windows_after_loss": 1,    // skip N windows after any loss

  // --- System ---
  "assets": ["BTC", "ETH", "SOL", "XRP"],
  "data_dir": "data",
  "show_dashboard": true,
  "dashboard_refresh_rate": 2.0,
  "run_analysis": false
}
```

Valid time values: `"t0"`, `"t2.5"`, `"t5"`, `"t7.5"`, `"t10"`, `"t12.5"`.

## CLI Usage

### Trading Bot (primary)

```bash
# Run with default config (config/config.json)
polynance-trade

# Use a different config file
polynance-trade --config path/to/config.json

# Reset trading state (wipes sim_trading.db and starts fresh)
polynance-trade --reset

# Show current config and exit
polynance-trade --show-config

# Create a default config file
polynance-trade --init-config

# Disable dashboard (log-only mode)
polynance-trade --no-dashboard

# Verbose logging
polynance-trade -v
```

### Data Collection Only (no trading)

```bash
polynance
```

### Performance Charts

```bash
# Display chart in terminal
polynance-chart

# Save chart to file
polynance-chart --output chart.png
```

### Analysis

```bash
polynance-analyze
```

### Backfill Database (after schema updates)

```bash
python scripts/backfill_db.py
```

## How It Works

### Data Pipeline

1. **Sampling** (every 30 seconds): Fetches Polymarket YES/NO/bid/ask/spread prices + Binance spot price for each asset
2. **Window construction**: Groups samples into 15-minute windows at :00/:15/:30/:45 boundaries, with samples at t=0, 2.5, 5, 7.5, 10, 12.5, 15 minutes
3. **Window finalization** (at t=15): Computes outcome (up/down), spot stats, PM prices at each timepoint, volatility regime, and stores to database
4. **Trade routing**: Based on `entry_mode`, routes each sample to the appropriate trader method (e.g., `on_sample_at_accel_entry` for accel_dbl at t=5)
5. **Trade resolution**: When a window completes, open trades are resolved using early-exit P&L (entry contract price vs exit contract price minus fees)

### P&L Calculation (Early Exit)

Since trades exit before the binary outcome resolves (at t=12.5 instead of t=15), P&L is computed from contract price movement:

```
n_contracts = bet_size / entry_contract_price
gross_pnl = n_contracts * (exit_contract_price - entry_contract_price)
net_pnl = gross_pnl - taker_fee - spread_cost
```

### Dashboard

The Rich terminal dashboard shows:
- **Header**: Current window phase (PRE-ENTRY / ENTRY / HOLDING / EXIT / DONE), countdown timers
- **Portfolio**: Bankroll, P&L, return %, max drawdown, today's stats
- **Metrics**: Win rate, profit factor, Sharpe, Sortino, Calmar, recovery factor, expectancy
- **Open Positions**: Currently held trades
- **Recent Trades**: Last 10 resolved trades with P&L
- **Per-Asset Summary**: Trades, win %, P&L, signal, and volatility regime per asset
- **Window Status**: Previous PM values, entry prices, current prices, and trade status per asset

## Data Storage

### Per-Asset Databases (`data/btc.db`, `data/eth.db`, `data/sol.db`, `data/xrp.db`)

**`samples`** table: Raw 30-second snapshots (pm_yes_price, pm_spread, spot_price, etc.)

**`windows`** table: Aggregated 15-minute windows with:
- Outcome (up/down) and spot stats (open, close, change_bps, range_bps, high, low)
- PM prices at all timepoints (t0, t2.5, t5, t7.5, t10, t12.5)
- Cross-window references: `prev_pm_t12_5`, `prev2_pm_t12_5` (previous windows' pm_yes@t12.5)
- `window_time`: time-only key for cross-asset queries (e.g., `"20260209_1530"`)
- `volatility_regime`: classified as `low` (<15 bps range), `normal` (15-40), `high` (40-80), `extreme` (>80)

### Trading Database (`data/sim_trading.db`)

**`sim_state`** table: Single-row portfolio state (bankroll, P&L, streaks, drawdown)

**`sim_trades`** table: Individual trades with:
- Trade details: window_id, asset, direction, entry/exit prices, bet_size
- P&L breakdown: gross_pnl, fee_paid, spread_cost, net_pnl
- Strategy metadata: `entry_mode`, `prev_pm`, `prev2_pm`, `spot_velocity`, `pm_momentum`

## Project Structure

```
polynance/
├── src/polynance/
│   ├── main.py                    # Application orchestrator, sample routing
│   ├── sampler.py                 # 30-sec data collection, window finalization
│   ├── clients/
│   │   ├── exchange.py            # ExchangeClient ABC + factory
│   │   ├── polymarket.py          # Polymarket CLOB API client
│   │   ├── polymarket_adapter.py  # Polymarket ExchangeClient adapter
│   │   ├── kalshi_adapter.py      # Kalshi ExchangeClient adapter (public REST API)
│   │   └── binance.py             # Binance spot price client
│   ├── db/
│   │   ├── database.py            # Per-asset SQLite (samples + windows tables)
│   │   └── models.py              # Sample, Window dataclasses
│   ├── trading/
│   │   ├── dry_run.py             # CLI entry point for trading bot
│   │   ├── trader.py              # SimulatedTrader engine (all entry modes)
│   │   ├── config.py              # TradingConfig dataclass + JSON loader
│   │   ├── database.py            # TradingDatabase (sim_state + sim_trades)
│   │   ├── models.py              # SimulatedTrade, TradingState, PerAssetStats
│   │   ├── dashboard.py           # Rich trading dashboard
│   │   ├── chart.py               # Matplotlib performance charts
│   │   └── bet_sizing.py          # Bet sizing strategies
│   ├── analysis/
│   │   ├── analyzer.py            # Per-asset statistical analysis
│   │   ├── hourly_analyzer.py     # Hourly aggregation
│   │   └── polymarket_trading.py  # Trading-specific analysis
│   └── dashboard/
│       └── terminal.py            # Data-collection-only dashboard
├── analysis/
│   ├── backtest_suite.py          # Strategy backtesting framework
│   ├── slippage_model.py          # Spread/slippage impact modeling
│   └── advanced_strategy_ideas.py # Experimental strategy backtests
├── scripts/
│   └── backfill_db.py             # Database migration/backfill utility
├── config/
│   ├── config.json                # Default config (accel_dbl, Polymarket)
│   ├── config_consensus.json      # Contrarian consensus (Polymarket)
│   ├── config_consensus_kalshi.json # Contrarian consensus (Kalshi)
│   └── config_triple.json         # Triple filter (Polymarket)
├── data/                          # SQLite databases (gitignored)
├── pyproject.toml                 # Package config and dependencies
└── README.md
```

## Backtesting

Historical backtests run against the collected database:

```bash
# Full strategy sweep (train/test split)
python analysis/backtest_suite.py

# Slippage/spread impact modeling
python analysis/slippage_model.py

# Advanced strategy ideas (spot velocity, triple contrarian, etc.)
python analysis/advanced_strategy_ideas.py
```

Results are printed to stdout. Charts saved to `analysis/reports/`.

## Dependencies

Core:
- `aiohttp`, `aiosqlite`, `websockets` (async networking)
- `pandas`, `numpy`, `scipy`, `scikit-learn` (data/analytics)
- `rich`, `plotext` (terminal UI)
- `matplotlib`, `seaborn` (charting)
- `python-dotenv`, `pytz` (utilities)

Dev (optional):
- `pytest`, `pytest-asyncio`, `black`, `ruff`

Install dev deps: `pip install -e ".[dev]"`

## Disclaimer

This is a **simulation only**. No real trades are executed on any exchange. Past performance does not guarantee future results. Prediction markets carry significant risk.

## License

Private - All rights reserved.
