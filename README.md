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

| Exchange | Config Value | Fee Model | Market Data | Live Trading |
|----------|-------------|-----------|-------------|-------------|
| Polymarket | `"exchange": "polymarket"` | Flat (fee_rate + spread) | Direct API | Via polymarket CLI |
| Kalshi | `"exchange": "kalshi"` | Probability-weighted | Public REST API | Not yet |

Set `"exchange"` and `"fee_model"` in your config JSON. Kalshi uses the public API (no authentication required for market data).

Run both exchanges simultaneously with separate configs:

```bash
polynance-trade --config config/config_consensus.json          # Polymarket
polynance-trade --config config/config_consensus_kalshi.json   # Kalshi
```

## Live Trading (Polymarket)

Polynance supports live order placement on Polymarket via the [polymarket CLI](https://github.com/Polymarket/polymarket-cli). **This is disabled by default** — the bot runs in dry-run (simulation) mode unless explicitly enabled.

### Setup

```bash
# 1. Install the polymarket CLI
curl -sSL https://raw.githubusercontent.com/Polymarket/polymarket-cli/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Set credentials in .env (or export directly)
POLYMARKET_PRIVATE_KEY="your-polygon-private-key"
POLYMARKET_SIGNATURE_TYPE=1              # 0=EOA, 1=proxy wallet, 2=Gnosis Safe
POLYMARKET_FUNDER_ADDRESS="your-proxy-wallet-address"  # proxy wallet holding funds

# 3. Enable in config
```

Add to your config JSON:

```json
{
  "exchange": "polymarket",
  "live_trading": true,
  "signature_type": 1,
  "redeem_on_window_complete": true
}
```

### How It Works

When `live_trading` is enabled:
1. The bot still runs the full simulation (tracking P&L, win rate, drawdown, etc.)
2. **In addition**, it places real orders on Polymarket via the polymarket CLI
3. Entry orders buy YES/NO contracts (GTC limit orders) at the signal price
4. Exit orders use **FAK (Fill-And-Kill) sell loops** that drain the position via best available bids
5. **Wallet balance** from Polymarket is the source of truth for bankroll tracking
6. **Settled positions** are auto-redeemed via the gasless relayer after each window
7. **Losing positions** are cleaned up hourly to keep the portfolio tidy
8. If a live order fails, the simulation continues unaffected

### Position Lifecycle

```
Entry signal → GTC buy order → poll for fill → hold position
  → Exit signal → FAK sell loop (up to 5 attempts) → position drained
  → Window resolves → check for redeemable positions → gasless redeem via relayer
  → Hourly → sweep zero-value losers → clean up portfolio
  → Balance refresh → bankroll synced from Polymarket
```

### Safety

- `live_trading` defaults to `false` — you must explicitly enable it
- The polymarket CLI must be installed separately (`~/.local/bin/polymarket`)
- Missing `POLYMARKET_PRIVATE_KEY` gracefully disables live trading with a warning
- Live order errors are logged but never crash the bot
- The simulation always runs regardless of live order outcomes
- Redemption is rate-limited (once per 15 min for winners, once per hour for losers)
- Relayer circuit breaker protects against quota exhaustion

### Authentication

The polymarket CLI handles all signing and CLOB API communication:
- **Private Key**: Your Polygon wallet private key (signs order transactions via CLI)
- **Signature Type**: Wallet type (0=EOA, 1=proxy wallet, 2=Gnosis Safe)
- **Funder Address**: Proxy wallet address that holds funds and positions

These are read from `.env` via python-dotenv or from environment variables.

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

  // --- Live trading (CAUTION: real money) ---
  "live_trading": false,            // true = place real orders via polymarket CLI (default: false)
  "signature_type": 1,              // 0=EOA, 1=proxy wallet, 2=Gnosis Safe
  "redeem_on_window_complete": true, // auto-redeem settled positions after each window

  // --- Trading mechanics ---
  "min_trajectory": 0.20,           // minimum pm move from t0 to entry (trajectory filter)
  "pause_windows_after_loss": 1,    // skip N windows after any loss

  // --- Regime & day filters ---
  "skip_regimes": ["high", "extreme"],  // skip windows after these vol regimes (low/normal/high/extreme)
  "skip_days": [5],                     // skip Saturdays (0=Mon, 5=Sat, 6=Sun)

  // --- Prior momentum filter (contrarian_consensus only) ---
  "prior_mom_filter": false,            // require trend still building at t0 (no look-ahead)
  "prior_mom_min": 0.03,               // minimum |prev_pm - prev2_pm| to pass

  // --- Tiered exit (contrarian_consensus only) ---
  "tiered_exit": false,                 // hold to binary resolution when n_agree >= threshold
  "tiered_resolution_threshold": 3,    // minimum agreeing assets for binary resolution

  // --- Consecutive loss circuit breaker ---
  "max_consec_losses": 0,              // pause 1 window after N consecutive group losses (0=off)

  // --- UTC hour whitelist ---
  "allowed_hours": [],                  // only trade in these UTC hours; [] = all hours

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

# Reset trading state (wipes trading.db and starts fresh)
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

### Trading Database (`data/trading.db`)

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
│   │   ├── exchange.py            # ExchangeClient ABC + factory + trading types
│   │   ├── polymarket.py          # Polymarket CLOB API client (market data)
│   │   ├── polymarket_adapter.py  # Polymarket adapter (data + live trading via CLI)
│   │   ├── polymarket_claims.py   # Position discovery + gasless redemption
│   │   ├── polymarket_relayer.py  # Gasless proxy wallet relayer (batch redeem)
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
│   ├── backtest_suite.py          # Strategy backtesting framework (legacy)
│   ├── valid_momentum_backtest.py # Study 2: clean E-series filters (no look-ahead)
│   ├── creative_backtest.py       # Study 3: novel exits/sizing (tiered, off-hours, accel)
│   ├── filter_backtest.py         # Combined: vol/loss filters, 54 strategies
│   ├── slippage_model.py          # Spread/slippage impact modeling
│   └── advanced_strategy_ideas.py # Experimental strategy backtests
├── scripts/
│   └── backfill_db.py             # Database migration/backfill utility
├── config/
│   ├── config.json                # Default config (accel_dbl, Polymarket)
│   ├── config_consensus.json      # Contrarian consensus (Polymarket)
│   ├── config_consensus_kalshi.json # Contrarian consensus (Kalshi)
│   ├── config_triple.json         # Triple filter (Polymarket)
│   ├── config_e8_antimart.json    # E8: high threshold + prior momentum + anti-mart
│   ├── config_tiered_vol.json     # Tiered exit + skip high/extreme vol
│   ├── config_e8_daily100.json    # E8: high threshold + prior momentum + daily limit
│   ├── config_e7_priormom.json    # E7: high threshold + prior momentum
│   ├── config_cb_tiered.json      # Tiered exit + 2-loss circuit breaker
│   └── config_offhours_vol.json   # Off-hours (20:00–08:00 UTC) + skip high/extreme vol
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

Backtest scripts:

| Script | Description |
|--------|-------------|
| `analysis/valid_momentum_backtest.py` | Study 2 — validates E-series filters (no look-ahead bias) |
| `analysis/creative_backtest.py` | Study 3 — novel exit/sizing strategies (tiered, off-hours, accel) |
| `analysis/filter_backtest.py` | Combined study — vol/loss filters on top of best strategies |

---

## Winning Strategies (Backtested Jan 24 – Mar 1, 2026)

37-day dataset · 4 assets (BTC/ETH/SOL/XRP) · 3,411 windows · 70/30 time-series train/test split at Feb 19.
All strategies: `entry_mode=contrarian_consensus`, entry at t0, exit at t12.5. Fees: 1% per leg, 0.5% spread.

### Strategy Profiles

| Config File | Test P&L | Test WR | Max DD | Description |
|-------------|----------|---------|--------|-------------|
| [`config_e8_antimart.json`](config/config_e8_antimart.json) | **+$2,221** | 57.7% | -30.2% | High threshold (prev≥0.85) + prior momentum filter + anti-martingale 1.5× |
| [`config_tiered_vol.json`](config/config_tiered_vol.json) | +$1,755 | 61.1% | -19.1% | Skip high+extreme vol + tiered exit (hold to resolution when 3+ assets agree) |
| [`config_e8_daily100.json`](config/config_e8_daily100.json) | +$1,641 | 60.1% | -23.8% | High threshold + prior momentum + skip extreme vol + $100/day loss limit |
| [`config_e7_priormom.json`](config/config_e7_priormom.json) | +$1,635 | 60.0% | -24.7% | High threshold (prev≥0.85) + prior momentum filter + $100/day loss limit |
| [`config_cb_tiered.json`](config/config_cb_tiered.json) | +$1,607 | 60.7% | -17.6% | Tiered exit + 2-consecutive-loss circuit breaker + $100/day loss limit |
| [`config_offhours_vol.json`](config/config_offhours_vol.json) | +$1,489 | **61.6%** | **-15.1%** | Skip high+extreme vol + UTC 20:00–08:00 only — **best risk-adjusted** |

All six are profitable in both train and test sets (robust to the 70/30 split).

### Usage

```bash
# Best risk-adjusted (lowest drawdown, 61.6% WR, overnight hours only)
polynance-trade --config config/config_offhours_vol.json

# Highest absolute P&L (larger drawdown due to anti-martingale scaling)
polynance-trade --config config/config_e8_antimart.json

# Best balance: strong P&L with controlled drawdown
polynance-trade --config config/config_tiered_vol.json
```

### New Parameters (added Feb 2026)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_mom_filter` | bool | `false` | Require the previous window's trend to still be building. For bear: `prev_pm − prev2_pm > prior_mom_min` for at least 1 asset. Uses prev and prev2 PM prices — **no look-ahead bias**. |
| `prior_mom_min` | float | `0.03` | Minimum delta for prior momentum filter. |
| `tiered_exit` | bool | `false` | When `n_agree ≥ tiered_resolution_threshold`, hold to binary outcome (0 or 1); otherwise take early exit at t12.5. Captures asymmetric payoff on strong consensus. |
| `tiered_resolution_threshold` | int | `3` | Minimum agreeing assets to trigger tiered binary resolution. |
| `max_consec_losses` | int | `0` | Pause one window after this many consecutive group losses (0 = disabled). **Note:** raw data shows WR *rises* after loss runs — this is a risk-management feature, not an edge-enhancer. |
| `allowed_hours` | list | `[]` | UTC hours in which to enter trades (empty = all hours). e.g. `[20,21,22,23,0,1,2,3,4,5,6,7]` for overnight-only trading. |
| `redeem_on_window_complete` | bool | `true` | Auto-redeem settled positions after each window (live trading only). Includes hourly loser cleanup. |

### Key Findings

**Volatility regime WR (raw data, all signals):**
- `normal`: 58.2% — best regime
- `low`: 55.2%
- `high`: 52.9%
- `extreme`: 48.5% — below break-even

Skipping `high` and `extreme` vol windows adds ~2.6pp WR. This is the single most impactful filter.

**Off-hours edge:** Windows starting between 20:00–08:00 UTC (US/EU markets less active) show consistently better WR than peak hours. Combining this with vol filtering gives the best risk-adjusted profile.

**Tiered exit payoff:** When 3+ assets agree, the market hasn't fully reverted by t12.5 but resolves correctly at binary outcome. Holding to resolution captures this extra payoff on high-confidence signals.

**Counter-intuitive: WR rises after losing streaks.** Consecutive group losses → higher WR on the next signal. Circuit breakers (`max_consec_losses`) sacrifice real edge for psychological comfort during drawdowns.

---

## Dependencies

Core:
- `aiohttp`, `aiosqlite`, `websockets` (async networking)
- `pandas`, `numpy`, `scipy`, `scikit-learn` (data/analytics)
- `rich`, `plotext` (terminal UI)
- `matplotlib`, `seaborn` (charting)
- `python-dotenv`, `pytz` (utilities)

Live trading (requires separate install):
- `polymarket` CLI (`~/.local/bin/polymarket`) — handles order signing and CLOB API
- `web3`, `eth-account` (for gasless relayer redemption signing)

Dev (optional):
- `pytest`, `pytest-asyncio`, `black`, `ruff`

Install extras: `pip install -e ".[dev]"`

## Disclaimer

By default this runs in **dry-run (simulation) mode** — no real trades are placed. When `live_trading` is enabled, real orders are submitted to Polymarket. **Use at your own risk.** Past performance does not guarantee future results. Prediction markets carry significant risk.

## License

Private - All rights reserved.
