# Polynance

A simulated trading bot for Polymarket 15-minute crypto prediction markets.

## Overview

Polynance collects real-time data from Polymarket's crypto prediction markets (BTC, ETH, SOL, XRP) and runs a simulated trading strategy based on extreme price signals.

**Strategy:** When the Polymarket YES price reaches extreme levels (≥0.80 bullish, ≤0.20 bearish) at the midpoint of a 15-minute window (t=7.5), take a position. Resolve at window close (t=15).

**Current Performance (simulated):**
- Win Rate: ~93%
- Profit Factor: ~1.9
- Uses anti-martingale bet sizing (2x on win, 0.5x on loss)

## Installation

```bash
# Clone the repo
git clone git@github.com:YOUR_USERNAME/polynance.git
cd polynance

# Install dependencies
pip install -e .
```

## Usage

### Data Collection + Simulated Trading

```bash
# Run the trading bot with live dashboard
polynance-trade

# Or run data collection only (no trading)
polynance
```

### Generate Performance Charts

```bash
# Display chart
polynance-chart

# Save to file
polynance-chart --output chart.png
```

### Configuration

Edit `config/trading.json`:

```json
{
  "bull_threshold": 0.80,
  "bear_threshold": 0.20,
  "initial_bankroll": 1000.0,
  "base_bet": 25.0,
  "fee_rate": 0.001,
  "spread_cost": 0.005,
  "win_multiplier": 2.0,
  "loss_multiplier": 0.5,
  "max_bet_pct": 0.05,
  "assets": ["BTC", "ETH", "SOL", "XRP"]
}
```

## Project Structure

```
polynance/
├── src/polynance/
│   ├── main.py              # Data collection entry point
│   ├── sampler.py           # 30-second data sampling
│   ├── trading/
│   │   ├── trader.py        # Simulated trading engine
│   │   ├── dashboard.py     # Rich terminal dashboard
│   │   ├── chart.py         # Matplotlib performance charts
│   │   ├── database.py      # Trade database
│   │   ├── bet_sizing.py    # Anti-martingale sizing
│   │   └── models.py        # Data models
│   └── dashboard/
│       └── terminal.py      # Data collection dashboard
├── analysis/
│   ├── backtest_suite.py    # Strategy backtesting
│   └── reports/             # Generated charts/reports
├── config/
│   └── trading.json         # Trading configuration
├── data/                    # SQLite databases (gitignored)
└── pyproject.toml
```

## Data Storage

- **Per-asset databases** (`data/btc.db`, `data/eth.db`, etc.): Raw window data with samples at t=0, 2.5, 5, 7.5, 10, 12.5 minutes
- **Trading database** (`data/sim_trading.db`): Simulated trade history and state

## Key Metrics

The dashboard and charts display:

| Metric | Description |
|--------|-------------|
| Win Rate | Percentage of winning trades |
| Profit Factor | Gross wins / Gross losses |
| Sharpe | Risk-adjusted return (annualized) |
| Sortino | Downside risk-adjusted return |
| Calmar | Return / Max Drawdown |
| Recovery Factor | Total P&L / Max Drawdown |
| Expectancy | Average P&L per trade |

## How It Works

1. **Data Collection**: Samples Polymarket prices every 30 seconds for each 15-minute window
2. **Signal Detection**: At t=7.5, checks if price crosses threshold (≥0.80 or ≤0.20)
3. **Trade Execution**: Opens simulated position with anti-martingale sizing
4. **Resolution**: At t=15, resolves trade based on window outcome (up/down)
5. **Fees**: Applies 0.1% taker fee + 0.5% estimated spread

## Backtesting

Run historical backtests:

```bash
python analysis/backtest_suite.py
```

Generates charts and reports in `analysis/reports/`.

## Disclaimer

This is a **simulation only**. No real trades are executed. Past performance does not guarantee future results. Prediction markets carry significant risk.

## License

Private - All rights reserved.
