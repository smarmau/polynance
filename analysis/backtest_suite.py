#!/usr/bin/env python3
"""
Comprehensive Backtest Suite for Polynance Trading Strategies
Generates full trading metrics and visualization charts.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "reports"
ASSETS = ["btc", "eth", "sol", "xrp"]
INITIAL_BANKROLL = 1000.0
BASE_BET = 25.0
FEE_RATE = 0.02
SPREAD_COST = 0.006

STRATEGIES = [
    ("t5_0.60_0.40", "pm_yes_t5", 0.60, 0.40),
    ("t7.5_0.60_0.40", "pm_yes_t7_5", 0.60, 0.40),
    ("t7.5_0.75_0.25", "pm_yes_t7_5", 0.75, 0.25),
    ("t7.5_0.80_0.20", "pm_yes_t7_5", 0.80, 0.20),
    ("t10_0.60_0.40", "pm_yes_t10", 0.60, 0.40),
]


@dataclass
class Trade:
    """Single trade record."""
    timestamp: datetime
    asset: str
    direction: str  # 'bull' or 'bear'
    entry_price: float
    outcome: str  # 'up' or 'down'
    win: bool
    pnl: float
    bet_size: float
    bankroll_after: float
    drawdown: float
    drawdown_pct: float


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Basic stats
    strategy_name: str
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float

    # PnL metrics
    total_pnl: float
    avg_pnl: float
    median_pnl: float
    std_pnl: float

    # Return metrics
    total_return_pct: float
    avg_return_pct: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    avg_drawdown: float
    max_consecutive_losses: int
    max_consecutive_wins: int

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade analysis
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float

    # Time-based
    trades_per_day: float
    best_day_pnl: float
    worst_day_pnl: float
    pct_profitable_days: float

    # Per-asset breakdown
    per_asset_stats: Dict

    # Equity curve data
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)


def load_all_data() -> pd.DataFrame:
    """Load and combine data from all asset databases."""
    all_data = []

    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue

        conn = sqlite3.connect(db_path)
        query = """
            SELECT
                window_id, window_start_utc, outcome, outcome_binary,
                spot_open, spot_close, spot_change_pct, spot_change_bps,
                pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5,
                pm_spread_t0, pm_spread_t5
            FROM windows
            WHERE outcome IS NOT NULL
            ORDER BY window_start_utc
        """
        df = pd.read_sql_query(query, conn)
        df['asset'] = asset.upper()
        df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
        all_data.append(df)
        conn.close()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('window_start_utc').reset_index(drop=True)
    return combined


def generate_trades(df: pd.DataFrame, price_col: str, bull_thresh: float, bear_thresh: float) -> List[Dict]:
    """Generate trade signals based on strategy parameters."""
    trades = []

    for _, row in df.iterrows():
        price = row[price_col]
        if pd.isna(price):
            continue

        trade = None
        if price >= bull_thresh:
            # Bull signal: buy YES
            win = row['outcome'] == 'up'
            if win:
                gross_pnl = (1 - price) - SPREAD_COST
                pnl = gross_pnl * (1 - FEE_RATE)
            else:
                pnl = -price - SPREAD_COST
            trade = {
                'timestamp': row['window_start_utc'],
                'asset': row['asset'],
                'direction': 'bull',
                'entry_price': price,
                'outcome': row['outcome'],
                'win': win,
                'pnl_per_dollar': pnl,
            }
        elif price <= bear_thresh:
            # Bear signal: buy NO
            win = row['outcome'] == 'down'
            if win:
                gross_pnl = price - SPREAD_COST
                pnl = gross_pnl * (1 - FEE_RATE)
            else:
                pnl = -(1 - price) - SPREAD_COST
            trade = {
                'timestamp': row['window_start_utc'],
                'asset': row['asset'],
                'direction': 'bear',
                'entry_price': price,
                'outcome': row['outcome'],
                'win': win,
                'pnl_per_dollar': pnl,
            }

        if trade:
            trades.append(trade)

    return trades


def run_backtest_flat(trades: List[Dict], bet_size: float = BASE_BET) -> Tuple[List[Trade], List[float]]:
    """Run backtest with flat bet sizing."""
    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    equity_curve = [INITIAL_BANKROLL]
    drawdown_curve = [0.0]
    trade_records = []

    for t in trades:
        pnl = t['pnl_per_dollar'] * bet_size
        bankroll += pnl
        peak = max(peak, bankroll)
        dd = bankroll - peak
        dd_pct = (dd / peak) * 100 if peak > 0 else 0

        trade_records.append(Trade(
            timestamp=t['timestamp'],
            asset=t['asset'],
            direction=t['direction'],
            entry_price=t['entry_price'],
            outcome=t['outcome'],
            win=t['win'],
            pnl=pnl,
            bet_size=bet_size,
            bankroll_after=bankroll,
            drawdown=dd,
            drawdown_pct=dd_pct,
        ))

        equity_curve.append(bankroll)
        drawdown_curve.append(dd_pct)

    return trade_records, equity_curve, drawdown_curve


def run_backtest_antimartingale(trades: List[Dict], base_bet: float = BASE_BET,
                                 win_mult: float = 2.0, loss_mult: float = 0.5,
                                 max_bet_pct: float = 0.05) -> Tuple[List[Trade], List[float]]:
    """Run backtest with anti-martingale bet sizing."""
    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    current_bet = base_bet
    equity_curve = [INITIAL_BANKROLL]
    drawdown_curve = [0.0]
    trade_records = []

    for t in trades:
        # Cap bet at max_bet_pct of bankroll
        actual_bet = min(current_bet, bankroll * max_bet_pct)
        actual_bet = max(actual_bet, 1.0)  # Minimum $1 bet

        pnl = t['pnl_per_dollar'] * actual_bet
        bankroll += pnl
        peak = max(peak, bankroll)
        dd = bankroll - peak
        dd_pct = (dd / peak) * 100 if peak > 0 else 0

        trade_records.append(Trade(
            timestamp=t['timestamp'],
            asset=t['asset'],
            direction=t['direction'],
            entry_price=t['entry_price'],
            outcome=t['outcome'],
            win=t['win'],
            pnl=pnl,
            bet_size=actual_bet,
            bankroll_after=bankroll,
            drawdown=dd,
            drawdown_pct=dd_pct,
        ))

        equity_curve.append(bankroll)
        drawdown_curve.append(dd_pct)

        # Adjust bet size for next trade
        if t['win']:
            current_bet = actual_bet * win_mult
        else:
            current_bet = actual_bet * loss_mult
        current_bet = max(current_bet, base_bet * 0.1)  # Floor at 10% of base

    return trade_records, equity_curve, drawdown_curve


def calculate_metrics(strategy_name: str, trade_records: List[Trade],
                      equity_curve: List[float], drawdown_curve: List[float]) -> BacktestMetrics:
    """Calculate comprehensive backtest metrics."""
    if not trade_records:
        return None

    n_trades = len(trade_records)
    wins = [t for t in trade_records if t.win]
    losses = [t for t in trade_records if not t.win]
    n_wins = len(wins)
    n_losses = len(losses)

    pnls = [t.pnl for t in trade_records]
    win_pnls = [t.pnl for t in wins]
    loss_pnls = [t.pnl for t in losses]

    # Basic metrics
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)
    median_pnl = np.median(pnls)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 0

    # Return metrics
    total_return_pct = (equity_curve[-1] - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1]) * 100
    avg_return_pct = np.mean(returns) if len(returns) > 0 else 0

    # Drawdown metrics
    max_dd = min(drawdown_curve)
    avg_dd = np.mean([d for d in drawdown_curve if d < 0]) if any(d < 0 for d in drawdown_curve) else 0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for t in trade_records:
        if t.win:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Risk-adjusted returns
    if std_pnl > 0:
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252 * 24 * 4)  # Annualized for 15-min windows
    else:
        sharpe = 0

    downside_returns = [r for r in returns if r < 0]
    downside_std = np.std(downside_returns) if downside_returns else 0
    sortino = (avg_return_pct / downside_std) * np.sqrt(252 * 24 * 4) if downside_std > 0 else 0

    calmar = total_return_pct / abs(max_dd) if max_dd < 0 else 0

    # Trade analysis
    gross_wins = sum(win_pnls) if win_pnls else 0
    gross_losses = abs(sum(loss_pnls)) if loss_pnls else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    win_rate = n_wins / n_trades
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Daily analysis
    df_trades = pd.DataFrame([{
        'date': t.timestamp.date(),
        'pnl': t.pnl,
        'win': t.win,
    } for t in trade_records])

    daily_pnl = df_trades.groupby('date')['pnl'].sum()
    n_days = len(daily_pnl)
    trades_per_day = n_trades / n_days if n_days > 0 else 0
    best_day = daily_pnl.max() if len(daily_pnl) > 0 else 0
    worst_day = daily_pnl.min() if len(daily_pnl) > 0 else 0
    profitable_days = (daily_pnl > 0).sum()
    pct_profitable_days = profitable_days / n_days * 100 if n_days > 0 else 0

    # Per-asset breakdown
    per_asset = {}
    for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
        asset_trades = [t for t in trade_records if t.asset == asset]
        if asset_trades:
            asset_wins = len([t for t in asset_trades if t.win])
            asset_pnl = sum(t.pnl for t in asset_trades)
            per_asset[asset] = {
                'n_trades': len(asset_trades),
                'wins': asset_wins,
                'win_rate': asset_wins / len(asset_trades),
                'total_pnl': asset_pnl,
                'avg_pnl': asset_pnl / len(asset_trades),
            }

    return BacktestMetrics(
        strategy_name=strategy_name,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        median_pnl=median_pnl,
        std_pnl=std_pnl,
        total_return_pct=total_return_pct,
        avg_return_pct=avg_return_pct,
        max_drawdown=min(t.drawdown for t in trade_records),
        max_drawdown_pct=max_dd,
        avg_drawdown=avg_dd,
        max_consecutive_losses=max_consec_losses,
        max_consecutive_wins=max_consec_wins,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        expectancy=expectancy,
        trades_per_day=trades_per_day,
        best_day_pnl=best_day,
        worst_day_pnl=worst_day,
        pct_profitable_days=pct_profitable_days,
        per_asset_stats=per_asset,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        timestamps=[t.timestamp for t in trade_records],
        trades=trade_records,
    )


def create_equity_curves_chart(all_metrics: Dict[str, Dict[str, BacktestMetrics]], output_path: Path):
    """Create equity curve comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Flat betting equity curves
    ax1 = axes[0, 0]
    for name, metrics_dict in all_metrics.items():
        m = metrics_dict['flat']
        ax1.plot(m.equity_curve, label=f"{name} ({m.win_rate:.1%})", linewidth=1.5)
    ax1.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curves - Flat $25 Betting', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Bankroll ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Anti-martingale equity curves
    ax2 = axes[0, 1]
    for name, metrics_dict in all_metrics.items():
        m = metrics_dict['am']
        ax2.plot(m.equity_curve, label=f"{name}", linewidth=1.5)
    ax2.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Equity Curves - Anti-Martingale Sizing', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Bankroll ($)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Drawdown curves - Flat
    ax3 = axes[1, 0]
    for name, metrics_dict in all_metrics.items():
        m = metrics_dict['flat']
        ax3.fill_between(range(len(m.drawdown_curve)), m.drawdown_curve, alpha=0.3)
        ax3.plot(m.drawdown_curve, label=f"{name} (Max: {m.max_drawdown_pct:.1f}%)", linewidth=1)
    ax3.set_title('Drawdown - Flat Betting', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Drawdown curves - AM
    ax4 = axes[1, 1]
    for name, metrics_dict in all_metrics.items():
        m = metrics_dict['am']
        ax4.fill_between(range(len(m.drawdown_curve)), m.drawdown_curve, alpha=0.3)
        ax4.plot(m.drawdown_curve, label=f"{name} (Max: {m.max_drawdown_pct:.1f}%)", linewidth=1)
    ax4.set_title('Drawdown - Anti-Martingale', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('Drawdown (%)')
    ax4.legend(loc='lower left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_metrics_comparison_chart(all_metrics: Dict[str, Dict[str, BacktestMetrics]], output_path: Path):
    """Create metrics comparison bar charts."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    strategies = list(all_metrics.keys())
    x = np.arange(len(strategies))
    width = 0.35

    # Win Rate
    ax = axes[0, 0]
    flat_wr = [all_metrics[s]['flat'].win_rate * 100 for s in strategies]
    ax.bar(x, flat_wr, width, color='steelblue')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Breakeven')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Strategy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.legend()
    for i, v in enumerate(flat_wr):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    # Sharpe Ratio
    ax = axes[0, 1]
    flat_sharpe = [all_metrics[s]['flat'].sharpe_ratio for s in strategies]
    am_sharpe = [all_metrics[s]['am'].sharpe_ratio for s in strategies]
    ax.bar(x - width/2, flat_sharpe, width, label='Flat', color='steelblue')
    ax.bar(x + width/2, am_sharpe, width, label='Anti-Martingale', color='darkorange')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.legend()

    # Total PnL
    ax = axes[0, 2]
    flat_pnl = [all_metrics[s]['flat'].total_pnl for s in strategies]
    am_pnl = [all_metrics[s]['am'].total_pnl for s in strategies]
    ax.bar(x - width/2, flat_pnl, width, label='Flat', color='steelblue')
    ax.bar(x + width/2, am_pnl, width, label='Anti-Martingale', color='darkorange')
    ax.set_ylabel('Total PnL ($)')
    ax.set_title('Total PnL Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.legend()

    # Max Drawdown
    ax = axes[1, 0]
    flat_dd = [abs(all_metrics[s]['flat'].max_drawdown_pct) for s in strategies]
    am_dd = [abs(all_metrics[s]['am'].max_drawdown_pct) for s in strategies]
    ax.bar(x - width/2, flat_dd, width, label='Flat', color='steelblue')
    ax.bar(x + width/2, am_dd, width, label='Anti-Martingale', color='darkorange')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown (Lower is Better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.legend()

    # Profit Factor
    ax = axes[1, 1]
    flat_pf = [min(all_metrics[s]['flat'].profit_factor, 10) for s in strategies]  # Cap at 10 for display
    am_pf = [min(all_metrics[s]['am'].profit_factor, 10) for s in strategies]
    ax.bar(x - width/2, flat_pf, width, label='Flat', color='steelblue')
    ax.bar(x + width/2, am_pf, width, label='Anti-Martingale', color='darkorange')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Breakeven')
    ax.set_ylabel('Profit Factor')
    ax.set_title('Profit Factor (>1 is Profitable)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax.legend()

    # Trade Count
    ax = axes[1, 2]
    trade_counts = [all_metrics[s]['flat'].n_trades for s in strategies]
    ax.bar(x, trade_counts, width, color='steelblue')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trade Count by Strategy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    for i, v in enumerate(trade_counts):
        ax.text(i, v + 20, str(v), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_daily_returns_heatmap(metrics: BacktestMetrics, output_path: Path):
    """Create daily returns heatmap."""
    if not metrics.trades:
        return

    df_trades = pd.DataFrame([{
        'date': t.timestamp.date(),
        'hour': t.timestamp.hour,
        'pnl': t.pnl,
    } for t in metrics.trades])

    daily_pnl = df_trades.groupby('date')['pnl'].sum().reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    daily_pnl['weekday'] = daily_pnl['date'].dt.day_name()
    daily_pnl['day_num'] = (daily_pnl['date'] - daily_pnl['date'].min()).dt.days

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Daily PnL bar chart
    ax1 = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in daily_pnl['pnl']]
    ax1.bar(range(len(daily_pnl)), daily_pnl['pnl'], color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title(f'Daily PnL - {metrics.strategy_name}', fontweight='bold')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('PnL ($)')
    ax1.grid(True, alpha=0.3)

    # PnL by day of week
    ax2 = axes[0, 1]
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_pnl = daily_pnl.groupby('weekday')['pnl'].mean().reindex(weekday_order)
    colors = ['green' if x > 0 else 'red' for x in weekday_pnl.values]
    ax2.bar(weekday_order, weekday_pnl.values, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Average PnL by Day of Week', fontweight='bold')
    ax2.set_ylabel('Avg PnL ($)')
    ax2.tick_params(axis='x', rotation=45)

    # Hourly distribution
    ax3 = axes[1, 0]
    hourly_pnl = df_trades.groupby('hour')['pnl'].sum()
    colors = ['green' if x > 0 else 'red' for x in hourly_pnl.values]
    ax3.bar(hourly_pnl.index, hourly_pnl.values, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Total PnL by Hour (UTC)', fontweight='bold')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Total PnL ($)')
    ax3.set_xticks(range(0, 24))

    # Cumulative PnL over time
    ax4 = axes[1, 1]
    daily_pnl['cumulative'] = daily_pnl['pnl'].cumsum()
    ax4.fill_between(range(len(daily_pnl)), daily_pnl['cumulative'], alpha=0.3, color='steelblue')
    ax4.plot(range(len(daily_pnl)), daily_pnl['cumulative'], color='steelblue', linewidth=2)
    ax4.set_title('Cumulative Daily PnL', fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Cumulative PnL ($)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_trade_distribution_chart(metrics: BacktestMetrics, output_path: Path):
    """Create trade distribution analysis chart."""
    if not metrics.trades:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pnls = [t.pnl for t in metrics.trades]

    # PnL distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(pnls, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax1.axvline(x=np.mean(pnls), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(pnls):.2f}')
    ax1.set_title('PnL Distribution', fontweight='bold')
    ax1.set_xlabel('PnL ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Win/Loss by asset
    ax2 = axes[0, 1]
    assets = ['BTC', 'ETH', 'SOL', 'XRP']
    wins_by_asset = []
    losses_by_asset = []
    for asset in assets:
        asset_trades = [t for t in metrics.trades if t.asset == asset]
        wins_by_asset.append(len([t for t in asset_trades if t.win]))
        losses_by_asset.append(len([t for t in asset_trades if not t.win]))

    x = np.arange(len(assets))
    width = 0.35
    ax2.bar(x - width/2, wins_by_asset, width, label='Wins', color='green', alpha=0.7)
    ax2.bar(x + width/2, losses_by_asset, width, label='Losses', color='red', alpha=0.7)
    ax2.set_title('Wins/Losses by Asset', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(assets)
    ax2.legend()

    # Entry price distribution
    ax3 = axes[0, 2]
    bull_prices = [t.entry_price for t in metrics.trades if t.direction == 'bull']
    bear_prices = [t.entry_price for t in metrics.trades if t.direction == 'bear']
    if bull_prices:
        ax3.hist(bull_prices, bins=20, alpha=0.6, label=f'Bull (n={len(bull_prices)})', color='green')
    if bear_prices:
        ax3.hist(bear_prices, bins=20, alpha=0.6, label=f'Bear (n={len(bear_prices)})', color='red')
    ax3.set_title('Entry Price Distribution', fontweight='bold')
    ax3.set_xlabel('PM YES Price')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # Consecutive wins/losses
    ax4 = axes[1, 0]
    streaks = []
    current_streak = 0
    current_type = None
    for t in metrics.trades:
        if current_type is None:
            current_type = t.win
            current_streak = 1
        elif t.win == current_type:
            current_streak += 1
        else:
            streaks.append((current_streak, 'W' if current_type else 'L'))
            current_type = t.win
            current_streak = 1
    if current_streak > 0:
        streaks.append((current_streak, 'W' if current_type else 'L'))

    win_streaks = [s[0] for s in streaks if s[1] == 'W']
    loss_streaks = [s[0] for s in streaks if s[1] == 'L']

    ax4.hist(win_streaks, bins=range(1, max(win_streaks)+2), alpha=0.6, label='Win Streaks', color='green')
    ax4.hist(loss_streaks, bins=range(1, max(loss_streaks)+2 if loss_streaks else 2), alpha=0.6, label='Loss Streaks', color='red')
    ax4.set_title('Streak Distribution', fontweight='bold')
    ax4.set_xlabel('Streak Length')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # Bet size distribution (for AM)
    ax5 = axes[1, 1]
    bet_sizes = [t.bet_size for t in metrics.trades]
    ax5.hist(bet_sizes, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=np.mean(bet_sizes), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(bet_sizes):.2f}')
    ax5.set_title('Bet Size Distribution', fontweight='bold')
    ax5.set_xlabel('Bet Size ($)')
    ax5.set_ylabel('Frequency')
    ax5.legend()

    # Rolling win rate
    ax6 = axes[1, 2]
    window = min(100, len(metrics.trades) // 5)
    if window > 10:
        rolling_wins = pd.Series([1 if t.win else 0 for t in metrics.trades]).rolling(window).mean() * 100
        ax6.plot(rolling_wins.values, color='steelblue', linewidth=1.5)
        ax6.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50%')
        ax6.axhline(y=metrics.win_rate * 100, color='green', linestyle='--', alpha=0.7, label=f'Overall: {metrics.win_rate*100:.1f}%')
        ax6.set_title(f'Rolling Win Rate ({window}-trade window)', fontweight='bold')
        ax6.set_xlabel('Trade #')
        ax6.set_ylabel('Win Rate (%)')
        ax6.legend()
        ax6.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_per_asset_chart(all_metrics: Dict[str, Dict[str, BacktestMetrics]], output_path: Path):
    """Create per-asset performance breakdown chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    assets = ['BTC', 'ETH', 'SOL', 'XRP']
    strategies = list(all_metrics.keys())

    # Win rate by asset for each strategy
    ax1 = axes[0, 0]
    x = np.arange(len(assets))
    width = 0.15
    for i, strat in enumerate(strategies):
        win_rates = []
        for asset in assets:
            if asset in all_metrics[strat]['flat'].per_asset_stats:
                win_rates.append(all_metrics[strat]['flat'].per_asset_stats[asset]['win_rate'] * 100)
            else:
                win_rates.append(0)
        ax1.bar(x + i * width, win_rates, width, label=strat)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate by Asset', fontweight='bold')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(assets)
    ax1.legend(loc='lower right', fontsize=8)

    # PnL by asset for each strategy
    ax2 = axes[0, 1]
    for i, strat in enumerate(strategies):
        pnls = []
        for asset in assets:
            if asset in all_metrics[strat]['flat'].per_asset_stats:
                pnls.append(all_metrics[strat]['flat'].per_asset_stats[asset]['total_pnl'])
            else:
                pnls.append(0)
        ax2.bar(x + i * width, pnls, width, label=strat)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Total PnL ($)')
    ax2.set_title('Total PnL by Asset (Flat Betting)', fontweight='bold')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(assets)
    ax2.legend(loc='upper right', fontsize=8)

    # Trade count by asset
    ax3 = axes[1, 0]
    # Use the strategy with most trades
    best_strat = max(strategies, key=lambda s: all_metrics[s]['flat'].n_trades)
    trade_counts = []
    for asset in assets:
        if asset in all_metrics[best_strat]['flat'].per_asset_stats:
            trade_counts.append(all_metrics[best_strat]['flat'].per_asset_stats[asset]['n_trades'])
        else:
            trade_counts.append(0)
    ax3.bar(assets, trade_counts, color='steelblue', alpha=0.7)
    ax3.set_ylabel('Number of Trades')
    ax3.set_title(f'Trade Distribution by Asset ({best_strat})', fontweight='bold')
    for i, v in enumerate(trade_counts):
        ax3.text(i, v + 5, str(v), ha='center')

    # Bull vs Bear signals by asset
    ax4 = axes[1, 1]
    best_metrics = all_metrics[best_strat]['flat']
    bull_counts = []
    bear_counts = []
    for asset in assets:
        asset_trades = [t for t in best_metrics.trades if t.asset == asset]
        bull_counts.append(len([t for t in asset_trades if t.direction == 'bull']))
        bear_counts.append(len([t for t in asset_trades if t.direction == 'bear']))

    ax4.bar(x - width, bull_counts, width * 2, label='Bull', color='green', alpha=0.7)
    ax4.bar(x + width, bear_counts, width * 2, label='Bear', color='red', alpha=0.7)
    ax4.set_ylabel('Number of Trades')
    ax4.set_title(f'Bull vs Bear Signals by Asset ({best_strat})', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(assets)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_dashboard(all_metrics: Dict[str, Dict[str, BacktestMetrics]],
                             data_info: Dict, output_path: Path):
    """Create a summary dashboard with all key metrics."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('POLYNANCE BACKTEST SUITE - SUMMARY DASHBOARD', fontsize=16, fontweight='bold', y=0.98)

    # Data info text box
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis('off')
    info_text = f"""DATA OVERVIEW
━━━━━━━━━━━━━━━━━━━━
Total Windows: {data_info['total_windows']:,}
Date Range: {data_info['date_range'][0][:10]}
         to {data_info['date_range'][1][:10]}
Assets: BTC, ETH, SOL, XRP
Up/Down: {data_info['up_pct']:.1f}% / {data_info['down_pct']:.1f}%"""
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Best strategy highlight
    ax_best = fig.add_subplot(gs[0, 1:3])
    ax_best.axis('off')

    # Find best strategy by Sharpe
    best_strat = max(all_metrics.keys(), key=lambda s: all_metrics[s]['am'].sharpe_ratio)
    best_m = all_metrics[best_strat]['am']
    best_flat = all_metrics[best_strat]['flat']

    best_text = f"""RECOMMENDED STRATEGY: {best_strat}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Win Rate: {best_flat.win_rate*100:.1f}%    Trades: {best_flat.n_trades:,}
Flat PnL: ${best_flat.total_pnl:,.2f}    AM PnL: ${best_m.total_pnl:,.2f}
Max DD (Flat): {best_flat.max_drawdown_pct:.1f}%    Max DD (AM): {best_m.max_drawdown_pct:.1f}%
Sharpe (Flat): {best_flat.sharpe_ratio:.1f}    Sharpe (AM): {best_m.sharpe_ratio:.1f}
Profit Factor: {best_flat.profit_factor:.2f}    Expectancy: ${best_flat.expectancy:.2f}"""
    ax_best.text(0.1, 0.9, best_text, transform=ax_best.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Settings
    ax_settings = fig.add_subplot(gs[0, 3])
    ax_settings.axis('off')
    settings_text = f"""BACKTEST SETTINGS
━━━━━━━━━━━━━━━━━━━━
Initial: ${INITIAL_BANKROLL:,.0f}
Base Bet: ${BASE_BET:.0f}
Fee Rate: {FEE_RATE*100:.0f}%
Spread: {SPREAD_COST*100:.1f}%
AM Mult: 2x/0.5x
Max Bet: 5% bankroll"""
    ax_settings.text(0.1, 0.9, settings_text, transform=ax_settings.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Main metrics table
    ax_table = fig.add_subplot(gs[1:3, :])
    ax_table.axis('off')

    # Create table data
    columns = ['Strategy', 'Trades', 'Win%', 'PnL (Flat)', 'PnL (AM)',
               'MaxDD%', 'Sharpe', 'Sortino', 'PF', 'Avg Win', 'Avg Loss', 'Expect']

    table_data = []
    for name in all_metrics.keys():
        m_flat = all_metrics[name]['flat']
        m_am = all_metrics[name]['am']
        table_data.append([
            name,
            f"{m_flat.n_trades:,}",
            f"{m_flat.win_rate*100:.1f}%",
            f"${m_flat.total_pnl:,.0f}",
            f"${m_am.total_pnl:,.0f}",
            f"{m_am.max_drawdown_pct:.1f}%",
            f"{m_am.sharpe_ratio:.1f}",
            f"{m_am.sortino_ratio:.1f}",
            f"{m_flat.profit_factor:.2f}",
            f"${m_flat.avg_win:.2f}",
            f"${m_flat.avg_loss:.2f}",
            f"${m_flat.expectancy:.2f}",
        ])

    table = ax_table.table(cellText=table_data, colLabels=columns,
                           cellLoc='center', loc='center',
                           colColours=['lightblue'] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Highlight best row
    best_idx = list(all_metrics.keys()).index(best_strat) + 1
    for j in range(len(columns)):
        table[(best_idx, j)].set_facecolor('lightgreen')

    # Equity curve (best strategy)
    ax_equity = fig.add_subplot(gs[3, :2])
    ax_equity.plot(best_flat.equity_curve, label='Flat', color='steelblue', linewidth=1.5)
    ax_equity.plot(best_m.equity_curve, label='Anti-Martingale', color='darkorange', linewidth=1.5)
    ax_equity.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5)
    ax_equity.set_title(f'Equity Curve - {best_strat}', fontweight='bold')
    ax_equity.set_xlabel('Trade #')
    ax_equity.set_ylabel('Bankroll ($)')
    ax_equity.legend()
    ax_equity.grid(True, alpha=0.3)

    # Per-asset win rate for best strategy
    ax_asset = fig.add_subplot(gs[3, 2:])
    assets = ['BTC', 'ETH', 'SOL', 'XRP']
    win_rates = [best_flat.per_asset_stats.get(a, {}).get('win_rate', 0) * 100 for a in assets]
    colors = ['green' if wr > 50 else 'red' for wr in win_rates]
    bars = ax_asset.bar(assets, win_rates, color=colors, alpha=0.7)
    ax_asset.axhline(y=50, color='red', linestyle='--', alpha=0.7)
    ax_asset.set_ylabel('Win Rate (%)')
    ax_asset.set_title(f'Win Rate by Asset - {best_strat}', fontweight='bold')
    ax_asset.set_ylim(0, 100)
    for bar, wr in zip(bars, win_rates):
        ax_asset.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                      f'{wr:.1f}%', ha='center', fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_text_report(all_metrics: Dict[str, Dict[str, BacktestMetrics]],
                         data_info: Dict, output_path: Path):
    """Generate a comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("POLYNANCE BACKTEST SUITE - COMPREHENSIVE REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Data Overview
    lines.append("DATA OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Total Windows: {data_info['total_windows']:,}")
    lines.append(f"Date Range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
    lines.append(f"Assets: BTC, ETH, SOL, XRP")
    lines.append(f"Outcome Balance: UP {data_info['up_pct']:.1f}% / DOWN {data_info['down_pct']:.1f}%")
    lines.append("")

    # Backtest Settings
    lines.append("BACKTEST SETTINGS")
    lines.append("-" * 40)
    lines.append(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
    lines.append(f"Base Bet Size: ${BASE_BET:.2f}")
    lines.append(f"Fee Rate: {FEE_RATE*100:.1f}%")
    lines.append(f"Spread Cost: {SPREAD_COST*100:.2f}%")
    lines.append(f"Anti-Martingale: 2x on win, 0.5x on loss, 5% max")
    lines.append("")

    # Strategy Comparison
    lines.append("=" * 80)
    lines.append("STRATEGY COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    for name in all_metrics.keys():
        m_flat = all_metrics[name]['flat']
        m_am = all_metrics[name]['am']

        lines.append(f"STRATEGY: {name}")
        lines.append("-" * 60)
        lines.append("")

        lines.append("  BASIC METRICS")
        lines.append(f"    Trades: {m_flat.n_trades:,}")
        lines.append(f"    Wins: {m_flat.n_wins:,} ({m_flat.win_rate*100:.2f}%)")
        lines.append(f"    Losses: {m_flat.n_losses:,} ({(1-m_flat.win_rate)*100:.2f}%)")
        lines.append("")

        lines.append("  PNL METRICS (Flat $25 Betting)")
        lines.append(f"    Total PnL: ${m_flat.total_pnl:,.2f}")
        lines.append(f"    Total Return: {m_flat.total_return_pct:.2f}%")
        lines.append(f"    Avg PnL/Trade: ${m_flat.avg_pnl:.2f}")
        lines.append(f"    Median PnL: ${m_flat.median_pnl:.2f}")
        lines.append(f"    Std Dev: ${m_flat.std_pnl:.2f}")
        lines.append("")

        lines.append("  PNL METRICS (Anti-Martingale)")
        lines.append(f"    Total PnL: ${m_am.total_pnl:,.2f}")
        lines.append(f"    Total Return: {m_am.total_return_pct:.2f}%")
        lines.append(f"    Avg PnL/Trade: ${m_am.avg_pnl:.2f}")
        lines.append("")

        lines.append("  RISK METRICS")
        lines.append(f"    Max Drawdown (Flat): ${m_flat.max_drawdown:.2f} ({m_flat.max_drawdown_pct:.2f}%)")
        lines.append(f"    Max Drawdown (AM): ${m_am.max_drawdown:.2f} ({m_am.max_drawdown_pct:.2f}%)")
        lines.append(f"    Avg Drawdown: {m_flat.avg_drawdown:.2f}%")
        lines.append(f"    Max Consecutive Wins: {m_flat.max_consecutive_wins}")
        lines.append(f"    Max Consecutive Losses: {m_flat.max_consecutive_losses}")
        lines.append("")

        lines.append("  RISK-ADJUSTED RETURNS")
        lines.append(f"    Sharpe Ratio (Flat): {m_flat.sharpe_ratio:.2f}")
        lines.append(f"    Sharpe Ratio (AM): {m_am.sharpe_ratio:.2f}")
        lines.append(f"    Sortino Ratio (AM): {m_am.sortino_ratio:.2f}")
        lines.append(f"    Calmar Ratio (AM): {m_am.calmar_ratio:.2f}")
        lines.append("")

        lines.append("  TRADE ANALYSIS")
        lines.append(f"    Profit Factor: {m_flat.profit_factor:.2f}")
        lines.append(f"    Avg Win: ${m_flat.avg_win:.2f}")
        lines.append(f"    Avg Loss: ${m_flat.avg_loss:.2f}")
        lines.append(f"    Win/Loss Ratio: {m_flat.win_loss_ratio:.2f}")
        lines.append(f"    Expectancy: ${m_flat.expectancy:.2f}")
        lines.append("")

        lines.append("  TIME-BASED ANALYSIS")
        lines.append(f"    Trades/Day: {m_flat.trades_per_day:.1f}")
        lines.append(f"    Best Day: ${m_flat.best_day_pnl:.2f}")
        lines.append(f"    Worst Day: ${m_flat.worst_day_pnl:.2f}")
        lines.append(f"    Profitable Days: {m_flat.pct_profitable_days:.1f}%")
        lines.append("")

        lines.append("  PER-ASSET BREAKDOWN")
        for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
            if asset in m_flat.per_asset_stats:
                a = m_flat.per_asset_stats[asset]
                lines.append(f"    {asset}: {a['n_trades']} trades, {a['win_rate']*100:.1f}% win, ${a['total_pnl']:.2f} PnL")
        lines.append("")
        lines.append("")

    # Summary Table
    lines.append("=" * 80)
    lines.append("SUMMARY COMPARISON TABLE")
    lines.append("=" * 80)
    lines.append("")

    header = f"{'Strategy':<20} {'Trades':>8} {'Win%':>8} {'PnL(F)':>10} {'PnL(AM)':>10} {'MaxDD':>8} {'Sharpe':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for name in all_metrics.keys():
        m_flat = all_metrics[name]['flat']
        m_am = all_metrics[name]['am']
        row = f"{name:<20} {m_flat.n_trades:>8,} {m_flat.win_rate*100:>7.1f}% ${m_flat.total_pnl:>8,.0f} ${m_am.total_pnl:>8,.0f} {m_am.max_drawdown_pct:>7.1f}% {m_am.sharpe_ratio:>8.1f}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"Saved: {output_path}")
    return report_text


def main():
    """Main execution function."""
    print("=" * 80)
    print("POLYNANCE BACKTEST SUITE")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print("Loading data...")
    df = load_all_data()
    print(f"Total windows: {len(df)}")
    print(f"Date range: {df['window_start_utc'].min()} to {df['window_start_utc'].max()}")
    print(f"Assets: {df['asset'].unique().tolist()}")

    up_count = (df['outcome'] == 'up').sum()
    down_count = (df['outcome'] == 'down').sum()
    total = len(df)
    print(f"Outcome balance: UP {up_count} ({up_count/total*100:.1f}%), DOWN {down_count} ({down_count/total*100:.1f}%)")
    print()

    data_info = {
        'total_windows': len(df),
        'date_range': [str(df['window_start_utc'].min()), str(df['window_start_utc'].max())],
        'up_count': up_count,
        'down_count': down_count,
        'up_pct': up_count / total * 100,
        'down_pct': down_count / total * 100,
    }

    # Run backtests for all strategies
    print("Running backtests...")
    all_metrics = {}

    for name, price_col, bull_thresh, bear_thresh in STRATEGIES:
        print(f"  Processing {name}...")
        trades = generate_trades(df, price_col, bull_thresh, bear_thresh)

        # Flat betting
        trade_records_flat, equity_flat, dd_flat = run_backtest_flat(trades)
        metrics_flat = calculate_metrics(f"{name}_flat", trade_records_flat, equity_flat, dd_flat)

        # Anti-martingale
        trade_records_am, equity_am, dd_am = run_backtest_antimartingale(trades)
        metrics_am = calculate_metrics(f"{name}_am", trade_records_am, equity_am, dd_am)

        all_metrics[name] = {
            'flat': metrics_flat,
            'am': metrics_am,
        }

        print(f"    Trades: {metrics_flat.n_trades}, Win Rate: {metrics_flat.win_rate*100:.1f}%")
        print(f"    Flat PnL: ${metrics_flat.total_pnl:.2f}, AM PnL: ${metrics_am.total_pnl:.2f}")

    print()

    # Generate charts
    print("Generating charts...")

    create_equity_curves_chart(all_metrics, OUTPUT_DIR / f"equity_curves_{timestamp}.png")
    create_metrics_comparison_chart(all_metrics, OUTPUT_DIR / f"metrics_comparison_{timestamp}.png")
    create_per_asset_chart(all_metrics, OUTPUT_DIR / f"per_asset_{timestamp}.png")
    create_summary_dashboard(all_metrics, data_info, OUTPUT_DIR / f"summary_dashboard_{timestamp}.png")

    # Generate daily returns for best strategy
    best_strat = max(all_metrics.keys(), key=lambda s: all_metrics[s]['am'].sharpe_ratio)
    create_daily_returns_heatmap(all_metrics[best_strat]['am'],
                                  OUTPUT_DIR / f"daily_returns_{best_strat}_{timestamp}.png")
    create_trade_distribution_chart(all_metrics[best_strat]['am'],
                                     OUTPUT_DIR / f"trade_distribution_{best_strat}_{timestamp}.png")

    # Generate text report
    print()
    print("Generating text report...")
    report = generate_text_report(all_metrics, data_info, OUTPUT_DIR / f"backtest_report_{timestamp}.txt")

    # Save JSON results
    print()
    print("Saving JSON results...")

    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    results_json = {
        'timestamp': timestamp,
        'data_info': data_info,
        'strategies': {}
    }

    for name, metrics_dict in all_metrics.items():
        results_json['strategies'][name] = {
            'flat': {
                'n_trades': metrics_dict['flat'].n_trades,
                'win_rate': metrics_dict['flat'].win_rate,
                'total_pnl': metrics_dict['flat'].total_pnl,
                'max_drawdown_pct': metrics_dict['flat'].max_drawdown_pct,
                'sharpe_ratio': metrics_dict['flat'].sharpe_ratio,
                'sortino_ratio': metrics_dict['flat'].sortino_ratio,
                'calmar_ratio': metrics_dict['flat'].calmar_ratio,
                'profit_factor': metrics_dict['flat'].profit_factor,
                'expectancy': metrics_dict['flat'].expectancy,
                'avg_win': metrics_dict['flat'].avg_win,
                'avg_loss': metrics_dict['flat'].avg_loss,
                'max_consecutive_wins': metrics_dict['flat'].max_consecutive_wins,
                'max_consecutive_losses': metrics_dict['flat'].max_consecutive_losses,
                'per_asset': metrics_dict['flat'].per_asset_stats,
            },
            'am': {
                'n_trades': metrics_dict['am'].n_trades,
                'win_rate': metrics_dict['am'].win_rate,
                'total_pnl': metrics_dict['am'].total_pnl,
                'max_drawdown_pct': metrics_dict['am'].max_drawdown_pct,
                'sharpe_ratio': metrics_dict['am'].sharpe_ratio,
                'sortino_ratio': metrics_dict['am'].sortino_ratio,
                'calmar_ratio': metrics_dict['am'].calmar_ratio,
                'profit_factor': metrics_dict['am'].profit_factor,
                'expectancy': metrics_dict['am'].expectancy,
            },
        }

    results_json = convert_types(results_json)

    with open(OUTPUT_DIR / f"backtest_results_{timestamp}.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / f'backtest_results_{timestamp}.json'}")

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Strategy':<20} {'Trades':>8} {'Win%':>8} {'PnL(F)':>10} {'PnL(AM)':>10} {'MaxDD':>8} {'Sharpe':>8}")
    print("-" * 80)
    for name in all_metrics.keys():
        m_flat = all_metrics[name]['flat']
        m_am = all_metrics[name]['am']
        print(f"{name:<20} {m_flat.n_trades:>8,} {m_flat.win_rate*100:>7.1f}% ${m_flat.total_pnl:>8,.0f} ${m_am.total_pnl:>8,.0f} {m_am.max_drawdown_pct:>7.1f}% {m_am.sharpe_ratio:>8.1f}")

    print()
    print(f"Best Strategy (by Sharpe): {best_strat}")
    print()
    print(f"Output files saved to: {OUTPUT_DIR}")
    print()

    return all_metrics


if __name__ == "__main__":
    main()
