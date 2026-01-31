#!/usr/bin/env python3
"""
Generate equity curve and performance charts from trading database.

Usage:
    polynance-chart [--output PATH] [--show]
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


def load_trades(db_path: Path) -> List[dict]:
    """Load all resolved trades from the trading database."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT
            trade_id, window_id, asset, direction, entry_price,
            bet_size, outcome, net_pnl, bankroll_after, created_at
        FROM sim_trades
        WHERE outcome IS NOT NULL
        ORDER BY created_at
    """)

    trades = []
    for row in cursor.fetchall():
        trades.append({
            'trade_id': row[0],
            'window_id': row[1],
            'asset': row[2],
            'direction': row[3],
            'entry_price': row[4],
            'bet_size': row[5],
            'outcome': row[6],
            'net_pnl': row[7],
            'bankroll_after': row[8],
            'created_at': row[9],
        })

    conn.close()
    return trades


def load_state(db_path: Path) -> dict:
    """Load current trading state."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT * FROM sim_state LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    return {
        'initial_bankroll': row[4],
        'current_bankroll': row[2],
        'total_pnl': row[2] - row[4],
        'total_wins': row[6],
        'total_losses': row[7],
        'max_drawdown_pct': row[12],
    }


def calculate_metrics(trades: List[dict], initial_bankroll: float) -> dict:
    """Calculate performance metrics from trades."""
    if not trades:
        return {}

    pnls = [t['net_pnl'] for t in trades if t['net_pnl'] is not None]
    wins = [t for t in trades if t['outcome'] == 'win']
    losses = [t for t in trades if t['outcome'] == 'loss']

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(trades) if trades else 0

    # Equity curve
    equity = [initial_bankroll]
    for t in trades:
        if t['bankroll_after'] is not None:
            equity.append(t['bankroll_after'])

    # Drawdown
    peak = initial_bankroll
    drawdowns = []
    for val in equity:
        peak = max(peak, val)
        dd_pct = ((val - peak) / peak) * 100 if peak > 0 else 0
        drawdowns.append(dd_pct)

    max_dd = min(drawdowns) if drawdowns else 0

    # Profit factor
    gross_wins = sum(t['net_pnl'] for t in wins if t['net_pnl'] and t['net_pnl'] > 0)
    gross_losses = abs(sum(t['net_pnl'] for t in losses if t['net_pnl'] and t['net_pnl'] < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Calmar
    total_return_pct = (total_pnl / initial_bankroll) * 100
    calmar = total_return_pct / abs(max_dd) if max_dd < 0 else 0

    # Recovery factor
    max_dd_dollars = min([equity[i] - max(equity[:i+1]) for i in range(len(equity))])
    recovery_factor = total_pnl / abs(max_dd_dollars) if max_dd_dollars < 0 else float('inf')

    return {
        'total_trades': len(trades),
        'total_wins': len(wins),
        'total_losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'max_drawdown_pct': max_dd,
        'profit_factor': profit_factor,
        'calmar': calmar,
        'recovery_factor': recovery_factor,
        'equity_curve': equity,
        'drawdown_curve': drawdowns,
    }


def create_chart(trades: List[dict], state: dict, metrics: dict, output_path: Path = None, show: bool = True):
    """Create comprehensive performance chart."""
    if not trades:
        print("No trades to chart.")
        return

    initial_bankroll = state.get('initial_bankroll', 1000)
    equity = metrics['equity_curve']
    drawdowns = metrics['drawdown_curve']

    # Set up the figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('POLYNANCE SIMULATED TRADING PERFORMANCE', fontsize=14, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Equity Curve (large, top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity, color='#00BFFF', linewidth=1.5, label='Equity')
    ax1.axhline(y=initial_bankroll, color='white', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.fill_between(range(len(equity)), initial_bankroll, equity,
                     where=[e >= initial_bankroll for e in equity],
                     color='green', alpha=0.3)
    ax1.fill_between(range(len(equity)), initial_bankroll, equity,
                     where=[e < initial_bankroll for e in equity],
                     color='red', alpha=0.3)
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Bankroll ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1a1a2e')

    # 2. Drawdown (middle left)
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.fill_between(range(len(drawdowns)), drawdowns, color='red', alpha=0.5)
    ax2.plot(drawdowns, color='red', linewidth=1)
    ax2.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
    ax2.set_title(f'Drawdown (Max: {metrics["max_drawdown_pct"]:.1f}%)', fontweight='bold')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#1a1a2e')

    # 3. Summary stats (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')

    pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] < 100 else "Inf"
    rf_str = f"{metrics['recovery_factor']:.2f}" if metrics['recovery_factor'] < 100 else "Inf"

    stats_text = f"""
PERFORMANCE SUMMARY
{'─' * 30}

Total Trades:    {metrics['total_trades']}
Wins / Losses:   {metrics['total_wins']} / {metrics['total_losses']}
Win Rate:        {metrics['win_rate']*100:.1f}%

Total P&L:       ${metrics['total_pnl']:+,.2f}
Max Drawdown:    {metrics['max_drawdown_pct']:.1f}%

Profit Factor:   {pf_str}
Calmar Ratio:    {metrics['calmar']:.2f}
Recovery Factor: {rf_str}
"""
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             color='white',
             bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#0f3460'))

    # 4. P&L Distribution (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    pnls = [t['net_pnl'] for t in trades if t['net_pnl'] is not None]
    colors = ['green' if p > 0 else 'red' for p in pnls]
    ax4.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
    ax4.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
    ax4.set_title('P&L per Trade', fontweight='bold')
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('P&L ($)')
    ax4.set_facecolor('#1a1a2e')

    # 5. Win Rate by Asset (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    assets = ['BTC', 'ETH', 'SOL', 'XRP']
    asset_wr = []
    asset_counts = []
    for asset in assets:
        asset_trades = [t for t in trades if t['asset'] == asset]
        if asset_trades:
            wins = len([t for t in asset_trades if t['outcome'] == 'win'])
            asset_wr.append(wins / len(asset_trades) * 100)
            asset_counts.append(len(asset_trades))
        else:
            asset_wr.append(0)
            asset_counts.append(0)

    bars = ax5.bar(assets, asset_wr, color=['green' if wr > 50 else 'red' for wr in asset_wr], alpha=0.7)
    ax5.axhline(y=50, color='white', linestyle='--', alpha=0.5)
    ax5.set_title('Win Rate by Asset', fontweight='bold')
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_ylim(0, 100)
    for bar, wr, count in zip(bars, asset_wr, asset_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{wr:.0f}%\n({count})', ha='center', fontsize=9, color='white')
    ax5.set_facecolor('#1a1a2e')

    # 6. Cumulative P&L by Asset (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    asset_pnl = []
    for asset in assets:
        asset_trades = [t for t in trades if t['asset'] == asset]
        total = sum(t['net_pnl'] for t in asset_trades if t['net_pnl'] is not None)
        asset_pnl.append(total)

    colors = ['green' if p > 0 else 'red' for p in asset_pnl]
    ax6.bar(assets, asset_pnl, color=colors, alpha=0.7)
    ax6.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
    ax6.set_title('Total P&L by Asset', fontweight='bold')
    ax6.set_ylabel('P&L ($)')
    for i, (asset, pnl) in enumerate(zip(assets, asset_pnl)):
        ax6.text(i, pnl + (5 if pnl >= 0 else -10), f'${pnl:+.0f}',
                 ha='center', fontsize=9, color='white')
    ax6.set_facecolor('#1a1a2e')

    # Style
    fig.patch.set_facecolor('#0f0f23')
    for ax in [ax1, ax2, ax4, ax5, ax6]:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    # Adjust layout (rect leaves room for suptitle)
    plt.subplots_adjust(top=0.93)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Chart saved to: {output_path}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate trading performance charts')
    parser.add_argument('--db', type=str, default='data/sim_trading.db',
                        help='Path to trading database')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for PNG (default: display only)')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Display chart window (default: True)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display chart window')

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run the trading bot first to generate trades.")
        return

    print(f"Loading trades from {db_path}...")
    trades = load_trades(db_path)
    state = load_state(db_path)

    if not trades:
        print("No trades found in database.")
        return

    print(f"Found {len(trades)} trades")

    initial_bankroll = state.get('initial_bankroll', 1000)
    metrics = calculate_metrics(trades, initial_bankroll)

    output_path = Path(args.output) if args.output else None
    show = not args.no_show

    create_chart(trades, state, metrics, output_path, show)


if __name__ == "__main__":
    main()
