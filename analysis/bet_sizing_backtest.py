#!/usr/bin/env python3
"""
Backtest different bet sizing cap percentages to find optimal configuration.
Tests: 5%, 10%, 15%, 20%, 25%, and no cap (100%).
"""

import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ["btc", "eth", "sol", "xrp"]
INITIAL_BANKROLL = 1000.0
BASE_BET = 25.0
FEE_RATE = 0.001  # 0.1% taker fee (matches actual Polymarket)
SPREAD_COST = 0.005  # 0.5% spread estimate

# Strategy parameters (t=7.5, 0.80/0.20 thresholds)
PRICE_COL = "pm_yes_t7_5"
BULL_THRESH = 0.80
BEAR_THRESH = 0.20

# Anti-martingale parameters
WIN_MULT = 2.0
LOSS_MULT = 0.5
FLOOR_PCT = 0.10  # 10% of base bet = $2.50

# Cap percentages to test
CAP_PERCENTAGES = [0.05, 0.10, 0.15, 0.20, 0.25, 1.0]  # 1.0 = no cap


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    cap_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_bet_used: float
    avg_bet: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    bet_sizes: List[float]


def load_all_data() -> pd.DataFrame:
    """Load and combine data from all asset databases."""
    all_data = []

    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            print(f"Warning: {db_path} not found")
            continue

        conn = sqlite3.connect(db_path)
        query = """
            SELECT
                window_id, window_start_utc, outcome,
                pm_yes_t7_5
            FROM windows
            WHERE outcome IS NOT NULL
            ORDER BY window_start_utc
        """
        df = pd.read_sql_query(query, conn)
        df['asset'] = asset.upper()
        df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
        all_data.append(df)
        conn.close()

    if not all_data:
        raise ValueError("No data found in any database")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('window_start_utc').reset_index(drop=True)
    return combined


def generate_trades(df: pd.DataFrame) -> List[Dict]:
    """Generate trade signals based on strategy parameters."""
    trades = []

    for _, row in df.iterrows():
        price = row[PRICE_COL]
        if pd.isna(price):
            continue

        trade = None
        if price >= BULL_THRESH:
            # Bull signal: buy YES
            win = row['outcome'] == 'up'
            entry_price = price
            if win:
                gross_pnl = (1 - entry_price)
            else:
                gross_pnl = -entry_price

            # Fee on contract premium (all trades)
            fee = entry_price * FEE_RATE
            pnl_per_dollar = gross_pnl - fee - SPREAD_COST

            trade = {
                'timestamp': row['window_start_utc'],
                'asset': row['asset'],
                'direction': 'bull',
                'entry_price': price,
                'outcome': row['outcome'],
                'win': win,
                'pnl_per_dollar': pnl_per_dollar,
            }
        elif price <= BEAR_THRESH:
            # Bear signal: buy NO
            win = row['outcome'] == 'down'
            entry_price = 1 - price  # NO price
            if win:
                gross_pnl = (1 - entry_price)
            else:
                gross_pnl = -entry_price

            # Fee on contract premium (all trades)
            fee = entry_price * FEE_RATE
            pnl_per_dollar = gross_pnl - fee - SPREAD_COST

            trade = {
                'timestamp': row['window_start_utc'],
                'asset': row['asset'],
                'direction': 'bear',
                'entry_price': price,
                'outcome': row['outcome'],
                'win': win,
                'pnl_per_dollar': pnl_per_dollar,
            }

        if trade:
            trades.append(trade)

    return trades


def run_backtest(trades: List[Dict], cap_pct: float) -> BacktestResult:
    """Run backtest with specified cap percentage."""
    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    current_bet = BASE_BET
    floor_bet = BASE_BET * FLOOR_PCT

    equity_curve = [INITIAL_BANKROLL]
    drawdown_curve = [0.0]
    bet_sizes = []
    pnls = []
    wins = []

    for t in trades:
        # Calculate actual bet (apply cap and floor)
        max_bet = bankroll * cap_pct
        actual_bet = min(current_bet, max_bet)
        actual_bet = max(actual_bet, floor_bet)
        actual_bet = min(actual_bet, bankroll)  # Can't bet more than we have

        if actual_bet < floor_bet and bankroll < floor_bet:
            actual_bet = bankroll  # Bet remaining if below floor

        bet_sizes.append(actual_bet)

        # Calculate P&L
        pnl = t['pnl_per_dollar'] * actual_bet
        pnls.append(pnl)
        wins.append(t['win'])

        bankroll += pnl
        peak = max(peak, bankroll)
        dd = bankroll - peak
        dd_pct = (dd / peak) * 100 if peak > 0 else 0

        equity_curve.append(bankroll)
        drawdown_curve.append(dd_pct)

        # Update bet size for next trade (anti-martingale)
        if t['win']:
            current_bet = actual_bet * WIN_MULT
        else:
            current_bet = actual_bet * LOSS_MULT
        current_bet = max(current_bet, floor_bet)

    # Calculate metrics
    n_trades = len(trades)
    n_wins = sum(wins)
    n_losses = n_trades - n_wins
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    total_pnl = sum(pnls)
    total_return_pct = (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100

    max_dd = min(drawdown_curve)
    max_dd_dollars = min([equity_curve[i] - max(equity_curve[:i+1]) for i in range(len(equity_curve))])

    # Sharpe ratio
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 * 96)
    else:
        sharpe = 0

    # Sortino ratio
    downside = [p for p in pnls if p < 0]
    if downside and np.std(downside) > 0:
        sortino = (np.mean(pnls) / np.std(downside)) * np.sqrt(252 * 96)
    else:
        sortino = 0

    # Calmar ratio
    calmar = total_return_pct / abs(max_dd) if max_dd < 0 else 0

    # Profit factor
    win_pnls = [p for p, w in zip(pnls, wins) if w]
    loss_pnls = [p for p, w in zip(pnls, wins) if not w]
    gross_wins = sum(win_pnls) if win_pnls else 0
    gross_losses = abs(sum(loss_pnls)) if loss_pnls else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0

    return BacktestResult(
        cap_pct=cap_pct,
        trades=n_trades,
        wins=n_wins,
        losses=n_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        max_drawdown=max_dd_dollars,
        max_drawdown_pct=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_bet_used=max(bet_sizes),
        avg_bet=np.mean(bet_sizes),
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        bet_sizes=bet_sizes,
    )


def create_comparison_chart(results: Dict[float, BacktestResult], output_path: Path):
    """Create comparison chart for all cap percentages."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('BET SIZING CAP COMPARISON - Anti-Martingale Strategy',
                 fontsize=14, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    caps = list(results.keys())
    cap_labels = [f"{int(c*100)}%" if c < 1 else "No Cap" for c in caps]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(caps)))

    # 1. Equity Curves
    ax1 = fig.add_subplot(gs[0, :2])
    for cap, color in zip(caps, colors):
        r = results[cap]
        label = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        ax1.plot(r.equity_curve, label=f"{label} (${r.total_pnl:,.0f})",
                 color=color, linewidth=1.5)
    ax1.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curves', fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Bankroll ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Summary Stats Table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    table_data = []
    for cap in caps:
        r = results[cap]
        cap_str = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "Inf"
        table_data.append([
            cap_str,
            f"${r.total_pnl:,.0f}",
            f"{r.max_drawdown_pct:.1f}%",
            f"{r.sharpe:.1f}",
            pf_str,
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=['Cap', 'Total P&L', 'Max DD', 'Sharpe', 'PF'],
        cellLoc='center',
        loc='center',
        colColours=['lightblue'] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax2.set_title('Performance Summary', fontweight='bold', pad=20)

    # 3. Drawdown Curves
    ax3 = fig.add_subplot(gs[1, :2])
    for cap, color in zip(caps, colors):
        r = results[cap]
        label = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        ax3.fill_between(range(len(r.drawdown_curve)), r.drawdown_curve,
                         alpha=0.2, color=color)
        ax3.plot(r.drawdown_curve, label=f"{label} ({r.max_drawdown_pct:.1f}%)",
                 color=color, linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Drawdown Curves', fontweight='bold')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Bet Size Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    for cap, color in zip(caps, colors):
        r = results[cap]
        label = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        ax4.hist(r.bet_sizes, bins=30, alpha=0.4, color=color, label=label)
    ax4.set_title('Bet Size Distribution', fontweight='bold')
    ax4.set_xlabel('Bet Size ($)')
    ax4.set_ylabel('Frequency')
    ax4.legend(loc='upper right', fontsize=8)

    # 5. Return vs Risk (Scatter)
    ax5 = fig.add_subplot(gs[2, 0])
    for cap, color in zip(caps, colors):
        r = results[cap]
        label = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        ax5.scatter(abs(r.max_drawdown_pct), r.total_return_pct,
                   color=color, s=150, label=label, edgecolors='black', linewidth=1)
    ax5.set_title('Return vs Max Drawdown', fontweight='bold')
    ax5.set_xlabel('Max Drawdown (%)')
    ax5.set_ylabel('Total Return (%)')
    ax5.legend(loc='lower right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Sharpe vs Return
    ax6 = fig.add_subplot(gs[2, 1])
    sharpes = [results[c].sharpe for c in caps]
    returns = [results[c].total_return_pct for c in caps]
    bars = ax6.bar(cap_labels, sharpes, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_title('Sharpe Ratio by Cap', fontweight='bold')
    ax6.set_ylabel('Sharpe Ratio')
    ax6.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sharpe=1')
    for bar, s in zip(bars, sharpes):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{s:.1f}', ha='center', fontsize=9)

    # 7. Risk-Adjusted Comparison (Calmar)
    ax7 = fig.add_subplot(gs[2, 2])
    calmars = [results[c].calmar for c in caps]
    bars = ax7.bar(cap_labels, calmars, color=colors, alpha=0.7, edgecolor='black')
    ax7.set_title('Calmar Ratio by Cap', fontweight='bold')
    ax7.set_ylabel('Calmar Ratio')
    for bar, c in zip(bars, calmars):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{c:.2f}', ha='center', fontsize=9)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("BET SIZING CAP BACKTEST")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = load_all_data()
    print(f"Total windows: {len(df)}")
    print(f"Date range: {df['window_start_utc'].min()} to {df['window_start_utc'].max()}")
    print()

    # Generate trades
    print("Generating trades...")
    trades = generate_trades(df)
    print(f"Total trades: {len(trades)}")
    wins = sum(1 for t in trades if t['win'])
    print(f"Win rate: {wins}/{len(trades)} = {wins/len(trades)*100:.1f}%")
    print()

    # Run backtests for each cap percentage
    print("Running backtests...")
    results = {}
    for cap in CAP_PERCENTAGES:
        cap_str = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        print(f"  Testing {cap_str}...", end=" ")
        result = run_backtest(trades, cap)
        results[cap] = result
        print(f"P&L: ${result.total_pnl:,.2f}, DD: {result.max_drawdown_pct:.1f}%, "
              f"Sharpe: {result.sharpe:.2f}")
    print()

    # Print detailed comparison
    print("=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)
    print()

    header = f"{'Cap':<10} {'P&L':>12} {'Return':>10} {'Max DD':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'PF':>8} {'Avg Bet':>10} {'Max Bet':>10}"
    print(header)
    print("-" * len(header))

    for cap in CAP_PERCENTAGES:
        r = results[cap]
        cap_str = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "Inf"
        print(f"{cap_str:<10} ${r.total_pnl:>10,.0f} {r.total_return_pct:>9.1f}% "
              f"{r.max_drawdown_pct:>9.1f}% {r.sharpe:>8.2f} {r.sortino:>8.2f} "
              f"{r.calmar:>8.2f} {pf_str:>8} ${r.avg_bet:>9.2f} ${r.max_bet_used:>9.2f}")

    print()

    # Find optimal by different criteria
    print("OPTIMAL CAPS:")
    print(f"  By Total P&L:    {max(results.keys(), key=lambda c: results[c].total_pnl)*100:.0f}%")
    print(f"  By Sharpe Ratio: {max(results.keys(), key=lambda c: results[c].sharpe)*100:.0f}%")
    print(f"  By Calmar Ratio: {max(results.keys(), key=lambda c: results[c].calmar)*100:.0f}%")
    print(f"  By Min Drawdown: {min(results.keys(), key=lambda c: abs(results[c].max_drawdown_pct))*100:.0f}%")
    print()

    # Risk-adjusted score (custom: return / sqrt(drawdown))
    print("RISK-ADJUSTED RANKING (Return / sqrt(|MaxDD|)):")
    ranked = sorted(results.keys(),
                   key=lambda c: results[c].total_return_pct / np.sqrt(abs(results[c].max_drawdown_pct) + 1),
                   reverse=True)
    for i, cap in enumerate(ranked, 1):
        r = results[cap]
        score = r.total_return_pct / np.sqrt(abs(r.max_drawdown_pct) + 1)
        cap_str = f"{int(cap*100)}%" if cap < 1 else "No Cap"
        print(f"  {i}. {cap_str}: {score:.2f} (Return: {r.total_return_pct:.1f}%, DD: {r.max_drawdown_pct:.1f}%)")
    print()

    # Create chart
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)
    create_comparison_chart(results, output_dir / "bet_cap_comparison.png")

    print(f"Chart saved to: {output_dir / 'bet_cap_comparison.png'}")
    print()

    return results


if __name__ == "__main__":
    main()
