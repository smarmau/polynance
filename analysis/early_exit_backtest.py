#!/usr/bin/env python3
"""
Early Exit Strategy Backtest

Analyzes whether cutting losses early at t=10 or t=12.5 improves returns
compared to holding until window close.

For trades entered at t=7.5:
- We can check price at t=10 and t=12.5
- If price moves against us by X%, exit early
- Compare P&L of early exit vs holding to resolution
"""

import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "reports"
ASSETS = ["btc", "eth", "sol", "xrp"]

# Strategy params (matching our live trader)
BULL_THRESHOLD = 0.80
BEAR_THRESHOLD = 0.20
FEE_RATE = 0.001  # 0.1% on premium
SPREAD_COST = 0.005  # 0.5% spread


@dataclass
class TradeWithPrices:
    """Trade record with all price points for analysis."""
    timestamp: datetime
    asset: str
    direction: str  # 'bull' or 'bear'
    entry_price: float  # pm_yes at t=7.5
    price_t10: Optional[float]  # pm_yes at t=10
    price_t12_5: Optional[float]  # pm_yes at t=12.5
    final_outcome: str  # 'up' or 'down'

    # Calculated fields
    would_win: bool = False
    hold_pnl: float = 0.0  # P&L if held to resolution


def load_trade_data() -> pd.DataFrame:
    """Load window data with all price points."""
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
                pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5
            FROM windows
            WHERE outcome IS NOT NULL
              AND pm_yes_t7_5 IS NOT NULL
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


def calculate_pnl(direction: str, entry_price: float, exit_price: float,
                  is_early_exit: bool = False) -> float:
    """
    Calculate P&L for a trade.

    For early exit: we sell our position at current market price
    For hold to resolution: binary outcome (win full or lose full)

    Args:
        direction: 'bull' (bought YES) or 'bear' (bought NO)
        entry_price: Price when entering (pm_yes for bull, 1-pm_yes for bear)
        exit_price: Current pm_yes price (for early exit) or final outcome price
        is_early_exit: If True, calculate based on selling position
    """
    if direction == 'bull':
        # Bought YES at entry_price
        if is_early_exit:
            # Sell YES at exit_price
            # P&L = (exit_price - entry_price) - fees
            gross_pnl = exit_price - entry_price
        else:
            # Hold to resolution - binary outcome
            # exit_price here represents whether we won (1.0) or lost (0.0)
            gross_pnl = exit_price - entry_price
    else:
        # Bought NO at (1 - entry_price), where entry_price is pm_yes
        no_entry = 1 - entry_price
        if is_early_exit:
            # Current NO price = 1 - current_pm_yes
            no_exit = 1 - exit_price
            gross_pnl = no_exit - no_entry
        else:
            # Hold to resolution
            no_exit = exit_price  # This is 1.0 if we won, 0.0 if lost
            gross_pnl = no_exit - no_entry

    # Apply fees
    premium = entry_price if direction == 'bull' else (1 - entry_price)
    fee = premium * FEE_RATE
    spread = SPREAD_COST

    return gross_pnl - fee - spread


def generate_trades(df: pd.DataFrame) -> List[TradeWithPrices]:
    """Generate trades with all price points for analysis."""
    trades = []

    for _, row in df.iterrows():
        price_7_5 = row['pm_yes_t7_5']
        if pd.isna(price_7_5):
            continue

        direction = None
        if price_7_5 >= BULL_THRESHOLD:
            direction = 'bull'
        elif price_7_5 <= BEAR_THRESHOLD:
            direction = 'bear'

        if direction is None:
            continue

        # Determine if trade would win
        if direction == 'bull':
            would_win = row['outcome'] == 'up'
            # Hold P&L: win = (1 - entry), lose = -entry
            if would_win:
                hold_pnl = calculate_pnl('bull', price_7_5, 1.0, is_early_exit=False)
            else:
                hold_pnl = calculate_pnl('bull', price_7_5, 0.0, is_early_exit=False)
        else:
            would_win = row['outcome'] == 'down'
            # For bear: if we win, NO pays out 1.0; if lose, NO pays 0.0
            if would_win:
                hold_pnl = calculate_pnl('bear', price_7_5, 1.0, is_early_exit=False)
            else:
                hold_pnl = calculate_pnl('bear', price_7_5, 0.0, is_early_exit=False)

        trade = TradeWithPrices(
            timestamp=row['window_start_utc'],
            asset=row['asset'],
            direction=direction,
            entry_price=price_7_5,
            price_t10=row.get('pm_yes_t10'),
            price_t12_5=row.get('pm_yes_t12_5'),
            final_outcome=row['outcome'],
            would_win=would_win,
            hold_pnl=hold_pnl,
        )
        trades.append(trade)

    return trades


def analyze_early_exit_thresholds(trades: List[TradeWithPrices]) -> Dict:
    """
    Analyze different early exit thresholds.

    For losing trades, test exiting when price moves against us by X%.
    """
    results = {
        'exit_at_t10': {},
        'exit_at_t12_5': {},
    }

    # Test different threshold levels
    # Threshold = how much price must move against us to trigger exit
    thresholds = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    for exit_time in ['t10', 't12_5']:
        for thresh in thresholds:
            key = f"{exit_time}_{thresh}"

            total_pnl_with_exit = 0
            total_pnl_hold = 0
            exits_triggered = 0
            exits_that_helped = 0  # Exit saved money vs holding
            exits_that_hurt = 0   # Exit lost money vs holding

            losing_trades_analyzed = 0

            for trade in trades:
                # Get price at exit time
                if exit_time == 't10':
                    exit_price = trade.price_t10
                else:
                    exit_price = trade.price_t12_5

                if pd.isna(exit_price):
                    # No price data, assume hold
                    total_pnl_with_exit += trade.hold_pnl
                    total_pnl_hold += trade.hold_pnl
                    continue

                total_pnl_hold += trade.hold_pnl

                # Check if this is a trade moving against us
                should_exit = False

                if trade.direction == 'bull':
                    # Bull trade: price dropping is bad
                    price_move = exit_price - trade.entry_price
                    if price_move < -thresh:  # Price dropped by more than threshold
                        should_exit = True
                else:
                    # Bear trade: price rising is bad
                    price_move = exit_price - trade.entry_price
                    if price_move > thresh:  # Price rose by more than threshold
                        should_exit = True

                if should_exit:
                    exits_triggered += 1
                    # Calculate early exit P&L
                    early_exit_pnl = calculate_pnl(
                        trade.direction,
                        trade.entry_price,
                        exit_price,
                        is_early_exit=True
                    )
                    total_pnl_with_exit += early_exit_pnl

                    # Did this help or hurt?
                    if early_exit_pnl > trade.hold_pnl:
                        exits_that_helped += 1
                    else:
                        exits_that_hurt += 1

                    if not trade.would_win:
                        losing_trades_analyzed += 1
                else:
                    # Hold to resolution
                    total_pnl_with_exit += trade.hold_pnl

            results[f'exit_at_{exit_time}'][thresh] = {
                'threshold': thresh,
                'total_pnl_with_exit': total_pnl_with_exit,
                'total_pnl_hold': total_pnl_hold,
                'pnl_difference': total_pnl_with_exit - total_pnl_hold,
                'exits_triggered': exits_triggered,
                'exits_helped': exits_that_helped,
                'exits_hurt': exits_that_hurt,
                'exit_accuracy': exits_that_helped / exits_triggered if exits_triggered > 0 else 0,
            }

    return results


def analyze_losing_trades_price_movement(trades: List[TradeWithPrices]) -> pd.DataFrame:
    """
    Analyze price movements for trades that end up losing.

    This helps identify patterns: do losing trades show early warning signs?
    """
    losing_trade_data = []

    for trade in trades:
        if trade.would_win:
            continue  # Only analyze losers

        if pd.isna(trade.price_t10) or pd.isna(trade.price_t12_5):
            continue

        if trade.direction == 'bull':
            # Bull losing = price went down
            move_to_t10 = trade.price_t10 - trade.entry_price
            move_to_t12_5 = trade.price_t12_5 - trade.entry_price
        else:
            # Bear losing = price went up (bad for us)
            # Invert so positive = bad for our position
            move_to_t10 = trade.price_t10 - trade.entry_price
            move_to_t12_5 = trade.price_t12_5 - trade.entry_price

        losing_trade_data.append({
            'timestamp': trade.timestamp,
            'asset': trade.asset,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'price_t10': trade.price_t10,
            'price_t12_5': trade.price_t12_5,
            'move_to_t10': move_to_t10,
            'move_to_t12_5': move_to_t12_5,
            'hold_pnl': trade.hold_pnl,
        })

    return pd.DataFrame(losing_trade_data)


def analyze_winning_trades_price_movement(trades: List[TradeWithPrices]) -> pd.DataFrame:
    """Analyze price movements for winning trades (to avoid false exits)."""
    winning_trade_data = []

    for trade in trades:
        if not trade.would_win:
            continue

        if pd.isna(trade.price_t10) or pd.isna(trade.price_t12_5):
            continue

        if trade.direction == 'bull':
            move_to_t10 = trade.price_t10 - trade.entry_price
            move_to_t12_5 = trade.price_t12_5 - trade.entry_price
        else:
            move_to_t10 = trade.price_t10 - trade.entry_price
            move_to_t12_5 = trade.price_t12_5 - trade.entry_price

        winning_trade_data.append({
            'timestamp': trade.timestamp,
            'asset': trade.asset,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'price_t10': trade.price_t10,
            'price_t12_5': trade.price_t12_5,
            'move_to_t10': move_to_t10,
            'move_to_t12_5': move_to_t12_5,
            'hold_pnl': trade.hold_pnl,
        })

    return pd.DataFrame(winning_trade_data)


def create_analysis_charts(trades: List[TradeWithPrices],
                           threshold_results: Dict,
                           losing_df: pd.DataFrame,
                           winning_df: pd.DataFrame,
                           output_path: Path):
    """Create visualization charts for early exit analysis."""

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('EARLY EXIT STRATEGY ANALYSIS', fontsize=16, fontweight='bold', y=0.98)

    # 1. Threshold comparison - T10 exits
    ax1 = fig.add_subplot(gs[0, 0])
    t10_data = threshold_results['exit_at_t10']
    thresholds = sorted(t10_data.keys())
    pnl_diff = [t10_data[t]['pnl_difference'] for t in thresholds]
    colors = ['green' if p > 0 else 'red' for p in pnl_diff]
    ax1.bar([f"{t*100:.0f}%" for t in thresholds], pnl_diff, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('P&L Impact: Early Exit at t=10', fontweight='bold')
    ax1.set_xlabel('Exit Threshold (price move against)')
    ax1.set_ylabel('P&L Difference vs Hold ($)')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Threshold comparison - T12.5 exits
    ax2 = fig.add_subplot(gs[0, 1])
    t12_data = threshold_results['exit_at_t12_5']
    thresholds = sorted(t12_data.keys())
    pnl_diff = [t12_data[t]['pnl_difference'] for t in thresholds]
    colors = ['green' if p > 0 else 'red' for p in pnl_diff]
    ax2.bar([f"{t*100:.0f}%" for t in thresholds], pnl_diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('P&L Impact: Early Exit at t=12.5', fontweight='bold')
    ax2.set_xlabel('Exit Threshold (price move against)')
    ax2.set_ylabel('P&L Difference vs Hold ($)')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Exit accuracy by threshold
    ax3 = fig.add_subplot(gs[0, 2])
    t10_acc = [t10_data[t]['exit_accuracy'] * 100 for t in sorted(t10_data.keys())]
    t12_acc = [t12_data[t]['exit_accuracy'] * 100 for t in sorted(t12_data.keys())]
    x = range(len(thresholds))
    width = 0.35
    ax3.bar([i - width/2 for i in x], t10_acc, width, label='Exit at t=10', color='steelblue')
    ax3.bar([i + width/2 for i in x], t12_acc, width, label='Exit at t=12.5', color='darkorange')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{t*100:.0f}%" for t in sorted(t10_data.keys())])
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% breakeven')
    ax3.set_title('Exit Decision Accuracy', fontweight='bold')
    ax3.set_xlabel('Exit Threshold')
    ax3.set_ylabel('% of Exits That Helped')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)

    # 4. Losing trades: price movement distribution at t=10
    ax4 = fig.add_subplot(gs[1, 0])
    if len(losing_df) > 0:
        bull_losers = losing_df[losing_df['direction'] == 'bull']['move_to_t10']
        bear_losers = losing_df[losing_df['direction'] == 'bear']['move_to_t10']

        if len(bull_losers) > 0:
            ax4.hist(bull_losers, bins=20, alpha=0.6, label=f'Bull losers (n={len(bull_losers)})', color='green')
        if len(bear_losers) > 0:
            ax4.hist(bear_losers, bins=20, alpha=0.6, label=f'Bear losers (n={len(bear_losers)})', color='red')

        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.set_title('LOSING Trades: Price Move by t=10', fontweight='bold')
        ax4.set_xlabel('Price Change from Entry')
        ax4.set_ylabel('Frequency')
        ax4.legend()

    # 5. Winning trades: price movement distribution at t=10
    ax5 = fig.add_subplot(gs[1, 1])
    if len(winning_df) > 0:
        bull_winners = winning_df[winning_df['direction'] == 'bull']['move_to_t10']
        bear_winners = winning_df[winning_df['direction'] == 'bear']['move_to_t10']

        if len(bull_winners) > 0:
            ax5.hist(bull_winners, bins=20, alpha=0.6, label=f'Bull winners (n={len(bull_winners)})', color='green')
        if len(bear_winners) > 0:
            ax5.hist(bear_winners, bins=20, alpha=0.6, label=f'Bear winners (n={len(bear_winners)})', color='red')

        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax5.set_title('WINNING Trades: Price Move by t=10', fontweight='bold')
        ax5.set_xlabel('Price Change from Entry')
        ax5.set_ylabel('Frequency')
        ax5.legend()

    # 6. Overlap analysis - can we distinguish losers from winners?
    ax6 = fig.add_subplot(gs[1, 2])
    if len(losing_df) > 0 and len(winning_df) > 0:
        # For bull trades
        bull_losing_moves = losing_df[losing_df['direction'] == 'bull']['move_to_t10'].values
        bull_winning_moves = winning_df[winning_df['direction'] == 'bull']['move_to_t10'].values

        if len(bull_losing_moves) > 0 and len(bull_winning_moves) > 0:
            ax6.hist(bull_losing_moves, bins=15, alpha=0.5, label='Losers', color='red', density=True)
            ax6.hist(bull_winning_moves, bins=15, alpha=0.5, label='Winners', color='green', density=True)
            ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax6.set_title('BULL Trades: Winner vs Loser Distribution', fontweight='bold')
            ax6.set_xlabel('Price Change by t=10')
            ax6.set_ylabel('Density')
            ax6.legend()

    # 7. Number of exits triggered by threshold
    ax7 = fig.add_subplot(gs[2, 0])
    t10_exits = [t10_data[t]['exits_triggered'] for t in sorted(t10_data.keys())]
    t12_exits = [t12_data[t]['exits_triggered'] for t in sorted(t12_data.keys())]
    ax7.bar([i - width/2 for i in x], t10_exits, width, label='Exit at t=10', color='steelblue')
    ax7.bar([i + width/2 for i in x], t12_exits, width, label='Exit at t=12.5', color='darkorange')
    ax7.set_xticks(x)
    ax7.set_xticklabels([f"{t*100:.0f}%" for t in sorted(t10_data.keys())])
    ax7.set_title('Number of Exits Triggered', fontweight='bold')
    ax7.set_xlabel('Exit Threshold')
    ax7.set_ylabel('Count')
    ax7.legend()
    ax7.tick_params(axis='x', rotation=45)

    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    # Find best threshold
    best_t10 = max(t10_data.keys(), key=lambda t: t10_data[t]['pnl_difference'])
    best_t12 = max(t12_data.keys(), key=lambda t: t12_data[t]['pnl_difference'])

    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.would_win])
    losing_trades = len([t for t in trades if not t.would_win])

    baseline_pnl = sum(t.hold_pnl for t in trades)

    summary_text = f"""
ANALYSIS SUMMARY
{'='*50}

Dataset:
  Total Trades: {total_trades}
  Winners: {winning_trades} ({winning_trades/total_trades*100:.1f}%)
  Losers: {losing_trades} ({losing_trades/total_trades*100:.1f}%)

Baseline (Hold All): ${baseline_pnl:.2f}

Best Exit at t=10: {best_t10*100:.0f}% threshold
  P&L Improvement: ${t10_data[best_t10]['pnl_difference']:+.2f}
  Exits Triggered: {t10_data[best_t10]['exits_triggered']}
  Exit Accuracy: {t10_data[best_t10]['exit_accuracy']*100:.1f}%

Best Exit at t=12.5: {best_t12*100:.0f}% threshold
  P&L Improvement: ${t12_data[best_t12]['pnl_difference']:+.2f}
  Exits Triggered: {t12_data[best_t12]['exits_triggered']}
  Exit Accuracy: {t12_data[best_t12]['exit_accuracy']*100:.1f}%

{'='*50}
RECOMMENDATION:
"""

    # Add recommendation
    best_overall_pnl = max(
        t10_data[best_t10]['pnl_difference'],
        t12_data[best_t12]['pnl_difference']
    )

    if best_overall_pnl > 0:
        if t10_data[best_t10]['pnl_difference'] > t12_data[best_t12]['pnl_difference']:
            summary_text += f"  Exit at t=10 with {best_t10*100:.0f}% threshold\n"
            summary_text += f"  Expected improvement: ${t10_data[best_t10]['pnl_difference']:+.2f}"
        else:
            summary_text += f"  Exit at t=12.5 with {best_t12*100:.0f}% threshold\n"
            summary_text += f"  Expected improvement: ${t12_data[best_t12]['pnl_difference']:+.2f}"
    else:
        summary_text += "  NO EARLY EXIT - Hold to resolution\n"
        summary_text += "  Early exits do not improve returns in backtest"

    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_detailed_report(trades: List[TradeWithPrices],
                          threshold_results: Dict,
                          losing_df: pd.DataFrame):
    """Print detailed analysis report."""

    print("=" * 80)
    print("EARLY EXIT STRATEGY BACKTEST REPORT")
    print("=" * 80)
    print()

    # Dataset summary
    total = len(trades)
    winners = len([t for t in trades if t.would_win])
    losers = total - winners

    print("DATASET SUMMARY")
    print("-" * 40)
    print(f"Total Trades: {total}")
    print(f"Winners: {winners} ({winners/total*100:.1f}%)")
    print(f"Losers: {losers} ({losers/total*100:.1f}%)")
    print()

    # Baseline P&L
    baseline = sum(t.hold_pnl for t in trades)
    print(f"Baseline P&L (Hold All): ${baseline:.2f}")
    print()

    # Threshold analysis
    print("EXIT AT t=10 ANALYSIS")
    print("-" * 40)
    print(f"{'Threshold':>10} {'P&L Diff':>12} {'Exits':>8} {'Helped':>8} {'Hurt':>8} {'Accuracy':>10}")
    print("-" * 60)

    for thresh in sorted(threshold_results['exit_at_t10'].keys()):
        r = threshold_results['exit_at_t10'][thresh]
        print(f"{thresh*100:>9.0f}% ${r['pnl_difference']:>10.2f} {r['exits_triggered']:>8} "
              f"{r['exits_helped']:>8} {r['exits_hurt']:>8} {r['exit_accuracy']*100:>9.1f}%")

    print()
    print("EXIT AT t=12.5 ANALYSIS")
    print("-" * 40)
    print(f"{'Threshold':>10} {'P&L Diff':>12} {'Exits':>8} {'Helped':>8} {'Hurt':>8} {'Accuracy':>10}")
    print("-" * 60)

    for thresh in sorted(threshold_results['exit_at_t12_5'].keys()):
        r = threshold_results['exit_at_t12_5'][thresh]
        print(f"{thresh*100:>9.0f}% ${r['pnl_difference']:>10.2f} {r['exits_triggered']:>8} "
              f"{r['exits_helped']:>8} {r['exits_hurt']:>8} {r['exit_accuracy']*100:>9.1f}%")

    print()

    # Losing trade analysis
    if len(losing_df) > 0:
        print("LOSING TRADE PRICE MOVEMENTS")
        print("-" * 40)

        for direction in ['bull', 'bear']:
            dir_df = losing_df[losing_df['direction'] == direction]
            if len(dir_df) == 0:
                continue

            print(f"\n{direction.upper()} Losers (n={len(dir_df)}):")
            print(f"  Price move to t=10:   mean={dir_df['move_to_t10'].mean():+.4f}, "
                  f"std={dir_df['move_to_t10'].std():.4f}")
            print(f"  Price move to t=12.5: mean={dir_df['move_to_t12_5'].mean():+.4f}, "
                  f"std={dir_df['move_to_t12_5'].std():.4f}")

            # What % showed warning signs?
            if direction == 'bull':
                warned_t10 = (dir_df['move_to_t10'] < -0.05).sum()
                warned_t12 = (dir_df['move_to_t12_5'] < -0.05).sum()
            else:
                warned_t10 = (dir_df['move_to_t10'] > 0.05).sum()
                warned_t12 = (dir_df['move_to_t12_5'] > 0.05).sum()

            print(f"  Showed 5% warning by t=10: {warned_t10} ({warned_t10/len(dir_df)*100:.1f}%)")
            print(f"  Showed 5% warning by t=12.5: {warned_t12} ({warned_t12/len(dir_df)*100:.1f}%)")

    print()
    print("=" * 80)


def main():
    """Main execution function."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    df = load_trade_data()
    print(f"Loaded {len(df)} windows")

    print("Generating trades...")
    trades = generate_trades(df)
    print(f"Generated {len(trades)} trades matching strategy criteria")

    if len(trades) == 0:
        print("No trades to analyze!")
        return

    print("Analyzing early exit thresholds...")
    threshold_results = analyze_early_exit_thresholds(trades)

    print("Analyzing losing trade patterns...")
    losing_df = analyze_losing_trades_price_movement(trades)
    winning_df = analyze_winning_trades_price_movement(trades)

    print("Generating charts...")
    create_analysis_charts(
        trades,
        threshold_results,
        losing_df,
        winning_df,
        OUTPUT_DIR / f"early_exit_analysis_{timestamp}.png"
    )

    print()
    print_detailed_report(trades, threshold_results, losing_df)

    return trades, threshold_results


if __name__ == "__main__":
    main()
