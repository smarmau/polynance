#!/usr/bin/env python3
"""
Strategy B Fine-Tuning: Momentum Scalping

Exhaustive grid search over:
  - Entry time: t0, t2.5, t5, t7.5
  - Exit time: t7.5, t10, t12.5 (must be > entry)
  - Entry threshold: bull 0.55-0.75, bear 0.25-0.45 (various)
  - Trajectory filter: none, 0.03, 0.05, 0.08, 0.10, 0.15
  - Momentum requirement: none, positive only

All strategies: buy PM position at entry, sell at exit (no resolution).
Double fees (entry + exit).
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import json

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "reports"
ASSETS = ["btc", "eth", "sol", "xrp"]

INITIAL_BANKROLL = 1000.0
BASE_BET = 50.0
FEE_RATE = 0.001
SPREAD_COST = 0.005


def load_all_data() -> pd.DataFrame:
    all_data = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        query = """
            SELECT
                window_id, window_start_utc, outcome,
                pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5,
                pm_spread_t0, pm_spread_t5
            FROM windows
            WHERE outcome IS NOT NULL
              AND pm_yes_t0 IS NOT NULL
              AND pm_yes_t2_5 IS NOT NULL
              AND pm_yes_t5 IS NOT NULL
              AND pm_yes_t7_5 IS NOT NULL
              AND pm_yes_t10 IS NOT NULL
              AND pm_yes_t12_5 IS NOT NULL
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


def pnl_early_exit(direction: str, entry_pm_yes: float, exit_pm_yes: float, bet_size: float) -> float:
    if direction == 'bull':
        entry_contract = entry_pm_yes
        exit_contract = exit_pm_yes
    else:
        entry_contract = 1.0 - entry_pm_yes
        exit_contract = 1.0 - exit_pm_yes

    if entry_contract <= 0.001:
        return 0.0

    n_contracts = bet_size / entry_contract
    gross = n_contracts * (exit_contract - entry_contract)

    entry_fee = entry_contract * n_contracts * FEE_RATE
    exit_fee = exit_contract * n_contracts * FEE_RATE
    entry_spread = SPREAD_COST * bet_size
    exit_spread = SPREAD_COST * (n_contracts * exit_contract)

    return gross - entry_fee - exit_fee - entry_spread - exit_spread


TIME_COLS = {
    't0': 'pm_yes_t0',
    't2.5': 'pm_yes_t2_5',
    't5': 'pm_yes_t5',
    't7.5': 'pm_yes_t7_5',
    't10': 'pm_yes_t10',
    't12.5': 'pm_yes_t12_5',
}

TIME_ORDER = ['t0', 't2.5', 't5', 't7.5', 't10', 't12.5']


def run_momentum_strategy(df: pd.DataFrame,
                          entry_time: str,
                          exit_time: str,
                          bull_thresh: float,
                          bear_thresh: float,
                          min_trajectory: float = 0.0,
                          trajectory_from: str = None) -> Dict:
    """
    Run a single momentum strategy variant.

    Args:
        entry_time: When to enter (t0, t2.5, t5, t7.5)
        exit_time: When to exit (t7.5, t10, t12.5) - must be after entry
        bull_thresh: PM YES threshold for bull entry (e.g. 0.60)
        bear_thresh: PM YES threshold for bear entry (e.g. 0.40)
        min_trajectory: Minimum PM price movement from trajectory_from to entry_time
        trajectory_from: Time point to measure trajectory from (default: t0)
    """
    entry_col = TIME_COLS[entry_time]
    exit_col = TIME_COLS[exit_time]

    if trajectory_from is None:
        # Default: measure trajectory from t0 for all entries
        traj_col = TIME_COLS['t0']
    else:
        traj_col = TIME_COLS[trajectory_from]

    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    n_trades = 0
    n_wins = 0
    n_losses = 0
    total_pnl = 0.0
    pnls = []
    equity = [INITIAL_BANKROLL]

    # Per-asset tracking
    per_asset = {a: {'trades': 0, 'wins': 0, 'pnl': 0.0} for a in ['BTC', 'ETH', 'SOL', 'XRP']}

    # Streak tracking
    max_consec_wins = 0
    max_consec_losses = 0
    cur_w = 0
    cur_l = 0

    for _, row in df.iterrows():
        entry_pm = row[entry_col]
        exit_pm = row[exit_col]
        traj_pm = row[traj_col]

        if pd.isna(entry_pm) or pd.isna(exit_pm) or pd.isna(traj_pm):
            continue

        # Direction check
        direction = None
        if entry_pm >= bull_thresh:
            direction = 'bull'
            trajectory = entry_pm - traj_pm
        elif entry_pm <= bear_thresh:
            direction = 'bear'
            trajectory = traj_pm - entry_pm  # Positive = moving in our direction
        else:
            continue

        # Trajectory filter
        if min_trajectory > 0 and trajectory < min_trajectory:
            continue

        # Execute trade
        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        bankroll += pnl
        peak = max(peak, bankroll)
        total_pnl += pnl
        pnls.append(pnl)
        equity.append(bankroll)

        won = pnl > 0
        n_trades += 1
        if won:
            n_wins += 1
            cur_w += 1
            cur_l = 0
            max_consec_wins = max(max_consec_wins, cur_w)
        else:
            n_losses += 1
            cur_l += 1
            cur_w = 0
            max_consec_losses = max(max_consec_losses, cur_l)

        asset = row['asset']
        per_asset[asset]['trades'] += 1
        if won:
            per_asset[asset]['wins'] += 1
        per_asset[asset]['pnl'] += pnl

    if n_trades == 0:
        return None

    pnls = np.array(pnls)
    equity = np.array(equity)

    win_pnls = pnls[pnls > 0]
    loss_pnls = pnls[pnls <= 0]

    win_rate = n_wins / n_trades
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if n_trades > 1 else 0.0
    avg_win = float(np.mean(win_pnls)) if len(win_pnls) > 0 else 0.0
    avg_loss = float(np.mean(loss_pnls)) if len(loss_pnls) > 0 else 0.0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    gross_wins = float(np.sum(win_pnls)) if len(win_pnls) > 0 else 0.0
    gross_losses = abs(float(np.sum(loss_pnls))) if len(loss_pnls) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    peak_eq = np.maximum.accumulate(equity)
    dd = equity - peak_eq
    dd_pct = (dd / peak_eq) * 100
    max_dd = float(np.min(dd))
    max_dd_pct = float(np.min(dd_pct))

    periods_per_year = 96 * 365
    sharpe = (avg_pnl / std_pnl) * np.sqrt(periods_per_year) if std_pnl > 0 else 0.0

    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = (avg_pnl / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    total_return_pct = (equity[-1] - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    calmar = total_return_pct / abs(max_dd_pct) if max_dd_pct < 0 else 0.0

    recovery_factor = total_pnl / abs(max_dd) if max_dd < 0 else (float('inf') if total_pnl > 0 else 0.0)

    # Per-asset win rates
    pa_summary = {}
    for asset, data in per_asset.items():
        if data['trades'] > 0:
            pa_summary[asset] = {
                'trades': data['trades'],
                'win_rate': data['wins'] / data['trades'],
                'pnl': data['pnl'],
                'avg_pnl': data['pnl'] / data['trades'],
            }

    return {
        'entry_time': entry_time,
        'exit_time': exit_time,
        'bull_thresh': bull_thresh,
        'bear_thresh': bear_thresh,
        'min_trajectory': min_trajectory,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_bankroll': float(equity[-1]),
        'total_return_pct': total_return_pct,
        'avg_pnl': avg_pnl,
        'std_pnl': std_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'max_dd': max_dd,
        'max_dd_pct': max_dd_pct,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'recovery_factor': recovery_factor,
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'per_asset': pa_summary,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} windows")
    print(f"Date range: {df['window_start_utc'].min()} to {df['window_start_utc'].max()}")
    print()

    # =========================================================================
    # GRID SEARCH
    # =========================================================================

    # Entry/exit time combinations
    time_combos = [
        ('t0', 't5'),
        ('t0', 't7.5'),
        ('t0', 't10'),
        ('t0', 't12.5'),
        ('t2.5', 't7.5'),
        ('t2.5', 't10'),
        ('t2.5', 't12.5'),
        ('t5', 't10'),
        ('t5', 't12.5'),
        ('t7.5', 't10'),
        ('t7.5', 't12.5'),
    ]

    # Threshold pairs (bull, bear) — symmetric around 0.50
    threshold_pairs = [
        (0.55, 0.45),
        (0.58, 0.42),
        (0.60, 0.40),
        (0.63, 0.37),
        (0.65, 0.35),
        (0.68, 0.32),
        (0.70, 0.30),
        (0.75, 0.25),
    ]

    # Trajectory filters
    trajectory_filters = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15]

    total_combos = len(time_combos) * len(threshold_pairs) * len(trajectory_filters)
    print(f"Running {total_combos} strategy combinations...")
    print()

    all_results = []
    count = 0

    for entry_t, exit_t in time_combos:
        for bull_th, bear_th in threshold_pairs:
            for traj in trajectory_filters:
                count += 1
                result = run_momentum_strategy(
                    df,
                    entry_time=entry_t,
                    exit_time=exit_t,
                    bull_thresh=bull_th,
                    bear_thresh=bear_th,
                    min_trajectory=traj,
                )
                if result is not None and result['n_trades'] >= 50:
                    all_results.append(result)

    print(f"Completed. {len(all_results)} viable strategies (>= 50 trades)")
    print()

    # =========================================================================
    # SORT AND DISPLAY TOP STRATEGIES
    # =========================================================================

    # Sort by different criteria
    def safe_val(r, key, default=0):
        v = r.get(key, default)
        if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
            return default
        return v

    print("=" * 140)
    print("TOP 25 STRATEGIES BY TOTAL P&L")
    print("=" * 140)
    by_pnl = sorted(all_results, key=lambda r: safe_val(r, 'total_pnl'), reverse=True)[:25]
    print_table(by_pnl)

    print()
    print("=" * 140)
    print("TOP 25 STRATEGIES BY SHARPE RATIO")
    print("=" * 140)
    by_sharpe = sorted(all_results, key=lambda r: safe_val(r, 'sharpe'), reverse=True)[:25]
    print_table(by_sharpe)

    print()
    print("=" * 140)
    print("TOP 25 STRATEGIES BY EXPECTANCY (per trade)")
    print("=" * 140)
    by_exp = sorted(all_results, key=lambda r: safe_val(r, 'expectancy'), reverse=True)[:25]
    print_table(by_exp)

    print()
    print("=" * 140)
    print("TOP 25 STRATEGIES BY CALMAR RATIO (return / max DD)")
    print("=" * 140)
    by_calmar = sorted(all_results, key=lambda r: safe_val(r, 'calmar'), reverse=True)[:25]
    print_table(by_calmar)

    print()
    print("=" * 140)
    print("TOP 25 BY COMPOSITE SCORE (Sharpe * ProfitFactor * sqrt(trades))")
    print("=" * 140)
    for r in all_results:
        pf = safe_val(r, 'profit_factor', 0)
        sh = safe_val(r, 'sharpe', 0)
        nt = r['n_trades']
        r['composite'] = sh * min(pf, 10) * np.sqrt(nt) if sh > 0 and pf > 0 else 0
    by_composite = sorted(all_results, key=lambda r: r.get('composite', 0), reverse=True)[:25]
    print_table(by_composite, show_composite=True)

    # =========================================================================
    # FOCUS: Best strategies with good trade count
    # =========================================================================
    print()
    print("=" * 140)
    print("BALANCED PICKS: Profitable, Sharpe > 3, Trades > 200, MaxDD > -60%")
    print("=" * 140)
    balanced = [r for r in all_results
                if r['total_pnl'] > 0
                and safe_val(r, 'sharpe') > 3.0
                and r['n_trades'] > 200
                and r['max_dd_pct'] > -60]
    balanced = sorted(balanced, key=lambda r: safe_val(r, 'sharpe'), reverse=True)[:25]
    if balanced:
        print_table(balanced)
    else:
        print("  No strategies meet all criteria. Relaxing...")
        balanced = [r for r in all_results
                    if r['total_pnl'] > 0
                    and safe_val(r, 'sharpe') > 2.0
                    and r['n_trades'] > 100
                    and r['max_dd_pct'] > -80]
        balanced = sorted(balanced, key=lambda r: safe_val(r, 'sharpe'), reverse=True)[:25]
        print_table(balanced)

    # =========================================================================
    # DETAILED VIEW: Top 5 overall
    # =========================================================================
    print()
    print("=" * 140)
    print("DETAILED VIEW: TOP 5 BY COMPOSITE SCORE")
    print("=" * 140)
    for i, r in enumerate(by_composite[:5]):
        print(f"\n{'─'*80}")
        print(f"  #{i+1}: Entry={r['entry_time']}  Exit={r['exit_time']}  "
              f"Bull>={r['bull_thresh']:.2f}  Bear<={r['bear_thresh']:.2f}  "
              f"Traj>={r['min_trajectory']:.2f}")
        print(f"{'─'*80}")
        print(f"  Trades: {r['n_trades']:,}    Win Rate: {r['win_rate']:.1%}")
        print(f"  Total P&L: ${r['total_pnl']:,.2f}    Return: {r['total_return_pct']:.1f}%")
        print(f"  Avg Win: ${r['avg_win']:.3f}    Avg Loss: ${r['avg_loss']:.3f}    R:R: {r['rr_ratio']:.2f}")
        print(f"  Profit Factor: {safe_val(r, 'profit_factor'):.2f}    Expectancy: ${r['expectancy']:.3f}")
        print(f"  Sharpe: {safe_val(r, 'sharpe'):.1f}    Sortino: {safe_val(r, 'sortino'):.1f}    Calmar: {safe_val(r, 'calmar'):.2f}")
        print(f"  Max DD: ${r['max_dd']:,.2f} ({r['max_dd_pct']:.1f}%)    Recovery: {safe_val(r, 'recovery_factor'):.2f}")
        print(f"  Max Consec Wins: {r['max_consec_wins']}    Max Consec Losses: {r['max_consec_losses']}")
        print(f"  Per-Asset:")
        for asset, data in sorted(r.get('per_asset', {}).items()):
            print(f"    {asset}: {data['trades']} trades, {data['win_rate']:.1%} WR, "
                  f"${data['pnl']:.2f} P&L, ${data['avg_pnl']:.3f}/trade")

    # =========================================================================
    # SAVE
    # =========================================================================
    json_results = []
    for r in by_composite[:50]:
        jr = {k: v for k, v in r.items() if k != 'equity_curve'}
        for k, v in jr.items():
            if isinstance(v, (np.integer, np.int64)):
                jr[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                jr[k] = float(v)
        json_results.append(jr)

    output_path = OUTPUT_DIR / f"strategy_b_tuning_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nSaved top 50 results to: {output_path}")


def print_table(results: List[Dict], show_composite: bool = False):
    header = (f"  {'Entry':>5} {'Exit':>5} {'Bull':>5} {'Bear':>5} {'Traj':>5} "
              f"{'Trades':>7} {'Win%':>7} {'TotalPnL':>10} {'AvgPnL':>8} "
              f"{'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'PF':>6} {'Exp':>7} "
              f"{'MaxDD%':>8} {'Sharpe':>7} {'Sortino':>7} {'Calmar':>7} "
              f"{'MaxCW':>5} {'MaxCL':>5}")
    if show_composite:
        header += f" {'Score':>8}"
    print(header)
    print(f"  {'─'*len(header)}")

    for r in results:
        pf = min(r.get('profit_factor', 0), 99.9)
        if isinstance(pf, float) and (np.isinf(pf) or np.isnan(pf)):
            pf = 99.9
        rf = r.get('recovery_factor', 0)
        if isinstance(rf, float) and (np.isinf(rf) or np.isnan(rf)):
            rf = 99.9
        calmar = r.get('calmar', 0)
        if isinstance(calmar, float) and (np.isinf(calmar) or np.isnan(calmar)):
            calmar = 99.9
        sharpe = r.get('sharpe', 0)
        if isinstance(sharpe, float) and (np.isinf(sharpe) or np.isnan(sharpe)):
            sharpe = 99.9
        sortino = r.get('sortino', 0)
        if isinstance(sortino, float) and (np.isinf(sortino) or np.isnan(sortino)):
            sortino = 99.9

        line = (f"  {r['entry_time']:>5} {r['exit_time']:>5} "
                f"{r['bull_thresh']:>5.2f} {r['bear_thresh']:>5.2f} {r['min_trajectory']:>5.2f} "
                f"{r['n_trades']:>7} {r['win_rate']:>6.1%} ${r['total_pnl']:>8,.0f} "
                f"${r['avg_pnl']:>7.3f} ${r['avg_win']:>7.3f} ${r['avg_loss']:>7.3f} "
                f"{r['rr_ratio']:>6.2f} {pf:>6.2f} ${r['expectancy']:>6.3f} "
                f"{r['max_dd_pct']:>7.1f}% {sharpe:>7.1f} {sortino:>7.1f} {calmar:>7.2f} "
                f"{r['max_consec_wins']:>5} {r['max_consec_losses']:>5}")
        if show_composite:
            line += f" {r.get('composite', 0):>8.0f}"
        print(line)


if __name__ == "__main__":
    main()
