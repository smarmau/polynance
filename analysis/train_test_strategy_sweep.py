#!/usr/bin/env python3
"""
Train/Test Strategy Sweep (70/30 split)

Systematically tests a wide range of strategy ideas on a 70% training set,
then validates on a 30% held-out test set to detect overfitting.

Strategies tested:
1. Simple momentum (various entry/exit/thresholds)
2. Contrarian (prev window reversal)
3. Spread-based (wide vs tight spreads)
4. Intra-window velocity (PM movement speed)
5. Time-of-day filters
6. Spot price momentum
7. Entry price range filters
8. Consecutive window patterns
9. Combined/hybrid approaches
"""

import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import json

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']
FEE_RATE = 0.001
SPREAD_COST = 0.005
BET_SIZE = 50.0


def pnl_early_exit(direction, entry_pm, exit_pm, bet=BET_SIZE):
    """Calculate P&L for early exit trade."""
    if direction == 'bull':
        entry_c, exit_c = entry_pm, exit_pm
    else:
        entry_c, exit_c = 1.0 - entry_pm, 1.0 - exit_pm
    if entry_c <= 0.001:
        return 0.0
    n = bet / entry_c
    gross = n * (exit_c - entry_c)
    fees = entry_c * n * FEE_RATE + exit_c * n * FEE_RATE
    spread = SPREAD_COST * bet + SPREAD_COST * (n * exit_c)
    return gross - fees - spread


def load_all_data():
    """Load all windows with samples data from all assets."""
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Get windows with all PM data
        rows = conn.execute("""
            SELECT w.window_id, w.asset, w.window_start_utc, w.outcome, w.outcome_binary,
                   w.pm_yes_t0, w.pm_yes_t2_5, w.pm_yes_t5, w.pm_yes_t7_5, w.pm_yes_t10, w.pm_yes_t12_5,
                   w.pm_spread_t0, w.pm_spread_t5,
                   w.pm_price_momentum_0_to_5, w.pm_price_momentum_5_to_10,
                   w.spot_open, w.spot_close, w.spot_change_bps, w.spot_range_bps
            FROM windows w
            WHERE w.outcome IS NOT NULL
            ORDER BY w.window_start_utc
        """).fetchall()

        all_rows.extend([dict(r) for r in rows])
        conn.close()

    # Sort by time
    all_rows.sort(key=lambda x: x['window_start_utc'])

    # Add sequential info per asset (prev window data, streaks, etc.)
    asset_windows = defaultdict(list)
    for r in all_rows:
        asset_windows[r['asset']].append(r)

    enriched = []
    for asset, wins in asset_windows.items():
        streak = 0
        for i, w in enumerate(wins):
            row = dict(w)

            # Previous window data
            if i > 0:
                prev = wins[i - 1]
                row['prev_pm_t12_5'] = prev['pm_yes_t12_5']
                row['prev_pm_t0'] = prev['pm_yes_t0']
                row['prev_pm_t5'] = prev['pm_yes_t5']
                row['prev_outcome'] = prev['outcome']
                row['prev_spot_change_bps'] = prev.get('spot_change_bps')

                # 2-back
                if i > 1:
                    row['prev2_outcome'] = wins[i - 2]['outcome']
                    row['prev2_pm_t12_5'] = wins[i - 2]['pm_yes_t12_5']
                else:
                    row['prev2_outcome'] = None
                    row['prev2_pm_t12_5'] = None
            else:
                row['prev_pm_t12_5'] = None
                row['prev_pm_t0'] = None
                row['prev_pm_t5'] = None
                row['prev_outcome'] = None
                row['prev_spot_change_bps'] = None
                row['prev2_outcome'] = None
                row['prev2_pm_t12_5'] = None

            # Intra-window velocity
            if row['pm_yes_t0'] is not None and row['pm_yes_t2_5'] is not None:
                row['velocity_0_2_5'] = row['pm_yes_t2_5'] - row['pm_yes_t0']
            else:
                row['velocity_0_2_5'] = None

            if row['pm_yes_t2_5'] is not None and row['pm_yes_t5'] is not None:
                row['velocity_2_5_5'] = row['pm_yes_t5'] - row['pm_yes_t2_5']
            else:
                row['velocity_2_5_5'] = None

            if row['pm_yes_t5'] is not None and row['pm_yes_t7_5'] is not None:
                row['velocity_5_7_5'] = row['pm_yes_t7_5'] - row['pm_yes_t5']
            else:
                row['velocity_5_7_5'] = None

            # Hour of day
            try:
                row['hour_utc'] = int(row['window_start_utc'][11:13])
            except:
                row['hour_utc'] = None

            enriched.append(row)

    enriched.sort(key=lambda x: x['window_start_utc'])
    return enriched


def calc_metrics(pnls):
    """Calculate strategy metrics from list of P&Ls."""
    if len(pnls) < 10:
        return None
    arr = np.array(pnls)
    n = len(arr)
    wins = int(np.sum(arr > 0))
    total = float(np.sum(arr))
    avg = float(np.mean(arr))
    std = float(np.std(arr))
    sharpe = avg / std * np.sqrt(n) if std > 0 else 0
    wr = wins / n
    avg_w = float(np.mean(arr[arr > 0])) if np.sum(arr > 0) > 0 else 0
    avg_l = float(np.mean(arr[arr <= 0])) if np.sum(arr <= 0) > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    return {
        'n_trades': n,
        'win_rate': wr,
        'total_pnl': total,
        'avg_pnl': avg,
        'sharpe': sharpe,
        'avg_win': avg_w,
        'avg_loss': avg_l,
        'max_dd': max_dd,
        'profit_factor': abs(np.sum(arr[arr > 0]) / np.sum(arr[arr <= 0])) if np.sum(arr <= 0) != 0 else float('inf'),
    }


def run_strategy(data, strategy_fn):
    """Run a strategy function over data, return list of P&Ls."""
    pnls = []
    for r in data:
        result = strategy_fn(r)
        if result is not None:
            direction, entry_pm, exit_pm = result
            pnl = pnl_early_exit(direction, entry_pm, exit_pm)
            pnls.append(pnl)
    return pnls


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

def make_momentum(entry_col, exit_col, thresh):
    """Simple momentum: if PM >= thresh → bull, <= 1-thresh → bear."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if pm is None or ex is None:
            return None
        if pm >= thresh:
            return ('bull', pm, ex)
        elif pm <= (1.0 - thresh):
            return ('bear', pm, ex)
        return None
    return strategy


def make_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th):
    """Contrarian: bet against prev window direction."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and pm <= bear_th:
            return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


def make_velocity_filter(entry_col, exit_col, vel_col, vel_min, thresh):
    """Only trade when intra-window velocity exceeds threshold."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        vel = r.get(vel_col)
        if pm is None or ex is None or vel is None:
            return None
        if abs(vel) < vel_min:
            return None
        if pm >= thresh:
            return ('bull', pm, ex)
        elif pm <= (1.0 - thresh):
            return ('bear', pm, ex)
        return None
    return strategy


def make_spread_filter(entry_col, exit_col, spread_col, max_spread, thresh):
    """Only trade when spread is tight (market is liquid/confident)."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        spread = r.get(spread_col)
        if pm is None or ex is None or spread is None:
            return None
        if spread > max_spread:
            return None
        if pm >= thresh:
            return ('bull', pm, ex)
        elif pm <= (1.0 - thresh):
            return ('bear', pm, ex)
        return None
    return strategy


def make_time_of_day_filter(entry_col, exit_col, thresh, hours):
    """Only trade during specific hours."""
    def strategy(r):
        hour = r.get('hour_utc')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if hour is None or pm is None or ex is None:
            return None
        if hour not in hours:
            return None
        if pm >= thresh:
            return ('bull', pm, ex)
        elif pm <= (1.0 - thresh):
            return ('bear', pm, ex)
        return None
    return strategy


def make_streak_filter(entry_col, exit_col, thresh, require_prev_outcome):
    """Only trade after specific prev window outcome."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        prev = r.get('prev_outcome')
        if pm is None or ex is None or prev is None:
            return None
        if prev != require_prev_outcome:
            return None
        if pm >= thresh:
            return ('bull', pm, ex)
        elif pm <= (1.0 - thresh):
            return ('bear', pm, ex)
        return None
    return strategy


def make_double_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th):
    """Double contrarian: prev AND prev2 both strong in same direction."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or prev2 is None or pm is None or ex is None:
            return None
        # Both prev windows strong UP → expect reversal DOWN
        if prev >= prev_thresh and prev2 >= prev_thresh:
            if pm <= bear_th:
                return ('bear', pm, ex)
        # Both prev windows strong DOWN → expect reversal UP
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh):
            if pm >= bull_th:
                return ('bull', pm, ex)
        return None
    return strategy


def make_entry_price_filter(entry_col, exit_col, thresh, min_entry, max_entry):
    """Only trade when entry contract price is in a specific range."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if pm is None or ex is None:
            return None
        direction = None
        if pm >= thresh:
            direction = 'bull'
            entry_c = pm
        elif pm <= (1.0 - thresh):
            direction = 'bear'
            entry_c = 1.0 - pm
        else:
            return None
        if entry_c < min_entry or entry_c > max_entry:
            return None
        return (direction, pm, ex)
    return strategy


def make_spot_momentum_filter(entry_col, exit_col, thresh, min_spot_bps):
    """Only trade when spot price has already moved significantly."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        prev_spot = r.get('prev_spot_change_bps')
        if pm is None or ex is None or prev_spot is None:
            return None
        # Contrarian on spot: after big spot move, bet reversal
        if prev_spot >= min_spot_bps and pm <= (1.0 - thresh):
            return ('bear', pm, ex)  # spot went up hard, bet down
        elif prev_spot <= -min_spot_bps and pm >= thresh:
            return ('bull', pm, ex)  # spot went down hard, bet up
        return None
    return strategy


def make_fade_extreme(entry_col, exit_col, extreme_thresh):
    """Fade extreme PM prices — bet that extreme t0/t5 prices revert."""
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if pm is None or ex is None:
            return None
        # PM is extremely bullish → fade it (bet bear)
        if pm >= extreme_thresh:
            return ('bear', pm, ex)
        # PM is extremely bearish → fade it (bet bull)
        elif pm <= (1.0 - extreme_thresh):
            return ('bull', pm, ex)
        return None
    return strategy


def make_momentum_confirm(entry1_col, entry2_col, exit_col, thresh1, thresh2):
    """Two-timepoint confirmation: both must agree on direction."""
    def strategy(r):
        pm1 = r.get(entry1_col)
        pm2 = r.get(entry2_col)
        ex = r.get(exit_col)
        if pm1 is None or pm2 is None or ex is None:
            return None
        # Both bullish
        if pm1 >= thresh1 and pm2 >= thresh2:
            return ('bull', pm2, ex)
        # Both bearish
        elif pm1 <= (1.0 - thresh1) and pm2 <= (1.0 - thresh2):
            return ('bear', pm2, ex)
        return None
    return strategy


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("Loading and enriching data...")
    data = load_all_data()
    print(f"Total enriched rows: {len(data)}")

    # 70/30 split by time
    n = len(data)
    split_idx = int(n * 0.70)
    train = data[:split_idx]
    test = data[split_idx:]

    train_start = train[0]['window_start_utc'][:16]
    train_end = train[-1]['window_start_utc'][:16]
    test_start = test[0]['window_start_utc'][:16]
    test_end = test[-1]['window_start_utc'][:16]

    print(f"Train: {len(train)} rows ({train_start} to {train_end})")
    print(f"Test:  {len(test)} rows ({test_start} to {test_end})")

    # Build strategy list
    strategies = {}

    # 1. MOMENTUM
    for et, ec in [('t5', 'pm_yes_t5'), ('t7.5', 'pm_yes_t7_5')]:
        for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
            for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                strategies[f"MOM_{et}→{xt}_{th}"] = make_momentum(ec, xc, th)

    # 2. CONTRARIAN
    for pt in [0.70, 0.75, 0.80, 0.85]:
        for et, ec in [('t0', 'pm_yes_t0'), ('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.50, 0.50), (0.55, 0.45), (0.60, 0.40)]:
                    strategies[f"CONTRA_{pt}_{et}→{xt}_b{bt}_r{brt}"] = make_contrarian(pt, ec, xc, bt, brt)

    # 3. VELOCITY FILTER
    for vel_min in [0.05, 0.10, 0.15, 0.20]:
        for th in [0.55, 0.60, 0.65]:
            strategies[f"VEL_0→2.5_{vel_min}_{th}_t5→t12.5"] = make_velocity_filter(
                'pm_yes_t5', 'pm_yes_t12_5', 'velocity_0_2_5', vel_min, th)
            strategies[f"VEL_2.5→5_{vel_min}_{th}_t5→t12.5"] = make_velocity_filter(
                'pm_yes_t5', 'pm_yes_t12_5', 'velocity_2_5_5', vel_min, th)

    # 4. SPREAD FILTER
    for max_sp in [0.005, 0.01, 0.02, 0.03]:
        for th in [0.60, 0.65, 0.70]:
            strategies[f"SPREAD_t0_{max_sp}_{th}_t5→t12.5"] = make_spread_filter(
                'pm_yes_t5', 'pm_yes_t12_5', 'pm_spread_t0', max_sp, th)
            strategies[f"SPREAD_t5_{max_sp}_{th}_t5→t12.5"] = make_spread_filter(
                'pm_yes_t5', 'pm_yes_t12_5', 'pm_spread_t5', max_sp, th)

    # 5. TIME OF DAY
    hour_blocks = {
        '00-04': set(range(0, 4)),
        '00-08': set(range(0, 8)),
        '04-12': set(range(4, 12)),
        '12-20': set(range(12, 20)),
        '16-24': set(range(16, 24)),
        '20-04': set(range(20, 24)) | set(range(0, 4)),
    }
    for block_name, hours in hour_blocks.items():
        for th in [0.60, 0.65, 0.70]:
            strategies[f"TOD_{block_name}_{th}_t5→t12.5"] = make_time_of_day_filter(
                'pm_yes_t5', 'pm_yes_t12_5', th, hours)
            strategies[f"TOD_{block_name}_{th}_t7.5→t12.5"] = make_time_of_day_filter(
                'pm_yes_t7_5', 'pm_yes_t12_5', th, hours)

    # 6. STREAK / PREV OUTCOME
    for prev_out in ['up', 'down']:
        for th in [0.60, 0.65, 0.70]:
            strategies[f"AFTER_{prev_out}_{th}_t5→t12.5"] = make_streak_filter(
                'pm_yes_t5', 'pm_yes_t12_5', th, prev_out)

    # 7. DOUBLE CONTRARIAN
    for pt in [0.70, 0.75, 0.80]:
        for bt, brt in [(0.50, 0.50), (0.55, 0.45)]:
            strategies[f"DBL_CONTRA_{pt}_t5→t12.5_b{bt}_r{brt}"] = make_double_contrarian(
                pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt)

    # 8. ENTRY PRICE FILTER
    for th in [0.60, 0.65, 0.70]:
        for min_e, max_e in [(0.55, 0.70), (0.60, 0.75), (0.65, 0.80), (0.70, 0.85)]:
            strategies[f"EPRICE_{th}_{min_e}-{max_e}_t5→t12.5"] = make_entry_price_filter(
                'pm_yes_t5', 'pm_yes_t12_5', th, min_e, max_e)

    # 9. SPOT MOMENTUM CONTRARIAN
    for min_bps in [5, 10, 15, 20]:
        for th in [0.50, 0.55, 0.60]:
            strategies[f"SPOT_CONTRA_{min_bps}bps_{th}_t5→t12.5"] = make_spot_momentum_filter(
                'pm_yes_t5', 'pm_yes_t12_5', th, min_bps)

    # 10. FADE EXTREME
    for ex_th in [0.70, 0.75, 0.80, 0.85]:
        for et, ec in [('t5', 'pm_yes_t5'), ('t7.5', 'pm_yes_t7_5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                strategies[f"FADE_{ex_th}_{et}→{xt}"] = make_fade_extreme(ec, xc, ex_th)

    # 11. TWO-POINT CONFIRMATION
    for th1 in [0.55, 0.60]:
        for th2 in [0.65, 0.70, 0.75]:
            strategies[f"CONFIRM_t2.5({th1})+t5({th2})→t12.5"] = make_momentum_confirm(
                'pm_yes_t2_5', 'pm_yes_t5', 'pm_yes_t12_5', th1, th2)
            strategies[f"CONFIRM_t5({th1})+t7.5({th2})→t12.5"] = make_momentum_confirm(
                'pm_yes_t5', 'pm_yes_t7_5', 'pm_yes_t12_5', th1, th2)

    print(f"\nTotal strategies to test: {len(strategies)}")
    print("Running sweep...")

    # Run all strategies on train and test
    results = []
    for name, fn in strategies.items():
        train_pnls = run_strategy(train, fn)
        test_pnls = run_strategy(test, fn)

        train_m = calc_metrics(train_pnls)
        test_m = calc_metrics(test_pnls)

        if train_m is None or test_m is None:
            continue

        results.append({
            'name': name,
            'train': train_m,
            'test': test_m,
        })

    # Sort by TEST P&L (the real measure)
    results.sort(key=lambda x: x['test']['total_pnl'], reverse=True)

    print(f"\nCompleted: {len(results)} strategies with sufficient trades\n")

    # Print top 40 by test P&L
    print("=" * 140)
    print(f"{'Strategy':<50} {'Tr Trades':>9} {'Tr Win%':>8} {'Tr P&L':>10} {'Tr Sharpe':>9} | "
          f"{'Te Trades':>9} {'Te Win%':>8} {'Te P&L':>10} {'Te Sharpe':>9} {'Te MaxDD':>9}")
    print("=" * 140)

    for r in results[:40]:
        tr = r['train']
        te = r['test']
        print(f"{r['name']:<50} {tr['n_trades']:>9} {tr['win_rate']*100:>7.1f}% ${tr['total_pnl']:>9.2f} {tr['sharpe']:>9.2f} | "
              f"{te['n_trades']:>9} {te['win_rate']*100:>7.1f}% ${te['total_pnl']:>9.2f} {te['sharpe']:>9.2f} ${te['max_dd']:>8.2f}")

    print("\n\n")

    # Also show strategies that are profitable on BOTH train AND test
    print("=" * 140)
    print("STRATEGIES PROFITABLE ON BOTH TRAIN AND TEST (sorted by test P&L)")
    print("=" * 140)

    both_profitable = [r for r in results if r['train']['total_pnl'] > 0 and r['test']['total_pnl'] > 0]
    both_profitable.sort(key=lambda x: x['test']['total_pnl'], reverse=True)

    print(f"\n{'Strategy':<50} {'Tr Trades':>9} {'Tr Win%':>8} {'Tr P&L':>10} {'Tr Sharpe':>9} | "
          f"{'Te Trades':>9} {'Te Win%':>8} {'Te P&L':>10} {'Te Sharpe':>9} {'Te MaxDD':>9}")
    print("-" * 140)

    for r in both_profitable[:30]:
        tr = r['train']
        te = r['test']
        print(f"{r['name']:<50} {tr['n_trades']:>9} {tr['win_rate']*100:>7.1f}% ${tr['total_pnl']:>9.2f} {tr['sharpe']:>9.2f} | "
              f"{te['n_trades']:>9} {te['win_rate']*100:>7.1f}% ${te['total_pnl']:>9.2f} {te['sharpe']:>9.2f} ${te['max_dd']:>8.2f}")

    print(f"\n\nTotal profitable on both: {len(both_profitable)} / {len(results)}")

    # Save full results to JSON
    output_path = DATA_DIR / "reports" / "train_test_sweep_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_results = []
    for r in results:
        json_results.append({
            'name': r['name'],
            'train': r['train'],
            'test': r['test'],
        })

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nFull results saved to {output_path}")
