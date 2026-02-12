#!/usr/bin/env python3
"""
Train/Test Strategy Sweep V2 (70/30 temporal split)

Expanded sweep based on live trading data analysis (Feb 2026):
- accel_dbl bled out (55.6% WR, avg loss 2.4x avg win)
- contrarian_consensus was profitable (86% WR)
- Bear-side contrarian edge stronger (60%) than bull (53%)
- PM t0 < 0.30 overwhelmingly goes DOWN (71-82%)
- Winning bull trades have higher spot_velocity + pm_momentum

New strategy families tested:
1-11. Original families (momentum, contrarian, velocity, etc.)
12. Cross-asset consensus simulation
13. Bear-only / bear-skewed contrarian
14. PM t0 confirmation filter
15. Double contrarian + PM t0 confirmation (combo)
16. Momentum-confirmed contrarian (velocity filter)
17. Asymmetric thresholds (tighter bear, looser bull)
18. Multi-signal combinations
19. Stop-loss simulation (using intermediate PM readings)
20. Regime-aware strategies (volatility filtering)
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


def pnl_resolution(direction, entry_pm, outcome, bet=BET_SIZE):
    """Calculate P&L for holding to binary resolution."""
    if direction == 'bull':
        entry_c = entry_pm
        won = (outcome == 'up')
    else:
        entry_c = 1.0 - entry_pm
        won = (outcome == 'down')
    if entry_c <= 0.001:
        return 0.0
    n = bet / entry_c
    fees = entry_c * n * FEE_RATE
    spread = SPREAD_COST * bet
    if won:
        # Payout is $1 per contract
        gross = n * (1.0 - entry_c)
        return gross - fees - spread
    else:
        return -bet - fees - spread


def load_all_data():
    """Load all windows with samples data from all assets."""
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT w.window_id, w.asset, w.window_start_utc, w.outcome, w.outcome_binary,
                   w.pm_yes_t0, w.pm_yes_t2_5, w.pm_yes_t5, w.pm_yes_t7_5, w.pm_yes_t10, w.pm_yes_t12_5,
                   w.pm_spread_t0, w.pm_spread_t5,
                   w.pm_price_momentum_0_to_5, w.pm_price_momentum_5_to_10,
                   w.spot_open, w.spot_close, w.spot_change_bps, w.spot_range_bps,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5,
                   w.volatility_regime, w.window_time
            FROM windows w
            WHERE w.outcome IS NOT NULL
            ORDER BY w.window_start_utc
        """).fetchall()

        all_rows.extend([dict(r) for r in rows])
        conn.close()

    # Sort by time
    all_rows.sort(key=lambda x: x['window_start_utc'])

    # Enrich per-asset sequential data
    asset_windows = defaultdict(list)
    for r in all_rows:
        asset_windows[r['asset']].append(r)

    enriched = []
    for asset, wins in asset_windows.items():
        for i, w in enumerate(wins):
            row = dict(w)

            # Previous window data (use DB columns if available, else compute)
            if row.get('prev_pm_t12_5') is None and i > 0:
                row['prev_pm_t12_5'] = wins[i - 1].get('pm_yes_t12_5')
            if row.get('prev2_pm_t12_5') is None and i > 1:
                row['prev2_pm_t12_5'] = wins[i - 2].get('pm_yes_t12_5')

            if i > 0:
                prev = wins[i - 1]
                row['prev_pm_t0'] = prev.get('pm_yes_t0')
                row['prev_pm_t5'] = prev.get('pm_yes_t5')
                row['prev_outcome'] = prev.get('outcome')
                row['prev_spot_change_bps'] = prev.get('spot_change_bps')
                row['prev_spot_range_bps'] = prev.get('spot_range_bps')
            else:
                row['prev_pm_t0'] = None
                row['prev_pm_t5'] = None
                row['prev_outcome'] = None
                row['prev_spot_change_bps'] = None
                row['prev_spot_range_bps'] = None

            if i > 1:
                row['prev2_outcome'] = wins[i - 2].get('outcome')
            else:
                row['prev2_outcome'] = None

            # Intra-window velocities
            if row.get('pm_yes_t0') is not None and row.get('pm_yes_t2_5') is not None:
                row['velocity_0_2_5'] = row['pm_yes_t2_5'] - row['pm_yes_t0']
            else:
                row['velocity_0_2_5'] = None

            if row.get('pm_yes_t2_5') is not None and row.get('pm_yes_t5') is not None:
                row['velocity_2_5_5'] = row['pm_yes_t5'] - row['pm_yes_t2_5']
            else:
                row['velocity_2_5_5'] = None

            if row.get('pm_yes_t5') is not None and row.get('pm_yes_t7_5') is not None:
                row['velocity_5_7_5'] = row['pm_yes_t7_5'] - row['pm_yes_t5']
            else:
                row['velocity_5_7_5'] = None

            # PM momentum (t0→t5)
            if row.get('pm_yes_t0') is not None and row.get('pm_yes_t5') is not None:
                row['pm_momentum_0_5'] = abs(row['pm_yes_t5'] - row['pm_yes_t0'])
            else:
                row['pm_momentum_0_5'] = None

            # Hour of day
            try:
                row['hour_utc'] = int(row['window_start_utc'][11:13])
            except:
                row['hour_utc'] = None

            enriched.append(row)

    enriched.sort(key=lambda x: x['window_start_utc'])

    # Build cross-asset consensus lookup (group by window_time)
    time_groups = defaultdict(dict)
    for r in enriched:
        wt = r.get('window_time')
        if wt is None:
            # Derive from window_id: "BTC_20260206_0445" → "20260206_0445"
            parts = r['window_id'].split('_')
            if len(parts) >= 3:
                wt = '_'.join(parts[1:])
        if wt:
            time_groups[wt][r['asset']] = r

    # Annotate each row with cross-asset info
    for r in enriched:
        wt = r.get('window_time')
        if wt is None:
            parts = r['window_id'].split('_')
            if len(parts) >= 3:
                wt = '_'.join(parts[1:])
        if wt and wt in time_groups:
            group = time_groups[wt]
            # Count how many assets have prev strong up/down
            n_strong_up = 0
            n_strong_down = 0
            for a, ar in group.items():
                p = ar.get('prev_pm_t12_5')
                if p is not None:
                    if p >= 0.75:
                        n_strong_up += 1
                    elif p <= 0.25:
                        n_strong_down += 1
            r['xasset_strong_up'] = n_strong_up
            r['xasset_strong_down'] = n_strong_down

            # Count double contrarian across assets
            n_dbl_up = 0
            n_dbl_down = 0
            for a, ar in group.items():
                p1 = ar.get('prev_pm_t12_5')
                p2 = ar.get('prev2_pm_t12_5')
                if p1 is not None and p2 is not None:
                    if p1 >= 0.75 and p2 >= 0.75:
                        n_dbl_up += 1
                    elif p1 <= 0.25 and p2 <= 0.25:
                        n_dbl_down += 1
            r['xasset_dbl_up'] = n_dbl_up
            r['xasset_dbl_down'] = n_dbl_down
        else:
            r['xasset_strong_up'] = 0
            r['xasset_strong_down'] = 0
            r['xasset_dbl_up'] = 0
            r['xasset_dbl_down'] = 0

    return enriched


def calc_metrics(pnls):
    """Calculate strategy metrics from list of P&Ls."""
    if len(pnls) < 5:
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

    # Expectancy
    expectancy = wr * avg_w + (1 - wr) * avg_l

    return {
        'n_trades': n,
        'win_rate': wr,
        'total_pnl': total,
        'avg_pnl': avg,
        'sharpe': sharpe,
        'avg_win': avg_w,
        'avg_loss': avg_l,
        'max_dd': max_dd,
        'profit_factor': abs(np.sum(arr[arr > 0]) / np.sum(arr[arr <= 0])) if np.sum(arr[arr <= 0]) != 0 else float('inf'),
        'expectancy': expectancy,
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


def run_strategy_resolution(data, strategy_fn):
    """Run strategy using binary resolution (hold to end) instead of early exit."""
    pnls = []
    for r in data:
        result = strategy_fn(r)
        if result is not None:
            direction, entry_pm, _ = result
            outcome = r.get('outcome')
            if outcome:
                pnl = pnl_resolution(direction, entry_pm, outcome)
                pnls.append(pnl)
    return pnls


# =============================================================================
# ORIGINAL STRATEGY BUILDERS (from V1)
# =============================================================================

def make_momentum(entry_col, exit_col, thresh):
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


def make_double_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th):
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or prev2 is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and prev2 >= prev_thresh:
            if pm <= bear_th:
                return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh):
            if pm >= bull_th:
                return ('bull', pm, ex)
        return None
    return strategy


def make_velocity_filter(entry_col, exit_col, vel_col, vel_min, thresh):
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


def make_streak_filter(entry_col, exit_col, thresh, require_prev_outcome):
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


def make_entry_price_filter(entry_col, exit_col, thresh, min_entry, max_entry):
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
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        prev_spot = r.get('prev_spot_change_bps')
        if pm is None or ex is None or prev_spot is None:
            return None
        if prev_spot >= min_spot_bps and pm <= (1.0 - thresh):
            return ('bear', pm, ex)
        elif prev_spot <= -min_spot_bps and pm >= thresh:
            return ('bull', pm, ex)
        return None
    return strategy


def make_fade_extreme(entry_col, exit_col, extreme_thresh):
    def strategy(r):
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if pm is None or ex is None:
            return None
        if pm >= extreme_thresh:
            return ('bear', pm, ex)
        elif pm <= (1.0 - extreme_thresh):
            return ('bull', pm, ex)
        return None
    return strategy


def make_momentum_confirm(entry1_col, entry2_col, exit_col, thresh1, thresh2):
    def strategy(r):
        pm1 = r.get(entry1_col)
        pm2 = r.get(entry2_col)
        ex = r.get(exit_col)
        if pm1 is None or pm2 is None or ex is None:
            return None
        if pm1 >= thresh1 and pm2 >= thresh2:
            return ('bull', pm2, ex)
        elif pm1 <= (1.0 - thresh1) and pm2 <= (1.0 - thresh2):
            return ('bear', pm2, ex)
        return None
    return strategy


# =============================================================================
# NEW STRATEGY BUILDERS (V2)
# =============================================================================

def make_bear_only_contrarian(prev_thresh, entry_col, exit_col, bear_th):
    """Bear-only contrarian: only take the stronger side (prev strong UP → bet DOWN)."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and pm <= bear_th:
            return ('bear', pm, ex)
        return None
    return strategy


def make_bear_only_double_contrarian(prev_thresh, entry_col, exit_col, bear_th):
    """Bear-only double contrarian: 2 consecutive strong UP → bet DOWN."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or prev2 is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and prev2 >= prev_thresh and pm <= bear_th:
            return ('bear', pm, ex)
        return None
    return strategy


def make_pm0_confirmed_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, pm0_bull_min, pm0_bear_max):
    """Contrarian + PM t0 confirmation.
    For bears: prev strong UP + current pm0 < pm0_bear_max (market already leaning down)
    For bulls: prev strong DOWN + current pm0 > pm0_bull_min (market already leaning up)
    """
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        pm0 = r.get('pm_yes_t0')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or pm0 is None or pm is None or ex is None:
            return None
        # Bear: prev UP + pm0 confirms bearish
        if prev >= prev_thresh and pm0 <= pm0_bear_max and pm <= bear_th:
            return ('bear', pm, ex)
        # Bull: prev DOWN + pm0 confirms bullish
        elif prev <= (1.0 - prev_thresh) and pm0 >= pm0_bull_min and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


def make_pm0_confirmed_double_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, pm0_bull_min, pm0_bear_max):
    """Double contrarian + PM t0 confirmation."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm0 = r.get('pm_yes_t0')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or prev2 is None or pm0 is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and prev2 >= prev_thresh and pm0 <= pm0_bear_max and pm <= bear_th:
            return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh) and pm0 >= pm0_bull_min and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


def make_consensus_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, min_agree):
    """Contrarian + cross-asset consensus: at least N assets must have strong prev."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        n_up = r.get('xasset_strong_up', 0)
        n_down = r.get('xasset_strong_down', 0)
        if prev is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and n_up >= min_agree and pm <= bear_th:
            return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and n_down >= min_agree and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


def make_consensus_double_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, min_dbl_agree):
    """Double contrarian + cross-asset consensus: N assets must have double-strong prev."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        n_dbl_up = r.get('xasset_dbl_up', 0)
        n_dbl_down = r.get('xasset_dbl_down', 0)
        if prev is None or prev2 is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and prev2 >= prev_thresh and n_dbl_up >= min_dbl_agree and pm <= bear_th:
            return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh) and n_dbl_down >= min_dbl_agree and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


def make_velocity_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, vel_col, min_vel):
    """Contrarian + velocity confirmation: only enter when intra-window movement supports."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        vel = r.get(vel_col)
        if prev is None or pm is None or ex is None or vel is None:
            return None
        # For bear: velocity should be negative (PM dropping = confirming bear)
        if prev >= prev_thresh and pm <= bear_th and vel <= -min_vel:
            return ('bear', pm, ex)
        # For bull: velocity should be positive (PM rising = confirming bull)
        elif prev <= (1.0 - prev_thresh) and pm >= bull_th and vel >= min_vel:
            return ('bull', pm, ex)
        return None
    return strategy


def make_asymmetric_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th):
    """Asymmetric thresholds: e.g., tighter filter for bull, looser for bear (or vice versa)."""
    # This is just make_contrarian with different bull/bear thresholds, but we enumerate more combos
    return make_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th)


def make_stop_loss_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, stop_col, stop_delta):
    """Contrarian with stop-loss check at an intermediate time.
    If position has moved against us by stop_delta at stop_col time, use stop_col as exit.
    """
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        stop_pm = r.get(stop_col)
        if prev is None or pm is None or ex is None or stop_pm is None:
            return None
        direction = None
        if prev >= prev_thresh and pm <= bear_th:
            direction = 'bear'
        elif prev <= (1.0 - prev_thresh) and pm >= bull_th:
            direction = 'bull'
        if direction is None:
            return None

        # Check stop
        if direction == 'bear':
            # Bear entry: bought NO at (1-pm). Stop if pm rises by stop_delta from entry
            if stop_pm >= pm + stop_delta:
                # Stopped out: exit at stop_col price
                return (direction, pm, stop_pm)
            else:
                return (direction, pm, ex)
        else:
            # Bull entry: bought YES at pm. Stop if pm drops by stop_delta from entry
            if stop_pm <= pm - stop_delta:
                return (direction, pm, stop_pm)
            else:
                return (direction, pm, ex)
    return strategy


def make_regime_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, allowed_regimes):
    """Contrarian filtered by volatility regime."""
    def strategy(r):
        regime = r.get('volatility_regime')
        if regime is None or regime not in allowed_regimes:
            return None
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


def make_double_contrarian_with_stop(prev_thresh, entry_col, exit_col, bull_th, bear_th, stop_col, stop_delta):
    """Double contrarian with stop-loss at intermediate time."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        stop_pm = r.get(stop_col)
        if prev is None or prev2 is None or pm is None or ex is None or stop_pm is None:
            return None
        direction = None
        if prev >= prev_thresh and prev2 >= prev_thresh and pm <= bear_th:
            direction = 'bear'
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh) and pm >= bull_th:
            direction = 'bull'
        if direction is None:
            return None

        if direction == 'bear':
            if stop_pm >= pm + stop_delta:
                return (direction, pm, stop_pm)
            else:
                return (direction, pm, ex)
        else:
            if stop_pm <= pm - stop_delta:
                return (direction, pm, stop_pm)
            else:
                return (direction, pm, ex)
    return strategy


def make_neutral_band_double_contrarian(prev_thresh, entry_col, exit_col, bull_th, bear_th, neutral_band):
    """Double contrarian + t0 near neutral (accel_dbl style).
    Only enter if pm_t0 is within neutral_band of 0.50."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm0 = r.get('pm_yes_t0')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        if prev is None or prev2 is None or pm0 is None or pm is None or ex is None:
            return None
        if abs(pm0 - 0.5) > neutral_band:
            return None
        if prev >= prev_thresh and prev2 >= prev_thresh and pm <= bear_th:
            return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh) and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


def make_triple_filter(prev_thresh, entry_col, exit_col, bull_th, bear_th, min_agree, pm0_bull_min, pm0_bear_max):
    """Triple filter: double contrarian + consensus + PM t0 confirmation."""
    def strategy(r):
        prev = r.get('prev_pm_t12_5')
        prev2 = r.get('prev2_pm_t12_5')
        pm0 = r.get('pm_yes_t0')
        pm = r.get(entry_col)
        ex = r.get(exit_col)
        n_dbl_up = r.get('xasset_dbl_up', 0)
        n_dbl_down = r.get('xasset_dbl_down', 0)
        if prev is None or prev2 is None or pm0 is None or pm is None or ex is None:
            return None
        if prev >= prev_thresh and prev2 >= prev_thresh and n_dbl_up >= min_agree and pm0 <= pm0_bear_max and pm <= bear_th:
            return ('bear', pm, ex)
        elif prev <= (1.0 - prev_thresh) and prev2 <= (1.0 - prev_thresh) and n_dbl_down >= min_agree and pm0 >= pm0_bull_min and pm >= bull_th:
            return ('bull', pm, ex)
        return None
    return strategy


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("POLYNANCE STRATEGY SWEEP V2 — 70/30 Train/Test Split")
    print("=" * 80)

    print("\nLoading and enriching data...")
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

    # =========================================================================
    # ORIGINAL FAMILIES (V1)
    # =========================================================================

    # 1. MOMENTUM
    for et, ec in [('t5', 'pm_yes_t5'), ('t7.5', 'pm_yes_t7_5')]:
        for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
            for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                strategies[f"MOM_{et}→{xt}_{th}"] = make_momentum(ec, xc, th)

    # 2. CONTRARIAN (single)
    for pt in [0.70, 0.75, 0.80, 0.85]:
        for et, ec in [('t0', 'pm_yes_t0'), ('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.50, 0.50), (0.55, 0.45), (0.60, 0.40)]:
                    strategies[f"CONTRA_{pt}_{et}→{xt}_b{bt}_r{brt}"] = make_contrarian(pt, ec, xc, bt, brt)

    # 3. VELOCITY FILTER
    for vel_min in [0.05, 0.10, 0.15]:
        for th in [0.55, 0.60, 0.65]:
            strategies[f"VEL_0→2.5_{vel_min}_{th}_t5→t12.5"] = make_velocity_filter(
                'pm_yes_t5', 'pm_yes_t12_5', 'velocity_0_2_5', vel_min, th)

    # 4. SPREAD FILTER
    for max_sp in [0.005, 0.01, 0.02]:
        for th in [0.60, 0.65, 0.70]:
            strategies[f"SPREAD_t5_{max_sp}_{th}_t5→t12.5"] = make_spread_filter(
                'pm_yes_t5', 'pm_yes_t12_5', 'pm_spread_t5', max_sp, th)

    # 5. STREAK / PREV OUTCOME
    for prev_out in ['up', 'down']:
        for th in [0.60, 0.65, 0.70]:
            strategies[f"AFTER_{prev_out}_{th}_t5→t12.5"] = make_streak_filter(
                'pm_yes_t5', 'pm_yes_t12_5', th, prev_out)

    # 6. DOUBLE CONTRARIAN
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t0', 'pm_yes_t0'), ('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.50, 0.50), (0.55, 0.45), (0.60, 0.40)]:
                    strategies[f"DBL_CONTRA_{pt}_{et}→{xt}_b{bt}_r{brt}"] = make_double_contrarian(
                        pt, ec, xc, bt, brt)

    # 7. FADE EXTREME
    for ex_th in [0.75, 0.80, 0.85]:
        for et, ec in [('t5', 'pm_yes_t5'), ('t7.5', 'pm_yes_t7_5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                strategies[f"FADE_{ex_th}_{et}→{xt}"] = make_fade_extreme(ec, xc, ex_th)

    # 8. TWO-POINT CONFIRMATION
    for th1 in [0.55, 0.60]:
        for th2 in [0.65, 0.70, 0.75]:
            strategies[f"CONFIRM_t2.5({th1})+t5({th2})→t12.5"] = make_momentum_confirm(
                'pm_yes_t2_5', 'pm_yes_t5', 'pm_yes_t12_5', th1, th2)

    # =========================================================================
    # NEW FAMILIES (V2) — based on live trading analysis
    # =========================================================================

    # 12. BEAR-ONLY CONTRARIAN (exploit the stronger bear-side edge)
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t0', 'pm_yes_t0'), ('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for brt in [0.50, 0.45, 0.40]:
                    strategies[f"BEAR_ONLY_{pt}_{et}→{xt}_r{brt}"] = make_bear_only_contrarian(
                        pt, ec, xc, brt)

    # 13. BEAR-ONLY DOUBLE CONTRARIAN
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t0', 'pm_yes_t0'), ('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for brt in [0.50, 0.45, 0.40]:
                    strategies[f"BEAR_DBL_{pt}_{et}→{xt}_r{brt}"] = make_bear_only_double_contrarian(
                        pt, ec, xc, brt)

    # 14. PM t0 CONFIRMED CONTRARIAN
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
                    for pm0_bull, pm0_bear in [(0.50, 0.50), (0.55, 0.45), (0.45, 0.55)]:
                        strategies[f"PM0_CONTRA_{pt}_{et}→{xt}_b{bt}_r{brt}_pm0b{pm0_bull}r{pm0_bear}"] = \
                            make_pm0_confirmed_contrarian(pt, ec, xc, bt, brt, pm0_bull, pm0_bear)

    # 15. PM t0 CONFIRMED DOUBLE CONTRARIAN
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
                    for pm0_bull, pm0_bear in [(0.50, 0.50), (0.55, 0.45), (0.45, 0.55)]:
                        strategies[f"PM0_DBL_{pt}_{et}→{xt}_b{bt}_r{brt}_pm0b{pm0_bull}r{pm0_bear}"] = \
                            make_pm0_confirmed_double_contrarian(pt, ec, xc, bt, brt, pm0_bull, pm0_bear)

    # 16. CONSENSUS CONTRARIAN (simulated cross-asset agreement)
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t0', 'pm_yes_t0'), ('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
                    for min_ag in [2, 3, 4]:
                        strategies[f"CONSENSUS_{pt}_{et}→{xt}_b{bt}_r{brt}_xa{min_ag}"] = \
                            make_consensus_contrarian(pt, ec, xc, bt, brt, min_ag)

    # 17. CONSENSUS DOUBLE CONTRARIAN
    for pt in [0.70, 0.75, 0.80]:
        for et, ec in [('t5', 'pm_yes_t5')]:
            for xt, xc in [('t10', 'pm_yes_t10'), ('t12.5', 'pm_yes_t12_5')]:
                for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
                    for min_ag in [2, 3]:
                        strategies[f"CONS_DBL_{pt}_{et}→{xt}_b{bt}_r{brt}_xa{min_ag}"] = \
                            make_consensus_double_contrarian(pt, ec, xc, bt, brt, min_ag)

    # 18. VELOCITY-CONFIRMED CONTRARIAN
    for pt in [0.75, 0.80]:
        for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
            for vel_col, vel_name in [('velocity_0_2_5', 'v0_2.5'), ('velocity_2_5_5', 'v2.5_5')]:
                for min_vel in [0.03, 0.05, 0.08, 0.10]:
                    strategies[f"VEL_CONTRA_{pt}_{vel_name}>{min_vel}_b{bt}_r{brt}_t5→t12.5"] = \
                        make_velocity_contrarian(pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt, vel_col, min_vel)

    # 19. STOP-LOSS CONTRARIAN (check at t7.5 or t10)
    for pt in [0.75, 0.80]:
        for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
            for stop_col, stop_name in [('pm_yes_t7_5', 'st7.5'), ('pm_yes_t10', 'st10')]:
                for stop_d in [0.08, 0.10, 0.12, 0.15]:
                    strategies[f"STOP_CONTRA_{pt}_{stop_name}Δ{stop_d}_b{bt}_r{brt}_t5→t12.5"] = \
                        make_stop_loss_contrarian(pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt, stop_col, stop_d)

    # 20. DOUBLE CONTRARIAN WITH STOP
    for pt in [0.75, 0.80]:
        for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
            for stop_col, stop_name in [('pm_yes_t7_5', 'st7.5'), ('pm_yes_t10', 'st10')]:
                for stop_d in [0.08, 0.10, 0.15]:
                    strategies[f"DBL_STOP_{pt}_{stop_name}Δ{stop_d}_b{bt}_r{brt}_t5→t12.5"] = \
                        make_double_contrarian_with_stop(pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt, stop_col, stop_d)

    # 21. NEUTRAL-BAND DOUBLE CONTRARIAN (like accel_dbl)
    for pt in [0.70, 0.75, 0.80]:
        for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
            for band in [0.10, 0.15, 0.20, 0.25]:
                strategies[f"NEUTRAL_DBL_{pt}_band{band}_b{bt}_r{brt}_t5→t12.5"] = \
                    make_neutral_band_double_contrarian(pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt, band)

    # 22. REGIME-FILTERED CONTRARIAN
    for pt in [0.75, 0.80]:
        for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
            for regime_set, rname in [
                ({'low', 'normal'}, 'low+norm'),
                ({'normal', 'high'}, 'norm+high'),
                ({'normal'}, 'norm'),
                ({'low', 'normal', 'high'}, 'not_extreme'),
            ]:
                strategies[f"REGIME_CONTRA_{pt}_{rname}_b{bt}_r{brt}_t5→t12.5"] = \
                    make_regime_contrarian(pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt, regime_set)

    # 23. TRIPLE FILTER (double contrarian + consensus + PM0)
    for pt in [0.70, 0.75]:
        for bt, brt in [(0.55, 0.45), (0.50, 0.50)]:
            for min_ag in [2, 3]:
                for pm0_bull, pm0_bear in [(0.50, 0.50), (0.45, 0.55)]:
                    strategies[f"TRIPLE_{pt}_xa{min_ag}_pm0b{pm0_bull}r{pm0_bear}_b{bt}_r{brt}_t5→t12.5"] = \
                        make_triple_filter(pt, 'pm_yes_t5', 'pm_yes_t12_5', bt, brt, min_ag, pm0_bull, pm0_bear)

    # 24. ASYMMETRIC CONTRARIAN (different thresholds for bull vs bear)
    for pt in [0.75, 0.80]:
        for et, ec in [('t5', 'pm_yes_t5')]:
            for xt, xc in [('t12.5', 'pm_yes_t12_5')]:
                # Tighter bear (more selective), looser bull
                for bt, brt in [(0.50, 0.35), (0.50, 0.40), (0.55, 0.35), (0.55, 0.40),
                                (0.60, 0.40), (0.60, 0.35)]:
                    strategies[f"ASYM_{pt}_{et}→{xt}_b{bt}_r{brt}"] = \
                        make_contrarian(pt, ec, xc, bt, brt)

    print(f"\nTotal strategies to test: {len(strategies)}")
    print("Running sweep...\n")

    # Run all strategies on train and test (both early exit AND resolution)
    results = []
    for i, (name, fn) in enumerate(strategies.items()):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(strategies)}...")

        train_pnls = run_strategy(train, fn)
        test_pnls = run_strategy(test, fn)

        train_m = calc_metrics(train_pnls)
        test_m = calc_metrics(test_pnls)

        if train_m is None or test_m is None:
            continue

        # Also run resolution-based for comparison
        train_pnls_res = run_strategy_resolution(train, fn)
        test_pnls_res = run_strategy_resolution(test, fn)
        train_m_res = calc_metrics(train_pnls_res)
        test_m_res = calc_metrics(test_pnls_res)

        results.append({
            'name': name,
            'train': train_m,
            'test': test_m,
            'train_res': train_m_res,
            'test_res': test_m_res,
        })

    # Sort by TEST P&L (the real measure)
    results.sort(key=lambda x: x['test']['total_pnl'], reverse=True)

    print(f"\nCompleted: {len(results)} strategies with sufficient trades\n")

    # Print top 50 by test P&L
    print("=" * 160)
    print(f"{'Strategy':<60} {'Tr N':>5} {'Tr WR%':>7} {'Tr P&L':>9} {'Tr Shp':>7} | "
          f"{'Te N':>5} {'Te WR%':>7} {'Te P&L':>9} {'Te Shp':>7} {'Te DD':>8} {'Te PF':>7} {'Te Exp':>7}")
    print("=" * 160)

    for r in results[:50]:
        tr = r['train']
        te = r['test']
        print(f"{r['name']:<60} {tr['n_trades']:>5} {tr['win_rate']*100:>6.1f}% ${tr['total_pnl']:>8.2f} {tr['sharpe']:>7.2f} | "
              f"{te['n_trades']:>5} {te['win_rate']*100:>6.1f}% ${te['total_pnl']:>8.2f} {te['sharpe']:>7.2f} ${te['max_dd']:>7.2f} {te['profit_factor']:>7.2f} ${te['expectancy']:>6.2f}")

    # Strategies profitable on BOTH train AND test
    print("\n\n")
    print("=" * 160)
    print("STRATEGIES PROFITABLE ON BOTH TRAIN AND TEST (sorted by test P&L)")
    print("=" * 160)

    both_profitable = [r for r in results if r['train']['total_pnl'] > 0 and r['test']['total_pnl'] > 0]
    both_profitable.sort(key=lambda x: x['test']['total_pnl'], reverse=True)

    print(f"\n{'Strategy':<60} {'Tr N':>5} {'Tr WR%':>7} {'Tr P&L':>9} {'Tr Shp':>7} | "
          f"{'Te N':>5} {'Te WR%':>7} {'Te P&L':>9} {'Te Shp':>7} {'Te DD':>8} {'Te PF':>7} {'Te Exp':>7}")
    print("-" * 160)

    for r in both_profitable[:40]:
        tr = r['train']
        te = r['test']
        print(f"{r['name']:<60} {tr['n_trades']:>5} {tr['win_rate']*100:>6.1f}% ${tr['total_pnl']:>8.2f} {tr['sharpe']:>7.2f} | "
              f"{te['n_trades']:>5} {te['win_rate']*100:>6.1f}% ${te['total_pnl']:>8.2f} {te['sharpe']:>7.2f} ${te['max_dd']:>7.2f} {te['profit_factor']:>7.2f} ${te['expectancy']:>6.2f}")

    print(f"\n\nTotal profitable on both: {len(both_profitable)} / {len(results)}")

    # Also show RESOLUTION-based results for top performers
    print("\n\n")
    print("=" * 160)
    print("TOP 30 STRATEGIES — RESOLUTION-BASED P&L (hold to binary outcome)")
    print("=" * 160)

    results_by_res = [r for r in results if r.get('test_res') is not None]
    results_by_res.sort(key=lambda x: x['test_res']['total_pnl'] if x['test_res'] else -9999, reverse=True)

    print(f"\n{'Strategy':<60} {'Tr N':>5} {'Tr WR%':>7} {'Tr P&L':>9} | "
          f"{'Te N':>5} {'Te WR%':>7} {'Te P&L':>9} {'Te Shp':>7} {'Te DD':>8}")
    print("-" * 160)

    for r in results_by_res[:30]:
        tr = r.get('train_res') or {}
        te = r.get('test_res') or {}
        if not tr or not te:
            continue
        print(f"{r['name']:<60} {tr.get('n_trades',0):>5} {tr.get('win_rate',0)*100:>6.1f}% ${tr.get('total_pnl',0):>8.2f} | "
              f"{te.get('n_trades',0):>5} {te.get('win_rate',0)*100:>6.1f}% ${te.get('total_pnl',0):>8.2f} {te.get('sharpe',0):>7.2f} ${te.get('max_dd',0):>7.2f}")

    # Strategies profitable on both train+test for BOTH early exit AND resolution
    print("\n\n")
    print("=" * 160)
    print("ROBUST STRATEGIES: Profitable on both train+test in BOTH early-exit AND resolution modes")
    print("=" * 160)

    robust = []
    for r in results:
        tr_ee = r.get('train', {})
        te_ee = r.get('test', {})
        tr_res = r.get('train_res')
        te_res = r.get('test_res')
        if (tr_ee.get('total_pnl', 0) > 0 and te_ee.get('total_pnl', 0) > 0 and
            tr_res and te_res and
            tr_res.get('total_pnl', 0) > 0 and te_res.get('total_pnl', 0) > 0):
            # Combined score: average of early-exit and resolution test P&L
            combined_score = (te_ee['total_pnl'] + te_res['total_pnl']) / 2
            robust.append((r, combined_score))

    robust.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Strategy':<55} {'EE Te N':>7} {'EE Te WR':>8} {'EE Te P&L':>10} | "
          f"{'Res Te N':>8} {'Res Te WR':>9} {'Res Te P&L':>11} {'Combined':>9}")
    print("-" * 160)

    for r, score in robust[:30]:
        te_ee = r['test']
        te_res = r['test_res']
        print(f"{r['name']:<55} {te_ee['n_trades']:>7} {te_ee['win_rate']*100:>7.1f}% ${te_ee['total_pnl']:>9.2f} | "
              f"{te_res['n_trades']:>8} {te_res['win_rate']*100:>8.1f}% ${te_res['total_pnl']:>10.2f} ${score:>8.2f}")

    print(f"\nTotal robust strategies: {len(robust)} / {len(results)}")

    # Save full results to JSON
    output_path = DATA_DIR / "reports" / "train_test_sweep_v2_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_results = []
    for r in results:
        entry = {
            'name': r['name'],
            'train': r['train'],
            'test': r['test'],
        }
        if r.get('train_res'):
            entry['train_resolution'] = r['train_res']
        if r.get('test_res'):
            entry['test_resolution'] = r['test_res']
        json_results.append(entry)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nFull results saved to {output_path}")
    print(f"\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"{'='*80}")
