#!/usr/bin/env python3
"""
Drawdown Protection Backtest — 70/30 train/test split

Root-cause analysis of Feb 27 catastrophic drawdown (-68.7%) revealed:
1. Recovery sizing (linear) turns profitable base bets into losers
2. No pause after losses compounds drawdowns
3. Overnight hours (01-08 UTC) are consistently unprofitable
4. Bear side at prev_pm >= 0.90 barely breaks even after fees
5. prev_pm 0.10-0.20 for bulls is a trap (32% WR)

This script tests drawdown protection mechanisms against the baseline
contrarian_consensus strategy using historical window data.
"""

import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import json
import sys

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']

# Fee model matching the live config
FEE_RATE = 0.01        # 1% fee on notional
SPREAD_COST = 0.005    # 0.5% spread

INITIAL_BANKROLL = 1000.0
BASE_BET = 25.0
TRAIN_RATIO = 0.70


def pnl_early_exit(direction, entry_pm, exit_pm, bet):
    """Calculate P&L for early exit trade (sell at exit_pm)."""
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
    """Load all windows from all asset databases, enriched with cross-asset data."""
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            print(f"WARNING: {db_path} not found, skipping")
            continue
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

    all_rows.sort(key=lambda x: x['window_start_utc'])

    # Enrich per-asset sequential data
    asset_windows = defaultdict(list)
    for r in all_rows:
        asset_windows[r['asset']].append(r)

    enriched = []
    for asset, wins in asset_windows.items():
        for i, w in enumerate(wins):
            row = dict(w)
            if row.get('prev_pm_t12_5') is None and i > 0:
                row['prev_pm_t12_5'] = wins[i - 1].get('pm_yes_t12_5')
            if row.get('prev2_pm_t12_5') is None and i > 1:
                row['prev2_pm_t12_5'] = wins[i - 2].get('pm_yes_t12_5')
            # Hour of day
            try:
                row['hour_utc'] = int(row['window_start_utc'][11:13])
            except Exception:
                row['hour_utc'] = None
            enriched.append(row)

    enriched.sort(key=lambda x: x['window_start_utc'])

    # Build cross-asset consensus lookup (group by window_time)
    time_groups = defaultdict(dict)
    for r in enriched:
        wt = r.get('window_time')
        if wt is None:
            parts = r['window_id'].split('_')
            if len(parts) >= 3:
                wt = '_'.join(parts[1:])
        if wt:
            time_groups[wt][r['asset']] = r

    for r in enriched:
        wt = r.get('window_time')
        if wt is None:
            parts = r['window_id'].split('_')
            if len(parts) >= 3:
                wt = '_'.join(parts[1:])
        if wt and wt in time_groups:
            group = time_groups[wt]
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
        else:
            r['xasset_strong_up'] = 0
            r['xasset_strong_down'] = 0

    return enriched


# ─────────────── Strategies ───────────────

class BaseStrategy:
    """Consensus contrarian strategy with configurable risk management."""

    def __init__(self, name, **params):
        self.name = name
        # Consensus thresholds
        self.prev_thresh = params.get('prev_thresh', 0.80)
        self.bull_thresh = params.get('bull_thresh', 0.50)
        self.bear_thresh = params.get('bear_thresh', 0.50)
        self.min_agree = params.get('min_agree', 3)
        self.entry_time = params.get('entry_time', 't0')
        self.exit_time = params.get('exit_time', 't12.5')

        # Risk management
        self.pause_after_loss = params.get('pause_after_loss', 0)
        self.skip_hours = params.get('skip_hours', set())
        self.max_drawdown_pct = params.get('max_drawdown_pct', 1.0)  # 1.0 = no limit
        self.recovery_sizing = params.get('recovery_sizing', 'none')
        self.recovery_step = params.get('recovery_step', 1.0)
        self.recovery_max_mult = params.get('recovery_max_mult', 3)
        self.tighter_prev = params.get('tighter_prev', False)  # use 0.90/0.10 instead
        self.skip_weak_bear = params.get('skip_weak_bear', False)  # skip bear when prev >= 0.90
        self.skip_xrp = params.get('skip_xrp', False)
        self.daily_loss_limit = params.get('daily_loss_limit', None)  # max $ loss per day
        self.max_trades_per_window = params.get('max_trades_per_window', 4)
        self.bull_only = params.get('bull_only', False)
        self.bear_only = params.get('bear_only', False)

    def _get_entry_pm(self, row):
        time_map = {
            't0': 'pm_yes_t0', 't2.5': 'pm_yes_t2_5',
            't5': 'pm_yes_t5', 't7.5': 'pm_yes_t7_5', 't10': 'pm_yes_t10',
        }
        return row.get(time_map.get(self.entry_time, 'pm_yes_t0'))

    def _get_exit_pm(self, row):
        time_map = {
            't10': 'pm_yes_t10', 't12.5': 'pm_yes_t12_5',
        }
        return row.get(time_map.get(self.exit_time, 'pm_yes_t12_5'))


def simulate(strategy, data):
    """
    Simulate a strategy with full bankroll tracking and risk management.
    Returns detailed results dict.
    """
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET

    pause_remaining = 0
    asset_consecutive_losses = defaultdict(int)
    halted = False

    trades = []
    daily_pnl = defaultdict(float)
    equity_curve = [INITIAL_BANKROLL]
    max_dd = 0
    max_dd_pct = 0

    # Group rows by window_time for consensus evaluation
    time_groups = defaultdict(dict)
    for r in data:
        wt = r.get('window_time')
        if wt is None:
            parts = r['window_id'].split('_')
            if len(parts) >= 3:
                wt = '_'.join(parts[1:])
        if wt:
            time_groups[wt][r['asset']] = r

    processed_windows = set()

    for wt in sorted(time_groups.keys()):
        if halted:
            break

        group = time_groups[wt]

        # Check hour filter
        sample_row = next(iter(group.values()))
        hour = sample_row.get('hour_utc')
        if hour is not None and hour in strategy.skip_hours:
            continue

        # Check pause
        if pause_remaining > 0:
            pause_remaining -= 1
            continue

        # Check drawdown halt
        if bankroll <= 0:
            halted = True
            break
        dd_pct = (bankroll - peak_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        if dd_pct < -strategy.max_drawdown_pct:
            halted = True
            break

        # Daily loss limit check
        day_key = sample_row['window_start_utc'][:10]
        if strategy.daily_loss_limit is not None and daily_pnl[day_key] < -strategy.daily_loss_limit:
            continue

        # Phase 1: Previous window consensus
        n_strong_up = 0
        n_strong_down = 0
        prev_thresh = 0.90 if strategy.tighter_prev else strategy.prev_thresh
        for asset, row in group.items():
            p = row.get('prev_pm_t12_5')
            if p is not None:
                if p >= prev_thresh:
                    n_strong_up += 1
                elif p <= (1.0 - prev_thresh):
                    n_strong_down += 1

        direction = None
        if n_strong_up >= strategy.min_agree:
            direction = 'bear'
        elif n_strong_down >= strategy.min_agree:
            direction = 'bull'

        if direction is None:
            continue

        if strategy.bull_only and direction != 'bull':
            continue
        if strategy.bear_only and direction != 'bear':
            continue

        # Phase 2: Current window confirmation
        confirming = []
        for asset, row in group.items():
            if strategy.skip_xrp and asset.lower() == 'xrp':
                continue

            entry_pm = strategy._get_entry_pm(row)
            exit_pm = strategy._get_exit_pm(row)
            if entry_pm is None or exit_pm is None:
                continue

            if direction == 'bear':
                if entry_pm <= strategy.bear_thresh:
                    # Skip weak bear signals (prev_pm 0.80-0.90 range)
                    prev = row.get('prev_pm_t12_5')
                    if strategy.skip_weak_bear and prev is not None and prev < 0.90:
                        continue
                    confirming.append((asset, row, entry_pm, exit_pm))
            else:  # bull
                if entry_pm >= strategy.bull_thresh:
                    confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < strategy.min_agree:
            continue

        # Execute trades
        window_trades = 0
        window_had_loss = False

        for asset, row, entry_pm, exit_pm in confirming:
            if window_trades >= strategy.max_trades_per_window:
                break

            # Determine bet size
            if strategy.recovery_sizing == 'linear':
                losses = asset_consecutive_losses.get(asset.lower(), 0)
                bet = base_bet + (base_bet * strategy.recovery_step * losses)
                bet = min(bet, base_bet * strategy.recovery_max_mult)
            elif strategy.recovery_sizing == 'mart_1.5x':
                losses = asset_consecutive_losses.get(asset.lower(), 0)
                bet = base_bet * (1.5 ** losses)
                bet = min(bet, base_bet * strategy.recovery_max_mult)
            else:
                bet = base_bet

            # Cap at max_bet_pct of bankroll (5%)
            bet = min(bet, bankroll * 0.05)
            if bet < 1.0:
                continue

            # Calculate P&L
            net = pnl_early_exit(direction, entry_pm, exit_pm, bet)
            won = net > 0

            bankroll += net
            peak_bankroll = max(peak_bankroll, bankroll)

            dd_dollars = bankroll - peak_bankroll
            dd_pct_current = dd_dollars / peak_bankroll if peak_bankroll > 0 else 0
            max_dd = min(max_dd, dd_dollars)
            max_dd_pct = min(max_dd_pct, dd_pct_current)

            day_key = row['window_start_utc'][:10]
            daily_pnl[day_key] += net

            trades.append({
                'window_id': row['window_id'],
                'asset': asset,
                'direction': direction,
                'entry_pm': entry_pm,
                'exit_pm': exit_pm,
                'bet': bet,
                'net_pnl': net,
                'won': won,
                'bankroll': bankroll,
                'hour': hour,
                'prev_pm': row.get('prev_pm_t12_5'),
                'day': day_key,
            })

            if won:
                asset_consecutive_losses[asset.lower()] = 0
            else:
                asset_consecutive_losses[asset.lower()] = asset_consecutive_losses.get(asset.lower(), 0) + 1
                window_had_loss = True

            window_trades += 1
            equity_curve.append(bankroll)

        if window_had_loss and strategy.pause_after_loss > 0:
            pause_remaining = strategy.pause_after_loss

    # Compute metrics
    if not trades:
        return None

    pnls = np.array([t['net_pnl'] for t in trades])
    n = len(pnls)
    wins = int(np.sum(pnls > 0))
    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if n > 1 else 1.0
    sharpe = avg_pnl / std_pnl * np.sqrt(n) if std_pnl > 0 else 0
    win_rate = wins / n if n > 0 else 0
    avg_win = float(np.mean(pnls[pnls > 0])) if np.sum(pnls > 0) > 0 else 0
    avg_loss = float(np.mean(pnls[pnls <= 0])) if np.sum(pnls <= 0) > 0 else 0
    pf = abs(float(np.sum(pnls[pnls > 0])) / float(np.sum(pnls[pnls <= 0]))) if np.sum(pnls <= 0) != 0 else 999

    # Max consecutive losses
    max_loss_streak = 0
    current_streak = 0
    for t in trades:
        if not t['won']:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0

    # Per-asset breakdown
    asset_stats = {}
    for asset in ASSETS:
        a_trades = [t for t in trades if t['asset'].lower() == asset]
        if a_trades:
            a_pnls = [t['net_pnl'] for t in a_trades]
            a_wins = sum(1 for p in a_pnls if p > 0)
            asset_stats[asset] = {
                'n': len(a_trades),
                'wins': a_wins,
                'wr': a_wins / len(a_trades),
                'pnl': sum(a_pnls),
            }

    # Direction breakdown
    dir_stats = {}
    for d in ['bull', 'bear']:
        d_trades = [t for t in trades if t['direction'] == d]
        if d_trades:
            d_pnls = [t['net_pnl'] for t in d_trades]
            d_wins = sum(1 for p in d_pnls if p > 0)
            dir_stats[d] = {
                'n': len(d_trades),
                'wins': d_wins,
                'wr': d_wins / len(d_trades),
                'pnl': sum(d_pnls),
            }

    return {
        'name': strategy.name,
        'trades': n,
        'wins': wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'profit_factor': pf,
        'max_dd_dollars': max_dd,
        'max_dd_pct': max_dd_pct * 100,
        'final_bankroll': bankroll,
        'roi_pct': (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100,
        'max_loss_streak': max_loss_streak,
        'halted': halted,
        'asset_stats': asset_stats,
        'dir_stats': dir_stats,
        'daily_pnl': dict(daily_pnl),
        'equity_curve': equity_curve,
    }


def print_results(results, label=""):
    """Print formatted results table."""
    if not results:
        print(f"\n{'='*100}")
        print(f"  {label} — NO RESULTS")
        return

    print(f"\n{'='*120}")
    print(f"  {label}")
    print(f"{'='*120}")

    header = f"{'Strategy':<42} {'Trades':>6} {'WR%':>6} {'PnL':>10} {'Avg':>8} {'Sharpe':>7} {'PF':>6} {'MaxDD$':>10} {'MaxDD%':>8} {'Final$':>10} {'ROI%':>8} {'MaxLoss':>7}"
    print(header)
    print('-' * 120)

    for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        line = (
            f"{r['name']:<42} "
            f"{r['trades']:>6} "
            f"{r['win_rate']*100:>5.1f}% "
            f"${r['total_pnl']:>9.2f} "
            f"${r['avg_pnl']:>7.2f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['profit_factor']:>5.2f}x "
            f"${r['max_dd_dollars']:>9.2f} "
            f"{r['max_dd_pct']:>7.1f}% "
            f"${r['final_bankroll']:>9.2f} "
            f"{r['roi_pct']:>7.1f}% "
            f"{r['max_loss_streak']:>7}"
        )
        print(line)


def print_detail(result):
    """Print detailed breakdown for a single strategy."""
    if not result:
        return
    print(f"\n  --- {result['name']} Detail ---")

    if result.get('asset_stats'):
        print(f"  Per-asset:")
        for asset, s in sorted(result['asset_stats'].items()):
            print(f"    {asset.upper():>4}: {s['n']:>4} trades, {s['wr']*100:>5.1f}% WR, ${s['pnl']:>8.2f}")

    if result.get('dir_stats'):
        print(f"  Per-direction:")
        for d, s in result['dir_stats'].items():
            print(f"    {d:>5}: {s['n']:>4} trades, {s['wr']*100:>5.1f}% WR, ${s['pnl']:>8.2f}")

    if result.get('daily_pnl'):
        print(f"  Daily P&L:")
        worst_day = min(result['daily_pnl'].items(), key=lambda x: x[1])
        best_day = max(result['daily_pnl'].items(), key=lambda x: x[1])
        losing_days = sum(1 for v in result['daily_pnl'].values() if v < 0)
        total_days = len(result['daily_pnl'])
        print(f"    Best day:  {best_day[0]} ${best_day[1]:>8.2f}")
        print(f"    Worst day: {worst_day[0]} ${worst_day[1]:>8.2f}")
        print(f"    Losing days: {losing_days}/{total_days}")

    if result.get('halted'):
        print(f"  ** HALTED by drawdown limit **")


def main():
    print("Loading data from all asset databases...")
    data = load_all_data()
    print(f"Loaded {len(data)} windows across {len(ASSETS)} assets")

    # Get unique window times for temporal split
    window_times = sorted(set(r['window_start_utc'] for r in data))
    split_idx = int(len(window_times) * TRAIN_RATIO)
    split_time = window_times[split_idx]

    train_data = [r for r in data if r['window_start_utc'] < split_time]
    test_data = [r for r in data if r['window_start_utc'] >= split_time]

    print(f"Train: {len(train_data)} rows, up to {split_time}")
    print(f"Test:  {len(test_data)} rows, from {split_time}")
    print(f"Split: {TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f}")

    # ─────────────── Define strategies ───────────────

    strategies = []

    # 1. BASELINE: Current live config (contrarian_consensus, no protection)
    strategies.append(BaseStrategy("1_BASELINE (current live config)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='linear', recovery_step=1.0, recovery_max_mult=3,
        pause_after_loss=0,
    ))

    # 2. FLAT BET: Same but no recovery sizing
    strategies.append(BaseStrategy("2_FLAT_BET (no recovery sizing)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=0,
    ))

    # 3. PAUSE_2: Flat bet + pause 2 windows after loss
    strategies.append(BaseStrategy("3_PAUSE_2 (flat + 2 window pause)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=2,
    ))

    # 4. PAUSE_1: Flat bet + pause 1 window after loss
    strategies.append(BaseStrategy("4_PAUSE_1 (flat + 1 window pause)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=1,
    ))

    # 5. HOUR_FILTER: Skip overnight hours (00-08 UTC)
    strategies.append(BaseStrategy("5_HOUR_FILTER (skip 00-08 UTC)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=0,
        skip_hours=set(range(0, 9)),
    ))

    # 6. HOUR+PAUSE: Skip overnight + pause after loss
    strategies.append(BaseStrategy("6_HOUR+PAUSE (skip 00-08 + pause 2)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=2,
        skip_hours=set(range(0, 9)),
    ))

    # 7. TIGHTER_PREV: Require prev_pm >= 0.90 (tighter filter)
    strategies.append(BaseStrategy("7_TIGHTER_PREV (0.90 threshold)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        tighter_prev=True,
    ))

    # 8. DD_HALT_30: Stop trading at 30% drawdown
    strategies.append(BaseStrategy("8_DD_HALT_30 (30% drawdown halt)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        max_drawdown_pct=0.30,
    ))

    # 9. DD_HALT_20: Stop trading at 20% drawdown
    strategies.append(BaseStrategy("9_DD_HALT_20 (20% drawdown halt)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        max_drawdown_pct=0.20,
    ))

    # 10. DAILY_LIMIT: Max $100 loss per day
    strategies.append(BaseStrategy("10_DAILY_LIMIT (max $100/day loss)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        daily_loss_limit=100,
    ))

    # 11. DAILY_LIMIT_50: Max $50 loss per day
    strategies.append(BaseStrategy("11_DAILY_LIMIT_50 (max $50/day loss)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        daily_loss_limit=50,
    ))

    # 12. SKIP_XRP: Drop XRP (only loser)
    strategies.append(BaseStrategy("12_SKIP_XRP (drop losing asset)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        skip_xrp=True,
    ))

    # 13. BULL_ONLY: Only take bull trades (bear barely breaks even)
    strategies.append(BaseStrategy("13_BULL_ONLY (skip bear trades)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        bull_only=True,
    ))

    # 14. T5_ENTRY: Enter at t5 instead of t0 (more confirmation)
    strategies.append(BaseStrategy("14_T5_ENTRY (delayed entry at t5)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t5', exit_time='t12.5',
        recovery_sizing='none',
    ))

    # 15. FULL_PROTECT: Combination of best protections
    strategies.append(BaseStrategy("15_FULL_PROTECT (hours+pause+daily)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=1,
        skip_hours=set(range(0, 9)),
        daily_loss_limit=100,
    ))

    # 16. CONSERVATIVE: Tighter everything
    strategies.append(BaseStrategy("16_CONSERVATIVE (tight+hours+pause)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=2,
        skip_hours=set(range(0, 9)),
        daily_loss_limit=75,
        tighter_prev=True,
    ))

    # 17. AGREE_4: Require all 4 assets to agree
    strategies.append(BaseStrategy("17_AGREE_4 (require 4/4 consensus)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=4, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
    ))

    # 18. AGREE_2_WIDE: Lower consensus, wider net
    strategies.append(BaseStrategy("18_AGREE_2 (require 2/4 consensus)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=2, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
    ))

    # 19. BEST_COMBO: What the data suggests is optimal
    strategies.append(BaseStrategy("19_BEST_COMBO (data-driven optimal)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=1,
        skip_hours=set(range(0, 7)),  # slightly narrower hour filter
        daily_loss_limit=100,
        skip_xrp=True,
    ))

    # 20. HOUR_FILTER_TIGHT: Skip hours 00-06 only
    strategies.append(BaseStrategy("20_HOUR_0_6 (skip 00-06 UTC only)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        skip_hours=set(range(0, 7)),
    ))

    # 21. SKIP_WEAK_BEAR: Only take bear when prev >= 0.90
    strategies.append(BaseStrategy("21_SKIP_WEAK_BEAR (bear only 0.90+)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        skip_weak_bear=True,
    ))

    # 22. ULTRA_SAFE: Maximum protection
    strategies.append(BaseStrategy("22_ULTRA_SAFE (all protections)",
        prev_thresh=0.80, bull_thresh=0.50, bear_thresh=0.50,
        min_agree=3, entry_time='t0', exit_time='t12.5',
        recovery_sizing='none',
        pause_after_loss=2,
        skip_hours=set(range(0, 9)),
        daily_loss_limit=50,
        tighter_prev=True,
        skip_xrp=True,
        max_drawdown_pct=0.25,
    ))

    # ─────────────── Run simulations ───────────────

    print(f"\nRunning {len(strategies)} strategies on train set...")
    train_results = []
    for s in strategies:
        r = simulate(s, train_data)
        if r:
            train_results.append(r)

    print(f"Running {len(strategies)} strategies on test set...")
    test_results = []
    for s in strategies:
        r = simulate(s, test_data)
        if r:
            test_results.append(r)

    # ─────────────── Print results ───────────────

    print_results(train_results, "TRAIN SET RESULTS (70%)")
    print_results(test_results, "TEST SET RESULTS (30%)")

    # Print comparison table
    print(f"\n{'='*140}")
    print(f"  TRAIN vs TEST COMPARISON (sorted by test P&L)")
    print(f"{'='*140}")
    header = (
        f"{'Strategy':<42} "
        f"{'Train PnL':>10} {'Train WR':>8} {'Train DD%':>9} "
        f"{'Test PnL':>10} {'Test WR':>8} {'Test DD%':>9} "
        f"{'Robust?':>8} {'Train#':>7} {'Test#':>6}"
    )
    print(header)
    print('-' * 140)

    # Match by name
    test_by_name = {r['name']: r for r in test_results}
    train_by_name = {r['name']: r for r in train_results}

    combined = []
    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr is None:
            continue
        robust = "YES" if tr['total_pnl'] > 0 and te['total_pnl'] > 0 else "NO"
        combined.append((name, tr, te, robust))
        print(
            f"{name:<42} "
            f"${tr['total_pnl']:>9.2f} {tr['win_rate']*100:>6.1f}% {tr['max_dd_pct']:>8.1f}% "
            f"${te['total_pnl']:>9.2f} {te['win_rate']*100:>6.1f}% {te['max_dd_pct']:>8.1f}% "
            f"{robust:>8} {tr['trades']:>7} {te['trades']:>6}"
        )

    # Print detailed breakdown of top 5 test performers
    print(f"\n{'='*80}")
    print(f"  DETAILED BREAKDOWN — TOP 5 TEST PERFORMERS")
    print(f"{'='*80}")

    sorted_test = sorted(test_results, key=lambda x: x['total_pnl'], reverse=True)
    for r in sorted_test[:5]:
        print_detail(r)

    # Risk-adjusted ranking
    print(f"\n{'='*120}")
    print(f"  RISK-ADJUSTED RANKING (Sharpe / sqrt(|MaxDD%|), test set)")
    print(f"{'='*120}")
    risk_adj = []
    for r in test_results:
        dd = abs(r['max_dd_pct']) if r['max_dd_pct'] != 0 else 1
        score = r['sharpe'] / (dd ** 0.5) if dd > 0 else 0
        risk_adj.append((r['name'], score, r['sharpe'], r['max_dd_pct'], r['total_pnl'], r['roi_pct']))

    risk_adj.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Strategy':<42} {'Score':>8} {'Sharpe':>8} {'MaxDD%':>8} {'PnL':>10} {'ROI%':>8}")
    print('-' * 90)
    for name, score, sharpe, dd, pnl, roi in risk_adj:
        print(f"{name:<42} {score:>8.3f} {sharpe:>8.2f} {dd:>7.1f}% ${pnl:>9.2f} {roi:>7.1f}%")

    # Save results
    output = {
        'split_time': split_time,
        'train_rows': len(train_data),
        'test_rows': len(test_data),
        'strategies': {}
    }
    for name, tr, te, robust in combined:
        output['strategies'][name] = {
            'train': {k: v for k, v in tr.items() if k not in ('equity_curve',)},
            'test': {k: v for k, v in te.items() if k not in ('equity_curve',)},
            'robust': robust == 'YES',
        }

    out_path = DATA_DIR / "reports" / "drawdown_protection_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
