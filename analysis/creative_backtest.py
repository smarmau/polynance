#!/usr/bin/env python3
"""
Creative / Non-Traditional Strategy Backtest — 70/30 split

Explores novel ideas that require no look-ahead, using only data genuinely
available at window-start (t0). All signals are documented with their
information availability justification.

SIGNAL AVAILABILITY GUIDE:
  ✓ t0-available:
    - prev_pm_t12_5    previous window's closing PM price
    - prev2_pm_t12_5   two windows ago closing PM price
    - pm_yes_t0        current window opening price (IS the entry price)
    - volatility_regime  computed from previous window's spot range
    - window_start_utc / hour_utc

  ✗ NOT at t0 (but valid for MID-WINDOW exit decisions):
    - pm_yes_t5        valid ONLY as exit check at t5 after entry at t0

NOVEL IDEA CATEGORIES:
  F1  Double exhaustion: prev AND prev2 both above threshold
  F2  Signal freshness: prev just crossed threshold (prev2 was neutral)
  F3  Consensus count: require 3/4 or 4/4 assets to agree
  F4  Signal strength: average prev_pm across triggered assets
  F5  Time-of-day session filter
  F6  Adaptive mid-window exit (profit-take OR stop-hold at t5)
  F7  Opening price level (pm_yes_t0 relative to 0.50)
  F8  Tiered exit: strong consensus → resolution, weak → early
  F9  Signal deceleration (prev < prev2 = already reverting)
  FC  Best combinations
"""

import sqlite3
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']
INITIAL_BANKROLL = 1000.0
BASE_BET = 25.0
TRAIN_RATIO = 0.70
FEE_RATE = 0.01
SPREAD_COST = 0.005


# ─── P&L FUNCTIONS ────────────────────────────────────────────────────────────

def pnl_early_exit(direction, entry_pm, exit_pm, bet):
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


def pnl_resolution(direction, entry_pm, outcome, bet):
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
        gross = n * (1.0 - entry_c)
        return gross - fees - spread
    else:
        return -bet - fees - spread


def is_profitable_at_t5(direction, entry_pm, t5_pm):
    """Check if position is in profit at t5. Valid: entered at t0, checking at t5."""
    if direction == 'bear':
        return t5_pm < entry_pm  # YES dropped = NO (our position) appreciated
    else:
        return t5_pm > entry_pm  # YES rose = YES (our position) appreciated


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_all_data():
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT w.window_id, w.asset, w.window_start_utc, w.outcome,
                   w.pm_yes_t0, w.pm_yes_t2_5, w.pm_yes_t5,
                   w.pm_yes_t7_5, w.pm_yes_t10, w.pm_yes_t12_5,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5,
                   w.volatility_regime, w.window_time,
                   w.pm_spread_t0, w.spot_change_bps
            FROM windows w
            WHERE w.outcome IS NOT NULL
            ORDER BY w.window_start_utc
        """).fetchall()
        all_rows.extend([dict(r) for r in rows])
        conn.close()

    all_rows.sort(key=lambda x: x['window_start_utc'])

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
            try:
                row['hour_utc'] = int(row['window_start_utc'][11:13])
            except Exception:
                row['hour_utc'] = None
            enriched.append(row)

    enriched.sort(key=lambda x: x['window_start_utc'])

    time_groups = defaultdict(dict)
    for r in enriched:
        wt = r.get('window_time')
        if wt is None:
            parts = r['window_id'].split('_', 1)
            wt = parts[1] if len(parts) > 1 else r['window_id']
        if wt:
            time_groups[wt][r['asset']] = r

    return enriched, time_groups


# ─── STRATEGY CLASS ────────────────────────────────────────────────────────────

class Strategy:
    def __init__(self, name, **p):
        self.name = name
        self.prev_thresh = p.get('prev_thresh', 0.80)
        self.min_agree = p.get('min_agree', 2)

        # ── F1: Double exhaustion ──────────────────────────────────────────
        # Require prev2 also above threshold (two consecutive extreme windows)
        # Note: prev2_pm_t12_5 available at t0 ✓
        self.require_double_exhaust = p.get('require_double_exhaust', False)
        self.double_exhaust_thresh = p.get('double_exhaust_thresh', 0.75)

        # ── F2: Signal freshness ───────────────────────────────────────────
        # Filter by whether signal just appeared (fresh) or has been building
        # Note: prev2_pm_t12_5 available at t0 ✓
        self.freshness_filter = p.get('freshness_filter', None)
        # None = no filter, 'fresh' = only new signals, 'stale' = only persistent
        self.freshness_neutral_thresh = p.get('freshness_neutral_thresh', 0.65)

        # ── F3: Consensus count ────────────────────────────────────────────
        # min_agree already controls this — just use higher values
        # No new code needed

        # ── F4: Signal strength ────────────────────────────────────────────
        # Require average prev_pm across triggered assets to be above a threshold
        # Note: prev_pm_t12_5 available at t0 ✓
        self.min_avg_strength = p.get('min_avg_strength', None)

        # ── F5: Time-of-day filter ─────────────────────────────────────────
        # UTC hour windows (window_start_utc is t0 itself ✓)
        self.allowed_hours = p.get('allowed_hours', None)  # None = all hours
        # e.g., list(range(0, 8)) for Asian session

        # ── F6: Adaptive exit at t5 ────────────────────────────────────────
        # ENTRY at t0. At t5, check position. Mid-window decision — NOT look-ahead.
        # 'profit_take': if profitable at t5 → exit at t5; else → hold to resolution
        # 'stop_hold':   if losing at t5 → exit at t5; else → hold to resolution
        # 'early_only':  standard early exit (baseline)
        self.adaptive_exit = p.get('adaptive_exit', 'early_only')

        # ── F7: Opening level filter ───────────────────────────────────────
        # Filter on pm_yes_t0 (the actual entry price, available at t0 ✓)
        # bear: require pm_yes_t0 > min_open_level (still elevated at open)
        # bull: require pm_yes_t0 < (1 - min_open_level) (still depressed at open)
        self.min_open_level = p.get('min_open_level', None)
        # bear: yes must be above 0.50, bull: yes must be below 0.50
        self.require_open_elevated = p.get('require_open_elevated', False)
        # bear: yes > 0.50 and below prev_pm (hasn't blown through midpoint yet)
        self.max_open_level = p.get('max_open_level', None)

        # ── F8: Tiered exit ────────────────────────────────────────────────
        # If n_agree >= tiered_resolution_threshold → resolution; else → early
        self.tiered_exit = p.get('tiered_exit', False)
        self.tiered_resolution_threshold = p.get('tiered_resolution_threshold', 3)

        # ── F9: Signal deceleration ────────────────────────────────────────
        # Only trade when signal is decelerating (prev < prev2 for bear)
        # Means: the overextension peaked last window, now fading = early sign of reversal
        # Note: both available at t0 ✓
        self.decel_filter = p.get('decel_filter', False)
        # accel_filter: only trade when signal is ACCELERATING (prev > prev2)
        self.accel_filter = p.get('accel_filter', False)

        # ── Sizing ─────────────────────────────────────────────────────────
        self.sizing_mode = p.get('sizing_mode', 'flat')
        self.anti_mart_mult = p.get('anti_mart_mult', 1.5)
        self.kelly_fraction = p.get('kelly_fraction', 0.25)
        self.kelly_window = p.get('kelly_window', 50)
        self.confidence_scale = p.get('confidence_scale', 2.0)

        # ── Risk ───────────────────────────────────────────────────────────
        self.daily_loss_limit = p.get('daily_loss_limit', None)
        self.skip_vol_regimes = set(p.get('skip_vol_regimes', []))
        self.bear_only = p.get('bear_only', False)
        self.bull_only = p.get('bull_only', False)


# ─── SIMULATION ───────────────────────────────────────────────────────────────

def simulate(strategy, time_groups):
    s = strategy
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET

    trades = []
    daily_pnl = defaultdict(float)
    max_dd = 0
    max_dd_pct = 0

    last_bet = base_bet
    last_won = True
    recent_results = deque(maxlen=s.kelly_window if s.sizing_mode == 'kelly' else 1)

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group or bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        day_key = sample_row['window_start_utc'][:10]

        # Daily loss limit
        if s.daily_loss_limit is not None and daily_pnl[day_key] < -s.daily_loss_limit:
            continue

        # F5: Time-of-day filter ── valid: window_start_utc IS t0 ✓
        if s.allowed_hours is not None:
            hour = sample_row.get('hour_utc')
            if hour is None:
                try:
                    hour = int(sample_row['window_start_utc'][11:13])
                except Exception:
                    hour = 0
            if hour not in s.allowed_hours:
                continue

        # ── Direction signal: previous window consensus ────────────────────
        n_strong_up = 0
        n_strong_down = 0
        strong_up_vals = []
        strong_down_vals = []

        for asset, row in group.items():
            p1 = row.get('prev_pm_t12_5')
            if p1 is None:
                continue
            if p1 >= s.prev_thresh:
                n_strong_up += 1
                strong_up_vals.append(p1)
            elif p1 <= (1.0 - s.prev_thresh):
                n_strong_down += 1
                strong_down_vals.append(p1)

        direction = None
        if n_strong_up >= s.min_agree:
            direction = 'bear'
        elif n_strong_down >= s.min_agree:
            direction = 'bull'
        if direction is None:
            continue
        if s.bear_only and direction != 'bear':
            continue
        if s.bull_only and direction != 'bull':
            continue

        n_agree = n_strong_up if direction == 'bear' else n_strong_down
        strong_vals = strong_up_vals if direction == 'bear' else strong_down_vals

        # F4: Signal strength filter ── avg prev_pm of triggered assets ✓
        if s.min_avg_strength is not None and strong_vals:
            if direction == 'bear':
                avg_strength = np.mean(strong_vals)
            else:
                avg_strength = 1.0 - np.mean(strong_vals)  # normalize to [0,1] above thresh
            if avg_strength < s.min_avg_strength:
                continue

        # ── Per-asset filters and confirmation ────────────────────────────
        confirming = []
        for asset, row in group.items():
            entry_pm = row.get('pm_yes_t0')   # entry at t0 ✓
            exit_pm = row.get('pm_yes_t12_5')  # exit target (end of window)
            if entry_pm is None or exit_pm is None:
                continue

            # Basic direction threshold: same convention as valid_momentum_backtest
            # Bear (fade bullish): only enter when YES has reverted to ≤ 0.50
            # Bull (fade bearish): only enter when YES is ≥ 0.50 (still depressed)
            if direction == 'bear' and entry_pm > 0.50:
                continue
            if direction == 'bull' and entry_pm < 0.50:
                continue

            p1 = row.get('prev_pm_t12_5')
            p2 = row.get('prev2_pm_t12_5')
            if p1 is None:
                continue

            # F1: Double exhaustion ── both prev and prev2 above thresh ✓
            if s.require_double_exhaust and p2 is not None:
                de_thresh = s.double_exhaust_thresh
                if direction == 'bear' and p2 < de_thresh:
                    continue
                if direction == 'bull' and p2 > (1.0 - de_thresh):
                    continue

            # F2: Signal freshness ✓
            if s.freshness_filter == 'fresh' and p2 is not None:
                # Fresh: prev just crossed threshold, prev2 was below
                neutral = s.freshness_neutral_thresh
                if direction == 'bear' and p2 >= neutral:
                    continue  # signal was already elevated two windows ago
                if direction == 'bull' and p2 <= (1.0 - neutral):
                    continue
            elif s.freshness_filter == 'stale' and p2 is not None:
                # Stale/persistent: prev2 was also above threshold
                neutral = s.freshness_neutral_thresh
                if direction == 'bear' and p2 < neutral:
                    continue
                if direction == 'bull' and p2 > (1.0 - neutral):
                    continue

            # F9: Deceleration / acceleration filter ✓
            if p2 is not None:
                if s.decel_filter:
                    # Decel: prev is LESS extreme than prev2
                    # bear: p1 < p2 (YES price dropping from last window)
                    if direction == 'bear' and p1 >= p2:
                        continue
                    if direction == 'bull' and p1 <= p2:
                        continue
                if s.accel_filter:
                    # Accel: prev is MORE extreme than prev2
                    if direction == 'bear' and p1 <= p2:
                        continue
                    if direction == 'bull' and p1 >= p2:
                        continue

            # F7: Opening level filter ── pm_yes_t0 is the entry price ✓
            if s.min_open_level is not None:
                if direction == 'bear' and entry_pm < s.min_open_level:
                    continue
                if direction == 'bull' and entry_pm > (1.0 - s.min_open_level):
                    continue
            if s.max_open_level is not None:
                if direction == 'bear' and entry_pm > s.max_open_level:
                    continue
                if direction == 'bull' and entry_pm < (1.0 - s.max_open_level):
                    continue

            # Volatility regime filter
            if s.skip_vol_regimes:
                if row.get('volatility_regime') in s.skip_vol_regimes:
                    continue

            confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < s.min_agree:
            continue

        # ── Bet sizing ─────────────────────────────────────────────────────
        if s.sizing_mode == 'anti_mart':
            if last_won:
                bet = min(last_bet * s.anti_mart_mult, base_bet * 3)
            else:
                bet = max(last_bet / s.anti_mart_mult, base_bet * 0.25)
        elif s.sizing_mode == 'kelly':
            if len(recent_results) >= 20:
                wins_k = sum(1 for r in recent_results if r > 0)
                p = wins_k / len(recent_results)
                avg_w = np.mean([r for r in recent_results if r > 0]) if wins_k > 0 else 0
                losses_k = len(recent_results) - wins_k
                avg_l = abs(np.mean([r for r in recent_results if r <= 0])) if losses_k > 0 else 1
                b = avg_w / avg_l if avg_l > 0 else 1
                kelly = max(0, (b * p - (1 - p)) / b) * s.kelly_fraction if b > 0 else 0
                bet = max(bankroll * kelly, base_bet * 0.25)
            else:
                bet = base_bet
        elif s.sizing_mode == 'strength_scale':
            # Scale bet by signal strength (average prev_pm above baseline)
            if strong_vals:
                avg_s = np.mean(strong_vals) if direction == 'bear' else 1.0 - np.mean(strong_vals)
                # avg_s is in [0.80, 1.0] for threshold=0.80
                scale = 1.0 + (s.confidence_scale - 1.0) * min((avg_s - 0.80) / 0.20, 1.0)
                bet = base_bet * scale
            else:
                bet = base_bet
        else:
            bet = base_bet

        bet = min(bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        # ── Execute trades ─────────────────────────────────────────────────
        for asset, row, entry_pm, exit_pm in confirming[:4]:
            trade_bet = bet

            if trade_bet < 1.0:
                continue

            # ── F6: Adaptive exit logic ──────────────────────────────────
            # Entry at t0. Decision at t5. NOT look-ahead — this is a mid-trade check.
            if s.adaptive_exit in ('profit_take', 'stop_hold'):
                t5_pm = row.get('pm_yes_t5')
                if t5_pm is not None:
                    profitable = is_profitable_at_t5(direction, entry_pm, t5_pm)
                    if s.adaptive_exit == 'profit_take':
                        if profitable:
                            # Take profit: exit at t5 market price
                            net = pnl_early_exit(direction, entry_pm, t5_pm, trade_bet)
                        else:
                            # Hold loser to binary resolution
                            net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
                    else:  # stop_hold
                        if not profitable:
                            # Stop loss: exit at t5
                            net = pnl_early_exit(direction, entry_pm, t5_pm, trade_bet)
                        else:
                            # Let winner ride to resolution
                            net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
                else:
                    net = pnl_early_exit(direction, entry_pm, exit_pm, trade_bet)

            # ── F8: Tiered exit based on consensus count ──────────────────
            elif s.tiered_exit:
                if n_agree >= s.tiered_resolution_threshold:
                    net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
                else:
                    net = pnl_early_exit(direction, entry_pm, exit_pm, trade_bet)

            else:
                # Standard early exit
                net = pnl_early_exit(direction, entry_pm, exit_pm, trade_bet)

            won = net > 0
            bankroll += net
            peak_bankroll = max(peak_bankroll, bankroll)

            dd_d = bankroll - peak_bankroll
            dd_p = dd_d / peak_bankroll if peak_bankroll > 0 else 0
            max_dd = min(max_dd, dd_d)
            max_dd_pct = min(max_dd_pct, dd_p)

            daily_pnl[day_key] += net
            recent_results.append(net)

            trades.append({
                'asset': asset, 'direction': direction,
                'entry_pm': entry_pm, 'exit_pm': exit_pm,
                'bet': trade_bet, 'net_pnl': net, 'won': won,
                'bankroll': bankroll, 'day': day_key,
            })
            last_bet = trade_bet
            last_won = won

    if not trades:
        return None

    pnls = np.array([t['net_pnl'] for t in trades])
    n = len(pnls)
    wins = int(np.sum(pnls > 0))
    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if n > 1 else 1.0
    sharpe = avg_pnl / std_pnl * np.sqrt(n) if std_pnl > 0 else 0
    win_rate = wins / n
    avg_w = float(np.mean(pnls[pnls > 0])) if np.sum(pnls > 0) > 0 else 0
    avg_l = float(np.mean(pnls[pnls <= 0])) if np.sum(pnls <= 0) > 0 else 0
    pf = (abs(float(np.sum(pnls[pnls > 0]))) / abs(float(np.sum(pnls[pnls <= 0])))
          if np.sum(pnls <= 0) != 0 else 999)

    max_ls = cs = 0
    for t in trades:
        if not t['won']:
            cs += 1
            max_ls = max(max_ls, cs)
        else:
            cs = 0

    losing_days = sum(1 for v in daily_pnl.values() if v < 0)
    total_days = len(daily_pnl)

    dir_stats = {}
    for d in ['bull', 'bear']:
        dt = [t for t in trades if t['direction'] == d]
        if dt:
            dp = [t['net_pnl'] for t in dt]
            dw = sum(1 for p in dp if p > 0)
            dir_stats[d] = {'n': len(dt), 'wins': dw, 'wr': dw / len(dt), 'pnl': sum(dp)}

    return {
        'name': s.name, 'trades': n, 'wins': wins, 'win_rate': win_rate,
        'total_pnl': total_pnl, 'avg_pnl': avg_pnl,
        'avg_win': avg_w, 'avg_loss': avg_l,
        'sharpe': sharpe, 'profit_factor': pf,
        'max_dd_dollars': max_dd, 'max_dd_pct': max_dd_pct * 100,
        'final_bankroll': bankroll,
        'roi_pct': (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100,
        'max_loss_streak': max_ls,
        'losing_days': losing_days, 'total_days': total_days,
        'dir_stats': dir_stats,
    }


# ─── OUTPUT ────────────────────────────────────────────────────────────────────

def print_table(results, label):
    if not results:
        print(f"\n  {label} — NO RESULTS")
        return
    print(f"\n{'='*160}")
    print(f"  {label}")
    print(f"{'='*160}")
    hdr = (f"{'Strategy':<52} {'#':>5} {'WR%':>6} {'PnL':>10} {'AvgW':>7} {'AvgL':>7} "
           f"{'Shrp':>6} {'PF':>5} {'DD%':>7} {'Final$':>9} {'ROI%':>7} {'MLS':>4}")
    print(hdr)
    print('-' * 160)
    for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        print(
            f"{r['name']:<52} "
            f"{r['trades']:>5} {r['win_rate']*100:>5.1f}% "
            f"${r['total_pnl']:>9.2f} ${r['avg_win']:>6.2f} ${r['avg_loss']:>6.2f} "
            f"{r['sharpe']:>6.2f} {r['profit_factor']:>4.2f}x "
            f"{r['max_dd_pct']:>6.1f}% ${r['final_bankroll']:>8.2f} "
            f"{r['roi_pct']:>6.1f}% {r['max_loss_streak']:>4}"
        )


def main():
    print("Loading data...")
    enriched, time_groups = load_all_data()

    all_times = sorted(time_groups.keys())
    split_idx = int(len(all_times) * TRAIN_RATIO)
    split_time = all_times[split_idx]
    train_groups = {k: v for k, v in time_groups.items() if k < split_time}
    test_groups = {k: v for k, v in time_groups.items() if k >= split_time}

    # Count windows with sufficient data
    n_with_prev = sum(1 for g in train_groups.values()
                      for r in g.values() if r.get('prev_pm_t12_5') is not None)
    print(f"Train: {len(train_groups)} windows ({n_with_prev} with prev data) | "
          f"Test: {len(test_groups)} | Split: {split_time}")
    print()

    strategies = []

    # ─── BASELINE ────────────────────────────────────────────────────────────
    strategies.append(Strategy("BASE_agree2_flat", min_agree=2))
    strategies.append(Strategy("BASE_agree2_daily100", min_agree=2, daily_loss_limit=100))

    # ─── F1: DOUBLE EXHAUSTION ────────────────────────────────────────────────
    # Two consecutive windows at extreme levels = more exhaustion → stronger reversal?
    # VALID: uses prev_pm_t12_5 and prev2_pm_t12_5, both available at t0
    strategies.append(Strategy(
        "F1_DOUBLE_0.75", min_agree=2,
        require_double_exhaust=True, double_exhaust_thresh=0.75,
    ))
    strategies.append(Strategy(
        "F1_DOUBLE_0.78", min_agree=2,
        require_double_exhaust=True, double_exhaust_thresh=0.78,
    ))
    strategies.append(Strategy(
        "F1_DOUBLE_0.80", min_agree=2,
        require_double_exhaust=True, double_exhaust_thresh=0.80,
    ))
    strategies.append(Strategy(
        "F1_DOUBLE_0.75+daily100", min_agree=2,
        require_double_exhaust=True, double_exhaust_thresh=0.75,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F1_DOUBLE_0.75+agree3", min_agree=3,
        require_double_exhaust=True, double_exhaust_thresh=0.75,
    ))

    # ─── F2: SIGNAL FRESHNESS ─────────────────────────────────────────────────
    # Fresh: signal just appeared (prev2 was neutral). Does a "new" extreme differ from persistent?
    # VALID: uses prev2_pm_t12_5 available at t0
    strategies.append(Strategy(
        "F2_FRESH_0.65", min_agree=2,
        freshness_filter='fresh', freshness_neutral_thresh=0.65,
    ))
    strategies.append(Strategy(
        "F2_FRESH_0.70", min_agree=2,
        freshness_filter='fresh', freshness_neutral_thresh=0.70,
    ))
    strategies.append(Strategy(
        "F2_FRESH_0.65+daily100", min_agree=2,
        freshness_filter='fresh', freshness_neutral_thresh=0.65,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F2_STALE_0.65", min_agree=2,
        freshness_filter='stale', freshness_neutral_thresh=0.65,
    ))

    # ─── F3: CONSENSUS COUNT (3+ OR 4/4) ─────────────────────────────────────
    # When 3 or 4 assets all show the same extreme → stronger edge?
    # VALID: just requires more assets above threshold at t0
    strategies.append(Strategy("F3_AGREE3", min_agree=3))
    strategies.append(Strategy("F3_AGREE4", min_agree=4))
    strategies.append(Strategy("F3_AGREE3+daily100", min_agree=3, daily_loss_limit=100))
    strategies.append(Strategy("F3_AGREE4+daily100", min_agree=4, daily_loss_limit=100))

    # ─── F4: SIGNAL STRENGTH (avg prev_pm of triggered assets) ───────────────
    # Higher average extreme price = stronger fade potential
    # VALID: prev_pm_t12_5 available at t0
    strategies.append(Strategy("F4_STRENGTH_0.82", min_agree=2, min_avg_strength=0.82))
    strategies.append(Strategy("F4_STRENGTH_0.83", min_agree=2, min_avg_strength=0.83))
    strategies.append(Strategy("F4_STRENGTH_0.85", min_agree=2, min_avg_strength=0.85))
    strategies.append(Strategy(
        "F4_STRENGTH_0.83+daily100", min_agree=2, min_avg_strength=0.83,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F4_STRENGTH_0.83+agree3", min_agree=3, min_avg_strength=0.83,
    ))

    # ─── F5: TIME-OF-DAY FILTER ───────────────────────────────────────────────
    # Crypto markets have different session dynamics. Does reversion work better
    # in certain UTC hours?
    # VALID: window_start_utc is t0 itself
    strategies.append(Strategy(
        "F5_ASIAN_0-8", min_agree=2,
        allowed_hours=list(range(0, 8)),
    ))
    strategies.append(Strategy(
        "F5_EURO_6-14", min_agree=2,
        allowed_hours=list(range(6, 14)),
    ))
    strategies.append(Strategy(
        "F5_US_12-22", min_agree=2,
        allowed_hours=list(range(12, 22)),
    ))
    strategies.append(Strategy(
        "F5_OFFHOURS_20-8", min_agree=2,
        allowed_hours=list(range(20, 24)) + list(range(0, 8)),
    ))
    strategies.append(Strategy(
        "F5_ACTIVE_8-20", min_agree=2,
        allowed_hours=list(range(8, 20)),
    ))

    # ─── F6: ADAPTIVE MID-WINDOW EXIT ────────────────────────────────────────
    # ENTRY at t0. At t5 (5 min later), check if position is profitable.
    # This is a mid-trade decision — NOT look-ahead for entry.
    #
    # profit_take: "banking quick wins, letting losers ride to resolution"
    #   - If up at t5 → exit at t5 (take profit)
    #   - If down at t5 → hold to binary resolution (pray for reversal)
    #
    # stop_hold: "stopping out bad trades, letting winners compound"
    #   - If down at t5 → stop out (cut loss)
    #   - If up at t5 → hold to resolution (let winner grow)
    strategies.append(Strategy("F6_PROFIT_TAKE", min_agree=2, adaptive_exit='profit_take'))
    strategies.append(Strategy(
        "F6_PROFIT_TAKE+daily100", min_agree=2,
        adaptive_exit='profit_take', daily_loss_limit=100,
    ))
    strategies.append(Strategy("F6_STOP_HOLD", min_agree=2, adaptive_exit='stop_hold'))
    strategies.append(Strategy(
        "F6_STOP_HOLD+daily100", min_agree=2,
        adaptive_exit='stop_hold', daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F6_PROFIT_TAKE+agree3", min_agree=3, adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "F6_PROFIT_TAKE+agree3+daily100", min_agree=3,
        adaptive_exit='profit_take', daily_loss_limit=100,
    ))

    # ─── F7: OPENING PRICE LEVEL ──────────────────────────────────────────────
    # pm_yes_t0 IS the entry price — available at t0. Base filter: YES ≤ 0.50 for bear.
    # These filters tighten or widen WITHIN that constraint.
    # For BEAR (entry_pm ≤ 0.50): how far has the reversal progressed at open?
    #   max_open_level=0.45: only enter if YES already dropped to ≤ 0.45 (strong reversal)
    #   max_open_level=0.48: modest drop required
    #   min_open_level=0.38: don't enter if YES is ALREADY too low (too far gone)
    # Note: min_open_level for BEAR sets a floor on how low YES can be
    strategies.append(Strategy(
        "F7_BEAR_MAX0.45", min_agree=2,
        max_open_level=0.45,  # bear: YES must be ≤ 0.45 (strong early reversal)
    ))
    strategies.append(Strategy(
        "F7_BEAR_MAX0.48", min_agree=2,
        max_open_level=0.48,  # bear: YES ≤ 0.48
    ))
    strategies.append(Strategy(
        "F7_BEAR_0.35-0.48", min_agree=2,
        min_open_level=0.35, max_open_level=0.48,  # not too far, not too close
    ))
    strategies.append(Strategy(
        "F7_BEAR_0.40-0.50", min_agree=2,
        min_open_level=0.40, max_open_level=0.50,  # near midpoint zone
    ))
    strategies.append(Strategy(
        "F7_BEAR_MAX0.48+daily100", min_agree=2,
        max_open_level=0.48, daily_loss_limit=100,
    ))

    # ─── F8: TIERED EXIT ─────────────────────────────────────────────────────
    # Use consensus COUNT to decide exit mode.
    # Hypothesis: when 3+ assets agree, the signal is stronger → hold to resolution
    # for maximum payout. When only 2 agree, exit early.
    strategies.append(Strategy(
        "F8_TIERED_3→RESOL", min_agree=2,
        tiered_exit=True, tiered_resolution_threshold=3,
    ))
    strategies.append(Strategy(
        "F8_TIERED_4→RESOL", min_agree=2,
        tiered_exit=True, tiered_resolution_threshold=4,
    ))
    strategies.append(Strategy(
        "F8_TIERED_3→RESOL+daily100", min_agree=2,
        tiered_exit=True, tiered_resolution_threshold=3,
        daily_loss_limit=100,
    ))

    # ─── F9: SIGNAL DECELERATION / ACCELERATION ──────────────────────────────
    # Decel: prev_pm_t12_5 < prev2_pm_t12_5 for bear (signal weakening)
    #   → The extreme peaked last window, early sign of reversion
    # Accel: prev > prev2 (signal strengthening = fresh extreme)
    # VALID: both columns available at t0
    strategies.append(Strategy("F9_DECEL", min_agree=2, decel_filter=True))
    strategies.append(Strategy("F9_ACCEL", min_agree=2, accel_filter=True))
    strategies.append(Strategy(
        "F9_DECEL+daily100", min_agree=2, decel_filter=True, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F9_ACCEL+daily100", min_agree=2, accel_filter=True, daily_loss_limit=100,
    ))

    # ─── FC: BEST COMBINATIONS ────────────────────────────────────────────────
    strategies.append(Strategy(
        "FC1_DOUBLE+AGREE3",
        min_agree=3, require_double_exhaust=True, double_exhaust_thresh=0.75,
    ))
    strategies.append(Strategy(
        "FC2_DOUBLE+PROFIT_TAKE",
        min_agree=2, require_double_exhaust=True, double_exhaust_thresh=0.75,
        adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC3_AGREE3+PROFIT_TAKE",
        min_agree=3, adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC3_AGREE3+PROFIT_TAKE+daily100",
        min_agree=3, adaptive_exit='profit_take', daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "FC4_FRESH+PROFIT_TAKE",
        min_agree=2, freshness_filter='fresh', freshness_neutral_thresh=0.65,
        adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC5_LOW_OPEN+PROFIT_TAKE",
        min_agree=2, max_open_level=0.45,
        adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC5_LOW_OPEN+PROFIT_TAKE+daily100",
        min_agree=2, max_open_level=0.45,
        adaptive_exit='profit_take', daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "FC6_DOUBLE+LOW_OPEN+PROFIT_TAKE",
        min_agree=2, require_double_exhaust=True, double_exhaust_thresh=0.75,
        max_open_level=0.45, adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC7_STRENGTH+AGREE3+PROFIT_TAKE",
        min_agree=3, min_avg_strength=0.83, adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC8_DECEL+PROFIT_TAKE",
        min_agree=2, decel_filter=True, adaptive_exit='profit_take',
    ))
    strategies.append(Strategy(
        "FC9_DOUBLE+DECEL+daily100",
        min_agree=2, require_double_exhaust=True, double_exhaust_thresh=0.75,
        decel_filter=True, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "FC10_US_HOURS+AGREE3",
        min_agree=3, allowed_hours=list(range(12, 22)),
    ))
    strategies.append(Strategy(
        "FC11_STRENGTH+LOW_OPEN+daily100",
        min_agree=2, min_avg_strength=0.83, max_open_level=0.48,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "FC12_ALL_BEST",
        min_agree=3, require_double_exhaust=True, double_exhaust_thresh=0.75,
        min_open_level=0.52, adaptive_exit='profit_take', daily_loss_limit=100,
    ))
    # Sizing variants on the most promising combos
    strategies.append(Strategy(
        "FC3_AGREE3+PT+ANTIMART",
        min_agree=3, adaptive_exit='profit_take',
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))
    strategies.append(Strategy(
        "FC5_LOW_OPEN+PT+ANTIMART",
        min_agree=2, max_open_level=0.45,
        adaptive_exit='profit_take',
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))

    print(f"Running {len(strategies)} strategies...")
    train_results, test_results = [], []
    for strat in strategies:
        tr = simulate(strat, train_groups)
        te = simulate(strat, test_groups)
        if tr:
            train_results.append(tr)
        if te:
            test_results.append(te)

    print_table(train_results, "TRAIN SET")
    print_table(test_results, "TEST SET")

    # ─── Train vs Test ─────────────────────────────────────────────────────────
    print(f"\n{'='*165}")
    print(f"  TRAIN vs TEST — sorted by test P&L")
    print(f"{'='*165}")
    print(
        f"{'Strategy':<52} "
        f"{'TrPnL':>9} {'TrWR':>6} {'TrDD%':>7} "
        f"{'TePnL':>9} {'TeWR':>6} {'TeDD%':>7} "
        f"{'Rob':>4} {'Tr#':>5} {'Te#':>5} "
        f"{'TeAvgW':>7} {'TeAvgL':>7}"
    )
    print('-' * 165)

    test_by_name = {r['name']: r for r in test_results}
    train_by_name = {r['name']: r for r in train_results}

    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if not tr:
            continue
        rob = "Y" if tr['total_pnl'] > 0 and te['total_pnl'] > 0 else "-"
        print(
            f"{name:<52} "
            f"${tr['total_pnl']:>8.0f} {tr['win_rate']*100:>5.1f}% {tr['max_dd_pct']:>6.1f}% "
            f"${te['total_pnl']:>8.0f} {te['win_rate']*100:>5.1f}% {te['max_dd_pct']:>6.1f}% "
            f"{rob:>4} {tr['trades']:>5} {te['trades']:>5} "
            f"${te['avg_win']:>6.2f} ${te['avg_loss']:>6.2f}"
        )

    # ─── Robust ────────────────────────────────────────────────────────────────
    print(f"\n{'='*130}")
    print(f"  ROBUST (profitable in BOTH train and test) — sorted by test P&L")
    print(f"{'='*130}")
    robust = [(n, train_by_name[n], test_by_name[n])
              for n in sorted(test_by_name, key=lambda n: test_by_name[n]['total_pnl'], reverse=True)
              if n in train_by_name
              and train_by_name[n]['total_pnl'] > 0
              and test_by_name[n]['total_pnl'] > 0]
    if not robust:
        print("  None.")
    for i, (name, tr, te) in enumerate(robust, 1):
        ds = te.get('dir_stats', {})
        b = ds.get('bear', {}); bu = ds.get('bull', {})
        bull_s = f"bull:{bu.get('wr',0)*100:.0f}%/{bu.get('n',0)}" if bu else ""
        bear_s = f"bear:{b.get('wr',0)*100:.0f}%/{b.get('n',0)}" if b else ""
        print(
            f"  {i:>2}. {name:<52} "
            f"Train: ${tr['total_pnl']:>8.0f} ({tr['win_rate']*100:.1f}%)  "
            f"Test: ${te['total_pnl']:>8.0f} ({te['win_rate']*100:.1f}%, {te['max_dd_pct']:.1f}%DD)  "
            f"{bull_s} {bear_s}"
        )

    # ─── Save ──────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "reports" / "creative_results.json"
    out_path.parent.mkdir(exist_ok=True)
    output = {'split_time': split_time, 'strategies': {}}
    for name in test_by_name:
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr:
            output['strategies'][name] = {
                'train': {k: v for k, v in tr.items()},
                'test': {k: v for k, v in te.items()},
                'robust': tr['total_pnl'] > 0 and te['total_pnl'] > 0,
            }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
