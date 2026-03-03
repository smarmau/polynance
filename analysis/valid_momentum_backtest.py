#!/usr/bin/env python3
"""
Valid Momentum Filters Backtest — 70/30 split

Explores genuinely valid t0 entry filters (no look-ahead bias).

The original advanced_exits_sizing_backtest.py used pm_price_momentum_0_to_5
(t0→t5 price change) as an entry filter while entering at t0. This is look-ahead.
Proof: A4_T5_MOM_CONFIRM (honest version) has *negative* test P&L.

Signals actually available at t0:
  - prev_pm_t12_5       previous window's closing price (basis of direction signal)
  - prev2_pm_t12_5      two windows ago closing price
  - pm_yes_t0           current window's opening price
  - volatility_regime   from previous window's spot range

Derived valid t0 signals:
  E1  Prior momentum: prev_pm_t12_5 - prev2_pm_t12_5
      "The trend we're fading was building in the prior window"
  E2  Opening gap:    pm_yes_t0 - prev_pm_t12_5
      "Price has already started reversing at the open (2.5-min gap)"
  E3  Signal strength: higher prev_thresh (0.82 / 0.85 / 0.90)
  E4  Volatility regime filter
  E5–E8  Combinations + best sizing from prior study
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


# ─── P&L FUNCTIONS ───────────────────────────────────────────────────────────

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
                   w.pm_price_momentum_0_to_5,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5,
                   w.volatility_regime, w.window_time,
                   w.pm_spread_t0
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
            parts = r['window_id'].split('_')
            if len(parts) >= 3:
                wt = '_'.join(parts[1:])
        if wt:
            time_groups[wt][r['asset']] = r

    return enriched, time_groups


# ─── STRATEGY CLASS ───────────────────────────────────────────────────────────

class Strategy:
    def __init__(self, name, **p):
        self.name = name
        self.prev_thresh = p.get('prev_thresh', 0.80)
        self.bull_thresh = p.get('bull_thresh', 0.50)
        self.bear_thresh = p.get('bear_thresh', 0.50)
        self.min_agree = p.get('min_agree', 2)

        self.entry_time = p.get('entry_time', 't0')
        self.exit_time = p.get('exit_time', 't12.5')
        self.exit_mode = p.get('exit_mode', 'early')

        # ── Valid t0 filters (no look-ahead) ──────────────────────────────

        # E1: Prior window momentum filter
        # For bear: prev_pm_t12_5 - prev2_pm_t12_5 > prior_mom_min
        #   (the trend we're fading was still building in the prior window)
        # For bull: prev2_pm_t12_5 - prev_pm_t12_5 > prior_mom_min
        self.prior_mom_filter = p.get('prior_mom_filter', False)
        self.prior_mom_min = p.get('prior_mom_min', 0.03)
        self.prior_mom_min_agree = p.get('prior_mom_min_agree', 1)  # assets needed

        # E2: Opening gap filter
        # For bear: pm_yes_t0 < prev_pm_t12_5 - gap_min (already starting to drop)
        # For bull: pm_yes_t0 > prev_pm_t12_5 + gap_min (already starting to rise)
        self.opening_gap_filter = p.get('opening_gap_filter', False)
        self.opening_gap_min = p.get('opening_gap_min', 0.005)

        # E4: Volatility regime exclusion
        self.skip_vol_regimes = set(p.get('skip_vol_regimes', []))

        # Bet sizing
        self.sizing_mode = p.get('sizing_mode', 'flat')
        self.anti_mart_mult = p.get('anti_mart_mult', 1.5)
        self.anti_mart_max = p.get('anti_mart_max', 3.0)
        self.kelly_window = p.get('kelly_window', 50)
        self.kelly_fraction = p.get('kelly_fraction', 0.25)
        self.bear_size_mult = p.get('bear_size_mult', 1.0)
        self.bull_size_mult = p.get('bull_size_mult', 1.0)
        self.confidence_scale = p.get('confidence_scale', 2.0)

        # Risk management
        self.daily_loss_limit = p.get('daily_loss_limit', None)
        self.bear_only = p.get('bear_only', False)
        self.bull_only = p.get('bull_only', False)

    def pm_key(self, t):
        return {
            't0': 'pm_yes_t0', 't2.5': 'pm_yes_t2_5', 't5': 'pm_yes_t5',
            't7.5': 'pm_yes_t7_5', 't10': 'pm_yes_t10', 't12.5': 'pm_yes_t12_5',
        }.get(t, 'pm_yes_t0')


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

    win_streak = 0
    recent_results = deque(maxlen=s.kelly_window if s.sizing_mode == 'kelly' else 1)

    entry_key = s.pm_key(s.entry_time)
    exit_key = s.pm_key(s.exit_time)

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group or bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        day_key = sample_row['window_start_utc'][:10]

        if s.daily_loss_limit is not None and daily_pnl[day_key] < -s.daily_loss_limit:
            continue

        # ── Phase 1: Previous window consensus (direction signal) ──────────
        n_strong_up = 0
        n_strong_down = 0
        for asset, row in group.items():
            p = row.get('prev_pm_t12_5')
            if p is not None:
                if p >= s.prev_thresh:
                    n_strong_up += 1
                elif p <= (1.0 - s.prev_thresh):
                    n_strong_down += 1

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

        # ── E1: Prior window momentum filter (group-level) ─────────────────
        if s.prior_mom_filter:
            n_confirm_prior = 0
            for asset, row in group.items():
                p1 = row.get('prev_pm_t12_5')
                p2 = row.get('prev2_pm_t12_5')
                if p1 is None or p2 is None:
                    continue
                delta = p1 - p2
                if direction == 'bear' and delta > s.prior_mom_min:
                    n_confirm_prior += 1
                elif direction == 'bull' and delta < -s.prior_mom_min:
                    n_confirm_prior += 1
            if n_confirm_prior < s.prior_mom_min_agree:
                continue

        # ── Phase 2: Per-asset confirmation ────────────────────────────────
        confirming = []
        for asset, row in group.items():
            entry_pm = row.get(entry_key)
            exit_pm = row.get(exit_key)
            if entry_pm is None or exit_pm is None:
                continue

            if direction == 'bear' and entry_pm > s.bear_thresh:
                continue
            if direction == 'bull' and entry_pm < s.bull_thresh:
                continue

            # E2: Opening gap filter (per-asset)
            if s.opening_gap_filter:
                t0_pm = row.get('pm_yes_t0')
                prev_close = row.get('prev_pm_t12_5')
                if t0_pm is None or prev_close is None:
                    continue
                gap = t0_pm - prev_close
                if direction == 'bear' and gap > -s.opening_gap_min:
                    continue
                if direction == 'bull' and gap < s.opening_gap_min:
                    continue

            # E4: Volatility regime filter (per-asset)
            if s.skip_vol_regimes:
                regime = row.get('volatility_regime')
                if regime in s.skip_vol_regimes:
                    continue

            confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < s.min_agree:
            continue

        # ── Bet sizing ─────────────────────────────────────────────────────
        if s.sizing_mode == 'anti_mart':
            if win_streak > 0:
                bet = base_bet * min(s.anti_mart_mult ** win_streak, s.anti_mart_max)
            else:
                bet = base_bet
        elif s.sizing_mode == 'kelly':
            if len(recent_results) >= 20:
                wins = sum(1 for r in recent_results if r > 0)
                losses = len(recent_results) - wins
                p = wins / len(recent_results)
                avg_w = np.mean([r for r in recent_results if r > 0]) if wins > 0 else 0
                avg_l = abs(np.mean([r for r in recent_results if r <= 0])) if losses > 0 else 1
                b = avg_w / avg_l if avg_l > 0 else 1
                kelly = (b * p - (1 - p)) / b if b > 0 else 0
                kelly = max(0, kelly) * s.kelly_fraction
                bet = max(bankroll * kelly, base_bet * 0.25)
            else:
                bet = base_bet
        elif s.sizing_mode == 'asymmetric':
            bet = base_bet * (s.bear_size_mult if direction == 'bear' else s.bull_size_mult)
        elif s.sizing_mode == 'confidence':
            bet = base_bet  # scaled per-trade below
        else:
            bet = base_bet

        bet = min(bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        # ── Execute trades ─────────────────────────────────────────────────
        for asset, row, entry_pm, exit_pm in confirming[:4]:
            trade_bet = bet

            if s.sizing_mode == 'confidence':
                # Use prior momentum as confidence proxy (valid at t0!)
                p1 = row.get('prev_pm_t12_5')
                p2 = row.get('prev2_pm_t12_5')
                if p1 is not None and p2 is not None:
                    abs_prior_mom = abs(p1 - p2)
                    scale = 1.0 + (s.confidence_scale - 1.0) * min(abs_prior_mom / 0.20, 1.0)
                    trade_bet = base_bet * scale
                trade_bet = min(trade_bet, bankroll * 0.05)

            if trade_bet < 1.0:
                continue

            if s.exit_mode == 'resolution':
                net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
            else:
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

            if won:
                win_streak += 1
            else:
                win_streak = 0

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


# ─── OUTPUT ───────────────────────────────────────────────────────────────────

def print_table(results, label):
    if not results:
        print(f"\n  {label} — NO RESULTS")
        return
    print(f"\n{'='*155}")
    print(f"  {label}")
    print(f"{'='*155}")
    print(
        f"{'Strategy':<50} {'#':>5} {'WR%':>6} {'PnL':>10} {'AvgW':>7} {'AvgL':>7} "
        f"{'Shrp':>6} {'PF':>5} {'DD%':>7} {'Final$':>9} {'ROI%':>7} {'MLS':>4}"
    )
    print('-' * 155)
    for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        print(
            f"{r['name']:<50} "
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
    print(f"Train: {len(train_groups)} windows | Test: {len(test_groups)} windows | Split: {split_time}")
    print()

    strategies = []

    # ─── BASELINES ───────────────────────────────────────────────────────────
    strategies.append(Strategy("BASE_agree2_flat", min_agree=2))
    strategies.append(Strategy("BASE_agree2_daily100", min_agree=2, daily_loss_limit=100))
    strategies.append(Strategy("BASE_agree3_flat", min_agree=3))

    # ─── E1: Prior window momentum filter ────────────────────────────────────
    # "The trend we're fading had positive momentum in the window before"
    for mom_min in [0.01, 0.02, 0.03, 0.05, 0.08]:
        strategies.append(Strategy(
            f"E1_PRIORMOM_{mom_min:.2f}",
            min_agree=2, prior_mom_filter=True, prior_mom_min=mom_min,
        ))
    strategies.append(Strategy(
        "E1_PRIORMOM_0.03+daily100",
        min_agree=2, prior_mom_filter=True, prior_mom_min=0.03, daily_loss_limit=100,
    ))
    # Require 2 assets to confirm prior momentum (stricter)
    strategies.append(Strategy(
        "E1_PRIORMOM_0.03+agree2",
        min_agree=2, prior_mom_filter=True, prior_mom_min=0.03, prior_mom_min_agree=2,
    ))

    # ─── E2: Opening gap filter ───────────────────────────────────────────────
    # "PM price has already started moving our way in the gap since last window"
    for gap_min in [0.003, 0.005, 0.01, 0.02]:
        strategies.append(Strategy(
            f"E2_GAP_{gap_min:.3f}",
            min_agree=2, opening_gap_filter=True, opening_gap_min=gap_min,
        ))
    strategies.append(Strategy(
        "E2_GAP_0.005+daily100",
        min_agree=2, opening_gap_filter=True, opening_gap_min=0.005, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "E2_GAP_0.010+daily100",
        min_agree=2, opening_gap_filter=True, opening_gap_min=0.01, daily_loss_limit=100,
    ))

    # ─── E3: Higher signal threshold ─────────────────────────────────────────
    # "Only fade when prior close was more extreme"
    for thresh in [0.82, 0.85, 0.88, 0.90]:
        strategies.append(Strategy(
            f"E3_THRESH_{thresh:.2f}",
            min_agree=2, prev_thresh=thresh,
        ))
    strategies.append(Strategy(
        "E3_THRESH_0.85+daily100",
        min_agree=2, prev_thresh=0.85, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "E3_THRESH_0.85+agree3",
        min_agree=3, prev_thresh=0.85,
    ))

    # ─── E4: Volatility regime filter ────────────────────────────────────────
    strategies.append(Strategy(
        "E4_SKIP_EXTREME",
        min_agree=2, skip_vol_regimes=['extreme'],
    ))
    strategies.append(Strategy(
        "E4_SKIP_HIGH_EXTREME",
        min_agree=2, skip_vol_regimes=['high', 'extreme'],
    ))
    strategies.append(Strategy(
        "E4_ONLY_NORMAL",
        min_agree=2, skip_vol_regimes=['low', 'high', 'extreme'],
    ))
    strategies.append(Strategy(
        "E4_LOW_NORMAL",
        min_agree=2, skip_vol_regimes=['high', 'extreme'],
    ))

    # ─── E5: Prior momentum + opening gap combo ───────────────────────────────
    strategies.append(Strategy(
        "E5_PRIORMOM+GAP",
        min_agree=2,
        prior_mom_filter=True, prior_mom_min=0.03,
        opening_gap_filter=True, opening_gap_min=0.005,
    ))
    strategies.append(Strategy(
        "E5_PRIORMOM+GAP+daily100",
        min_agree=2,
        prior_mom_filter=True, prior_mom_min=0.03,
        opening_gap_filter=True, opening_gap_min=0.005,
        daily_loss_limit=100,
    ))

    # ─── E6: Valid filters + bet sizing ──────────────────────────────────────
    strategies.append(Strategy(
        "E6_PRIORMOM+ANTIMART",
        min_agree=2, prior_mom_filter=True, prior_mom_min=0.03,
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))
    strategies.append(Strategy(
        "E6_PRIORMOM+CONF1.5x",
        min_agree=2, prior_mom_filter=True, prior_mom_min=0.03,
        sizing_mode='confidence', confidence_scale=1.5,
    ))
    strategies.append(Strategy(
        "E6_GAP+ANTIMART",
        min_agree=2, opening_gap_filter=True, opening_gap_min=0.005,
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))
    strategies.append(Strategy(
        "E6_GAP+CONF1.5x",
        min_agree=2, opening_gap_filter=True, opening_gap_min=0.005,
        sizing_mode='confidence', confidence_scale=1.5,
    ))
    strategies.append(Strategy(
        "E6_GAP+CONF2.0x",
        min_agree=2, opening_gap_filter=True, opening_gap_min=0.005,
        sizing_mode='confidence', confidence_scale=2.0,
    ))

    # ─── E7: Prior momentum + higher threshold ────────────────────────────────
    strategies.append(Strategy(
        "E7_PRIORMOM+THRESH_0.85",
        min_agree=2, prior_mom_filter=True, prior_mom_min=0.03,
        prev_thresh=0.85,
    ))
    strategies.append(Strategy(
        "E7_GAP+THRESH_0.85",
        min_agree=2, opening_gap_filter=True, opening_gap_min=0.005,
        prev_thresh=0.85,
    ))
    strategies.append(Strategy(
        "E7_PRIORMOM+THRESH_0.85+daily100",
        min_agree=2, prior_mom_filter=True, prior_mom_min=0.03,
        prev_thresh=0.85, daily_loss_limit=100,
    ))

    # ─── E8: Kitchen sink (all valid filters) ────────────────────────────────
    strategies.append(Strategy(
        "E8_ALL_FILTERS",
        min_agree=2,
        prior_mom_filter=True, prior_mom_min=0.03,
        opening_gap_filter=True, opening_gap_min=0.005,
        prev_thresh=0.85,
        skip_vol_regimes=['extreme'],
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "E8_ALL_FILTERS+ANTIMART",
        min_agree=2,
        prior_mom_filter=True, prior_mom_min=0.03,
        opening_gap_filter=True, opening_gap_min=0.005,
        prev_thresh=0.85,
        skip_vol_regimes=['extreme'],
        daily_loss_limit=100,
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))

    print(f"Running {len(strategies)} strategies...")
    train_results = []
    test_results = []
    for strat in strategies:
        tr = simulate(strat, train_groups)
        te = simulate(strat, test_groups)
        if tr:
            train_results.append(tr)
        if te:
            test_results.append(te)

    print_table(train_results, "TRAIN SET")
    print_table(test_results, "TEST SET")

    # ─── Train vs Test comparison ─────────────────────────────────────────────
    print(f"\n{'='*160}")
    print(f"  TRAIN vs TEST — sorted by test P&L")
    print(f"{'='*160}")
    print(
        f"{'Strategy':<50} "
        f"{'TrPnL':>9} {'TrWR':>6} {'TrDD%':>7} "
        f"{'TePnL':>9} {'TeWR':>6} {'TeDD%':>7} "
        f"{'Rob':>4} {'Tr#':>5} {'Te#':>5} "
        f"{'TeAvgW':>7} {'TeAvgL':>7}"
    )
    print('-' * 160)

    test_by_name = {r['name']: r for r in test_results}
    train_by_name = {r['name']: r for r in train_results}

    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if not tr:
            continue
        rob = "Y" if tr['total_pnl'] > 0 and te['total_pnl'] > 0 else "-"
        print(
            f"{name:<50} "
            f"${tr['total_pnl']:>8.0f} {tr['win_rate']*100:>5.1f}% {tr['max_dd_pct']:>6.1f}% "
            f"${te['total_pnl']:>8.0f} {te['win_rate']*100:>5.1f}% {te['max_dd_pct']:>6.1f}% "
            f"{rob:>4} {tr['trades']:>5} {te['trades']:>5} "
            f"${te['avg_win']:>6.2f} ${te['avg_loss']:>6.2f}"
        )

    # ─── Robust strategies ────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  ROBUST (profitable in BOTH train and test)")
    print(f"{'='*120}")
    robust = []
    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr and tr['total_pnl'] > 0 and te['total_pnl'] > 0:
            robust.append((name, tr, te))

    if not robust:
        print("  None found.")
    for i, (name, tr, te) in enumerate(robust, 1):
        ds = te.get('dir_stats', {})
        bull_info = f"bull:{ds['bull']['wr']*100:.0f}%/{ds['bull']['n']}" if 'bull' in ds else ""
        bear_info = f"bear:{ds['bear']['wr']*100:.0f}%/{ds['bear']['n']}" if 'bear' in ds else ""
        print(
            f"  {i:>2}. {name:<50} "
            f"Train: ${tr['total_pnl']:>8.0f} ({tr['win_rate']*100:.1f}%)  "
            f"Test: ${te['total_pnl']:>8.0f} ({te['win_rate']*100:.1f}% WR, {te['max_dd_pct']:.1f}% DD)  "
            f"{bull_info} {bear_info}"
        )

    # ─── Save ─────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "reports" / "valid_momentum_results.json"
    out_path.parent.mkdir(exist_ok=True)
    output = {'split_time': split_time, 'strategies': {}}
    for name in test_by_name:
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr:
            output['strategies'][name] = {
                'train': {k: v for k, v in tr.items() if k != 'equity_curve'},
                'test': {k: v for k, v in te.items() if k != 'equity_curve'},
                'robust': tr['total_pnl'] > 0 and te['total_pnl'] > 0,
            }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
