#!/usr/bin/env python3
"""
Filter Backtest — Volatility & Loss Filters + Study 2 Candidates
70/30 time-series split, extended data Jan 24 → Mar 1

WHAT THIS TESTS
===============
This script answers: do volatility filters and loss-streak circuit breakers
improve performance on our best valid strategies?

Data findings (pre-coded from EDA on 12k+ trades):
  Regime WR (agree2 signal):
    normal:  58.2%  ← best regime
    low:     55.2%
    high:    52.9%
    extreme: 48.5%  ← below break-even

  Spot range bps (prior window) → WR:
    0-20:  54.4%   20-40: 59.7%  ← sweet spot
    40-60: 54.5%   60-80: 49.8%
    80+:   46-44%  ← consistent losers

  Consecutive loss streaks → next WR (COUNTER-INTUITIVE):
    0 prior losses: 60.7%
    1 prior loss:   57.9%
    2 prior losses: 62.3%
    3 prior losses: 67.6%   ← signal strengthens on losing streaks
  → Circuit breakers expected to HURT. Testing anyway to quantify.

STUDY 2 PORTS (E-series)
  E1: Prior momentum filter (prev_pm - prev2_pm)
  E4: Volatility regime exclusion
  E7: Prior momentum + higher threshold + daily100
  E8: All filters combined

NEW FILTERS (G/H-series applied to Study 3 best)
  G_VOL: volatility regime cutoff
  G_RNG: max prior window spot range bps
  H_CB:  consecutive-loss circuit breaker

SIGNAL AVAILABILITY — all filters here are t0-valid:
  ✓ prev_pm_t12_5     previous window close
  ✓ prev2_pm_t12_5    two windows ago close
  ✓ pm_yes_t0         entry price (IS t0)
  ✓ volatility_regime from prior window spot range
  ✓ prev_spot_range_bps prior window's bps range (precomputed from prior row)
  ✓ window_start_utc  IS t0
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

# Volatility regime ordering for threshold comparisons
VOL_ORDER = {'low': 0, 'normal': 1, 'high': 2, 'extreme': 3}


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
    if direction == 'bear':
        return t5_pm < entry_pm
    else:
        return t5_pm > entry_pm


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
                   w.pm_yes_t0, w.pm_yes_t5, w.pm_yes_t12_5,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5,
                   w.volatility_regime, w.window_time,
                   w.pm_spread_t0, w.spot_range_bps
            FROM windows w
            WHERE w.outcome IS NOT NULL
            ORDER BY w.window_start_utc
        """).fetchall()
        all_rows.extend([dict(r) for r in rows])
        conn.close()

    all_rows.sort(key=lambda x: x['window_start_utc'])

    # ── Per-asset enrichment: fill prev columns and precompute prev_spot_range_bps
    asset_windows = defaultdict(list)
    for r in all_rows:
        asset_windows[r['asset']].append(r)

    enriched = []
    for asset, wins in asset_windows.items():
        for i, w in enumerate(wins):
            row = dict(w)
            # Fill prev columns from prior row if DB didn't populate them
            if row.get('prev_pm_t12_5') is None and i > 0:
                row['prev_pm_t12_5'] = wins[i - 1].get('pm_yes_t12_5')
            if row.get('prev2_pm_t12_5') is None and i > 1:
                row['prev2_pm_t12_5'] = wins[i - 2].get('pm_yes_t12_5')
            # ✓ prev_spot_range_bps: prior window's spot range, valid at t0
            row['prev_spot_range_bps'] = wins[i - 1].get('spot_range_bps') if i > 0 else None
            # UTC hour
            try:
                row['hour_utc'] = int(row['window_start_utc'][11:13])
            except Exception:
                row['hour_utc'] = None
            enriched.append(row)

    enriched.sort(key=lambda x: x['window_start_utc'])

    # ── Group by time slot
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

        # ── Core signal ────────────────────────────────────────────────────
        self.prev_thresh = p.get('prev_thresh', 0.80)
        self.min_agree = p.get('min_agree', 2)
        self.bear_only = p.get('bear_only', False)
        self.bull_only = p.get('bull_only', False)

        # ── Exit mode ──────────────────────────────────────────────────────
        # 'early': exit at pm_yes_t12_5 (mid-market end of window)
        # 'resolution': hold to binary outcome
        # 'tiered': resolution if n_agree >= tiered_resolution_threshold, else early
        # 'profit_take': if up at t5 → exit t5, else → resolution
        self.exit_mode = p.get('exit_mode', 'early')
        self.tiered_resolution_threshold = p.get('tiered_resolution_threshold', 3)

        # ── Entry constraints ──────────────────────────────────────────────
        # Bear: entry_pm <= bear_thresh (YES has reverted to ≤ 0.50)
        # Bull: entry_pm >= bull_thresh (YES has recovered to ≥ 0.50)
        self.bear_thresh = p.get('bear_thresh', 0.50)
        self.bull_thresh = p.get('bull_thresh', 0.50)
        # Optional tighter opening level (per-asset)
        self.max_open_level = p.get('max_open_level', None)  # bear: YES ≤ this

        # ── E1: Prior momentum filter (group-level) ────────────────────────
        # For bear: require that (prev_pm - prev2_pm) > prior_mom_min for
        # at least prior_mom_min_agree assets → signal was still building last window
        # ✓ uses prev_pm_t12_5, prev2_pm_t12_5 — both available at t0
        self.prior_mom_filter = p.get('prior_mom_filter', False)
        self.prior_mom_min = p.get('prior_mom_min', 0.03)
        self.prior_mom_min_agree = p.get('prior_mom_min_agree', 1)

        # ── F4: Signal strength ────────────────────────────────────────────
        # ✓ uses prev_pm_t12_5 — available at t0
        self.min_avg_strength = p.get('min_avg_strength', None)

        # ── F9: Decel / Accel ─────────────────────────────────────────────
        # decel: prev < prev2 for bear (signal already fading last window)
        # accel: prev > prev2 for bear (signal still building = fresh extreme)
        # ✓ both columns available at t0
        self.decel_filter = p.get('decel_filter', False)
        self.accel_filter = p.get('accel_filter', False)

        # ── F5: Time-of-day ───────────────────────────────────────────────
        # ✓ window_start_utc IS t0
        self.allowed_hours = p.get('allowed_hours', None)

        # ── G: Volatility filters ──────────────────────────────────────────
        # skip_vol_regimes: set of regime names to skip
        # ✓ volatility_regime is computed from the PRIOR window's spot range
        self.skip_vol_regimes = set(p.get('skip_vol_regimes', []))

        # max_vol_regime: skip if regime is strictly ABOVE this level
        # e.g. 'normal' → skip high + extreme; 'high' → skip only extreme
        # ✓ same as above — valid at t0
        self.max_vol_regime = p.get('max_vol_regime', None)

        # max_prev_spot_range_bps: skip if prior window's spot range exceeds threshold
        # ✓ prev_spot_range_bps precomputed in load_all_data from prior row
        self.max_prev_spot_range_bps = p.get('max_prev_spot_range_bps', None)

        # ── H: Loss-streak circuit breaker ────────────────────────────────
        # max_consec_losses: if last N GROUP trades (windows that fired) were all
        # losses, sit out the next window (mandatory 1-window pause, then reset).
        # DATA SAYS THIS SHOULD HURT — WR rises after losses, not falls.
        # Testing anyway to quantify the cost of "conventional" risk management.
        self.max_consec_losses = p.get('max_consec_losses', None)

        # ── Sizing ─────────────────────────────────────────────────────────
        self.sizing_mode = p.get('sizing_mode', 'flat')
        self.anti_mart_mult = p.get('anti_mart_mult', 1.5)
        self.daily_loss_limit = p.get('daily_loss_limit', None)


# ─── SIMULATION ───────────────────────────────────────────────────────────────

def simulate(strategy, time_groups):
    s = strategy
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET

    trades = []
    daily_pnl = defaultdict(float)
    max_dd_pct = 0.0

    last_bet = base_bet
    last_won = True

    # H: circuit breaker state — tracks GROUP-level consecutive losses
    # (only counts windows where we actually took trades)
    group_consec_losses = 0

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group or bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        day_key = sample_row['window_start_utc'][:10]

        # ── Daily loss limit ──────────────────────────────────────────────
        if s.daily_loss_limit is not None and daily_pnl[day_key] < -s.daily_loss_limit:
            continue

        # ── F5: Time-of-day ───────────────────────────────────────────────
        if s.allowed_hours is not None:
            hour = sample_row.get('hour_utc')
            if hour is None:
                try:
                    hour = int(sample_row['window_start_utc'][11:13])
                except Exception:
                    hour = 0
            if hour not in s.allowed_hours:
                continue

        # ── H: Circuit breaker ────────────────────────────────────────────
        # After max_consec_losses consecutive group losses, sit out one window.
        # Skipped window resets the streak so we try again next time.
        if s.max_consec_losses is not None and group_consec_losses >= s.max_consec_losses:
            group_consec_losses = 0  # mandatory pause: skip this window, reset
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

        # ── F4: Signal strength (group-level) ─────────────────────────────
        if s.min_avg_strength is not None and strong_vals:
            if direction == 'bear':
                avg_strength = np.mean(strong_vals)
            else:
                avg_strength = 1.0 - np.mean(strong_vals)
            if avg_strength < s.min_avg_strength:
                continue

        # ── E1: Prior momentum filter (group-level) ────────────────────────
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

        # ── Per-asset filters ──────────────────────────────────────────────
        confirming = []
        for asset, row in group.items():
            entry_pm = row.get('pm_yes_t0')
            exit_pm = row.get('pm_yes_t12_5')
            if entry_pm is None or exit_pm is None:
                continue

            # Basic direction entry constraint (reverted to midpoint)
            if direction == 'bear' and entry_pm > s.bear_thresh:
                continue
            if direction == 'bull' and entry_pm < s.bull_thresh:
                continue

            # Optional tighter opening level
            if s.max_open_level is not None:
                if direction == 'bear' and entry_pm > s.max_open_level:
                    continue
                if direction == 'bull' and entry_pm < (1.0 - s.max_open_level):
                    continue

            p1 = row.get('prev_pm_t12_5')
            p2 = row.get('prev2_pm_t12_5')
            if p1 is None:
                continue

            # ── F9: Decel / Accel ─────────────────────────────────────────
            if p2 is not None:
                if s.decel_filter:
                    if direction == 'bear' and p1 >= p2:
                        continue
                    if direction == 'bull' and p1 <= p2:
                        continue
                if s.accel_filter:
                    if direction == 'bear' and p1 <= p2:
                        continue
                    if direction == 'bull' and p1 >= p2:
                        continue

            # ── G: Volatility regime filter ────────────────────────────────
            regime = row.get('volatility_regime')
            if s.skip_vol_regimes and regime in s.skip_vol_regimes:
                continue
            if s.max_vol_regime is not None and regime is not None:
                if VOL_ORDER.get(regime, 1) > VOL_ORDER.get(s.max_vol_regime, 3):
                    continue

            # ── G: Prior spot range filter ─────────────────────────────────
            if s.max_prev_spot_range_bps is not None:
                prb = row.get('prev_spot_range_bps')
                if prb is not None and prb > s.max_prev_spot_range_bps:
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
        else:
            bet = base_bet

        bet = min(bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        # ── Execute trades ─────────────────────────────────────────────────
        group_pnl = 0.0
        for asset, row, entry_pm, exit_pm in confirming[:4]:
            trade_bet = bet
            if trade_bet < 1.0:
                continue

            # Exit logic
            if s.exit_mode == 'tiered':
                if n_agree >= s.tiered_resolution_threshold:
                    net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
                else:
                    net = pnl_early_exit(direction, entry_pm, exit_pm, trade_bet)

            elif s.exit_mode == 'resolution':
                net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)

            elif s.exit_mode == 'profit_take':
                t5_pm = row.get('pm_yes_t5')
                if t5_pm is not None and is_profitable_at_t5(direction, entry_pm, t5_pm):
                    net = pnl_early_exit(direction, entry_pm, t5_pm, trade_bet)
                else:
                    net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)

            else:  # 'early' (default)
                net = pnl_early_exit(direction, entry_pm, exit_pm, trade_bet)

            won = net > 0
            bankroll += net
            group_pnl += net
            peak_bankroll = max(peak_bankroll, bankroll)

            dd_p = (bankroll - peak_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            max_dd_pct = min(max_dd_pct, dd_p)

            daily_pnl[day_key] += net
            trades.append({
                'asset': asset, 'direction': direction,
                'entry_pm': entry_pm, 'exit_pm': exit_pm,
                'bet': trade_bet, 'net_pnl': net, 'won': won,
                'bankroll': bankroll, 'day': day_key,
            })
            last_bet = trade_bet
            last_won = won

        # ── H: Update group-level loss streak ─────────────────────────────
        if s.max_consec_losses is not None and confirming:
            if group_pnl > 0:
                group_consec_losses = 0
            else:
                group_consec_losses += 1

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
    avg_w = float(np.mean(pnls[pnls > 0])) if wins > 0 else 0
    avg_l = float(np.mean(pnls[pnls <= 0])) if (n - wins) > 0 else 0
    pf = (abs(float(np.sum(pnls[pnls > 0]))) / abs(float(np.sum(pnls[pnls <= 0])))
          if np.sum(pnls <= 0) != 0 else 999)

    max_ls = cs = 0
    for t in trades:
        if not t['won']:
            cs += 1; max_ls = max(max_ls, cs)
        else:
            cs = 0

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
        'max_dd_pct': max_dd_pct * 100,
        'final_bankroll': bankroll,
        'roi_pct': (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100,
        'max_loss_streak': max_ls,
        'dir_stats': dir_stats,
    }


# ─── OUTPUT ────────────────────────────────────────────────────────────────────

def print_table(results, label):
    if not results:
        print(f"\n  {label} — NO RESULTS")
        return
    W = 168
    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"{'='*W}")
    hdr = (f"{'Strategy':<52} {'#':>5} {'WR%':>6} {'PnL':>10} {'AvgW':>7} {'AvgL':>7} "
           f"{'Shrp':>6} {'PF':>5} {'DD%':>7} {'Final$':>9} {'ROI%':>7} {'MLS':>4}")
    print(hdr)
    print('-' * W)
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
    _, time_groups = load_all_data()

    all_times = sorted(time_groups.keys())
    split_idx = int(len(all_times) * TRAIN_RATIO)
    split_time = all_times[split_idx]
    train_groups = {k: v for k, v in time_groups.items() if k < split_time}
    test_groups  = {k: v for k, v in time_groups.items() if k >= split_time}

    n_train_prev = sum(1 for g in train_groups.values()
                       for r in g.values() if r.get('prev_pm_t12_5') is not None)
    print(f"Train: {len(train_groups)} windows ({n_train_prev} with prev data) | "
          f"Test: {len(test_groups)} | Split: {split_time}")
    print()

    strategies = []

    # ═══════════════════════════════════════════════════════════════════════════
    # BASELINES — same as creative_backtest for apples-to-apples comparison
    # ═══════════════════════════════════════════════════════════════════════════
    strategies.append(Strategy("BASE_agree2_flat"))
    strategies.append(Strategy("BASE_agree2_daily100", daily_loss_limit=100))

    # ═══════════════════════════════════════════════════════════════════════════
    # STUDY 2 BEST (E-series) — ported to this framework
    # These ran on older split (Feb 16) with less data.
    # Re-running here on extended data to see if they hold.
    # ═══════════════════════════════════════════════════════════════════════════

    # E3: Higher signal threshold (0.85 instead of 0.80)
    strategies.append(Strategy(
        "E3_THRESH_0.85+daily100",
        prev_thresh=0.85, daily_loss_limit=100,
    ))

    # E4: Volatility regime exclusion
    strategies.append(Strategy(
        "E4_SKIP_EXTREME",
        skip_vol_regimes={'extreme'},
    ))
    strategies.append(Strategy(
        "E4_SKIP_HIGH_EXT",
        skip_vol_regimes={'high', 'extreme'},
    ))
    strategies.append(Strategy(
        "E4_ONLY_NORMAL",
        skip_vol_regimes={'low', 'high', 'extreme'},
    ))
    strategies.append(Strategy(
        "E4_SKIP_EXTREME+daily100",
        skip_vol_regimes={'extreme'}, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "E4_SKIP_HIGH_EXT+daily100",
        skip_vol_regimes={'high', 'extreme'}, daily_loss_limit=100,
    ))

    # E1: Prior momentum filter
    # "Signal was still building in the prior window" = fresh trend to fade
    strategies.append(Strategy(
        "E1_PRIORMOM_0.03+daily100",
        prior_mom_filter=True, prior_mom_min=0.03, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "E1_PRIORMOM_0.05+daily100",
        prior_mom_filter=True, prior_mom_min=0.05, daily_loss_limit=100,
    ))

    # E7: Prior momentum + higher threshold + daily100 (Study 2's #2 overall)
    strategies.append(Strategy(
        "E7_PRIORMOM+THRESH_0.85+daily100",
        prev_thresh=0.85, prior_mom_filter=True, prior_mom_min=0.03,
        daily_loss_limit=100,
    ))

    # E8: All Study 2 filters combined + anti-martingale (Study 2's #1 overall)
    strategies.append(Strategy(
        "E8_ALL+daily100",
        prev_thresh=0.85, prior_mom_filter=True, prior_mom_min=0.03,
        skip_vol_regimes={'extreme'}, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "E8_ALL+ANTIMART",
        prev_thresh=0.85, prior_mom_filter=True, prior_mom_min=0.03,
        skip_vol_regimes={'extreme'},
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # STUDY 3 BEST — re-run for baseline comparison
    # ═══════════════════════════════════════════════════════════════════════════
    strategies.append(Strategy(
        "F8_TIERED_3+daily100",
        exit_mode='tiered', tiered_resolution_threshold=3, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F9_ACCEL+daily100",
        accel_filter=True, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "F5_OFFHOURS_20-8",
        allowed_hours=list(range(20, 24)) + list(range(0, 8)),
    ))
    strategies.append(Strategy(
        "F4_STRENGTH_0.83+daily100",
        min_avg_strength=0.83, daily_loss_limit=100,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # G: VOLATILITY FILTERS ON TOP OF STUDY 3 BEST
    # ═══════════════════════════════════════════════════════════════════════════

    # G1: Regime cutoff — apply to best Study 3 strategies
    for base_name, base_kwargs in [
        ("BASE_daily100",   dict(daily_loss_limit=100)),
        ("F8_TIERED_3+d100", dict(exit_mode='tiered', tiered_resolution_threshold=3, daily_loss_limit=100)),
        ("F9_ACCEL+d100",   dict(accel_filter=True, daily_loss_limit=100)),
        ("F5_OFFHOURS",     dict(allowed_hours=list(range(20, 24)) + list(range(0, 8)))),
    ]:
        strategies.append(Strategy(
            f"G1_SKIP_EXT+{base_name}",
            skip_vol_regimes={'extreme'}, **base_kwargs,
        ))
        strategies.append(Strategy(
            f"G1_SKIP_HI_EXT+{base_name}",
            skip_vol_regimes={'high', 'extreme'}, **base_kwargs,
        ))
        strategies.append(Strategy(
            f"G1_MAX_NORMAL+{base_name}",
            max_vol_regime='normal', **base_kwargs,
        ))

    # G2: Prior spot range filter — the new continuous vol proxy
    # sweet spot: 20-40 bps (59.7% WR). Losers: 80+ bps (46-44% WR).
    for threshold in [60, 80, 100]:
        strategies.append(Strategy(
            f"G2_RANGE_{threshold}+daily100",
            max_prev_spot_range_bps=threshold, daily_loss_limit=100,
        ))
        strategies.append(Strategy(
            f"G2_RANGE_{threshold}+F8_TIERED",
            exit_mode='tiered', tiered_resolution_threshold=3,
            max_prev_spot_range_bps=threshold, daily_loss_limit=100,
        ))
        strategies.append(Strategy(
            f"G2_RANGE_{threshold}+F9_ACCEL",
            accel_filter=True,
            max_prev_spot_range_bps=threshold, daily_loss_limit=100,
        ))

    # G3: Combining best regime filter with best range filter
    strategies.append(Strategy(
        "G3_EXT+RANGE80+daily100",
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "G3_EXT+RANGE80+F8_TIERED",
        exit_mode='tiered', tiered_resolution_threshold=3,
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "G3_EXT+RANGE80+F9_ACCEL",
        accel_filter=True,
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80, daily_loss_limit=100,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # H: LOSS-STREAK CIRCUIT BREAKERS (expected to hurt — testing to quantify)
    # DATA: WR rises after consecutive losses → pausing = throwing away trades
    # ═══════════════════════════════════════════════════════════════════════════
    for cb in [2, 3, 4]:
        strategies.append(Strategy(
            f"H_CB{cb}+daily100",
            max_consec_losses=cb, daily_loss_limit=100,
        ))
        strategies.append(Strategy(
            f"H_CB{cb}+F8_TIERED",
            exit_mode='tiered', tiered_resolution_threshold=3,
            max_consec_losses=cb, daily_loss_limit=100,
        ))

    # ═══════════════════════════════════════════════════════════════════════════
    # BEST COMBINATIONS — vol filter + Study 3 novel exits
    # ═══════════════════════════════════════════════════════════════════════════
    # Tiered exit is our strongest novel finding. Does vol filtering make it better?
    strategies.append(Strategy(
        "COMBO_TIERED+EXT+RANGE80",
        exit_mode='tiered', tiered_resolution_threshold=3,
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "COMBO_ACCEL+EXT+RANGE80",
        accel_filter=True,
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "COMBO_PRIORMOM+TIERED+EXT",
        exit_mode='tiered', tiered_resolution_threshold=3,
        prior_mom_filter=True, prior_mom_min=0.03,
        skip_vol_regimes={'extreme'}, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "COMBO_PRIORMOM+ACCEL+EXT",
        accel_filter=True,
        prior_mom_filter=True, prior_mom_min=0.03,
        skip_vol_regimes={'extreme'}, daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "COMBO_STRENGTH+EXT+RANGE80",
        min_avg_strength=0.83,
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80,
        daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "COMBO_OFFHOURS+EXT+RANGE80",
        allowed_hours=list(range(20, 24)) + list(range(0, 8)),
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80,
    ))
    strategies.append(Strategy(
        "COMBO_ALL_BEST",
        exit_mode='tiered', tiered_resolution_threshold=3,
        accel_filter=True,
        skip_vol_regimes={'extreme'}, max_prev_spot_range_bps=80,
        daily_loss_limit=100,
    ))

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════════════════════════════════════
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
    print_table(test_results,  "TEST SET")

    # ─── Train vs Test ─────────────────────────────────────────────────────────
    W = 170
    print(f"\n{'='*W}")
    print(f"  TRAIN vs TEST — sorted by test P&L")
    print(f"{'='*W}")
    print(
        f"{'Strategy':<52} "
        f"{'TrPnL':>9} {'TrWR':>6} {'TrDD%':>7} "
        f"{'TePnL':>9} {'TeWR':>6} {'TeDD%':>7} "
        f"{'Rob':>4} {'Tr#':>5} {'Te#':>5} "
        f"{'TeAvgW':>7} {'TeAvgL':>7}"
    )
    print('-' * W)

    test_by_name  = {r['name']: r for r in test_results}
    train_by_name = {r['name']: r for r in train_results}

    for name in sorted(test_by_name, key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
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

    # ─── Robust strategies ─────────────────────────────────────────────────────
    print(f"\n{'='*135}")
    print(f"  ROBUST (profitable in BOTH train and test) — sorted by test P&L")
    print(f"{'='*135}")
    robust = [
        (n, train_by_name[n], test_by_name[n])
        for n in sorted(test_by_name, key=lambda n: test_by_name[n]['total_pnl'], reverse=True)
        if n in train_by_name
        and train_by_name[n]['total_pnl'] > 0
        and test_by_name[n]['total_pnl'] > 0
    ]
    if not robust:
        print("  None.")
    for i, (name, tr, te) in enumerate(robust, 1):
        ds = te.get('dir_stats', {})
        b = ds.get('bear', {}); bu = ds.get('bull', {})
        bull_s = f"bull:{bu.get('wr',0)*100:.0f}%/{bu.get('n',0)}" if bu else ""
        bear_s = f"bear:{b.get('wr',0)*100:.0f}%/{b.get('n',0)}" if b else ""
        print(
            f"  {i:>2}. {name:<52} "
            f"Train: ${tr['total_pnl']:>7.0f} ({tr['win_rate']*100:.1f}%)  "
            f"Test: ${te['total_pnl']:>7.0f} ({te['win_rate']*100:.1f}%, {te['max_dd_pct']:.1f}%DD)  "
            f"{bull_s} {bear_s}"
        )

    # ─── Save ──────────────────────────────────────────────────────────────────
    out_path = DATA_DIR / "reports" / "filter_results.json"
    out_path.parent.mkdir(exist_ok=True)
    output = {'split_time': split_time, 'strategies': {}}
    for name in test_by_name:
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr:
            output['strategies'][name] = {
                'train': tr, 'test': te,
                'robust': tr['total_pnl'] > 0 and te['total_pnl'] > 0,
            }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
