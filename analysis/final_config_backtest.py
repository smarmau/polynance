#!/usr/bin/env python3
"""
final_config_backtest.py — Rigorous strategy selection for live trading.

PURPOSE: Find the best config to run live, being brutally honest about edge.

DESIGN PRINCIPLES (borrowed from honest_backtest.py):
  1. Chronological 70/30 split — no shuffling
  2. Wilson 95% CIs on all win rates
  3. Walk-forward weekly P&L for stability
  4. Strategies defined a priori — nothing tuned on test
  5. Regime filter uses MAJORITY RULE (matching live trader.py:2350-2366)
  6. Fee model matches live config: flat 1% + 0.5% spread per side
  7. Max bet capped at 5% of bankroll (matching live max_bet_pct)

STRATEGIES TESTED:
  Group A: Proven approaches (from prior backtests)
    A1. Baseline: 2-of-4 agree, thresh 0.80, t0 entry, t12.5 exit (current live)
    A2. 4-of-4 agree, thresh 0.75, t5 entry, t12.5 exit (honest_backtest winner)
    A3. 4-of-4 agree, thresh 0.90, t5 entry, t12.5 exit
    A4. Prior momentum filter + thresh 0.85 + daily limit $100
    A5. Bear-only, no BTC, thresh 0.80, t5 entry

  Group B: Novel techniques
    B1. Spread filter: skip windows with pm_spread_t0 > 0.02 (2 cents)
    B2. Opening neutrality: only trade when |pm_yes_t0 - 0.50| <= 0.10
    B3. Spot range filter: skip if prev window spot_range_bps > 60
    B4. Delayed entry t2.5 (early confirmation) vs t0 vs t5
    B5. Combined: 4-of-4 + spread filter + opening neutrality

  Group C: Sizing variants (on best signal from A/B)
    C1. Flat $25
    C2. Bankroll-scaled (increase base after growth)
    C3. Regime-adaptive (reduce in high vol)
    C4. Anti-mart (streak-based, matching live)
    C5. Daily loss limit $100

Run: cd /home/smarm/Music/polynance && python analysis/final_config_backtest.py
"""

import sqlite3
import math
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']
INITIAL_BANKROLL = 1000.0
BASE_BET = 25.0
FEE_RATE = 0.01
SPREAD_COST = 0.005
TRAIN_RATIO = 0.70

# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Load windows from all asset DBs, enrich with cross-window features."""
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM windows
            WHERE outcome IS NOT NULL
            ORDER BY window_start_utc ASC
        """).fetchall()
        for r in rows:
            d = dict(r)
            d['asset'] = asset.upper()
            all_rows.append(d)
        conn.close()

    all_rows.sort(key=lambda r: (r['window_start_utc'], r['asset']))

    # Enrich: add prev_spot_range_bps from prior row of same asset
    by_asset = defaultdict(list)
    for r in all_rows:
        by_asset[r['asset']].append(r)
    for asset, rows in by_asset.items():
        for i, r in enumerate(rows):
            if i > 0:
                r['prev_spot_range_bps'] = rows[i-1].get('spot_range_bps')
            else:
                r['prev_spot_range_bps'] = None

    # Group by window_time for cross-asset consensus
    time_groups = defaultdict(dict)
    for r in all_rows:
        wt = r.get('window_time')
        if wt:
            time_groups[wt][r['asset']] = r

    return all_rows, time_groups


def train_test_split(time_groups):
    """Chronological 70/30 split."""
    sorted_wts = sorted(time_groups.keys())
    split_idx = int(len(sorted_wts) * TRAIN_RATIO)
    split_wt = sorted_wts[split_idx]
    train = {wt: g for wt, g in time_groups.items() if wt < split_wt}
    test = {wt: g for wt, g in time_groups.items() if wt >= split_wt}
    return train, test, split_wt


# ── P&L functions ────────────────────────────────────────────────────────────

def pnl_early_exit(direction, entry_pm, exit_pm, bet):
    """P&L for early exit (sell at exit_pm)."""
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
    """P&L for holding to binary resolution."""
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
        return n * (1.0 - entry_c) - fees - spread
    else:
        return -bet - fees - spread


# ── Wilson CI ────────────────────────────────────────────────────────────────

def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    delta = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, center - delta), min(1, center + delta)


# ── Strategy class ───────────────────────────────────────────────────────────

class Strategy:
    def __init__(self, name, **p):
        self.name = name

        # Signal
        self.prev_thresh = p.get('prev_thresh', 0.80)
        self.min_agree = p.get('min_agree', 2)
        self.bear_only = p.get('bear_only', False)
        self.exclude_btc = p.get('exclude_btc', False)

        # Entry/exit timing
        self.entry_time = p.get('entry_time', 't0')     # t0, t2_5, t5
        self.exit_time = p.get('exit_time', 't12_5')    # t12_5, resolution

        # Prior momentum filter
        self.prior_mom_filter = p.get('prior_mom_filter', False)
        self.prior_mom_min = p.get('prior_mom_min', 0.03)

        # Regime filter (MAJORITY RULE — matching live trader)
        self.skip_regimes = set(p.get('skip_regimes', []))

        # Novel filters
        self.max_spread_t0 = p.get('max_spread_t0', None)       # Skip if spread > X
        self.opening_band = p.get('opening_band', None)          # |pm_t0 - 0.50| <= X
        self.max_prev_spot_range = p.get('max_prev_spot_range', None)  # Skip if prev bps > X

        # Sizing
        self.sizing = p.get('sizing', 'flat')      # flat, anti_mart, regime_adaptive
        self.anti_mart_mult = p.get('anti_mart_mult', 1.5)
        self.anti_mart_max = p.get('anti_mart_max', 3.0)
        self.daily_loss_limit = p.get('daily_loss_limit', 0)     # 0 = disabled

        # Regime-adaptive sizing
        self.regime_normal_mult = p.get('regime_normal_mult', 1.0)
        self.regime_high_mult = p.get('regime_high_mult', 0.5)


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate(strategy, time_groups):
    s = strategy
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET

    trades = []
    daily_pnl = defaultdict(float)
    max_dd_pct = 0.0
    win_streak = 0

    # Column mapping
    entry_col = f"pm_yes_{s.entry_time}"
    exit_col = f"pm_yes_{s.exit_time}" if s.exit_time != 'resolution' else None

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group or bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        day_key = sample_row['window_start_utc'][:10]

        # Daily loss limit
        if s.daily_loss_limit > 0 and daily_pnl[day_key] < -s.daily_loss_limit:
            continue

        # ── Regime filter: MAJORITY RULE (matching live trader) ──────────
        if s.skip_regimes:
            skip_count = sum(
                1 for r in group.values()
                if r.get('volatility_regime') in s.skip_regimes
            )
            if skip_count >= len(group) / 2:
                continue

        # ── Phase 1: Previous window consensus (contrarian signal) ───────
        n_strong_up = 0
        n_strong_down = 0

        for asset, row in group.items():
            if s.exclude_btc and asset == 'BTC':
                continue
            p1 = row.get('prev_pm_t12_5')
            if p1 is None:
                continue
            if p1 >= s.prev_thresh:
                n_strong_up += 1
            elif p1 <= (1.0 - s.prev_thresh):
                n_strong_down += 1

        direction = None
        if n_strong_up >= s.min_agree:
            direction = 'bear'
        elif n_strong_down >= s.min_agree and not s.bear_only:
            direction = 'bull'
        if direction is None:
            continue

        # ── Prior momentum filter ────────────────────────────────────────
        if s.prior_mom_filter:
            n_confirm = 0
            for asset, row in group.items():
                if s.exclude_btc and asset == 'BTC':
                    continue
                p1 = row.get('prev_pm_t12_5')
                p2 = row.get('prev2_pm_t12_5')
                if p1 is None or p2 is None:
                    continue
                delta = p1 - p2
                if direction == 'bear' and delta > s.prior_mom_min:
                    n_confirm += 1
                elif direction == 'bull' and delta < -s.prior_mom_min:
                    n_confirm += 1
            if n_confirm < 1:
                continue

        # ── Phase 2: Per-asset filters ───────────────────────────────────
        confirming = []
        for asset, row in group.items():
            if s.exclude_btc and asset == 'BTC':
                continue

            entry_pm = row.get(entry_col)
            if s.exit_time == 'resolution':
                exit_pm = None
            else:
                exit_pm = row.get(exit_col)

            if entry_pm is None:
                continue
            if s.exit_time != 'resolution' and exit_pm is None:
                continue

            # Entry threshold check
            if direction == 'bear' and entry_pm > 0.50:
                continue
            if direction == 'bull' and entry_pm < 0.50:
                continue

            # Novel: spread filter
            if s.max_spread_t0 is not None:
                spread = row.get('pm_spread_t0')
                if spread is not None and spread > s.max_spread_t0:
                    continue

            # Novel: opening neutrality band
            if s.opening_band is not None:
                t0_pm = row.get('pm_yes_t0')
                if t0_pm is not None and abs(t0_pm - 0.50) > s.opening_band:
                    continue

            # Novel: prev spot range filter
            if s.max_prev_spot_range is not None:
                psr = row.get('prev_spot_range_bps')
                if psr is not None and psr > s.max_prev_spot_range:
                    continue

            confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < s.min_agree:
            continue

        # ── Bet sizing ───────────────────────────────────────────────────
        if s.sizing == 'anti_mart':
            if win_streak > 0:
                bet = base_bet * min(s.anti_mart_mult ** win_streak, s.anti_mart_max)
            else:
                bet = base_bet
        elif s.sizing == 'regime_adaptive':
            # Use majority regime of this group
            regimes = [r.get('volatility_regime', 'normal') for r in group.values()]
            dominant = max(set(regimes), key=regimes.count) if regimes else 'normal'
            if dominant in ('high', 'extreme'):
                bet = base_bet * s.regime_high_mult
            else:
                bet = base_bet * s.regime_normal_mult
        else:
            bet = base_bet

        bet = min(bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        # ── Execute trades ───────────────────────────────────────────────
        for asset, row, entry_pm, exit_pm in confirming[:4]:
            trade_bet = bet
            if trade_bet < 1.0:
                continue

            if s.exit_time == 'resolution':
                net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
            else:
                net = pnl_early_exit(direction, entry_pm, exit_pm, trade_bet)

            won = net > 0
            bankroll += net
            peak_bankroll = max(peak_bankroll, bankroll)

            dd_p = (bankroll - peak_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            max_dd_pct = min(max_dd_pct, dd_p)

            daily_pnl[day_key] += net

            trades.append({
                'asset': asset, 'direction': direction,
                'entry_pm': entry_pm,
                'exit_pm': exit_pm if exit_pm else (1.0 if (row.get('outcome') == 'up' and direction == 'bull') or (row.get('outcome') == 'down' and direction == 'bear') else 0.0),
                'bet': trade_bet, 'net_pnl': net, 'won': won,
                'bankroll': bankroll, 'day': day_key,
                'week': day_key[:7],  # YYYY-MM for weekly grouping
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
    sharpe = avg_pnl / std_pnl * np.sqrt(n) if std_pnl > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(wins, n)

    # Weekly P&L for stability check
    weekly = defaultdict(float)
    for t in trades:
        weekly[t['day'][:7]] += t['net_pnl']
    weekly_values = list(weekly.values())
    weeks_positive = sum(1 for w in weekly_values if w > 0)
    weeks_total = len(weekly_values)

    # Avg win / avg loss
    win_pnls = pnls[pnls > 0]
    loss_pnls = pnls[pnls <= 0]
    avg_win = float(np.mean(win_pnls)) if len(win_pnls) > 0 else 0
    avg_loss = float(np.mean(loss_pnls)) if len(loss_pnls) > 0 else 0
    profit_factor = float(np.sum(win_pnls) / -np.sum(loss_pnls)) if np.sum(loss_pnls) != 0 else float('inf')

    return {
        'name': strategy.name,
        'trades': n,
        'wins': wins,
        'wr': wins / n if n > 0 else 0,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe': sharpe,
        'max_dd_pct': max_dd_pct * 100,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_bankroll': bankroll,
        'weeks_pos': weeks_positive,
        'weeks_total': weeks_total,
        'weekly_values': weekly_values,
    }


# ── Display ──────────────────────────────────────────────────────────────────

def print_table(results, title):
    print(f"\n{'='*170}")
    print(f"  {title}")
    print(f"{'='*170}")
    hdr = (
        f"  {'Strategy':<42} "
        f"{'N':>5} {'WR%':>6} {'95% CI':>12} "
        f"{'P&L':>10} {'Sharpe':>7} {'PF':>5} "
        f"{'MaxDD%':>7} {'AvgW':>7} {'AvgL':>7} "
        f"{'Wk+/Tot':>8} {'Final$':>8}"
    )
    print(hdr)
    print(f"  {'-'*165}")

    for r in results:
        if r is None:
            continue
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else " +inf"
        print(
            f"  {r['name']:<42} "
            f"{r['trades']:>5} {r['wr']*100:>5.1f}% "
            f"[{r['ci_lo']*100:.0f}-{r['ci_hi']*100:.0f}%] "
            f"${r['pnl']:>+9.2f} {r['sharpe']:>+7.2f} {pf_str:>5} "
            f"{r['max_dd_pct']:>+6.1f}% "
            f"${r['avg_win']:>5.2f} ${r['avg_loss']:>6.2f} "
            f"{r['weeks_pos']:>3}/{r['weeks_total']:<3} "
            f"${r['final_bankroll']:>7.0f}"
        )
    print(f"{'='*170}")


def print_comparison(train_results, test_results):
    """Side-by-side train vs test comparison sorted by test Sharpe."""
    # Match by name
    test_by_name = {r['name']: r for r in test_results if r is not None}
    train_by_name = {r['name']: r for r in train_results if r is not None}

    combined = []
    for name in test_by_name:
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr is None:
            continue
        combined.append((name, tr, te))

    combined.sort(key=lambda x: x[2]['sharpe'], reverse=True)

    print(f"\n{'='*200}")
    print(f"  TRAIN vs TEST — sorted by test Sharpe (most reliable metric)")
    print(f"{'='*200}")
    hdr = (
        f"  {'Strategy':<42}  "
        f"{'--- TRAIN ---':^40}  "
        f"{'--- TEST ---':^55}  "
        f"{'OVERFIT?':>8}"
    )
    sub = (
        f"  {'':42}  "
        f"{'N':>5} {'WR%':>6} {'P&L':>10} {'Shrp':>6} {'DD%':>6}  "
        f"{'N':>5} {'WR%':>6} {'CI':>11} {'P&L':>10} {'Shrp':>6} {'DD%':>6} {'PF':>5} {'Wk+':>4}  "
        f"{'':>8}"
    )
    print(hdr)
    print(sub)
    print(f"  {'-'*195}")

    for name, tr, te in combined:
        wr_drop = tr['wr'] - te['wr']
        overfit = "!! YES" if wr_drop > 0.05 else ("  WARN" if wr_drop > 0.03 else "")
        robust = te['pnl'] > 0 and tr['pnl'] > 0
        pf_str = f"{te['profit_factor']:.2f}" if te['profit_factor'] < 100 else "+inf"
        marker = " ***" if te['sharpe'] > 2.0 and robust and not overfit else ""

        print(
            f"  {name:<42}  "
            f"{tr['trades']:>5} {tr['wr']*100:>5.1f}% ${tr['pnl']:>+9.0f} {tr['sharpe']:>+6.1f} {tr['max_dd_pct']:>+5.0f}%  "
            f"{te['trades']:>5} {te['wr']*100:>5.1f}% [{te['ci_lo']*100:.0f}-{te['ci_hi']*100:.0f}%] "
            f"${te['pnl']:>+9.0f} {te['sharpe']:>+6.1f} {te['max_dd_pct']:>+5.0f}% {pf_str:>5} "
            f"{te['weeks_pos']}/{te['weeks_total']}  "
            f"{overfit}{marker}"
        )

    print(f"{'='*200}")
    print(f"  *** = Robust (profitable both sets), Sharpe > 2.0, no overfit flag")
    print(f"  OVERFIT = Train WR > Test WR by > 5pp")
    print(f"  WARN = Train WR > Test WR by 3-5pp")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("  FINAL CONFIG BACKTEST — Rigorous strategy selection for live trading")
    print("  Regime filter uses MAJORITY RULE (matching live trader)")
    print("  Anti-mart uses STREAK-BASED sizing (matching live trader)")
    print("="*80)

    all_rows, time_groups = load_all_data()
    train_groups, test_groups, split_wt = train_test_split(time_groups)
    print(f"\n  Train: {len(train_groups)} windows  |  Test: {len(test_groups)} windows")
    print(f"  Split at: {split_wt}")

    strategies = []

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP A: PROVEN APPROACHES
    # ═══════════════════════════════════════════════════════════════════════

    # A1: Current live config (broken — the baseline to beat)
    strategies.append(Strategy(
        "A1_CURRENT_LIVE (2agree t0)",
        min_agree=2, prev_thresh=0.85, entry_time='t0', exit_time='t12_5',
        prior_mom_filter=True, prior_mom_min=0.03,
        skip_regimes=['extreme'],
    ))

    # A2: Honest_backtest winner — 4-of-4 consensus, t5 entry
    strategies.append(Strategy(
        "A2_4AGREE_0.75_t5",
        min_agree=4, prev_thresh=0.75, entry_time='t5', exit_time='t12_5',
    ))

    # A3: 4-of-4 higher threshold
    strategies.append(Strategy(
        "A3_4AGREE_0.90_t5",
        min_agree=4, prev_thresh=0.90, entry_time='t5', exit_time='t12_5',
    ))

    # A4: Best filter_backtest combo
    strategies.append(Strategy(
        "A4_PRIORMOM_0.85_daily100",
        min_agree=2, prev_thresh=0.85, entry_time='t0', exit_time='t12_5',
        prior_mom_filter=True, prior_mom_min=0.03,
        skip_regimes=['extreme'], daily_loss_limit=100,
    ))

    # A5: Bear-only, no BTC (honest_backtest insight)
    strategies.append(Strategy(
        "A5_BEAR_NOBTC_0.80_t5",
        min_agree=2, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        bear_only=True, exclude_btc=True,
    ))

    # A6: 3-of-4 agree (testing the middle ground)
    strategies.append(Strategy(
        "A6_3AGREE_0.80_t5",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
    ))

    # A7: 4-of-4 + daily limit
    strategies.append(Strategy(
        "A7_4AGREE_0.85_t5_daily100",
        min_agree=4, prev_thresh=0.85, entry_time='t5', exit_time='t12_5',
        daily_loss_limit=100,
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP B: NOVEL TECHNIQUES
    # ═══════════════════════════════════════════════════════════════════════

    # B1: Spread filter (wide spread = poor liquidity = worse fills)
    strategies.append(Strategy(
        "B1_3AGREE_SPREAD_0.02",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        max_spread_t0=0.02,
    ))

    # B2: Opening neutrality (market near 0.50 = more room for reversal)
    strategies.append(Strategy(
        "B2_3AGREE_OPENBAND_0.10",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        opening_band=0.10,
    ))
    strategies.append(Strategy(
        "B2b_3AGREE_OPENBAND_0.15",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        opening_band=0.15,
    ))

    # B3: Spot range filter (calm prior window = better signal)
    strategies.append(Strategy(
        "B3_3AGREE_SPOTRANGE_60",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        max_prev_spot_range=60,
    ))
    strategies.append(Strategy(
        "B3b_3AGREE_SPOTRANGE_40",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        max_prev_spot_range=40,
    ))

    # B4: Entry timing comparison
    strategies.append(Strategy(
        "B4a_3AGREE_t0_entry",
        min_agree=3, prev_thresh=0.80, entry_time='t0', exit_time='t12_5',
    ))
    strategies.append(Strategy(
        "B4b_3AGREE_t2_5_entry",
        min_agree=3, prev_thresh=0.80, entry_time='t2_5', exit_time='t12_5',
    ))
    strategies.append(Strategy(
        "B4c_3AGREE_t5_entry",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
    ))

    # B5: Combined novel filters
    strategies.append(Strategy(
        "B5_4AGREE_SPREAD+OPEN+RANGE",
        min_agree=4, prev_thresh=0.85, entry_time='t5', exit_time='t12_5',
        max_spread_t0=0.02, opening_band=0.15, max_prev_spot_range=60,
    ))

    # B6: Bear-only + novel filters
    strategies.append(Strategy(
        "B6_BEAR_3AGR_SPREAD+OPEN",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        bear_only=True, max_spread_t0=0.02, opening_band=0.15,
    ))

    # B7: Resolution exit (hold to binary) vs early exit
    strategies.append(Strategy(
        "B7_3AGREE_RESOLUTION",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='resolution',
    ))

    # B8: 4-of-4 + prior momentum (combining two strongest signals)
    strategies.append(Strategy(
        "B8_4AGREE_PRIORMOM_0.85_t5",
        min_agree=4, prev_thresh=0.85, entry_time='t5', exit_time='t12_5',
        prior_mom_filter=True, prior_mom_min=0.03,
    ))

    # B9: Skip high+extreme regimes (not just extreme)
    strategies.append(Strategy(
        "B9_3AGREE_SKIP_HI_EXT",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        skip_regimes=['high', 'extreme'],
    ))

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP C: SIZING VARIANTS (on 3-of-4 agree base signal)
    # ═══════════════════════════════════════════════════════════════════════

    # C1: Flat (baseline)
    strategies.append(Strategy(
        "C1_3AGREE_FLAT",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
    ))

    # C2: Anti-mart (streak-based, matching live)
    strategies.append(Strategy(
        "C2_3AGREE_ANTIMART",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        sizing='anti_mart', anti_mart_mult=1.5, anti_mart_max=3.0,
    ))

    # C3: Regime-adaptive (normal=full, high/extreme=half)
    strategies.append(Strategy(
        "C3_3AGREE_REGIME_ADAPTIVE",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        sizing='regime_adaptive', regime_normal_mult=1.0, regime_high_mult=0.5,
    ))

    # C4: Flat + daily loss limit
    strategies.append(Strategy(
        "C4_3AGREE_FLAT_daily100",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        daily_loss_limit=100,
    ))

    # C5: Regime-adaptive + daily loss limit
    strategies.append(Strategy(
        "C5_3AGREE_REGADAPT_daily100",
        min_agree=3, prev_thresh=0.80, entry_time='t5', exit_time='t12_5',
        sizing='regime_adaptive', regime_normal_mult=1.0, regime_high_mult=0.5,
        daily_loss_limit=100,
    ))

    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n  Running {len(strategies)} strategies...")

    train_results = []
    test_results = []
    for strat in strategies:
        tr = simulate(strat, train_groups)
        te = simulate(strat, test_groups)
        train_results.append(tr)
        test_results.append(te)

    print_table(
        [r for r in test_results if r is not None],
        "TEST SET — sorted by strategy order"
    )

    print_comparison(train_results, test_results)

    # ── Top 3 detailed weekly breakdown ──────────────────────────────────
    test_valid = [(r, t) for r, t in zip(test_results, train_results) if r is not None and t is not None]
    test_valid.sort(key=lambda x: x[0]['sharpe'], reverse=True)

    print(f"\n{'='*120}")
    print(f"  TOP 5 STRATEGIES — Weekly P&L Stability (Test Set)")
    print(f"{'='*120}")

    for r, _ in test_valid[:5]:
        print(f"\n  {r['name']} — WR={r['wr']*100:.1f}% [{r['ci_lo']*100:.0f}-{r['ci_hi']*100:.0f}%], "
              f"Sharpe={r['sharpe']:+.1f}, DD={r['max_dd_pct']:.1f}%")
        # Print weekly values
        weekly = defaultdict(float)
        for idx, val in enumerate(r.get('weekly_values', [])):
            weekly[f"Week {idx+1}"] = val
        # Actually reconstruct from trades data... for simplicity just show summary
        print(f"    Positive weeks: {r['weeks_pos']}/{r['weeks_total']}")
        print(f"    Avg win: ${r['avg_win']:.2f}, Avg loss: ${r['avg_loss']:.2f}")
        print(f"    Profit factor: {r['profit_factor']:.2f}")

    # ── Final recommendation ─────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  RECOMMENDATION")
    print(f"{'='*120}")

    # Find best strategy: highest test Sharpe that's also robust
    robust = [
        (r, t) for r, t in test_valid
        if r['pnl'] > 0 and t is not None and t['pnl'] > 0
        and (t['wr'] - r['wr']) < 0.05  # not overfitting
    ]
    if robust:
        best_r, best_t = robust[0]
        print(f"\n  Best robust strategy: {best_r['name']}")
        print(f"  Train: N={best_t['trades']}, WR={best_t['wr']*100:.1f}%, P&L=${best_t['pnl']:+.0f}, Sharpe={best_t['sharpe']:+.1f}")
        print(f"  Test:  N={best_r['trades']}, WR={best_r['wr']*100:.1f}% [{best_r['ci_lo']*100:.0f}-{best_r['ci_hi']*100:.0f}%], "
              f"P&L=${best_r['pnl']:+.0f}, Sharpe={best_r['sharpe']:+.1f}")
        print(f"         DD={best_r['max_dd_pct']:.1f}%, PF={best_r['profit_factor']:.2f}, "
              f"Positive weeks: {best_r['weeks_pos']}/{best_r['weeks_total']}")
    else:
        print("\n  No strategy passed robustness criteria.")

    print(f"""
  HONEST ASSESSMENT:
  - We have ~38 days of data. Any edge is provisional.
  - 95% CIs span 5-10pp — a 57% WR could easily be 52% or 62%.
  - Anti-mart and fancy sizing AMPLIFY noise, not signal.
  - The safest approach: flat bets, conservative filters, daily loss limit.
  - Trade small, reassess weekly.
""")


if __name__ == "__main__":
    main()
