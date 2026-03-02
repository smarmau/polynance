#!/usr/bin/env python3
"""
Regime-Adaptive Backtest — 70/30 train/test split

Key insight from round 1: the strategy had a strong edge in train (Jan-mid Feb)
that degraded in test (mid Feb onward). This suggests regime dependence.

Tests:
1. Rolling win-rate circuit breaker (pause when trailing WR < threshold)
2. Anti-recovery sizing (reduce bets during drawdowns)
3. Fee sensitivity analysis (are we overestimating fees?)
4. Equity curve slope detection (pause when slope turns negative)
5. Dynamic daily loss limits (scale with bankroll)
6. Momentum regime filter (only trade when recent performance is positive)
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


def pnl_early_exit(direction, entry_pm, exit_pm, bet, fee_rate=0.01, spread_cost=0.005):
    """Calculate P&L for early exit trade."""
    if direction == 'bull':
        entry_c, exit_c = entry_pm, exit_pm
    else:
        entry_c, exit_c = 1.0 - entry_pm, 1.0 - exit_pm
    if entry_c <= 0.001:
        return 0.0
    n = bet / entry_c
    gross = n * (exit_c - entry_c)
    fees = entry_c * n * fee_rate + exit_c * n * fee_rate
    spread = spread_cost * bet + spread_cost * (n * exit_c)
    return gross - fees - spread


def load_all_data():
    """Load all windows from all asset databases."""
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT w.window_id, w.asset, w.window_start_utc, w.outcome,
                   w.pm_yes_t0, w.pm_yes_t2_5, w.pm_yes_t5, w.pm_yes_t7_5, w.pm_yes_t10, w.pm_yes_t12_5,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5,
                   w.volatility_regime, w.window_time
            FROM windows w
            WHERE w.outcome IS NOT NULL
            ORDER BY w.window_start_utc
        """).fetchall()
        all_rows.extend([dict(r) for r in rows])
        conn.close()

    all_rows.sort(key=lambda x: x['window_start_utc'])

    # Enrich
    asset_windows = defaultdict(list)
    for r in all_rows:
        asset_windows[r['asset']].append(r)

    enriched = []
    for asset, wins in asset_windows.items():
        for i, w in enumerate(wins):
            row = dict(w)
            if row.get('prev_pm_t12_5') is None and i > 0:
                row['prev_pm_t12_5'] = wins[i - 1].get('pm_yes_t12_5')
            try:
                row['hour_utc'] = int(row['window_start_utc'][11:13])
            except Exception:
                row['hour_utc'] = None
            enriched.append(row)

    enriched.sort(key=lambda x: x['window_start_utc'])

    # Cross-asset consensus
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


class RegimeAdaptiveStrategy:
    """Strategy with regime-adaptive risk management."""

    def __init__(self, name, **params):
        self.name = name
        # Core consensus params
        self.prev_thresh = params.get('prev_thresh', 0.80)
        self.bull_thresh = params.get('bull_thresh', 0.50)
        self.bear_thresh = params.get('bear_thresh', 0.50)
        self.min_agree = params.get('min_agree', 3)
        self.entry_time_key = params.get('entry_time', 't0')
        self.exit_time_key = params.get('exit_time', 't12.5')

        # Fee model
        self.fee_rate = params.get('fee_rate', 0.01)
        self.spread_cost = params.get('spread_cost', 0.005)

        # Regime-adaptive params
        self.rolling_wr_window = params.get('rolling_wr_window', 0)  # 0 = disabled
        self.rolling_wr_min = params.get('rolling_wr_min', 0.53)     # pause below this
        self.rolling_wr_resume = params.get('rolling_wr_resume', 0.55)  # resume above this

        # Anti-recovery: reduce bets during drawdowns
        self.anti_recovery = params.get('anti_recovery', False)
        self.anti_recovery_dd_thresh = params.get('anti_recovery_dd_thresh', 0.10)  # 10% DD
        self.anti_recovery_scale = params.get('anti_recovery_scale', 0.50)  # halve bets

        # Daily loss limit (dynamic or fixed)
        self.daily_loss_limit = params.get('daily_loss_limit', None)
        self.daily_loss_pct = params.get('daily_loss_pct', None)  # % of bankroll

        # Equity slope detection
        self.slope_window = params.get('slope_window', 0)  # 0 = disabled
        self.slope_min = params.get('slope_min', 0.0)  # pause when slope < 0

        # Flat betting
        self.recovery_sizing = params.get('recovery_sizing', 'none')

        # Hour filter
        self.skip_hours = params.get('skip_hours', set())

        # Pause after loss
        self.pause_after_loss = params.get('pause_after_loss', 0)

    def _pm_key(self, time_key):
        return {
            't0': 'pm_yes_t0', 't2.5': 'pm_yes_t2_5', 't5': 'pm_yes_t5',
            't7.5': 'pm_yes_t7_5', 't10': 'pm_yes_t10', 't12.5': 'pm_yes_t12_5',
        }.get(time_key, 'pm_yes_t0')


def simulate_regime(strategy, time_groups):
    """Simulate with regime-adaptive risk management."""
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET

    pause_remaining = 0
    trades = []
    daily_pnl = defaultdict(float)
    equity_curve = [INITIAL_BANKROLL]

    # Rolling window tracking
    recent_outcomes = deque(maxlen=strategy.rolling_wr_window if strategy.rolling_wr_window > 0 else 1)
    regime_paused = False  # paused by rolling WR circuit breaker

    # Slope tracking
    slope_equity = deque(maxlen=strategy.slope_window if strategy.slope_window > 0 else 1)
    slope_equity.append(INITIAL_BANKROLL)

    max_dd = 0
    max_dd_pct = 0

    entry_key = strategy._pm_key(strategy.entry_time_key)
    exit_key = strategy._pm_key(strategy.exit_time_key)

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group:
            continue

        if bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        hour = sample_row.get('hour_utc')
        day_key = sample_row['window_start_utc'][:10]

        # Hour filter
        if hour is not None and hour in strategy.skip_hours:
            continue

        # Pause after loss
        if pause_remaining > 0:
            pause_remaining -= 1
            continue

        # Rolling WR circuit breaker
        if strategy.rolling_wr_window > 0 and len(recent_outcomes) >= strategy.rolling_wr_window:
            wr = sum(recent_outcomes) / len(recent_outcomes)
            if regime_paused:
                if wr >= strategy.rolling_wr_resume:
                    regime_paused = False
                else:
                    continue
            else:
                if wr < strategy.rolling_wr_min:
                    regime_paused = True
                    continue

        # Equity slope check
        if strategy.slope_window > 0 and len(slope_equity) >= strategy.slope_window:
            slope_vals = list(slope_equity)
            x = np.arange(len(slope_vals))
            slope = np.polyfit(x, slope_vals, 1)[0]
            if slope < strategy.slope_min:
                continue

        # Daily loss limit
        if strategy.daily_loss_limit is not None:
            if daily_pnl[day_key] < -strategy.daily_loss_limit:
                continue
        if strategy.daily_loss_pct is not None:
            limit = bankroll * strategy.daily_loss_pct
            if daily_pnl[day_key] < -limit:
                continue

        # Phase 1: Previous window consensus
        n_strong_up = 0
        n_strong_down = 0
        for asset, row in group.items():
            p = row.get('prev_pm_t12_5')
            if p is not None:
                if p >= strategy.prev_thresh:
                    n_strong_up += 1
                elif p <= (1.0 - strategy.prev_thresh):
                    n_strong_down += 1

        direction = None
        if n_strong_up >= strategy.min_agree:
            direction = 'bear'
        elif n_strong_down >= strategy.min_agree:
            direction = 'bull'
        if direction is None:
            continue

        # Phase 2: Current window confirmation
        confirming = []
        for asset, row in group.items():
            entry_pm = row.get(entry_key)
            exit_pm = row.get(exit_key)
            if entry_pm is None or exit_pm is None:
                continue
            if direction == 'bear' and entry_pm <= strategy.bear_thresh:
                confirming.append((asset, row, entry_pm, exit_pm))
            elif direction == 'bull' and entry_pm >= strategy.bull_thresh:
                confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < strategy.min_agree:
            continue

        # Determine bet size
        bet = base_bet

        # Anti-recovery: reduce bets during drawdowns
        if strategy.anti_recovery:
            dd_pct = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            if dd_pct > strategy.anti_recovery_dd_thresh:
                bet = bet * strategy.anti_recovery_scale

        bet = min(bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        # Execute trades
        window_had_loss = False
        for asset, row, entry_pm, exit_pm in confirming[:4]:
            net = pnl_early_exit(direction, entry_pm, exit_pm, bet,
                                 strategy.fee_rate, strategy.spread_cost)
            won = net > 0
            bankroll += net
            peak_bankroll = max(peak_bankroll, bankroll)

            dd_dollars = bankroll - peak_bankroll
            dd_pct_val = dd_dollars / peak_bankroll if peak_bankroll > 0 else 0
            max_dd = min(max_dd, dd_dollars)
            max_dd_pct = min(max_dd_pct, dd_pct_val)

            daily_pnl[day_key] += net
            equity_curve.append(bankroll)
            slope_equity.append(bankroll)

            recent_outcomes.append(1 if won else 0)

            trades.append({
                'asset': asset, 'direction': direction,
                'entry_pm': entry_pm, 'exit_pm': exit_pm,
                'bet': bet, 'net_pnl': net, 'won': won,
                'bankroll': bankroll, 'day': day_key,
            })

            if not won:
                window_had_loss = True

        if window_had_loss and strategy.pause_after_loss > 0:
            pause_remaining = strategy.pause_after_loss

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
    pf = abs(float(np.sum(pnls[pnls > 0])) / float(np.sum(pnls[pnls <= 0]))) if np.sum(pnls <= 0) != 0 else 999

    max_loss_streak = 0
    cs = 0
    for t in trades:
        if not t['won']:
            cs += 1
            max_loss_streak = max(max_loss_streak, cs)
        else:
            cs = 0

    losing_days = sum(1 for v in daily_pnl.values() if v < 0)
    total_days = len(daily_pnl)
    worst_day = min(daily_pnl.values()) if daily_pnl else 0
    best_day = max(daily_pnl.values()) if daily_pnl else 0

    return {
        'name': strategy.name,
        'trades': n, 'wins': wins, 'win_rate': win_rate,
        'total_pnl': total_pnl, 'avg_pnl': avg_pnl,
        'sharpe': sharpe, 'profit_factor': pf,
        'max_dd_dollars': max_dd, 'max_dd_pct': max_dd_pct * 100,
        'final_bankroll': bankroll,
        'roi_pct': (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100,
        'max_loss_streak': max_loss_streak,
        'losing_days': losing_days, 'total_days': total_days,
        'worst_day': worst_day, 'best_day': best_day,
    }


def print_results(results, label=""):
    if not results:
        print(f"\n  {label} — NO RESULTS")
        return
    print(f"\n{'='*140}")
    print(f"  {label}")
    print(f"{'='*140}")
    header = (
        f"{'Strategy':<50} {'Trades':>6} {'WR%':>6} {'PnL':>10} {'Avg':>8} "
        f"{'Sharpe':>7} {'PF':>6} {'MaxDD%':>8} {'Final$':>10} {'ROI%':>8} "
        f"{'LDays':>6} {'MaxLS':>6}"
    )
    print(header)
    print('-' * 140)
    for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        ld = f"{r.get('losing_days', '?')}/{r.get('total_days', '?')}"
        print(
            f"{r['name']:<50} "
            f"{r['trades']:>6} {r['win_rate']*100:>5.1f}% "
            f"${r['total_pnl']:>9.2f} ${r['avg_pnl']:>7.2f} "
            f"{r['sharpe']:>7.2f} {r['profit_factor']:>5.2f}x "
            f"{r['max_dd_pct']:>7.1f}% ${r['final_bankroll']:>9.2f} "
            f"{r['roi_pct']:>7.1f}% {ld:>6} {r['max_loss_streak']:>6}"
        )


def main():
    print("Loading data...")
    enriched, time_groups = load_all_data()
    print(f"Loaded {len(enriched)} windows, {len(time_groups)} unique time slots")

    # Temporal split
    all_times = sorted(time_groups.keys())
    split_idx = int(len(all_times) * TRAIN_RATIO)
    split_time = all_times[split_idx]

    train_groups = {k: v for k, v in time_groups.items() if k < split_time}
    test_groups = {k: v for k, v in time_groups.items() if k >= split_time}

    print(f"Train: {len(train_groups)} windows, Test: {len(test_groups)} windows")
    print(f"Split at: {split_time}")

    strategies = []

    # === SECTION 1: Fee Sensitivity ===
    for fee_rate in [0.001, 0.005, 0.01, 0.02]:
        strategies.append(RegimeAdaptiveStrategy(
            f"FEE_{fee_rate:.3f}",
            fee_rate=fee_rate, min_agree=3,
        ))

    # === SECTION 2: Rolling WR Circuit Breaker ===
    for window in [30, 50, 75, 100]:
        for min_wr in [0.52, 0.53, 0.55]:
            strategies.append(RegimeAdaptiveStrategy(
                f"ROLL_WR_{window}w_{min_wr:.2f}min",
                rolling_wr_window=window, rolling_wr_min=min_wr,
                rolling_wr_resume=min_wr + 0.02,
                min_agree=3,
            ))

    # === SECTION 3: Anti-Recovery (reduce bets during drawdown) ===
    for dd_thresh in [0.05, 0.10, 0.15, 0.20]:
        for scale in [0.25, 0.50, 0.75]:
            strategies.append(RegimeAdaptiveStrategy(
                f"ANTI_RECOV_{dd_thresh:.0%}dd_{scale:.0%}scale",
                anti_recovery=True,
                anti_recovery_dd_thresh=dd_thresh,
                anti_recovery_scale=scale,
                min_agree=3,
            ))

    # === SECTION 4: Equity Slope Detection ===
    for slope_win in [20, 40, 60]:
        strategies.append(RegimeAdaptiveStrategy(
            f"SLOPE_{slope_win}w",
            slope_window=slope_win, slope_min=0.0,
            min_agree=3,
        ))

    # === SECTION 5: Dynamic Daily Loss Limit (% of bankroll) ===
    for pct in [0.03, 0.05, 0.08, 0.10]:
        strategies.append(RegimeAdaptiveStrategy(
            f"DYN_DAILY_{pct:.0%}",
            daily_loss_pct=pct, min_agree=3,
        ))

    # === SECTION 6: Combinations ===

    # Best from round 1: daily limit + low fee
    strategies.append(RegimeAdaptiveStrategy(
        "COMBO_daily100+lowfee",
        daily_loss_limit=100, fee_rate=0.001, min_agree=3,
    ))

    # Rolling WR + daily limit
    strategies.append(RegimeAdaptiveStrategy(
        "COMBO_roll50+daily100",
        rolling_wr_window=50, rolling_wr_min=0.53, rolling_wr_resume=0.55,
        daily_loss_limit=100, min_agree=3,
    ))

    # Anti-recovery + rolling WR
    strategies.append(RegimeAdaptiveStrategy(
        "COMBO_antirecov+roll50",
        anti_recovery=True, anti_recovery_dd_thresh=0.10, anti_recovery_scale=0.50,
        rolling_wr_window=50, rolling_wr_min=0.53, rolling_wr_resume=0.55,
        min_agree=3,
    ))

    # Kitchen sink: anti-recovery + rolling WR + daily limit
    strategies.append(RegimeAdaptiveStrategy(
        "COMBO_full_adaptive",
        anti_recovery=True, anti_recovery_dd_thresh=0.10, anti_recovery_scale=0.50,
        rolling_wr_window=50, rolling_wr_min=0.53, rolling_wr_resume=0.55,
        daily_loss_limit=100,
        min_agree=3,
    ))

    # Agree=2 with protections (best from round 1 was agree=2)
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2_flat",
        min_agree=2,
    ))
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2+daily100",
        min_agree=2, daily_loss_limit=100,
    ))
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2+roll50+daily100",
        min_agree=2,
        rolling_wr_window=50, rolling_wr_min=0.53, rolling_wr_resume=0.55,
        daily_loss_limit=100,
    ))
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2+antirecov10+daily100",
        min_agree=2,
        anti_recovery=True, anti_recovery_dd_thresh=0.10, anti_recovery_scale=0.50,
        daily_loss_limit=100,
    ))
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2+full_adaptive",
        min_agree=2,
        anti_recovery=True, anti_recovery_dd_thresh=0.10, anti_recovery_scale=0.50,
        rolling_wr_window=50, rolling_wr_min=0.53, rolling_wr_resume=0.55,
        daily_loss_limit=100,
    ))

    # Agree=2 with lower fee
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2+lowfee",
        min_agree=2, fee_rate=0.001,
    ))

    # === SECTION 7: Pause after loss combos ===
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE2+pause1+daily100",
        min_agree=2, pause_after_loss=1, daily_loss_limit=100,
    ))
    strategies.append(RegimeAdaptiveStrategy(
        "AGREE3+pause1+daily100",
        min_agree=3, pause_after_loss=1, daily_loss_limit=100,
    ))

    # === SECTION 8: Baseline reference ===
    strategies.append(RegimeAdaptiveStrategy(
        "BASELINE_agree3",
        min_agree=3,
    ))

    print(f"\nRunning {len(strategies)} strategies...")

    train_results = []
    test_results = []
    for s in strategies:
        tr = simulate_regime(s, train_groups)
        te = simulate_regime(s, test_groups)
        if tr:
            train_results.append(tr)
        if te:
            test_results.append(te)

    print_results(train_results, "TRAIN SET")
    print_results(test_results, "TEST SET")

    # Comparison table
    print(f"\n{'='*150}")
    print(f"  TRAIN vs TEST COMPARISON (sorted by test P&L)")
    print(f"{'='*150}")
    print(
        f"{'Strategy':<50} "
        f"{'Tr PnL':>9} {'Tr WR':>6} {'Tr DD%':>7} "
        f"{'Te PnL':>9} {'Te WR':>6} {'Te DD%':>7} "
        f"{'Robust':>7} {'Tr#':>5} {'Te#':>5}"
    )
    print('-' * 150)

    test_by_name = {r['name']: r for r in test_results}
    train_by_name = {r['name']: r for r in train_results}

    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr is None:
            continue
        robust = "YES" if tr['total_pnl'] > 0 and te['total_pnl'] > 0 else "no"
        print(
            f"{name:<50} "
            f"${tr['total_pnl']:>8.0f} {tr['win_rate']*100:>5.1f}% {tr['max_dd_pct']:>6.1f}% "
            f"${te['total_pnl']:>8.0f} {te['win_rate']*100:>5.1f}% {te['max_dd_pct']:>6.1f}% "
            f"{robust:>7} {tr['trades']:>5} {te['trades']:>5}"
        )

    # Highlight robust strategies
    print(f"\n{'='*100}")
    print(f"  ROBUST STRATEGIES (profitable in BOTH train and test)")
    print(f"{'='*100}")
    robust_count = 0
    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr and tr['total_pnl'] > 0 and te['total_pnl'] > 0:
            robust_count += 1
            print(
                f"  {robust_count:>2}. {name:<48} "
                f"Train: ${tr['total_pnl']:>8.0f} ({tr['win_rate']*100:.1f}% WR, {tr['max_dd_pct']:.1f}% DD)  "
                f"Test: ${te['total_pnl']:>8.0f} ({te['win_rate']*100:.1f}% WR, {te['max_dd_pct']:.1f}% DD)"
            )
    if robust_count == 0:
        print("  (none)")

    # Fee sensitivity summary
    print(f"\n{'='*100}")
    print(f"  FEE SENSITIVITY (impact of fee model on test P&L)")
    print(f"{'='*100}")
    for name in sorted(test_by_name.keys()):
        if name.startswith("FEE_"):
            tr = train_by_name.get(name)
            te = test_by_name[name]
            if tr:
                print(
                    f"  {name:<20} Train: ${tr['total_pnl']:>9.2f} ({tr['win_rate']*100:.1f}% WR)  "
                    f"Test: ${te['total_pnl']:>9.2f} ({te['win_rate']*100:.1f}% WR)  "
                    f"Edge lost: {'YES' if te['total_pnl'] < 0 else 'NO'}"
                )


if __name__ == '__main__':
    main()
