#!/usr/bin/env python3
"""
Advanced Exits, Entry Filters, and Bet Sizing Backtest — 70/30 split

New ideas based on data analysis:

ENTRIES:
1. Momentum confirmation: pm_momentum_0_to_5 must confirm direction (79% WR vs 28%)
2. t0 sweet spot: bear enters only when t0 in [0.40-0.50], bull when [0.50-0.60]
3. Asymmetric: bear-only (stronger edge) or bear-heavy sizing
4. t5 entry with momentum confirmation (wait for signal, then enter)

EXITS:
5. Stop-loss at t5: exit if position underwater by >X cents
6. Stop-loss at t7.5: same but later check
7. Trailing stop: exit if position was profitable but reverses
8. Hold to resolution (binary payout) instead of early exit

SIZING:
9. Fractional Kelly (f*/4) using rolling WR
10. Anti-martingale 1.5x (up on win, down on loss)
11. Asymmetric sizing: 1.5x on bear, 0.75x on bull
12. Confidence-scaled: larger bets when momentum is strong
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


def pnl_early_exit(direction, entry_pm, exit_pm, bet):
    """P&L for selling position at exit_pm."""
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
        gross = n * (1.0 - entry_c)
        return gross - fees - spread
    else:
        return -bet - fees - spread


def load_all_data():
    """Load and enrich all windows."""
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
                   w.pm_price_momentum_0_to_5, w.pm_price_momentum_5_to_10,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5,
                   w.volatility_regime, w.window_time,
                   w.spot_change_bps
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

    # Cross-asset grouping
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


class Strategy:
    def __init__(self, name, **p):
        self.name = name
        self.prev_thresh = p.get('prev_thresh', 0.80)
        self.bull_thresh = p.get('bull_thresh', 0.50)
        self.bear_thresh = p.get('bear_thresh', 0.50)
        self.min_agree = p.get('min_agree', 2)

        # Entry timing
        self.entry_time = p.get('entry_time', 't0')
        self.exit_time = p.get('exit_time', 't12.5')
        self.exit_mode = p.get('exit_mode', 'early')  # 'early' or 'resolution'

        # Momentum filter
        self.require_momentum = p.get('require_momentum', False)
        self.momentum_min = p.get('momentum_min', 0.03)  # min abs momentum to confirm

        # t0 sweet spot
        self.t0_sweet_spot = p.get('t0_sweet_spot', False)
        self.bear_t0_range = p.get('bear_t0_range', (0.35, 0.55))
        self.bull_t0_range = p.get('bull_t0_range', (0.45, 0.65))

        # Stop loss at intermediate time
        self.stop_loss_time = p.get('stop_loss_time', None)  # 't5', 't7.5'
        self.stop_loss_thresh = p.get('stop_loss_thresh', 0.05)  # exit if underwater by this

        # Direction filter
        self.bear_only = p.get('bear_only', False)
        self.bull_only = p.get('bull_only', False)

        # Bet sizing
        self.sizing_mode = p.get('sizing_mode', 'flat')
        # 'flat', 'anti_mart', 'kelly', 'confidence', 'asymmetric'
        self.anti_mart_mult = p.get('anti_mart_mult', 1.5)
        self.kelly_window = p.get('kelly_window', 50)
        self.kelly_fraction = p.get('kelly_fraction', 0.25)  # quarter Kelly
        self.bear_size_mult = p.get('bear_size_mult', 1.0)
        self.bull_size_mult = p.get('bull_size_mult', 1.0)
        self.confidence_scale = p.get('confidence_scale', 2.0)  # max multiplier for strong momentum

        # Risk management
        self.daily_loss_limit = p.get('daily_loss_limit', None)
        self.daily_loss_pct = p.get('daily_loss_pct', None)
        self.pause_after_loss = p.get('pause_after_loss', 0)

    def pm_key(self, t):
        return {
            't0': 'pm_yes_t0', 't2.5': 'pm_yes_t2_5', 't5': 'pm_yes_t5',
            't7.5': 'pm_yes_t7_5', 't10': 'pm_yes_t10', 't12.5': 'pm_yes_t12_5',
        }.get(t, 'pm_yes_t0')


def simulate(strategy, time_groups):
    s = strategy
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET

    pause_remaining = 0
    trades = []
    daily_pnl = defaultdict(float)
    equity_curve = [INITIAL_BANKROLL]
    max_dd = 0
    max_dd_pct = 0

    # For anti-martingale
    last_bet = base_bet
    last_won = True

    # For Kelly
    recent_results = deque(maxlen=s.kelly_window if s.sizing_mode == 'kelly' else 1)

    entry_key = s.pm_key(s.entry_time)
    exit_key = s.pm_key(s.exit_time)
    stop_key = s.pm_key(s.stop_loss_time) if s.stop_loss_time else None

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group or bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        day_key = sample_row['window_start_utc'][:10]

        if pause_remaining > 0:
            pause_remaining -= 1
            continue

        # Daily loss limit
        if s.daily_loss_limit is not None and daily_pnl[day_key] < -s.daily_loss_limit:
            continue
        if s.daily_loss_pct is not None and daily_pnl[day_key] < -(bankroll * s.daily_loss_pct):
            continue

        # Phase 1: Previous window consensus
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

        # Phase 2: Confirmation + filters per asset
        confirming = []
        for asset, row in group.items():
            entry_pm = row.get(entry_key)
            exit_pm = row.get(exit_key)
            if entry_pm is None or exit_pm is None:
                continue

            # Basic threshold check
            if direction == 'bear' and entry_pm > s.bear_thresh:
                continue
            if direction == 'bull' and entry_pm < s.bull_thresh:
                continue

            # t0 sweet spot filter
            if s.t0_sweet_spot:
                t0 = row.get('pm_yes_t0')
                if t0 is not None:
                    if direction == 'bear' and not (s.bear_t0_range[0] <= t0 <= s.bear_t0_range[1]):
                        continue
                    if direction == 'bull' and not (s.bull_t0_range[0] <= t0 <= s.bull_t0_range[1]):
                        continue

            # Momentum confirmation filter
            if s.require_momentum:
                mom = row.get('pm_price_momentum_0_to_5')
                if mom is None:
                    continue
                if direction == 'bear' and mom > -s.momentum_min:
                    # For bear: need negative momentum (price going down)
                    continue
                if direction == 'bull' and mom < s.momentum_min:
                    # For bull: need positive momentum (price going up)
                    continue

            confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < s.min_agree:
            continue

        # Determine bet size based on sizing mode
        if s.sizing_mode == 'anti_mart':
            if last_won:
                bet = min(last_bet * s.anti_mart_mult, base_bet * 3)
            else:
                bet = max(last_bet / s.anti_mart_mult, base_bet * 0.25)
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
                bet = bankroll * kelly
                bet = max(bet, base_bet * 0.25)  # floor
            else:
                bet = base_bet
        elif s.sizing_mode == 'confidence':
            # Scale bet by momentum strength
            bet = base_bet  # will be scaled per-trade below
        else:
            bet = base_bet

        # Direction-based asymmetric sizing
        if s.sizing_mode == 'asymmetric':
            bet = base_bet * (s.bear_size_mult if direction == 'bear' else s.bull_size_mult)

        bet = min(bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        # Execute trades
        window_had_loss = False
        for asset, row, entry_pm, exit_pm in confirming[:4]:
            trade_bet = bet

            # Confidence-scaled sizing
            if s.sizing_mode == 'confidence':
                mom = row.get('pm_price_momentum_0_to_5')
                if mom is not None:
                    abs_mom = abs(mom)
                    # Scale from 1x at momentum=0.03 to confidence_scale at momentum=0.20+
                    scale = 1.0 + (s.confidence_scale - 1.0) * min(abs_mom / 0.20, 1.0)
                    trade_bet = base_bet * scale
                trade_bet = min(trade_bet, bankroll * 0.05)

            if trade_bet < 1.0:
                continue

            # Check stop loss at intermediate time
            actual_exit_pm = exit_pm
            stopped_out = False
            if s.stop_loss_time and stop_key:
                stop_pm = row.get(stop_key)
                if stop_pm is not None:
                    # Check if position is underwater
                    if direction == 'bear':
                        position_pnl = (1 - entry_pm) - (1 - stop_pm)  # NO position
                        # Underwater if stop_pm > entry_pm (YES went up = bad for bear)
                        if stop_pm - entry_pm > s.stop_loss_thresh:
                            actual_exit_pm = stop_pm
                            stopped_out = True
                    else:
                        position_pnl = stop_pm - entry_pm  # YES position
                        if entry_pm - stop_pm > s.stop_loss_thresh:
                            actual_exit_pm = stop_pm
                            stopped_out = True

            # Calculate P&L
            if s.exit_mode == 'resolution' and not stopped_out:
                net = pnl_resolution(direction, entry_pm, row.get('outcome'), trade_bet)
            else:
                net = pnl_early_exit(direction, entry_pm, actual_exit_pm, trade_bet)

            won = net > 0
            bankroll += net
            peak_bankroll = max(peak_bankroll, bankroll)

            dd_d = bankroll - peak_bankroll
            dd_p = dd_d / peak_bankroll if peak_bankroll > 0 else 0
            max_dd = min(max_dd, dd_d)
            max_dd_pct = min(max_dd_pct, dd_p)

            daily_pnl[day_key] += net
            equity_curve.append(bankroll)
            recent_results.append(net)

            trades.append({
                'asset': asset, 'direction': direction,
                'entry_pm': entry_pm, 'exit_pm': actual_exit_pm,
                'bet': trade_bet, 'net_pnl': net, 'won': won,
                'bankroll': bankroll, 'day': day_key,
                'stopped': stopped_out,
            })

            if not won:
                window_had_loss = True

            last_bet = trade_bet
            last_won = won

        if window_had_loss and s.pause_after_loss > 0:
            pause_remaining = s.pause_after_loss

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
    pf = abs(float(np.sum(pnls[pnls > 0])) / float(np.sum(pnls[pnls <= 0]))) if np.sum(pnls <= 0) != 0 else 999

    max_ls = 0
    cs = 0
    for t in trades:
        if not t['won']:
            cs += 1
            max_ls = max(max_ls, cs)
        else:
            cs = 0

    stopped_count = sum(1 for t in trades if t.get('stopped'))
    losing_days = sum(1 for v in daily_pnl.values() if v < 0)
    total_days = len(daily_pnl)

    # Direction breakdown
    dir_stats = {}
    for d in ['bull', 'bear']:
        dt = [t for t in trades if t['direction'] == d]
        if dt:
            dp = [t['net_pnl'] for t in dt]
            dw = sum(1 for p in dp if p > 0)
            dir_stats[d] = {'n': len(dt), 'wins': dw, 'wr': dw/len(dt), 'pnl': sum(dp)}

    return {
        'name': s.name, 'trades': n, 'wins': wins, 'win_rate': win_rate,
        'total_pnl': total_pnl, 'avg_pnl': avg_pnl,
        'avg_win': avg_w, 'avg_loss': avg_l,
        'sharpe': sharpe, 'profit_factor': pf,
        'max_dd_dollars': max_dd, 'max_dd_pct': max_dd_pct * 100,
        'final_bankroll': bankroll,
        'roi_pct': (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100,
        'max_loss_streak': max_ls,
        'stopped_count': stopped_count,
        'losing_days': losing_days, 'total_days': total_days,
        'dir_stats': dir_stats,
    }


def print_table(results, label):
    if not results:
        print(f"\n  {label} — NO RESULTS")
        return
    print(f"\n{'='*155}")
    print(f"  {label}")
    print(f"{'='*155}")
    print(
        f"{'Strategy':<52} {'#':>5} {'WR%':>6} {'PnL':>10} {'AvgW':>7} {'AvgL':>7} "
        f"{'Shrp':>6} {'PF':>5} {'DD%':>7} {'Final$':>9} {'ROI%':>7} "
        f"{'MLS':>4} {'Stop':>5} {'LD':>5}"
    )
    print('-' * 155)
    for r in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        ld = f"{r.get('losing_days','?')}/{r.get('total_days','?')}"
        ds = r.get('dir_stats', {})
        bull_wr = f"{ds['bull']['wr']*100:.0f}%" if 'bull' in ds else '-'
        bear_wr = f"{ds['bear']['wr']*100:.0f}%" if 'bear' in ds else '-'
        print(
            f"{r['name']:<52} "
            f"{r['trades']:>5} {r['win_rate']*100:>5.1f}% "
            f"${r['total_pnl']:>9.2f} ${r['avg_win']:>6.2f} ${r['avg_loss']:>6.2f} "
            f"{r['sharpe']:>6.2f} {r['profit_factor']:>4.2f}x "
            f"{r['max_dd_pct']:>6.1f}% ${r['final_bankroll']:>8.2f} "
            f"{r['roi_pct']:>6.1f}% {r['max_loss_streak']:>4} "
            f"{r['stopped_count']:>5} {ld:>5}"
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

    strategies = []

    # ─── BASELINES ───
    strategies.append(Strategy("BASE_agree2_flat", min_agree=2))
    strategies.append(Strategy("BASE_agree2_daily100", min_agree=2, daily_loss_limit=100))
    strategies.append(Strategy("BASE_agree3_flat", min_agree=3))

    # ─── SECTION A: ENTRY FILTERS ───

    # A1: Momentum confirmation
    for mom_min in [0.03, 0.05, 0.10]:
        strategies.append(Strategy(
            f"A1_MOM_CONFIRM_{mom_min:.2f}",
            min_agree=2, require_momentum=True, momentum_min=mom_min,
        ))
        strategies.append(Strategy(
            f"A1_MOM_CONFIRM_{mom_min:.2f}+daily100",
            min_agree=2, require_momentum=True, momentum_min=mom_min,
            daily_loss_limit=100,
        ))

    # A2: t0 sweet spot
    strategies.append(Strategy(
        "A2_T0_SWEET_SPOT",
        min_agree=2, t0_sweet_spot=True,
    ))
    strategies.append(Strategy(
        "A2_T0_SWEET_WIDE",
        min_agree=2, t0_sweet_spot=True,
        bear_t0_range=(0.30, 0.60), bull_t0_range=(0.40, 0.70),
    ))

    # A3: Momentum + t0 sweet spot
    strategies.append(Strategy(
        "A3_MOM+SWEET",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        t0_sweet_spot=True,
    ))
    strategies.append(Strategy(
        "A3_MOM+SWEET+daily100",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        t0_sweet_spot=True, daily_loss_limit=100,
    ))

    # A4: t5 entry (wait for momentum to develop)
    strategies.append(Strategy(
        "A4_T5_ENTRY",
        min_agree=2, entry_time='t5',
    ))
    strategies.append(Strategy(
        "A4_T5_MOM_CONFIRM",
        min_agree=2, entry_time='t5', require_momentum=True, momentum_min=0.03,
    ))

    # A5: Bear only (stronger edge)
    strategies.append(Strategy(
        "A5_BEAR_ONLY",
        min_agree=2, bear_only=True,
    ))
    strategies.append(Strategy(
        "A5_BEAR_MOM",
        min_agree=2, bear_only=True, require_momentum=True, momentum_min=0.03,
    ))

    # ─── SECTION B: EXIT STRATEGIES ───

    # B1: Stop-loss at t5
    for thresh in [0.03, 0.05, 0.08, 0.10]:
        strategies.append(Strategy(
            f"B1_STOP_T5_{thresh:.2f}",
            min_agree=2, stop_loss_time='t5', stop_loss_thresh=thresh,
        ))

    # B2: Stop-loss at t7.5
    for thresh in [0.05, 0.08, 0.10]:
        strategies.append(Strategy(
            f"B2_STOP_T7.5_{thresh:.2f}",
            min_agree=2, stop_loss_time='t7.5', stop_loss_thresh=thresh,
        ))

    # B3: Hold to resolution (binary payout)
    strategies.append(Strategy(
        "B3_RESOLUTION",
        min_agree=2, exit_mode='resolution',
    ))
    strategies.append(Strategy(
        "B3_RESOLUTION+daily100",
        min_agree=2, exit_mode='resolution', daily_loss_limit=100,
    ))
    strategies.append(Strategy(
        "B3_RESOL+MOM",
        min_agree=2, exit_mode='resolution', require_momentum=True, momentum_min=0.03,
    ))
    strategies.append(Strategy(
        "B3_RESOL+MOM+daily100",
        min_agree=2, exit_mode='resolution', require_momentum=True, momentum_min=0.03,
        daily_loss_limit=100,
    ))

    # B4: Stop-loss + momentum combo
    strategies.append(Strategy(
        "B4_MOM+STOP_T5_0.05",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        stop_loss_time='t5', stop_loss_thresh=0.05,
    ))
    strategies.append(Strategy(
        "B4_MOM+STOP_T5_0.08",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        stop_loss_time='t5', stop_loss_thresh=0.08,
    ))

    # B5: Resolution with stop-loss (stop out losers, let winners resolve)
    strategies.append(Strategy(
        "B5_RESOL+STOP_T5_0.05",
        min_agree=2, exit_mode='resolution',
        stop_loss_time='t5', stop_loss_thresh=0.05,
    ))
    strategies.append(Strategy(
        "B5_RESOL+STOP_T5_0.08",
        min_agree=2, exit_mode='resolution',
        stop_loss_time='t5', stop_loss_thresh=0.08,
    ))
    strategies.append(Strategy(
        "B5_RESOL+STOP_T7.5_0.08",
        min_agree=2, exit_mode='resolution',
        stop_loss_time='t7.5', stop_loss_thresh=0.08,
    ))

    # ─── SECTION C: BET SIZING ───

    # C1: Anti-martingale
    for mult in [1.25, 1.5, 2.0]:
        strategies.append(Strategy(
            f"C1_ANTIMART_{mult:.2f}x",
            min_agree=2, sizing_mode='anti_mart', anti_mart_mult=mult,
        ))

    # C2: Fractional Kelly
    for frac in [0.125, 0.25, 0.50]:
        strategies.append(Strategy(
            f"C2_KELLY_{frac:.3f}",
            min_agree=2, sizing_mode='kelly', kelly_fraction=frac,
        ))

    # C3: Asymmetric sizing (favor bear)
    strategies.append(Strategy(
        "C3_ASYM_bear1.5_bull0.75",
        min_agree=2, sizing_mode='asymmetric', bear_size_mult=1.5, bull_size_mult=0.75,
    ))
    strategies.append(Strategy(
        "C3_ASYM_bear2.0_bull0.50",
        min_agree=2, sizing_mode='asymmetric', bear_size_mult=2.0, bull_size_mult=0.50,
    ))

    # C4: Confidence-scaled (scale by momentum strength)
    for scale in [1.5, 2.0, 3.0]:
        strategies.append(Strategy(
            f"C4_CONFIDENCE_{scale:.1f}x",
            min_agree=2, sizing_mode='confidence', confidence_scale=scale,
            require_momentum=True, momentum_min=0.03,
        ))

    # ─── SECTION D: BEST COMBINATIONS ───

    # D1: Momentum + daily limit + anti-mart
    strategies.append(Strategy(
        "D1_MOM+DAILY+ANTIMART",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        daily_loss_limit=100,
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))

    # D2: Momentum + resolution + daily
    strategies.append(Strategy(
        "D2_MOM+RESOL+DAILY",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        exit_mode='resolution', daily_loss_limit=100,
    ))

    # D3: Full combo: momentum + t0 sweet + resolution + stop + daily
    strategies.append(Strategy(
        "D3_MOM+SWEET+RESOL+STOP+DAILY",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        t0_sweet_spot=True,
        exit_mode='resolution',
        stop_loss_time='t5', stop_loss_thresh=0.08,
        daily_loss_limit=100,
    ))

    # D4: Confidence-scaled momentum + daily + resolution
    strategies.append(Strategy(
        "D4_CONF+MOM+RESOL+DAILY",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        sizing_mode='confidence', confidence_scale=2.0,
        exit_mode='resolution', daily_loss_limit=100,
    ))

    # D5: Bear-heavy + momentum + daily
    strategies.append(Strategy(
        "D5_ASYM+MOM+DAILY",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        sizing_mode='asymmetric', bear_size_mult=1.5, bull_size_mult=0.75,
        daily_loss_limit=100,
    ))

    # D6: Kitchen sink (pause + momentum + sweet + daily + anti-mart)
    strategies.append(Strategy(
        "D6_KITCHEN_SINK",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        t0_sweet_spot=True,
        daily_loss_limit=100, pause_after_loss=1,
        sizing_mode='anti_mart', anti_mart_mult=1.5,
    ))

    # D7: Momentum + resolution + Kelly
    strategies.append(Strategy(
        "D7_MOM+RESOL+KELLY",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        exit_mode='resolution',
        sizing_mode='kelly', kelly_fraction=0.25,
    ))

    # D8: Resolution + stop + momentum + daily + Kelly
    strategies.append(Strategy(
        "D8_RESOL+STOP+MOM+DAILY+KELLY",
        min_agree=2, require_momentum=True, momentum_min=0.03,
        exit_mode='resolution',
        stop_loss_time='t5', stop_loss_thresh=0.08,
        daily_loss_limit=100,
        sizing_mode='kelly', kelly_fraction=0.25,
    ))

    print(f"\nRunning {len(strategies)} strategies...")
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

    # Comparison
    print(f"\n{'='*160}")
    print(f"  TRAIN vs TEST (sorted by test P&L)")
    print(f"{'='*160}")
    print(
        f"{'Strategy':<52} "
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
            f"{name:<52} "
            f"${tr['total_pnl']:>8.0f} {tr['win_rate']*100:>5.1f}% {tr['max_dd_pct']:>6.1f}% "
            f"${te['total_pnl']:>8.0f} {te['win_rate']*100:>5.1f}% {te['max_dd_pct']:>6.1f}% "
            f"{rob:>4} {tr['trades']:>5} {te['trades']:>5} "
            f"${te['avg_win']:>6.2f} ${te['avg_loss']:>6.2f}"
        )

    # Robust strategies
    print(f"\n{'='*120}")
    print(f"  ROBUST STRATEGIES (profitable in BOTH train and test)")
    print(f"{'='*120}")
    i = 0
    for name in sorted(test_by_name.keys(), key=lambda n: test_by_name[n]['total_pnl'], reverse=True):
        tr = train_by_name.get(name)
        te = test_by_name[name]
        if tr and tr['total_pnl'] > 0 and te['total_pnl'] > 0:
            i += 1
            ds = te.get('dir_stats', {})
            bull_info = f"bull:{ds['bull']['wr']*100:.0f}%/{ds['bull']['n']}" if 'bull' in ds else ""
            bear_info = f"bear:{ds['bear']['wr']*100:.0f}%/{ds['bear']['n']}" if 'bear' in ds else ""
            print(
                f"  {i:>2}. {name:<50} "
                f"Train: ${tr['total_pnl']:>8.0f} ({tr['win_rate']*100:.1f}%)  "
                f"Test: ${te['total_pnl']:>8.0f} ({te['win_rate']*100:.1f}% WR, {te['max_dd_pct']:.1f}% DD)  "
                f"{bull_info} {bear_info}"
            )

    # Save
    out_path = DATA_DIR / "reports" / "advanced_exits_sizing_results.json"
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
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
