#!/usr/bin/env python3
"""
Bet Sizing Strategy Comparison Backtest

Simulates contrarian_consensus strategy with 10 different bet sizing approaches.
Uses 70/30 temporal train/test split. Tracks per-asset loss streaks.

Strategy: contrarian_consensus
  - prev_thresh: 0.80 (strong UP >= 0.80, strong DOWN <= 0.20)
  - bull_thresh: 0.50, bear_thresh: 0.50
  - consensus_min_agree: 3 of 4 assets
  - Entry at t0 pm_yes, Exit at t12.5 pm_yes
  - fee_rate: 0.001, spread_cost: 0.005
"""

import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']
FEE_RATE = 0.001
SPREAD_COST = 0.005
INITIAL_BANKROLL = 1000.0

# Strategy parameters
PREV_THRESH = 0.80
BULL_THRESH = 0.50
BEAR_THRESH = 0.50
CONSENSUS_MIN_AGREE = 3
ENTRY_COL = 'pm_yes_t0'
EXIT_COL = 'pm_yes_t12_5'


def load_all_data():
    """Load all windows data from all assets and build cross-asset consensus."""
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT w.window_id, w.asset, w.window_start_utc, w.outcome,
                   w.pm_yes_t0, w.pm_yes_t2_5, w.pm_yes_t5, w.pm_yes_t7_5,
                   w.pm_yes_t10, w.pm_yes_t12_5,
                   w.prev_pm_t12_5, w.prev2_pm_t12_5, w.window_time
            FROM windows w
            WHERE w.outcome IS NOT NULL
            ORDER BY w.window_start_utc
        """).fetchall()
        all_rows.extend([dict(r) for r in rows])
        conn.close()

    all_rows.sort(key=lambda x: x['window_start_utc'])

    # Build per-asset sequential data for prev fields if missing
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

    # Annotate each row with cross-asset consensus counts
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
                    if p >= PREV_THRESH:
                        n_strong_up += 1
                    elif p <= (1.0 - PREV_THRESH):
                        n_strong_down += 1
            r['xasset_strong_up'] = n_strong_up
            r['xasset_strong_down'] = n_strong_down
        else:
            r['xasset_strong_up'] = 0
            r['xasset_strong_down'] = 0

    return enriched


def generate_signals(data):
    """Generate trade signals for contrarian_consensus strategy.

    Returns list of dicts with: asset, direction, entry_pm, exit_pm, window_time, timestamp
    """
    signals = []
    for r in data:
        prev = r.get('prev_pm_t12_5')
        pm_entry = r.get(ENTRY_COL)
        pm_exit = r.get(EXIT_COL)
        n_up = r.get('xasset_strong_up', 0)
        n_down = r.get('xasset_strong_down', 0)

        if prev is None or pm_entry is None or pm_exit is None:
            continue

        direction = None
        # Strong UP prev -> expect reversal -> BEAR entry
        if prev >= PREV_THRESH and n_up >= CONSENSUS_MIN_AGREE and pm_entry <= BEAR_THRESH:
            direction = 'bear'
        # Strong DOWN prev -> expect reversal -> BULL entry
        elif prev <= (1.0 - PREV_THRESH) and n_down >= CONSENSUS_MIN_AGREE and pm_entry >= BULL_THRESH:
            direction = 'bull'

        if direction is not None:
            signals.append({
                'asset': r['asset'],
                'direction': direction,
                'entry_pm': pm_entry,
                'exit_pm': pm_exit,
                'timestamp': r['window_start_utc'],
                'window_time': r.get('window_time', ''),
            })

    return signals


def calc_trade_pnl(direction, entry_pm, exit_pm, bet_size):
    """Calculate P&L for an early-exit trade.

    For bull: entry_contract = pm_yes_t0, exit_contract = pm_yes_t12.5
    For bear: entry_contract = 1 - pm_yes_t0, exit_contract = 1 - pm_yes_t12.5
    n_contracts = bet_size / entry_contract
    gross = n_contracts * (exit_contract - entry_contract)
    fees = fee_rate * bet_size + spread_cost * bet_size (entry) + fee_rate * (n*exit_c) + spread_cost * (n*exit_c) (exit)
    """
    if direction == 'bull':
        entry_c = entry_pm
        exit_c = exit_pm
    else:
        entry_c = 1.0 - entry_pm
        exit_c = 1.0 - exit_pm

    if entry_c <= 0.001:
        return 0.0

    n = bet_size / entry_c
    gross = n * (exit_c - entry_c)

    # Fees: flat model per user spec
    # Entry fees: fee_rate * bet_size + spread_cost * bet_size
    # Exit fees: fee_rate * bet_size + spread_cost * bet_size
    # But more accurately from the sweep code:
    # fees = entry_c * n * FEE_RATE + exit_c * n * FEE_RATE
    # spread = SPREAD_COST * bet_size + SPREAD_COST * (n * exit_c)
    fees = entry_c * n * FEE_RATE + exit_c * n * FEE_RATE
    spread = SPREAD_COST * bet_size + SPREAD_COST * (n * exit_c)

    return gross - fees - spread


# =============================================================================
# BET SIZING STRATEGIES
# =============================================================================

class FlatSizer:
    """Flat bet sizing - constant amount per trade."""
    def __init__(self, amount):
        self.amount = amount
        self.name = f"Flat ${amount:.0f}"

    def get_bet(self, asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history):
        return min(self.amount, bankroll)

    def on_result(self, asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history):
        pass  # no state to update


class LinearStepUpSizer:
    """Linear step-up: base + step * consecutive_losses, capped at cap_mult * base."""
    def __init__(self, base, step, cap_mult):
        self.base = base
        self.step = step
        self.cap_mult = cap_mult
        self.name = f"Linear ${base:.0f}+${step:.0f}/loss cap{cap_mult}x"

    def get_bet(self, asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history):
        losses = asset_loss_streaks.get(asset, 0)
        bet = self.base + self.step * losses
        bet = min(bet, self.base * self.cap_mult)
        return min(bet, bankroll)

    def on_result(self, asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history):
        if won:
            asset_loss_streaks[asset] = 0
        else:
            asset_loss_streaks[asset] = asset_loss_streaks.get(asset, 0) + 1


class MartingaleSizer:
    """Martingale: base * mult^consecutive_losses, capped."""
    def __init__(self, base, mult, cap_mult):
        self.base = base
        self.mult = mult
        self.cap_mult = cap_mult
        self.name = f"Martingale {mult}x from ${base:.0f} cap{cap_mult}x"

    def get_bet(self, asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history):
        losses = asset_loss_streaks.get(asset, 0)
        bet = self.base * (self.mult ** losses)
        bet = min(bet, self.base * self.cap_mult)
        return min(bet, bankroll)

    def on_result(self, asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history):
        if won:
            asset_loss_streaks[asset] = 0
        else:
            asset_loss_streaks[asset] = asset_loss_streaks.get(asset, 0) + 1


class FibonacciSizer:
    """Fibonacci bet sizing from base: $25, $25, $50, $75, $125, $200 (capped)."""
    def __init__(self, base):
        self.base = base
        # Build fib sequence scaled to base
        self.fib_mult = [1, 1, 2, 3, 5, 8]  # caps at 8x base
        self.name = f"Fibonacci from ${base:.0f}"

    def get_bet(self, asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history):
        losses = asset_loss_streaks.get(asset, 0)
        idx = min(losses, len(self.fib_mult) - 1)
        bet = self.base * self.fib_mult[idx]
        return min(bet, bankroll)

    def on_result(self, asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history):
        if won:
            asset_loss_streaks[asset] = 0
        else:
            asset_loss_streaks[asset] = asset_loss_streaks.get(asset, 0) + 1


class AntiMartingaleSizer:
    """Anti-martingale: increase bet on WIN, reset on loss. Per-asset."""
    def __init__(self, base, win_mult=1.5, cap_mult=5):
        self.base = base
        self.win_mult = win_mult
        self.cap_mult = cap_mult
        self.name = f"Anti-Martingale ${base:.0f}"
        self._current_bet = {}  # per-asset current bet

    def get_bet(self, asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history):
        bet = self._current_bet.get(asset, self.base)
        bet = min(bet, self.base * self.cap_mult)
        return min(bet, bankroll)

    def on_result(self, asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history):
        if won:
            # Increase on win
            current = self._current_bet.get(asset, self.base)
            self._current_bet[asset] = current * self.win_mult
            asset_loss_streaks[asset] = 0
            asset_win_streaks[asset] = asset_win_streaks.get(asset, 0) + 1
        else:
            # Reset on loss
            self._current_bet[asset] = self.base
            asset_loss_streaks[asset] = asset_loss_streaks.get(asset, 0) + 1
            asset_win_streaks[asset] = 0


class KellyLiteSizer:
    """Kelly-lite: $25 * (2 * rolling_win_rate - 1), floored $10, capped $125.
    Uses last 20 trades per-asset rolling window."""
    def __init__(self, base=25.0, window=20, floor=10.0, cap=125.0):
        self.base = base
        self.window = window
        self.floor = floor
        self.cap = cap
        self.name = f"Kelly-lite ${base:.0f} (w={window})"
        self._asset_results = defaultdict(list)  # per-asset list of booleans

    def get_bet(self, asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history):
        results = self._asset_results.get(asset, [])
        if len(results) < 5:
            # Not enough data, use base
            return min(self.base, bankroll)
        recent = results[-self.window:]
        wr = sum(recent) / len(recent)
        kelly_frac = 2 * wr - 1
        bet = self.base * max(kelly_frac, 0)
        bet = max(bet, self.floor)
        bet = min(bet, self.cap)
        return min(bet, bankroll)

    def on_result(self, asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history):
        self._asset_results[asset].append(won)
        if won:
            asset_loss_streaks[asset] = 0
        else:
            asset_loss_streaks[asset] = asset_loss_streaks.get(asset, 0) + 1


def run_simulation(signals, sizer):
    """Run a full simulation with the given sizing strategy over signals.

    Returns dict of metrics.
    """
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    max_drawdown = 0.0
    asset_loss_streaks = {}
    asset_win_streaks = {}
    asset_max_loss_streaks = {}
    trade_history = []

    pnls = []
    wins = 0
    total_trades = 0
    went_bust = False

    for sig in signals:
        if bankroll < 1.0:
            went_bust = True
            break

        asset = sig['asset']
        direction = sig['direction']
        entry_pm = sig['entry_pm']
        exit_pm = sig['exit_pm']

        bet_size = sizer.get_bet(asset, bankroll, asset_loss_streaks, asset_win_streaks, trade_history)
        if bet_size < 0.01:
            continue

        pnl = calc_trade_pnl(direction, entry_pm, exit_pm, bet_size)
        won = pnl > 0

        bankroll += pnl
        pnls.append(pnl)
        total_trades += 1
        if won:
            wins += 1

        sizer.on_result(asset, won, bet_size, asset_loss_streaks, asset_win_streaks, trade_history)
        trade_history.append({'asset': asset, 'won': won, 'pnl': pnl, 'bet': bet_size})

        # Track peak and drawdown
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = peak_bankroll - bankroll
        if dd > max_drawdown:
            max_drawdown = dd

        # Track per-asset max loss streaks
        for a in ASSETS:
            a_upper = a.upper()
            streak = asset_loss_streaks.get(a_upper, 0)
            if streak > asset_max_loss_streaks.get(a_upper, 0):
                asset_max_loss_streaks[a_upper] = streak

    # Compute metrics
    win_rate = wins / total_trades if total_trades > 0 else 0
    total_pnl = sum(pnls)
    max_dd_pct = (max_drawdown / peak_bankroll * 100) if peak_bankroll > 0 else 0
    pnl_dd_ratio = total_pnl / max_drawdown if max_drawdown > 0.01 else (999.0 if total_pnl > 0 else 0.0)
    max_consec_loss = max(asset_max_loss_streaks.values()) if asset_max_loss_streaks else 0
    final_bankroll = bankroll

    return {
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown,
        'max_dd_pct': max_dd_pct,
        'pnl_dd_ratio': pnl_dd_ratio,
        'n_trades': total_trades,
        'win_rate': win_rate,
        'max_consec_loss': max_consec_loss,
        'peak_bankroll': peak_bankroll,
        'final_bankroll': final_bankroll,
        'went_bust': went_bust,
    }


def reset_sizer(sizer):
    """Create a fresh copy of the sizer (reset state).
    We instantiate new objects since some sizers have internal state."""
    if isinstance(sizer, FlatSizer):
        return FlatSizer(sizer.amount)
    elif isinstance(sizer, LinearStepUpSizer):
        return LinearStepUpSizer(sizer.base, sizer.step, sizer.cap_mult)
    elif isinstance(sizer, MartingaleSizer):
        return MartingaleSizer(sizer.base, sizer.mult, sizer.cap_mult)
    elif isinstance(sizer, FibonacciSizer):
        return FibonacciSizer(sizer.base)
    elif isinstance(sizer, AntiMartingaleSizer):
        return AntiMartingaleSizer(sizer.base, sizer.win_mult, sizer.cap_mult)
    elif isinstance(sizer, KellyLiteSizer):
        return KellyLiteSizer(sizer.base, sizer.window, sizer.floor, sizer.cap)
    return sizer


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 120)
    print("BET SIZING STRATEGY COMPARISON BACKTEST")
    print("Strategy: contrarian_consensus | prev_thresh=0.80 | bull/bear=0.50 | consensus=3/4 | entry=t0, exit=t12.5")
    print(f"Starting bankroll: ${INITIAL_BANKROLL:.0f} | Fee={FEE_RATE} | Spread={SPREAD_COST}")
    print("=" * 120)

    print("\nLoading data...")
    data = load_all_data()
    print(f"Total rows: {len(data)}")

    # 70/30 temporal split
    n = len(data)
    split_idx = int(n * 0.70)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"Train: {len(train_data)} rows ({train_data[0]['window_start_utc'][:16]} to {train_data[-1]['window_start_utc'][:16]})")
    print(f"Test:  {len(test_data)} rows ({test_data[0]['window_start_utc'][:16]} to {test_data[-1]['window_start_utc'][:16]})")

    # Generate signals
    train_signals = generate_signals(train_data)
    test_signals = generate_signals(test_data)
    print(f"\nTrain signals: {len(train_signals)}")
    print(f"Test signals:  {len(test_signals)}")

    # Define sizing strategies
    sizers = [
        FlatSizer(25.0),                                    # a. Flat $25
        FlatSizer(50.0),                                    # b. Flat $50
        LinearStepUpSizer(25.0, 25.0, 5),                   # c. Linear $25+$25/loss cap 5x
        LinearStepUpSizer(25.0, 25.0, 3),                   # d. Linear $25+$25/loss cap 3x
        LinearStepUpSizer(15.0, 15.0, 5),                   # e. Linear $15+$15/loss cap 5x
        MartingaleSizer(25.0, 1.5, 5),                       # f. Martingale 1.5x from $25 cap 5x
        MartingaleSizer(25.0, 2.0, 4),                       # g. Martingale 2x from $25 cap 4x
        FibonacciSizer(25.0),                                # h. Fibonacci from $25
        AntiMartingaleSizer(25.0, win_mult=1.5, cap_mult=5), # i. Anti-Martingale
        KellyLiteSizer(25.0, window=20, floor=10.0, cap=125.0), # j. Kelly-lite
    ]

    labels = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
    ]

    # Run simulations
    results = []
    for label, sizer_template in zip(labels, sizers):
        # Train
        sizer_train = reset_sizer(sizer_template)
        train_metrics = run_simulation(train_signals, sizer_train)

        # Test (fresh sizer)
        sizer_test = reset_sizer(sizer_template)
        test_metrics = run_simulation(test_signals, sizer_test)

        results.append({
            'label': label,
            'name': sizer_template.name,
            'train': train_metrics,
            'test': test_metrics,
        })

    # Sort by test PnL/DD ratio
    results.sort(key=lambda x: x['test']['pnl_dd_ratio'], reverse=True)

    # Print results table
    print("\n")
    print("=" * 180)
    print(f"{'':>3} {'Strategy':<38} | {'--- TRAIN ---':^55} | {'--- TEST ---':^70}")
    print(f"{'':>3} {'':38} | {'PnL':>9} {'MaxDD':>9} {'PnL/DD':>8} {'#Tr':>5} {'WR%':>6} {'MaxL':>5} {'Peak$':>8} {'Final$':>8} {'Bust':>5} | "
          f"{'PnL':>9} {'MaxDD':>9} {'PnL/DD':>8} {'#Tr':>5} {'WR%':>6} {'MaxL':>5} {'Peak$':>8} {'Final$':>8} {'Bust':>5}")
    print("=" * 180)

    for r in results:
        tr = r['train']
        te = r['test']
        bust_tr = 'YES' if tr['went_bust'] else 'no'
        bust_te = 'YES' if te['went_bust'] else 'no'
        tr_ratio = f"{tr['pnl_dd_ratio']:.2f}" if tr['pnl_dd_ratio'] < 900 else 'inf'
        te_ratio = f"{te['pnl_dd_ratio']:.2f}" if te['pnl_dd_ratio'] < 900 else 'inf'

        print(f" {r['label']}) {r['name']:<37} | "
              f"${tr['total_pnl']:>8.2f} ${tr['max_drawdown']:>8.2f} {tr_ratio:>8} {tr['n_trades']:>5} {tr['win_rate']*100:>5.1f} {tr['max_consec_loss']:>5} ${tr['peak_bankroll']:>7.0f} ${tr['final_bankroll']:>7.0f} {bust_tr:>5} | "
              f"${te['total_pnl']:>8.2f} ${te['max_drawdown']:>8.2f} {te_ratio:>8} {te['n_trades']:>5} {te['win_rate']*100:>5.1f} {te['max_consec_loss']:>5} ${te['peak_bankroll']:>7.0f} ${te['final_bankroll']:>7.0f} {bust_te:>5}")

    print("=" * 180)

    # Additional summary
    print("\n\nSUMMARY (sorted by test PnL/DD ratio - higher is better):")
    print("-" * 90)
    print(f"{'#':>3} {'Strategy':<38} {'Test PnL':>10} {'Test DD':>10} {'Test PnL/DD':>12} {'Test WR':>8}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        te = r['test']
        ratio_str = f"{te['pnl_dd_ratio']:.3f}" if te['pnl_dd_ratio'] < 900 else 'inf'
        print(f"{i:>3} {r['label']}) {r['name']:<35} ${te['total_pnl']:>9.2f} ${te['max_drawdown']:>9.2f} {ratio_str:>12} {te['win_rate']*100:>7.1f}%")

    print("-" * 90)

    # Highlight the winner
    best = results[0]
    print(f"\n>>> BEST by PnL/DD ratio: {best['label']}) {best['name']}")
    print(f"    Test PnL: ${best['test']['total_pnl']:.2f}, MaxDD: ${best['test']['max_drawdown']:.2f}, "
          f"Ratio: {best['test']['pnl_dd_ratio']:.3f}, WR: {best['test']['win_rate']*100:.1f}%")

    # Also show best by raw PnL
    by_pnl = sorted(results, key=lambda x: x['test']['total_pnl'], reverse=True)
    best_pnl = by_pnl[0]
    print(f"\n>>> BEST by raw PnL: {best_pnl['label']}) {best_pnl['name']}")
    print(f"    Test PnL: ${best_pnl['test']['total_pnl']:.2f}, MaxDD: ${best_pnl['test']['max_drawdown']:.2f}, "
          f"Ratio: {best_pnl['test']['pnl_dd_ratio']:.3f}, WR: {best_pnl['test']['win_rate']*100:.1f}%")

    # Show any strategies that went bust
    busted = [r for r in results if r['test']['went_bust']]
    if busted:
        print(f"\n>>> WENT BUST on test: {', '.join(r['label'] + ') ' + r['name'] for r in busted)}")
    else:
        print("\n>>> No strategies went bust on test set.")

    print("\n" + "=" * 120)
    print("BACKTEST COMPLETE")
    print("=" * 120)
