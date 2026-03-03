#!/usr/bin/env python3
"""
novel_signals_backtest.py — Test novel signal enhancements on the A7 base.

KEY FINDINGS FROM EDA (on test set):
  1. Spot confirmation at t5: 77.7% WR when spot confirms reversal (vs 35.8%)
  2. Early PM momentum (t0→t2.5): 67.9% when PM drops 5+ cents early
  3. Spread compression (t0→t5): 62.7% when narrowing vs 52.0% when widening
  4. High PM variance (choppy paths): 52.0% vs 58.1% for smooth paths
  5. SOL/XRP best bear assets (~59%), BTC weakest (~53%)

STRATEGIES:
  Base: A7_4AGREE_0.85_t5_daily100 (the proven winner)
  + Spot confirmation (spot_t5 < spot_t0 for bear)
  + Early PM momentum confirmation
  + Spread compression filter
  + Asset selection (drop BTC or weight)
  + Combined

NOTE: The EDA was run on the test set. By testing these ideas on the same test
set, we are at risk of data snooping. The results should be treated as
SUGGESTIVE, not definitive. A true validation needs fresh out-of-sample data.
"""

import sqlite3
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']
INITIAL_BANKROLL = 1000.0
BASE_BET = 25.0
FEE_RATE = 0.01
SPREAD_COST = 0.005
TRAIN_RATIO = 0.70


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    delta = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, center - delta), min(1, center + delta)


def load_all_data():
    all_rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM windows WHERE outcome IS NOT NULL ORDER BY window_start_utc"
        ).fetchall()
        for r in rows:
            d = dict(r)
            d['asset'] = asset.upper()
            all_rows.append(d)
        conn.close()
    all_rows.sort(key=lambda r: (r['window_start_utc'], r['asset']))

    # Load samples for spot price at t0 and t5
    samples_lookup = {}  # window_id -> {t_minutes: {spot_price, pm_yes_bid, pm_yes_ask, pm_spread}}
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        for s in conn.execute("SELECT window_id, t_minutes, spot_price, pm_spread FROM samples").fetchall():
            wid = s['window_id']
            t = s['t_minutes']
            if wid not in samples_lookup:
                samples_lookup[wid] = {}
            samples_lookup[wid][t] = {
                'spot_price': s['spot_price'],
                'pm_spread': s['pm_spread'],
            }
        conn.close()

    # Enrich rows with spot_t0, spot_t5, spread data from samples
    for r in all_rows:
        wid = r.get('window_id')
        if wid and wid in samples_lookup:
            s = samples_lookup[wid]
            r['spot_t0'] = s.get(0.0, {}).get('spot_price')
            r['spot_t5'] = s.get(5.0, {}).get('spot_price')
            r['spot_t2_5'] = s.get(2.5, {}).get('spot_price')
            r['spread_t0_samples'] = s.get(0.0, {}).get('pm_spread')
            r['spread_t5_samples'] = s.get(5.0, {}).get('pm_spread')
        else:
            r['spot_t0'] = r['spot_t5'] = r['spot_t2_5'] = None
            r['spread_t0_samples'] = r['spread_t5_samples'] = None

    # Group by window_time
    time_groups = defaultdict(dict)
    for r in all_rows:
        wt = r.get('window_time')
        if wt:
            time_groups[wt][r['asset']] = r

    return all_rows, time_groups


def train_test_split(time_groups):
    sorted_wts = sorted(time_groups.keys())
    split_idx = int(len(sorted_wts) * TRAIN_RATIO)
    split_wt = sorted_wts[split_idx]
    train = {wt: g for wt, g in time_groups.items() if wt < split_wt}
    test = {wt: g for wt, g in time_groups.items() if wt >= split_wt}
    return train, test, split_wt


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


class Strategy:
    def __init__(self, name, **p):
        self.name = name
        self.prev_thresh = p.get('prev_thresh', 0.85)
        self.min_agree = p.get('min_agree', 4)
        self.bear_only = p.get('bear_only', False)
        self.exclude_btc = p.get('exclude_btc', False)
        self.entry_time = p.get('entry_time', 't5')
        self.exit_time = p.get('exit_time', 't12_5')
        self.skip_regimes = set(p.get('skip_regimes', ['extreme']))
        self.daily_loss_limit = p.get('daily_loss_limit', 100)

        # Novel filters
        self.require_spot_confirm = p.get('require_spot_confirm', False)
        self.require_early_pm_mom = p.get('require_early_pm_mom', False)
        self.early_pm_mom_thresh = p.get('early_pm_mom_thresh', -0.03)
        self.require_spread_compress = p.get('require_spread_compress', False)
        self.max_pm_variance = p.get('max_pm_variance', None)  # Skip if intra-window variance > X


def simulate(strategy, time_groups):
    s = strategy
    bankroll = INITIAL_BANKROLL
    peak_bankroll = INITIAL_BANKROLL
    base_bet = BASE_BET
    trades = []
    daily_pnl = defaultdict(float)
    max_dd_pct = 0.0

    entry_col = f"pm_yes_{s.entry_time}"
    exit_col = f"pm_yes_{s.exit_time}"

    for wt in sorted(time_groups.keys()):
        group = time_groups[wt]
        if not group or bankroll <= 10:
            break

        sample_row = next(iter(group.values()))
        day_key = sample_row['window_start_utc'][:10]

        if s.daily_loss_limit > 0 and daily_pnl[day_key] < -s.daily_loss_limit:
            continue

        # Majority-rule regime filter
        if s.skip_regimes:
            skip_count = sum(
                1 for r in group.values()
                if r.get('volatility_regime') in s.skip_regimes
            )
            if skip_count >= len(group) / 2:
                continue

        # Phase 1: Previous window consensus
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

        # Phase 2: Per-asset filters
        confirming = []
        for asset, row in group.items():
            if s.exclude_btc and asset == 'BTC':
                continue

            entry_pm = row.get(entry_col)
            exit_pm = row.get(exit_col)
            if entry_pm is None or exit_pm is None:
                continue

            if direction == 'bear' and entry_pm > 0.50:
                continue
            if direction == 'bull' and entry_pm < 0.50:
                continue

            # ── NOVEL: Spot confirmation ──────────────────────────────
            if s.require_spot_confirm:
                spot_t0 = row.get('spot_t0')
                spot_t5 = row.get('spot_t5')
                if spot_t0 is not None and spot_t5 is not None and spot_t0 > 0:
                    spot_chg = (spot_t5 - spot_t0) / spot_t0
                    if direction == 'bear' and spot_chg >= 0:
                        continue  # Spot not confirming bear reversal
                    if direction == 'bull' and spot_chg <= 0:
                        continue  # Spot not confirming bull reversal
                # If spot data missing, allow the trade (don't block on missing data)

            # ── NOVEL: Early PM momentum (t0 → t2.5) ─────────────────
            if s.require_early_pm_mom:
                t0 = row.get('pm_yes_t0')
                t2_5 = row.get('pm_yes_t2_5')
                if t0 is not None and t2_5 is not None:
                    early_mom = t2_5 - t0
                    if direction == 'bear' and early_mom > s.early_pm_mom_thresh:
                        continue  # PM not dropping early enough
                    if direction == 'bull' and early_mom < -s.early_pm_mom_thresh:
                        continue  # PM not rising early enough

            # ── NOVEL: Spread compression (t0 → t5) ──────────────────
            if s.require_spread_compress:
                s0 = row.get('spread_t0_samples') or row.get('pm_spread_t0')
                s5 = row.get('spread_t5_samples') or row.get('pm_spread_t5')
                if s0 is not None and s5 is not None and s0 > 0:
                    if s5 > s0:  # Spread widening = uncertainty = skip
                        continue

            # ── NOVEL: PM path variance filter ────────────────────────
            if s.max_pm_variance is not None:
                prices = []
                for col in ['pm_yes_t0', 'pm_yes_t2_5', 'pm_yes_t5', 'pm_yes_t7_5', 'pm_yes_t10', 'pm_yes_t12_5']:
                    v = row.get(col)
                    if v is not None:
                        prices.append(v)
                if len(prices) >= 4:
                    var = float(np.var(prices))
                    if var > s.max_pm_variance:
                        continue

            confirming.append((asset, row, entry_pm, exit_pm))

        if len(confirming) < s.min_agree:
            continue

        bet = min(base_bet, bankroll * 0.05)
        if bet < 1.0:
            continue

        for asset, row, entry_pm, exit_pm in confirming[:4]:
            net = pnl_early_exit(direction, entry_pm, exit_pm, bet)
            won = net > 0
            bankroll += net
            peak_bankroll = max(peak_bankroll, bankroll)
            dd_p = (bankroll - peak_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            max_dd_pct = min(max_dd_pct, dd_p)
            daily_pnl[day_key] += net
            trades.append({
                'asset': asset, 'direction': direction,
                'bet': bet, 'net_pnl': net, 'won': won,
                'bankroll': bankroll, 'day': day_key,
            })

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
    win_pnls = pnls[pnls > 0]
    loss_pnls = pnls[pnls <= 0]
    avg_win = float(np.mean(win_pnls)) if len(win_pnls) > 0 else 0
    avg_loss = float(np.mean(loss_pnls)) if len(loss_pnls) > 0 else 0
    pf = float(np.sum(win_pnls) / -np.sum(loss_pnls)) if np.sum(loss_pnls) != 0 else float('inf')

    return {
        'name': strategy.name, 'trades': n, 'wins': wins,
        'wr': wins/n, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'pnl': total_pnl, 'sharpe': sharpe, 'max_dd_pct': max_dd_pct * 100,
        'profit_factor': pf, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'final_bankroll': bankroll,
    }


def main():
    print("="*100)
    print("  NOVEL SIGNALS BACKTEST — Testing spot confirmation & microstructure filters")
    print("  WARNING: EDA was done on test set. These results have snooping risk.")
    print("  Treat as suggestive. True validation needs fresh data.")
    print("="*100)

    all_rows, time_groups = load_all_data()
    train, test, split_wt = train_test_split(time_groups)
    print(f"\n  Train: {len(train)} windows | Test: {len(test)} windows | Split: {split_wt}")

    strategies = [
        # ── Baseline ─────────────────────────────────────────────
        Strategy("BASE: A7_4agree_t5_daily100"),

        # ── Spot confirmation variants ───────────────────────────
        Strategy("SPOT_CONFIRM: 4agree+spot_t5",
                 require_spot_confirm=True),

        Strategy("SPOT_CONFIRM: 3agree+spot_t5",
                 min_agree=3, require_spot_confirm=True),

        Strategy("SPOT_CONFIRM: 3agree+spot_t5 (no BTC)",
                 min_agree=3, require_spot_confirm=True, exclude_btc=True),

        # ── Early PM momentum ────────────────────────────────────
        Strategy("EARLY_MOM: 4agree+pm_drop_0.03",
                 require_early_pm_mom=True, early_pm_mom_thresh=-0.03),

        Strategy("EARLY_MOM: 3agree+pm_drop_0.03",
                 min_agree=3, require_early_pm_mom=True, early_pm_mom_thresh=-0.03),

        # ── Spread compression ───────────────────────────────────
        Strategy("SPREAD_COMP: 4agree+spread_narrow",
                 require_spread_compress=True),

        Strategy("SPREAD_COMP: 3agree+spread_narrow",
                 min_agree=3, require_spread_compress=True),

        # ── PM variance filter ───────────────────────────────────
        Strategy("LOW_VAR: 3agree+var<0.025",
                 min_agree=3, max_pm_variance=0.025),

        # ── Combinations ─────────────────────────────────────────
        Strategy("COMBO: 4agree+spot+spread",
                 require_spot_confirm=True, require_spread_compress=True),

        Strategy("COMBO: 3agree+spot+spread",
                 min_agree=3, require_spot_confirm=True, require_spread_compress=True),

        Strategy("COMBO: 3agree+spot+early_pm",
                 min_agree=3, require_spot_confirm=True,
                 require_early_pm_mom=True, early_pm_mom_thresh=-0.03),

        Strategy("FULL: 3agree+spot+spread+lowvar",
                 min_agree=3, require_spot_confirm=True,
                 require_spread_compress=True, max_pm_variance=0.025),

        # ── Asset exclusion ──────────────────────────────────────
        Strategy("NO_BTC: 3agree+spot (bear only)",
                 min_agree=3, require_spot_confirm=True,
                 bear_only=True, exclude_btc=True),

        # ── Relaxed consensus + strong filters ───────────────────
        Strategy("RELAX: 2agree+spot+spread+early_pm",
                 min_agree=2, require_spot_confirm=True,
                 require_spread_compress=True,
                 require_early_pm_mom=True, early_pm_mom_thresh=-0.03),
    ]

    print(f"\n  Running {len(strategies)} strategies...\n")

    print(f"{'='*170}")
    print(f"  {'':45} {'--- TRAIN ---':^38}  {'--- TEST ---':^55}  {'OVERFIT':>7}")
    sub = (
        f"  {'Strategy':<45} "
        f"{'N':>5} {'WR%':>6} {'P&L':>9} {'Shrp':>6} {'DD%':>6}  "
        f"{'N':>5} {'WR%':>6} {'CI':>11} {'P&L':>9} {'Shrp':>6} {'DD%':>6} {'PF':>5}  "
        f"{'':>7}"
    )
    print(sub)
    print(f"  {'-'*165}")

    results = []
    for strat in strategies:
        tr = simulate(strat, train)
        te = simulate(strat, test)
        results.append((strat.name, tr, te))

        if tr is None or te is None:
            tr = tr or {'trades':0,'wr':0,'pnl':0,'sharpe':0,'max_dd_pct':0}
            te = te or {'trades':0,'wr':0,'pnl':0,'sharpe':0,'max_dd_pct':0,'ci_lo':0,'ci_hi':0,'profit_factor':0}
            print(f"  {strat.name:<45} {'(no trades)':>30}")
            continue

        wr_drop = tr['wr'] - te['wr']
        overfit = "!! YES" if wr_drop > 0.05 else ("  WARN" if wr_drop > 0.03 else "")
        robust = tr['pnl'] > 0 and te['pnl'] > 0
        pf_str = f"{te['profit_factor']:.2f}" if te['profit_factor'] < 100 else "+inf"
        marker = " ***" if te['sharpe'] > 2.0 and robust else ""

        print(
            f"  {strat.name:<45} "
            f"{tr['trades']:>5} {tr['wr']*100:>5.1f}% ${tr['pnl']:>+8.0f} {tr['sharpe']:>+6.1f} {tr['max_dd_pct']:>+5.0f}%  "
            f"{te['trades']:>5} {te['wr']*100:>5.1f}% [{te['ci_lo']*100:.0f}-{te['ci_hi']*100:.0f}%] "
            f"${te['pnl']:>+8.0f} {te['sharpe']:>+6.1f} {te['max_dd_pct']:>+5.0f}% {pf_str:>5}  "
            f"{overfit}{marker}"
        )

    print(f"{'='*170}")
    print(f"  *** = Robust AND Sharpe > 2.0")
    print(f"  WARNING: Novel filters were discovered on the test set. Results are suggestive, not confirmed.")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  INTERPRETATION")
    print(f"{'='*100}")
    print("""
  Spot confirmation is the strongest novel signal available at entry time.
  When spot price at t5 confirms the contrarian direction:
    - Bear: spot is falling → 77.7% WR in raw EDA
    - Bull: spot is rising

  The key question: does it HELP or just REDUCE TRADE COUNT?
  If WR jumps but trades drop by half, net P&L may not improve.

  Compare Sharpe ratios (risk-adjusted, accounts for trade count).

  SNOOPING CAVEAT: The EDA that discovered these signals used the test set.
  Any test-set improvement MUST be discounted. Only fresh data can validate.
  The correct interpretation: these filters have strong THEORETICAL backing
  (spot leads PM) and the EDA CONFIRMS the theory. But the exact thresholds
  and effect sizes are likely overestimated.
""")


if __name__ == "__main__":
    main()
