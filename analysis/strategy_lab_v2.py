#!/usr/bin/env python3
"""
strategy_lab_v2.py — Fee-Optimized Strategy Lab
Feb 22, 2026

Building on strategy_lab.py findings. Now that we have correct Polymarket crypto
fees (C × 0.25 × (p×(1-p))^2 per side, ~3-5% round-trip), this lab tests
strategies designed to MINIMIZE fee impact:

  1) Hold-to-resolution: eliminate exit fees entirely (binary P&L)
  2) Multi-window holding: skip sell/re-buy when consecutive signals agree
  3) Higher prev thresholds: fewer trades, bigger edge per trade
  4) Fee-aware minimum edge: only enter when expected profit > estimated fee
  5) Later entry timing: enter at t2.5/t5 for confirmation + better prices
  6) Combined fee-optimized strategies

All use real Polymarket crypto fee formula. NO lookahead bias.
"""

import sqlite3
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
BET_SIZE = 25.0

COLS = [
    "window_id", "asset", "window_start_utc", "window_time",
    "pm_yes_t0", "pm_yes_t2_5", "pm_yes_t5", "pm_yes_t7_5", "pm_yes_t10", "pm_yes_t12_5",
    "pm_spread_t0", "pm_spread_t5",
    "pm_price_momentum_0_to_5", "pm_price_momentum_5_to_10",
    "spot_open", "spot_close", "spot_change_pct", "spot_change_bps",
    "spot_high", "spot_low", "spot_range_bps",
    "prev_pm_t12_5", "prev2_pm_t12_5",
    "volatility_regime",
    "outcome", "outcome_binary",
]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all():
    rows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset.lower()}.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            f"SELECT {','.join(COLS)} FROM windows "
            f"WHERE outcome IS NOT NULL ORDER BY window_start_utc ASC"
        )
        for r in cur.fetchall():
            rows.append(dict(r))
        conn.close()
    rows.sort(key=lambda r: (r["window_start_utc"], r["asset"]))
    return rows


def enrich_cross_window(rows):
    """Add cross-window features that are legitimately known at t=0."""
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset"]].append(r)

    for asset, asset_rows in by_asset.items():
        asset_rows.sort(key=lambda r: r["window_start_utc"])
        for i, r in enumerate(asset_rows):
            if i > 0:
                prev = asset_rows[i - 1]
                if r["spot_open"] and prev["spot_open"] and prev["spot_open"] > 0:
                    r["xw_spot_momentum_bps"] = ((r["spot_open"] - prev["spot_open"]) / prev["spot_open"]) * 10000
                else:
                    r["xw_spot_momentum_bps"] = None
                r["prev_vol_regime"] = prev["volatility_regime"]
                r["prev_spot_range_bps"] = prev["spot_range_bps"]
                r["prev_outcome_binary"] = prev["outcome_binary"]
                r["prev_spot_close"] = prev["spot_close"]
                # Store previous window's direction signal for multi-window holding
                r["prev_window_id"] = prev["window_id"]
            else:
                r["xw_spot_momentum_bps"] = None
                r["prev_vol_regime"] = None
                r["prev_spot_range_bps"] = None
                r["prev_outcome_binary"] = None
                r["prev_spot_close"] = None
                r["prev_window_id"] = None

    return rows


def temporal_split(rows, frac=0.70):
    ts = sorted(set(r["window_start_utc"] for r in rows))
    cut = ts[int(len(ts) * frac)]
    train = [r for r in rows if r["window_start_utc"] < cut]
    test = [r for r in rows if r["window_start_utc"] >= cut]
    return train, test, cut


# ── Fee Engine ───────────────────────────────────────────────────────────────

def pm_fee(dollar_amount, price):
    """Polymarket 15-min crypto: C × 0.25 × (p × (1-p))^2 per side. C = dollar amount."""
    if price <= 0.0 or price >= 1.0:
        return 0.0
    raw = dollar_amount * 0.25 * (price * (1.0 - price)) ** 2
    return max(round(raw, 4), 0.0001)


def estimate_fee_pct(price):
    """Estimate fee as % of bet for a given contract price (entry side only).
    Since C=dollar_amount and entry dollar_amount=bet, fee_pct = 0.25 * (p*(1-p))^2."""
    return 0.25 * (price * (1.0 - price)) ** 2 if price > 0 else 0


# ── P&L Engines ──────────────────────────────────────────────────────────────

def pnl_early_exit(direction, entry_pm, exit_pm, bet=BET_SIZE):
    """Standard early exit at t12.5: pay entry + exit fees."""
    if direction == "bull":
        entry_c, exit_c = entry_pm, exit_pm
    else:
        entry_c, exit_c = 1.0 - entry_pm, 1.0 - exit_pm
    if entry_c <= 0.01 or entry_c >= 0.99:
        return 0.0
    contracts = bet / entry_c
    gross = contracts * (exit_c - entry_c)
    entry_dollars = contracts * entry_c  # = bet
    exit_dollars = contracts * exit_c
    fees = pm_fee(entry_dollars, entry_c) + pm_fee(exit_dollars, exit_c)
    return gross - fees


def pnl_hold_to_resolution(direction, entry_pm, outcome_binary, bet=BET_SIZE):
    """Hold to resolution: entry fee only. Binary outcome."""
    if direction == "bull":
        entry_c = entry_pm
        won = (outcome_binary == 1)  # price went up = YES wins
    else:
        entry_c = 1.0 - entry_pm
        won = (outcome_binary == 0)  # price went down = NO wins
    if entry_c <= 0.01 or entry_c >= 0.99:
        return 0.0
    contracts = bet / entry_c
    entry_dollars = contracts * entry_c  # = bet
    entry_fee = pm_fee(entry_dollars, entry_c)
    if won:
        # Contract resolves to $1, we paid entry_c per contract
        return contracts * (1.0 - entry_c) - entry_fee
    else:
        # Contract resolves to $0, we lose our cost
        return -(contracts * entry_c) - entry_fee


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(trades):
    if len(trades) < 3:
        return {"n": len(trades), "wr": 0.0, "pnl": 0.0, "avg_pnl": 0.0,
                "max_dd": 0.0, "pnl_dd": 0.0}
    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n
    total = sum(pnls)
    avg = total / n
    eq, pk, dd = 0.0, 0.0, 0.0
    for p in pnls:
        eq += p
        pk = max(pk, eq)
        dd = max(dd, pk - eq)
    pnl_dd = total / dd if dd > 0 else (float("inf") if total > 0 else 0.0)
    return {"n": n, "wr": wr, "pnl": total, "avg_pnl": avg, "max_dd": dd, "pnl_dd": pnl_dd}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_wt(r):
    wt = r.get("window_time")
    if wt is None:
        parts = r["window_id"].split("_")
        if len(parts) >= 3:
            wt = "_".join(parts[1:])
    return wt


def build_consensus_index(rows):
    by_wt = defaultdict(list)
    for r in rows:
        wt = get_wt(r)
        if wt:
            by_wt[wt].append(r)
    index = {}
    for wt, wrows in by_wt.items():
        n_bear_prev = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] >= 0.80)
        n_bull_prev = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] <= 0.20)
        index[wt] = {"n_bear_prev": n_bear_prev, "n_bull_prev": n_bull_prev}
    return index


def get_consensus_direction(r, cidx, prev_thresh=0.80, min_agree=3):
    """Get contrarian consensus direction for a row, or None."""
    prev = r.get("prev_pm_t12_5")
    if prev is None:
        return None
    wt = get_wt(r)
    if wt is None or wt not in cidx:
        return None
    ci = cidx[wt]
    if prev >= prev_thresh and ci["n_bear_prev"] >= min_agree:
        return "bear"
    elif prev <= (1.0 - prev_thresh) and ci["n_bull_prev"] >= min_agree:
        return "bull"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: HOLD TO RESOLUTION
# No exit fee — binary outcome. Test at various prev_thresh levels.
# ══════════════════════════════════════════════════════════════════════════════

def run_hold_to_resolution(rows, cidx, params):
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    entry_t_key = params.get("entry_t", "pm_yes_t0")
    adaptive_n = params.get("adaptive_n", 0)
    all_candidates = []

    for r in rows:
        entry_pm = r.get(entry_t_key)
        ob = r.get("outcome_binary")
        if entry_pm is None or ob is None:
            continue
        if not (0.01 < entry_pm < 0.99):
            continue
        direction = get_consensus_direction(r, cidx, prev_thresh, min_agree)
        if direction is None:
            continue

        p = pnl_hold_to_resolution(direction, entry_pm, ob)
        trade = {"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                 "ts": r["window_start_utc"]}
        all_candidates.append(trade)

    if adaptive_n > 0:
        trades = []
        for i, t in enumerate(all_candidates):
            if i < adaptive_n:
                continue
            history = all_candidates[i - adaptive_n:i]
            bull_h = [h for h in history if h["dir"] == "bull"]
            bear_h = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_h if h["win"]) / len(bull_h) if bull_h else 0.5
            bear_wr = sum(1 for h in bear_h if h["win"]) / len(bear_h) if bear_h else 0.5
            if (t["dir"] == "bull" and bull_wr >= bear_wr) or \
               (t["dir"] == "bear" and bear_wr > bull_wr):
                trades.append(t)
        return trades
    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: MULTI-WINDOW HOLDING
# When consecutive windows signal the same direction, hold through instead of
# selling and re-buying. Saves one full round-trip of fees.
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_window_hold(rows, cidx, params):
    """
    Simulate multi-window holding: when direction persists, we DON'T sell at t12.5
    and re-buy at t0 — we just hold. This saves exit+entry fees.

    For P&L:
    - First window in a streak: pay entry fee only
    - Middle windows: NO fees (holding through)
    - Last window: pay exit fee only (sell at t12.5) or hold to resolution
    """
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    hold_to_res = params.get("hold_to_resolution", False)
    adaptive_n = params.get("adaptive_n", 0)

    # Group rows by asset and time
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset"]].append(r)

    all_candidates = []

    for asset, asset_rows in by_asset.items():
        asset_rows.sort(key=lambda r: r["window_start_utc"])

        # Track streaks of same-direction signals
        streak_dir = None
        streak_entry_pm = None
        streak_entry_c = None
        streak_contracts = None
        streak_start_ts = None
        streak_windows = 0

        for i, r in enumerate(asset_rows):
            entry_pm = r.get("pm_yes_t0")
            exit_pm = r.get("pm_yes_t12_5")
            ob = r.get("outcome_binary")
            if entry_pm is None:
                # End any active streak
                if streak_dir is not None and streak_windows > 0:
                    # Close out the streak with last known values
                    streak_dir = None
                    streak_windows = 0
                continue

            direction = get_consensus_direction(r, cidx, prev_thresh, min_agree)

            # Check if we continue or break the streak
            if direction == streak_dir and streak_dir is not None:
                # SAME direction — hold through! No fees for this continuation.
                streak_windows += 1
                # The P&L for this window is just the price movement, no fees
                if exit_pm is not None and 0.01 < exit_pm < 0.99:
                    if streak_dir == "bull":
                        # We hold YES contracts, PM moved from prev exit to this exit
                        window_pnl = streak_contracts * (exit_pm - entry_pm)
                    else:
                        window_pnl = streak_contracts * ((1.0 - exit_pm) - (1.0 - entry_pm))
                    all_candidates.append({
                        "pnl": window_pnl, "win": window_pnl > 0, "dir": streak_dir,
                        "asset": asset, "ts": r["window_start_utc"], "type": "hold_through"
                    })
            else:
                # Direction changed or no signal — close old streak, maybe start new
                if streak_dir is not None and streak_windows > 0:
                    # Pay exit fee on the PREVIOUS window's exit
                    prev_r = asset_rows[i - 1]
                    prev_exit = prev_r.get("pm_yes_t12_5")
                    if prev_exit is not None and streak_contracts:
                        if streak_dir == "bull":
                            exit_dollars = streak_contracts * prev_exit
                            exit_fee = pm_fee(exit_dollars, prev_exit)
                        else:
                            exit_price = 1.0 - prev_exit
                            exit_dollars = streak_contracts * exit_price
                            exit_fee = pm_fee(exit_dollars, exit_price)
                        # Deduct exit fee from last trade in streak
                        if all_candidates and all_candidates[-1]["asset"] == asset:
                            all_candidates[-1]["pnl"] -= exit_fee
                            all_candidates[-1]["win"] = all_candidates[-1]["pnl"] > 0

                # Start new streak if we have a direction
                if direction is not None and exit_pm is not None and 0.01 < entry_pm < 0.99:
                    streak_dir = direction
                    streak_entry_pm = entry_pm
                    if direction == "bull":
                        streak_entry_c = entry_pm
                    else:
                        streak_entry_c = 1.0 - entry_pm
                    streak_contracts = BET_SIZE / streak_entry_c if streak_entry_c > 0.01 else 0
                    streak_start_ts = r["window_start_utc"]
                    streak_windows = 1

                    # First window: pay entry fee, compute P&L
                    entry_dollars = streak_contracts * streak_entry_c  # = BET_SIZE
                    entry_fee = pm_fee(entry_dollars, streak_entry_c)
                    if direction == "bull":
                        gross = streak_contracts * (exit_pm - entry_pm)
                    else:
                        gross = streak_contracts * ((1.0 - exit_pm) - (1.0 - entry_pm))
                    all_candidates.append({
                        "pnl": gross - entry_fee, "win": (gross - entry_fee) > 0,
                        "dir": direction, "asset": asset, "ts": r["window_start_utc"],
                        "type": "streak_start"
                    })
                else:
                    streak_dir = None
                    streak_windows = 0

    # Sort by timestamp for adaptive filter
    all_candidates.sort(key=lambda t: t["ts"])

    if adaptive_n > 0:
        trades = []
        for i, t in enumerate(all_candidates):
            if i < adaptive_n:
                continue
            history = all_candidates[i - adaptive_n:i]
            bull_h = [h for h in history if h["dir"] == "bull"]
            bear_h = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_h if h["win"]) / len(bull_h) if bull_h else 0.5
            bear_wr = sum(1 for h in bear_h if h["win"]) / len(bear_h) if bear_h else 0.5
            if (t["dir"] == "bull" and bull_wr >= bear_wr) or \
               (t["dir"] == "bear" and bear_wr > bull_wr):
                trades.append(t)
        return trades
    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: HIGHER PREV THRESHOLD
# (Tested via params in standard contrarian — just using run_contrarian_v2)
# ══════════════════════════════════════════════════════════════════════════════

def run_contrarian_v2(rows, cidx, params):
    """Standard contrarian consensus with early exit, correct fees. Like v1 but simpler."""
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    entry_t_key = params.get("entry_t", "pm_yes_t0")
    exit_t_key = params.get("exit_t", "pm_yes_t12_5")
    adaptive_n = params.get("adaptive_n", 0)
    min_fee_edge = params.get("min_fee_edge", 0)  # minimum expected edge over fees

    all_candidates = []
    for r in rows:
        entry_pm = r.get(entry_t_key)
        exit_pm = r.get(exit_t_key)
        if entry_pm is None or exit_pm is None:
            continue
        if not (0.01 < entry_pm < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        direction = get_consensus_direction(r, cidx, prev_thresh, min_agree)
        if direction is None:
            continue

        # Fee-aware filter: estimate if this trade's entry price makes fees too expensive
        if min_fee_edge > 0:
            if direction == "bull":
                entry_c = entry_pm
            else:
                entry_c = 1.0 - entry_pm
            fee_pct = estimate_fee_pct(entry_c)
            # Need at least min_fee_edge% expected gross profit above 2x fee_pct (round trip)
            if fee_pct * 2 > min_fee_edge:
                continue

        p = pnl_early_exit(direction, entry_pm, exit_pm)
        trade = {"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                 "ts": r["window_start_utc"]}
        all_candidates.append(trade)

    if adaptive_n > 0:
        trades = []
        for i, t in enumerate(all_candidates):
            if i < adaptive_n:
                continue
            history = all_candidates[i - adaptive_n:i]
            bull_h = [h for h in history if h["dir"] == "bull"]
            bear_h = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_h if h["win"]) / len(bull_h) if bull_h else 0.5
            bear_wr = sum(1 for h in bear_h if h["win"]) / len(bear_h) if bear_h else 0.5
            if (t["dir"] == "bull" and bull_wr >= bear_wr) or \
               (t["dir"] == "bear" and bear_wr > bull_wr):
                trades.append(t)
        return trades
    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: LATER ENTRY TIMING
# Enter at t2.5 or t5 for direction confirmation + potentially better prices.
# ══════════════════════════════════════════════════════════════════════════════

def run_later_entry(rows, cidx, params):
    """Enter at t2.5 or t5 instead of t0. Confirms direction, may get better price."""
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    entry_t_key = params.get("entry_t", "pm_yes_t2_5")
    exit_t_key = params.get("exit_t", "pm_yes_t12_5")
    adaptive_n = params.get("adaptive_n", 0)
    # Require entry PM has moved in our direction (confirmation)
    require_confirmation = params.get("require_confirmation", False)

    all_candidates = []
    for r in rows:
        # Direction is determined at t0 (consensus from prev window)
        direction = get_consensus_direction(r, cidx, prev_thresh, min_agree)
        if direction is None:
            continue

        entry_pm = r.get(entry_t_key)
        exit_pm = r.get(exit_t_key)
        pm0 = r.get("pm_yes_t0")
        if entry_pm is None or exit_pm is None or pm0 is None:
            continue
        if not (0.01 < entry_pm < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        # Confirmation: entry PM has moved in contrarian direction vs t0
        if require_confirmation:
            if direction == "bull" and entry_pm <= pm0:
                continue  # PM hasn't moved up yet — skip
            if direction == "bear" and entry_pm >= pm0:
                continue  # PM hasn't moved down yet — skip

        p = pnl_early_exit(direction, entry_pm, exit_pm)
        trade = {"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                 "ts": r["window_start_utc"]}
        all_candidates.append(trade)

    if adaptive_n > 0:
        trades = []
        for i, t in enumerate(all_candidates):
            if i < adaptive_n:
                continue
            history = all_candidates[i - adaptive_n:i]
            bull_h = [h for h in history if h["dir"] == "bull"]
            bear_h = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_h if h["win"]) / len(bull_h) if bull_h else 0.5
            bear_wr = sum(1 for h in bear_h if h["win"]) / len(bear_h) if bear_h else 0.5
            if (t["dir"] == "bull" and bull_wr >= bear_wr) or \
               (t["dir"] == "bear" and bear_wr > bull_wr):
                trades.append(t)
        return trades
    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6: HYBRID — Hold to resolution + later entry
# ══════════════════════════════════════════════════════════════════════════════

def run_hold_later_entry(rows, cidx, params):
    """Later entry + hold to resolution. Best of both worlds: confirmation + no exit fee."""
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    entry_t_key = params.get("entry_t", "pm_yes_t2_5")
    adaptive_n = params.get("adaptive_n", 0)
    require_confirmation = params.get("require_confirmation", False)

    all_candidates = []
    for r in rows:
        direction = get_consensus_direction(r, cidx, prev_thresh, min_agree)
        if direction is None:
            continue

        entry_pm = r.get(entry_t_key)
        ob = r.get("outcome_binary")
        pm0 = r.get("pm_yes_t0")
        if entry_pm is None or ob is None or pm0 is None:
            continue
        if not (0.01 < entry_pm < 0.99):
            continue

        if require_confirmation:
            if direction == "bull" and entry_pm <= pm0:
                continue
            if direction == "bear" and entry_pm >= pm0:
                continue

        p = pnl_hold_to_resolution(direction, entry_pm, ob)
        trade = {"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                 "ts": r["window_start_utc"]}
        all_candidates.append(trade)

    if adaptive_n > 0:
        trades = []
        for i, t in enumerate(all_candidates):
            if i < adaptive_n:
                continue
            history = all_candidates[i - adaptive_n:i]
            bull_h = [h for h in history if h["dir"] == "bull"]
            bear_h = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_h if h["win"]) / len(bull_h) if bull_h else 0.5
            bear_wr = sum(1 for h in bear_h if h["win"]) / len(bear_h) if bear_h else 0.5
            if (t["dir"] == "bull" and bull_wr >= bear_wr) or \
               (t["dir"] == "bear" and bear_wr > bull_wr):
                trades.append(t)
        return trades
    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 115
    print(SEP)
    print("  POLYNANCE STRATEGY LAB v2 — Fee-Optimized Strategies")
    print("  Clean backtest: NO lookahead, 70/30 temporal split")
    print("  Correct fees: C × 0.25 × (p×(1-p))^2 per side (~1.5% at p=0.50)")
    print(SEP)

    print("\nLoading data...")
    all_rows = load_all()
    print(f"  Total windows: {len(all_rows)}")
    print(f"  Date range: {all_rows[0]['window_start_utc'][:16]} to {all_rows[-1]['window_start_utc'][:16]}")

    print("\nEnriching cross-window features...")
    all_rows = enrich_cross_window(all_rows)

    train, test, cutoff = temporal_split(all_rows, 0.70)
    print(f"\n  70/30 SPLIT at {cutoff[:16]}")
    print(f"  Train: {len(train)} rows  Test: {len(test)} rows\n")

    train_cidx = build_consensus_index(train)
    test_cidx = build_consensus_index(test)

    results = []

    def run_and_record(name, func, rows_tr, cidx_tr, rows_te, cidx_te, params):
        tr_trades = func(rows_tr, cidx_tr, params)
        te_trades = func(rows_te, cidx_te, params)
        tr_m = calc_metrics(tr_trades)
        te_m = calc_metrics(te_trades)
        results.append((name, tr_m, te_m))

    # ── Section A: BASELINES (from v1, for comparison) ────────────────────────
    print("  Running Section A: Baselines...")

    run_and_record("BASELINE: early_exit p80_xa3",
        run_contrarian_v2, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3})

    run_and_record("BASELINE: early_exit p80_xa3 adapt50",
        run_contrarian_v2, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "adaptive_n": 50})

    # ── Section B: HOLD TO RESOLUTION ─────────────────────────────────────────
    print("  Running Section B: Hold to resolution...")

    for pt in [0.80, 0.85, 0.90, 0.95]:
        run_and_record(f"HOLD_RES: p{int(pt*100)}_xa3",
            run_hold_to_resolution, train, train_cidx, test, test_cidx,
            {"prev_thresh": pt, "min_agree": 3})

    for pt in [0.80, 0.85, 0.90]:
        run_and_record(f"HOLD_RES: p{int(pt*100)}_xa4",
            run_hold_to_resolution, train, train_cidx, test, test_cidx,
            {"prev_thresh": pt, "min_agree": 4})

    # Hold to res + adaptive
    run_and_record("HOLD_RES: p80_xa3 adapt50",
        run_hold_to_resolution, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "adaptive_n": 50})

    run_and_record("HOLD_RES: p80_xa3 adapt25",
        run_hold_to_resolution, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "adaptive_n": 25})

    run_and_record("HOLD_RES: p85_xa3 adapt50",
        run_hold_to_resolution, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.85, "min_agree": 3, "adaptive_n": 50})

    run_and_record("HOLD_RES: p90_xa3 adapt50",
        run_hold_to_resolution, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.90, "min_agree": 3, "adaptive_n": 50})

    # ── Section C: HIGHER PREV THRESHOLD (early exit) ─────────────────────────
    print("  Running Section C: Higher prev thresholds...")

    for pt in [0.85, 0.90, 0.95]:
        run_and_record(f"HIGH_THRESH: p{int(pt*100)}_xa3",
            run_contrarian_v2, train, train_cidx, test, test_cidx,
            {"prev_thresh": pt, "min_agree": 3})

    for pt in [0.85, 0.90, 0.95]:
        run_and_record(f"HIGH_THRESH: p{int(pt*100)}_xa3 adapt50",
            run_contrarian_v2, train, train_cidx, test, test_cidx,
            {"prev_thresh": pt, "min_agree": 3, "adaptive_n": 50})

    # ── Section D: MULTI-WINDOW HOLDING ───────────────────────────────────────
    print("  Running Section D: Multi-window holding...")

    run_and_record("MULTI_HOLD: p80_xa3",
        run_multi_window_hold, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3})

    run_and_record("MULTI_HOLD: p85_xa3",
        run_multi_window_hold, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.85, "min_agree": 3})

    run_and_record("MULTI_HOLD: p80_xa3 adapt50",
        run_multi_window_hold, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "adaptive_n": 50})

    # ── Section E: LATER ENTRY TIMING ─────────────────────────────────────────
    print("  Running Section E: Later entry timing...")

    # Entry at t2.5
    run_and_record("LATE_ENTRY: t2.5 p80_xa3",
        run_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t2_5"})

    run_and_record("LATE_ENTRY: t5 p80_xa3",
        run_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t5"})

    # With confirmation
    run_and_record("LATE_ENTRY: t2.5 confirm p80_xa3",
        run_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t2_5", "require_confirmation": True})

    run_and_record("LATE_ENTRY: t5 confirm p80_xa3",
        run_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t5", "require_confirmation": True})

    # Later entry + adaptive
    run_and_record("LATE_ENTRY: t2.5 p80_xa3 adapt50",
        run_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t2_5", "adaptive_n": 50})

    run_and_record("LATE_ENTRY: t5 p80_xa3 adapt50",
        run_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t5", "adaptive_n": 50})

    # ── Section F: HYBRID — Hold to resolution + later entry ──────────────────
    print("  Running Section F: Hybrid hold+later entry...")

    run_and_record("HYBRID: hold_res t2.5 p80_xa3",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t2_5"})

    run_and_record("HYBRID: hold_res t5 p80_xa3",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t5"})

    run_and_record("HYBRID: hold_res t2.5 confirm p80_xa3",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t2_5", "require_confirmation": True})

    run_and_record("HYBRID: hold_res t5 confirm p80_xa3",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t5", "require_confirmation": True})

    # Hybrid + adaptive
    run_and_record("HYBRID: hold_res t2.5 p80_xa3 adapt50",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t2_5", "adaptive_n": 50})

    run_and_record("HYBRID: hold_res t0 p80_xa3 adapt50",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.80, "min_agree": 3, "entry_t": "pm_yes_t0", "adaptive_n": 50})

    run_and_record("HYBRID: hold_res t0 p85_xa3 adapt50",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.85, "min_agree": 3, "entry_t": "pm_yes_t0", "adaptive_n": 50})

    run_and_record("HYBRID: hold_res t0 p90_xa3 adapt50",
        run_hold_later_entry, train, train_cidx, test, test_cidx,
        {"prev_thresh": 0.90, "min_agree": 3, "entry_t": "pm_yes_t0", "adaptive_n": 50})

    # ── Section G: FEE-AWARE EDGE FILTER ──────────────────────────────────────
    print("  Running Section G: Fee-aware edge filter...")

    # Only enter when estimated round-trip fee < X% of bet
    for edge in [0.02, 0.025, 0.03]:
        run_and_record(f"FEE_FILTER: max_rt_fee<{edge:.1%} p80_xa3",
            run_contrarian_v2, train, train_cidx, test, test_cidx,
            {"prev_thresh": 0.80, "min_agree": 3, "min_fee_edge": edge})

    for edge in [0.02, 0.025]:
        run_and_record(f"FEE_FILTER: max_rt_fee<{edge:.1%} p80_xa3 adapt50",
            run_contrarian_v2, train, train_cidx, test, test_cidx,
            {"prev_thresh": 0.80, "min_agree": 3, "min_fee_edge": edge, "adaptive_n": 50})

    # ══════════════════════════════════════════════════════════════════════════
    # FEE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  Running fee analysis on baseline trades...")
    baseline_trades = run_contrarian_v2(all_rows, build_consensus_index(all_rows),
                                        {"prev_thresh": 0.80, "min_agree": 3})
    if baseline_trades:
        total_gross = 0
        total_fees = 0
        for t_row in all_rows:
            entry_pm = t_row.get("pm_yes_t0")
            exit_pm = t_row.get("pm_yes_t12_5")
            if entry_pm and exit_pm and 0.01 < entry_pm < 0.99 and 0.01 < exit_pm < 0.99:
                direction = get_consensus_direction(t_row, build_consensus_index(all_rows), 0.80, 3)
                if direction:
                    if direction == "bull":
                        ec, xc = entry_pm, exit_pm
                    else:
                        ec, xc = 1.0 - entry_pm, 1.0 - exit_pm
                    if 0.01 < ec < 0.99:
                        contracts = BET_SIZE / ec
                        gross = contracts * (xc - ec)
                        entry_dollars = contracts * ec  # = BET_SIZE
                        exit_dollars = contracts * xc
                        fees = pm_fee(entry_dollars, ec) + pm_fee(exit_dollars, xc)
                        total_gross += gross
                        total_fees += fees

        print(f"  Total gross P&L (before fees): ${total_gross:+,.2f}")
        print(f"  Total fees paid:               ${total_fees:,.2f}")
        print(f"  Net P&L (after fees):          ${total_gross - total_fees:+,.2f}")
        print(f"  Fees as % of gross:            {total_fees / abs(total_gross) * 100:.1f}%" if total_gross != 0 else "")

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    results.sort(key=lambda x: x[2]["pnl_dd"], reverse=True)

    print(f"\n{SEP}")
    print(f"  FULL RESULTS — sorted by Test PnL/DD")
    print(f"{SEP}")
    hdr = f"  {'Strategy':<52s} | {'Tr N':>5s} {'Tr WR':>6s} {'Tr P&L':>11s} | {'Te N':>5s} {'Te WR':>6s} {'Te P&L':>11s} {'Te DD':>8s} {'PnL/DD':>7s}"
    print(hdr)
    print("  " + "-" * 113)
    for name, tr, te in results:
        flag = " ***" if tr["pnl"] > 0 and te["pnl"] > 0 and te["n"] >= 10 else ""
        print(f"  {name:<52s} | {tr['n']:>5d} {tr['wr']:>5.1%} $ {tr['pnl']:>+9.2f} "
              f"| {te['n']:>5d} {te['wr']:>5.1%} $ {te['pnl']:>+9.2f} ${te['max_dd']:>7.2f} {te['pnl_dd']:>7.2f}{flag}")

    # Winners only
    winners = [(n, tr, te) for n, tr, te in results if tr["pnl"] > 0 and te["pnl"] > 0 and te["n"] >= 10]
    if winners:
        print(f"\n{SEP}")
        print(f"  WINNERS: Profitable on BOTH train AND test (N>=10) — {len(winners)}/{len(results)}")
        print(f"{SEP}")
        hdr2 = f"  {'#':>3s} {'Strategy':<52s} | {'Te N':>5s} {'Te WR':>6s} {'Te P&L':>11s} {'Te DD':>8s} {'PnL/DD':>7s} | {'Tr WR':>6s} {'Tr P&L':>11s}"
        print(hdr2)
        print("  " + "-" * 113)
        for i, (name, tr, te) in enumerate(winners, 1):
            print(f"  {i:>3d} {name:<52s} | {te['n']:>5d} {te['wr']:>5.1%} $ {te['pnl']:>+9.2f} ${te['max_dd']:>7.2f} {te['pnl_dd']:>7.2f}"
                  f" | {tr['wr']:>5.1%} $ {tr['pnl']:>+9.2f}")

    # Avg win/loss analysis for top strategies
    print(f"\n{SEP}")
    print(f"  RISK PROFILE: Avg Win vs Avg Loss for top strategies")
    print(f"{SEP}")
    for name, tr, te in results[:10]:
        if te["n"] >= 10:
            # Re-run to get individual trades
            print(f"  {name}: N={te['n']}, WR={te['wr']:.1%}, "
                  f"Avg={te['avg_pnl']:+.2f}/trade, PnL/DD={te['pnl_dd']:.2f}")

    print(f"\n{SEP}")
    print(f"  STRATEGY LAB v2 COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
