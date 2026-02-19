#!/usr/bin/env python3
"""
confluence_analysis.py -- Confluence Filter Analysis for Contrarian Strategy
Feb 19, 2026

Tests additional signals/indicators that could improve the contrarian consensus strategy.
Goal: find filters that work regardless of bull/bear regime.

Uses 70/30 temporal split (same as fresh_sweep.py).
P&L: $25 flat bet, early exit at t12.5, double fee 2*0.001*25.

Tests:
  1. Spot Velocity Filter (spot confirms contrarian direction)
  2. PM Spread (liquidity) Filter
  3. PM Momentum Confirmation (t0->t5 momentum agrees)
  4. Volatility Regime Interaction
  5. Cross-Window Momentum / Double Contrarian Strength
  6. Adaptive Direction Filter (rolling WR regime detection)
  7. Multi-Signal Confluence Score
"""

import sqlite3
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
FEE_PER_SIDE = 0.001
BET_SIZE = 25.0

COLS = [
    "window_id", "asset", "window_start_utc", "window_time",
    "pm_yes_t0", "pm_yes_t2_5", "pm_yes_t5", "pm_yes_t7_5", "pm_yes_t10", "pm_yes_t12_5",
    "pm_spread_t0", "pm_spread_t5",
    "pm_price_momentum_0_to_5", "pm_price_momentum_5_to_10",
    "spot_open", "spot_close", "spot_change_pct", "spot_change_bps", "spot_range_bps",
    "spot_high", "spot_low",
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


def temporal_split(rows, frac=0.70):
    ts = sorted(set(r["window_start_utc"] for r in rows))
    cut = ts[int(len(ts) * frac)]
    train = [r for r in rows if r["window_start_utc"] < cut]
    test  = [r for r in rows if r["window_start_utc"] >= cut]
    return train, test, cut


# ── P&L Engine ───────────────────────────────────────────────────────────────

def pnl_trade(direction, entry_pm, exit_pm, bet=BET_SIZE):
    """
    direction: 'bull' or 'bear'
    For bull: buy YES at entry_pm, sell at exit_pm
    For bear: buy NO at (1-entry_pm), sell at (1-exit_pm)
    contracts = bet / entry_contract
    pnl = contracts * (exit_contract - entry_contract) - 2 * FEE_PER_SIDE * bet
    """
    if direction == "bull":
        entry_c = entry_pm
        exit_c = exit_pm
    else:
        entry_c = 1.0 - entry_pm
        exit_c = 1.0 - exit_pm
    if entry_c <= 0.01 or entry_c >= 0.99:
        return 0.0
    contracts = bet / entry_c
    gross = contracts * (exit_c - entry_c)
    fees = 2 * FEE_PER_SIDE * bet
    return gross - fees


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(trades):
    if len(trades) < 3:
        return {"n": len(trades), "wr": 0.0, "pnl": 0.0, "avg_pnl": 0.0,
                "max_dd": 0.0, "pnl_dd": 0.0, "sharpe": 0.0, "pf": 0.0}
    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n
    total = sum(pnls)
    avg = total / n
    var_ = sum((p - avg)**2 for p in pnls) / max(n - 1, 1)
    std = math.sqrt(var_) if var_ > 0 else 0.0
    sharpe = avg / std * math.sqrt(n) if std > 0 else 0.0

    gross_w = sum(p for p in pnls if p > 0)
    gross_l = sum(-p for p in pnls if p < 0)
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    eq, pk, dd = 0.0, 0.0, 0.0
    for p in pnls:
        eq += p
        pk = max(pk, eq)
        dd = max(dd, pk - eq)
    pnl_dd = total / dd if dd > 0 else (float("inf") if total > 0 else 0.0)

    return {"n": n, "wr": wr, "pnl": total, "avg_pnl": avg, "max_dd": dd,
            "pnl_dd": pnl_dd, "sharpe": sharpe, "pf": pf}


def fmt_metrics(m, label=""):
    pnl_dd_s = f"{m['pnl_dd']:.2f}" if not math.isinf(m['pnl_dd']) else "+inf"
    pf_s = f"{m['pf']:.2f}" if not math.isinf(m['pf']) else "+inf"
    s = f"N={m['n']:<5} WR={m['wr']*100:>5.1f}%  P&L=${m['pnl']:>+8.2f}  " \
        f"Avg=${m['avg_pnl']:>+.3f}  DD=${m['max_dd']:.2f}  PnL/DD={pnl_dd_s}  PF={pf_s}"
    if label:
        s = f"{label}: {s}"
    return s


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
        n_up = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] >= 0.80)
        n_down = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] <= 0.20)
        index[wt] = {"n_up": n_up, "n_down": n_down}
    return index


def identify_contrarian_trades(rows, prev_thresh=0.80, consensus_idx=None, min_agree=3):
    """
    Identify base contrarian consensus trades.
    Returns list of dicts with trade info + the original row data for filtering.
    direction = 'bull' if prev_pm >= prev_thresh (strong previous UP -> fade to bear... wait)

    Per the user's P&L spec:
      direction = "bull" if prev_pm_t12_5 >= 0.80  -- NO, re-read:
      "direction = 'bear' if prev_pm_t12_5 >= 0.80" (prev was strong bull, fade to bear)
      "direction = 'bull' if prev_pm_t12_5 <= 0.20" (prev was strong bear, fade to bull)
    """
    trades = []
    for r in rows:
        prev = r.get("prev_pm_t12_5")
        entry_pm = r.get("pm_yes_t0")
        exit_pm = r.get("pm_yes_t12_5")
        if prev is None or entry_pm is None or exit_pm is None:
            continue
        if not (0.01 < entry_pm < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        direction = None
        # Check consensus if using it
        if consensus_idx is not None:
            wt = get_wt(r)
            if wt is None or wt not in consensus_idx:
                continue
            ci = consensus_idx[wt]
            if prev >= prev_thresh and ci["n_up"] >= min_agree:
                direction = "bear"
            elif prev <= (1.0 - prev_thresh) and ci["n_down"] >= min_agree:
                direction = "bull"
        else:
            if prev >= prev_thresh:
                direction = "bear"
            elif prev <= (1.0 - prev_thresh):
                direction = "bull"

        if direction is None:
            continue

        p = pnl_trade(direction, entry_pm, exit_pm)
        trades.append({
            "pnl": p,
            "win": p > 0,
            "dir": direction,
            "asset": r["asset"],
            "ts": r["window_start_utc"],
            "entry_pm": entry_pm,
            "exit_pm": exit_pm,
            "prev_pm": prev,
            "prev2_pm": r.get("prev2_pm_t12_5"),
            "spot_change_bps": r.get("spot_change_bps"),
            "spot_change_pct": r.get("spot_change_pct"),
            "spot_open": r.get("spot_open"),
            "spot_close": r.get("spot_close"),
            "spot_range_bps": r.get("spot_range_bps"),
            "pm_spread_t0": r.get("pm_spread_t0"),
            "pm_spread_t5": r.get("pm_spread_t5"),
            "pm_momentum_0_5": r.get("pm_price_momentum_0_to_5"),
            "pm_momentum_5_10": r.get("pm_price_momentum_5_to_10"),
            "volatility_regime": r.get("volatility_regime"),
            "outcome": r.get("outcome"),
            "outcome_binary": r.get("outcome_binary"),
            "window_id": r["window_id"],
            "row": r,
        })
    return trades


def print_header(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_comparison(label, train_trades, test_trades, indent=4):
    sp = " " * indent
    tr_m = calc_metrics(train_trades)
    te_m = calc_metrics(test_trades)
    print(f"{sp}{label}")
    print(f"{sp}  TRAIN: {fmt_metrics(tr_m)}")
    print(f"{sp}  TEST:  {fmt_metrics(te_m)}")
    both_ok = tr_m["pnl"] > 0 and te_m["pnl"] > 0 and te_m["n"] >= 5
    if both_ok:
        print(f"{sp}  >>> PROFITABLE ON BOTH TRAIN AND TEST <<<")
    return tr_m, te_m, both_ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 0: BASELINE — reproduce the contrarian consensus signal
# ══════════════════════════════════════════════════════════════════════════════

def test0_baseline(train, test, full_cidx):
    print_header("TEST 0: BASELINE — Contrarian Consensus (prev>=0.80, xa3, t0->t12.5)")

    train_trades = identify_contrarian_trades(train, 0.80, full_cidx, 3)
    test_trades = identify_contrarian_trades(test, 0.80, full_cidx, 3)
    print_comparison("Contrarian Consensus Baseline", train_trades, test_trades)

    # Also show simple contrarian (no consensus) for reference
    train_simple = identify_contrarian_trades(train, 0.80, None)
    test_simple = identify_contrarian_trades(test, 0.80, None)
    print_comparison("Simple Contrarian (no consensus) Reference", train_simple, test_simple)

    # Direction breakdown
    for d in ["bull", "bear"]:
        tr_d = [t for t in train_trades if t["dir"] == d]
        te_d = [t for t in test_trades if t["dir"] == d]
        print_comparison(f"  {d.upper()} only", tr_d, te_d, indent=6)

    return train_trades, test_trades


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: SPOT VELOCITY FILTER
# ══════════════════════════════════════════════════════════════════════════════

def test1_spot_velocity(train_trades, test_trades):
    print_header("TEST 1: SPOT VELOCITY FILTER")
    print("  Does spot price at t0 already moving in the contrarian direction confirm the fade?")
    print("  Bull entries: spot moving UP (spot_change_bps > 0) confirms")
    print("  Bear entries: spot moving DOWN (spot_change_bps < 0) confirms")
    print()

    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        valid = [t for t in trades if t["spot_change_bps"] is not None]
        agrees = []
        disagrees = []
        for t in valid:
            spot_up = t["spot_change_bps"] > 0
            if (t["dir"] == "bull" and spot_up) or (t["dir"] == "bear" and not spot_up):
                agrees.append(t)
            else:
                disagrees.append(t)
        print(f"  {label}:")
        m_agree = calc_metrics(agrees)
        m_disagree = calc_metrics(disagrees)
        print(f"    Spot AGREES with contrarian: {fmt_metrics(m_agree)}")
        print(f"    Spot DISAGREES:              {fmt_metrics(m_disagree)}")
        if m_agree["n"] > 0 and m_disagree["n"] > 0:
            delta_wr = m_agree["wr"] - m_disagree["wr"]
            print(f"    WR delta (agree - disagree): {delta_wr*100:+.1f}pp")
        print()

    # Also test thresholds for spot velocity magnitude
    print("  --- Spot velocity magnitude thresholds ---")
    thresholds = [0, 1, 2, 5, 10]
    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"\n  {label}:")
        valid = [t for t in trades if t["spot_change_bps"] is not None]
        for thresh in thresholds:
            strong_agrees = []
            for t in valid:
                bps = t["spot_change_bps"]
                if t["dir"] == "bull" and bps > thresh:
                    strong_agrees.append(t)
                elif t["dir"] == "bear" and bps < -thresh:
                    strong_agrees.append(t)
            m = calc_metrics(strong_agrees)
            if m["n"] >= 3:
                print(f"    Spot agrees >|{thresh}|bps: {fmt_metrics(m)}")
            else:
                print(f"    Spot agrees >|{thresh}|bps: N={m['n']} (insufficient)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: PM SPREAD (LIQUIDITY) FILTER
# ══════════════════════════════════════════════════════════════════════════════

def test2_spread_filter(train_trades, test_trades):
    print_header("TEST 2: PM SPREAD (LIQUIDITY) FILTER")
    print("  Wide spreads = uncertain markets. Does tight spread improve quality?")
    print()

    # Compute median spread from training data
    train_spreads = [t["pm_spread_t0"] for t in train_trades if t["pm_spread_t0"] is not None]
    if not train_spreads:
        print("  No spread data available!")
        return
    train_spreads.sort()
    median_spread = train_spreads[len(train_spreads) // 2]
    print(f"  Median PM spread at t0 (from train contrarian trades): {median_spread:.4f}")
    print(f"  Spread range: {min(train_spreads):.4f} - {max(train_spreads):.4f}")
    print()

    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        valid = [t for t in trades if t["pm_spread_t0"] is not None]
        tight = [t for t in valid if t["pm_spread_t0"] < median_spread]
        wide = [t for t in valid if t["pm_spread_t0"] >= median_spread]
        print(f"  {label}:")
        print(f"    Tight spread (< {median_spread:.4f}): {fmt_metrics(calc_metrics(tight))}")
        print(f"    Wide spread  (>= {median_spread:.4f}): {fmt_metrics(calc_metrics(wide))}")
        print()

    # Test at different spread percentiles
    print("  --- Spread percentile thresholds ---")
    percentiles = [25, 50, 75]
    for pct in percentiles:
        idx = int(len(train_spreads) * pct / 100)
        thresh = train_spreads[min(idx, len(train_spreads) - 1)]
        for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
            valid = [t for t in trades if t["pm_spread_t0"] is not None]
            below = [t for t in valid if t["pm_spread_t0"] <= thresh]
            m = calc_metrics(below)
            print(f"    {label} spread <= p{pct} ({thresh:.4f}): {fmt_metrics(m)}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: PM MOMENTUM CONFIRMATION (t0->t5)
# ══════════════════════════════════════════════════════════════════════════════

def test3_pm_momentum(train_trades, test_trades):
    print_header("TEST 3: PM MOMENTUM CONFIRMATION (t0 -> t5)")
    print("  pm_price_momentum_0_to_5: if PM is moving in contrarian direction, does it confirm?")
    print("  For bull: momentum > 0 means PM rising (confirms bull)")
    print("  For bear: momentum < 0 means PM falling (confirms bear)")
    print()

    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        valid = [t for t in trades if t["pm_momentum_0_5"] is not None]
        agrees = []
        disagrees = []
        for t in valid:
            mom = t["pm_momentum_0_5"]
            if (t["dir"] == "bull" and mom > 0) or (t["dir"] == "bear" and mom < 0):
                agrees.append(t)
            else:
                disagrees.append(t)

        print(f"  {label}:")
        print(f"    PM momentum CONFIRMS contrarian:  {fmt_metrics(calc_metrics(agrees))}")
        print(f"    PM momentum OPPOSES contrarian:    {fmt_metrics(calc_metrics(disagrees))}")
        if agrees and disagrees:
            m_a = calc_metrics(agrees)
            m_d = calc_metrics(disagrees)
            print(f"    WR delta: {(m_a['wr'] - m_d['wr'])*100:+.1f}pp")
        print()

    # Momentum magnitude thresholds
    print("  --- Momentum magnitude thresholds ---")
    mom_thresholds = [0.0, 0.02, 0.05, 0.10]
    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"\n  {label}:")
        valid = [t for t in trades if t["pm_momentum_0_5"] is not None]
        for thresh in mom_thresholds:
            strong = []
            for t in valid:
                mom = t["pm_momentum_0_5"]
                if (t["dir"] == "bull" and mom > thresh) or (t["dir"] == "bear" and mom < -thresh):
                    strong.append(t)
            m = calc_metrics(strong)
            if m["n"] >= 3:
                print(f"    PM momentum confirms >|{thresh:.2f}|: {fmt_metrics(m)}")
            else:
                print(f"    PM momentum confirms >|{thresh:.2f}|: N={m['n']} (insufficient)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: VOLATILITY REGIME INTERACTION
# ══════════════════════════════════════════════════════════════════════════════

def test4_volatility_regime(train_trades, test_trades):
    print_header("TEST 4: VOLATILITY REGIME INTERACTION")
    print("  Does contrarian work better in certain vol regimes?")
    print()

    regimes = ["low", "normal", "high", "extreme"]

    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"  {label}:")
        for regime in regimes:
            rt = [t for t in trades if t["volatility_regime"] == regime]
            m = calc_metrics(rt)
            if m["n"] >= 3:
                flag = "  <<< AVOID" if m["wr"] < 0.45 and m["n"] >= 10 else ""
                flag = "  <<< STRONG" if m["wr"] > 0.55 and m["n"] >= 10 else flag
                print(f"    {regime:<10}: {fmt_metrics(m)}{flag}")
            else:
                print(f"    {regime:<10}: N={m['n']} (insufficient)")

        # Also check unknown/None
        unknown = [t for t in trades if t["volatility_regime"] not in regimes]
        if unknown:
            print(f"    {'other':<10}: {fmt_metrics(calc_metrics(unknown))}")
        print()

    # Direction x Regime
    print("  --- Direction x Regime (TEST) ---")
    for d in ["bull", "bear"]:
        for regime in regimes:
            rt = [t for t in test_trades if t["dir"] == d and t["volatility_regime"] == regime]
            m = calc_metrics(rt)
            if m["n"] >= 3:
                print(f"    {d.upper():<5} {regime:<10}: {fmt_metrics(m)}")
            elif rt:
                w = sum(1 for t in rt if t["win"])
                print(f"    {d.upper():<5} {regime:<10}: N={m['n']}, W={w} (insufficient)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: CROSS-WINDOW MOMENTUM / DOUBLE CONTRARIAN STRENGTH
# ══════════════════════════════════════════════════════════════════════════════

def test5_double_contrarian_strength(train_trades, test_trades):
    print_header("TEST 5: DOUBLE CONTRARIAN STRENGTH & PREV PM DEGREE")
    print("  How much does the DEGREE of previous window extreme matter?")
    print("  Also: does requiring double contrarian (prev2 also extreme) improve signal?")
    print()

    # 5a: Prev PM strength buckets
    print("  --- 5a: Prev PM t12.5 Strength Buckets ---")
    bear_buckets = [
        ("0.80-0.85 (bear)", 0.80, 0.85, "bear"),
        ("0.85-0.90 (bear)", 0.85, 0.90, "bear"),
        ("0.90-0.95 (bear)", 0.90, 0.95, "bear"),
        ("0.95-1.00 (bear)", 0.95, 1.01, "bear"),
    ]
    bull_buckets = [
        ("0.00-0.05 (bull)", 0.00, 0.05, "bull"),
        ("0.05-0.10 (bull)", 0.05, 0.10, "bull"),
        ("0.10-0.15 (bull)", 0.10, 0.15, "bull"),
        ("0.15-0.20 (bull)", 0.15, 0.20, "bull"),
    ]
    all_buckets = bear_buckets + bull_buckets

    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"\n  {label}:")
        for bname, lo, hi, expected_dir in all_buckets:
            bt = [t for t in trades if t["dir"] == expected_dir and t["prev_pm"] is not None
                  and lo <= t["prev_pm"] < hi]
            m = calc_metrics(bt)
            if m["n"] >= 3:
                print(f"    [{bname:<20}]: {fmt_metrics(m)}")
            elif bt:
                w = sum(1 for t in bt if t["win"])
                print(f"    [{bname:<20}]: N={m['n']}, W={w} (insufficient)")

    # 5b: Double contrarian (prev2 also extreme)
    print("\n\n  --- 5b: Double Contrarian (prev AND prev2 extreme) ---")
    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"\n  {label}:")

        # Single contrarian only (prev extreme, prev2 NOT extreme)
        single_only = [t for t in trades if t["prev2_pm"] is not None and
                       not ((t["dir"] == "bear" and t["prev2_pm"] >= 0.80) or
                            (t["dir"] == "bull" and t["prev2_pm"] <= 0.20))]
        # Double contrarian (BOTH prev and prev2 extreme)
        double = [t for t in trades if t["prev2_pm"] is not None and
                  ((t["dir"] == "bear" and t["prev2_pm"] >= 0.80) or
                   (t["dir"] == "bull" and t["prev2_pm"] <= 0.20))]
        # No prev2 data
        no_prev2 = [t for t in trades if t["prev2_pm"] is None]

        print(f"    Single contrarian only (prev2 NOT extreme): {fmt_metrics(calc_metrics(single_only))}")
        print(f"    Double contrarian (prev AND prev2 extreme):  {fmt_metrics(calc_metrics(double))}")
        print(f"    No prev2 data:                               {fmt_metrics(calc_metrics(no_prev2))}")

    # 5c: Double contrarian with different prev2 thresholds
    print("\n\n  --- 5c: Double Contrarian with Variable prev2 Thresholds ---")
    prev2_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"\n  {label}:")
        for p2t in prev2_thresholds:
            double = [t for t in trades if t["prev2_pm"] is not None and
                      ((t["dir"] == "bear" and t["prev2_pm"] >= p2t) or
                       (t["dir"] == "bull" and t["prev2_pm"] <= (1.0 - p2t)))]
            m = calc_metrics(double)
            if m["n"] >= 3:
                print(f"    prev2 threshold {p2t:.2f}: {fmt_metrics(m)}")
            else:
                print(f"    prev2 threshold {p2t:.2f}: N={m['n']} (insufficient)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: ADAPTIVE DIRECTION FILTER (Rolling WR Regime Detection)
# ══════════════════════════════════════════════════════════════════════════════

def test6_adaptive_direction(all_rows, cutoff, full_cidx):
    print_header("TEST 6: ADAPTIVE DIRECTION FILTER (Rolling WR Regime Detection)")
    print("  Rolling WR for bull vs bear over last N windows.")
    print("  If bull WR > bear WR over trailing window, go bull-only (and vice versa).")
    print("  Test N = 25, 50, 100")
    print()

    # Get all contrarian trades in chronological order
    all_trades = identify_contrarian_trades(all_rows, 0.80, full_cidx, 3)

    for window_size in [25, 50, 100]:
        print(f"\n  --- Rolling Window N = {window_size} ---")

        # Compute rolling bull/bear WR at each trade
        adaptive_trades = []
        for i, t in enumerate(all_trades):
            if i < window_size:
                continue  # Not enough history yet

            # Look at last N trades
            history = all_trades[i - window_size : i]
            bull_hist = [h for h in history if h["dir"] == "bull"]
            bear_hist = [h for h in history if h["dir"] == "bear"]

            bull_wr = sum(1 for h in bull_hist if h["win"]) / len(bull_hist) if bull_hist else 0.5
            bear_wr = sum(1 for h in bear_hist if h["win"]) / len(bear_hist) if bear_hist else 0.5

            # Adaptive: only take the direction with higher trailing WR
            if t["dir"] == "bull" and bull_wr >= bear_wr:
                adaptive_trades.append(t)
            elif t["dir"] == "bear" and bear_wr >= bull_wr:
                adaptive_trades.append(t)
            # else skip

        # Split adaptive trades into train/test
        train_adaptive = [t for t in adaptive_trades if t["ts"] < cutoff]
        test_adaptive = [t for t in adaptive_trades if t["ts"] >= cutoff]

        # Also show non-adaptive for comparison
        train_all = [t for t in all_trades if t["ts"] < cutoff and all_trades.index(t) >= window_size]
        test_all = [t for t in all_trades if t["ts"] >= cutoff]

        print(f"    Non-adaptive (all directions):")
        print(f"      TRAIN: {fmt_metrics(calc_metrics(train_all))}")
        print(f"      TEST:  {fmt_metrics(calc_metrics(test_all))}")
        print(f"    Adaptive (favor higher trailing WR direction):")
        print(f"      TRAIN: {fmt_metrics(calc_metrics(train_adaptive))}")
        print(f"      TEST:  {fmt_metrics(calc_metrics(test_adaptive))}")

        # Show what % of trades are filtered out
        test_kept = len(test_adaptive)
        test_total = len(test_all)
        pct_kept = test_kept / test_total * 100 if test_total > 0 else 0
        print(f"    Test trades kept: {test_kept}/{test_total} ({pct_kept:.0f}%)")

    # Also test: "only take when BOTH directions profitable in trailing window"
    print("\n\n  --- Alternative: Only trade when BOTH bull+bear WR > 50% in trailing window ---")
    for window_size in [25, 50, 100]:
        all_trades_list = identify_contrarian_trades(all_rows, 0.80, full_cidx, 3)
        filtered_trades = []
        for i, t in enumerate(all_trades_list):
            if i < window_size:
                continue
            history = all_trades_list[i - window_size : i]
            bull_hist = [h for h in history if h["dir"] == "bull"]
            bear_hist = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_hist if h["win"]) / len(bull_hist) if bull_hist else 0.5
            bear_wr = sum(1 for h in bear_hist if h["win"]) / len(bear_hist) if bear_hist else 0.5
            if bull_wr > 0.50 and bear_wr > 0.50:
                filtered_trades.append(t)

        train_f = [t for t in filtered_trades if t["ts"] < cutoff]
        test_f = [t for t in filtered_trades if t["ts"] >= cutoff]
        print(f"    N={window_size}:")
        print(f"      TRAIN: {fmt_metrics(calc_metrics(train_f))}")
        print(f"      TEST:  {fmt_metrics(calc_metrics(test_f))}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: MULTI-SIGNAL CONFLUENCE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def test7_confluence_score(train_trades, test_trades, train_all_spreads):
    print_header("TEST 7: MULTI-SIGNAL CONFLUENCE SCORE")
    print("  Combine: spot confirms + tight spread + right vol regime + PM momentum confirms")
    print("  Score 0-4: how many signals agree. Does requiring 2+ or 3+ improve?")
    print()

    # Compute median spread from all training spreads for threshold
    if train_all_spreads:
        train_all_spreads.sort()
        median_spread = train_all_spreads[len(train_all_spreads) // 2]
    else:
        median_spread = 0.01  # fallback

    # Determine "good" volatility regimes from training data
    # (compute WR per regime on train to decide)
    regime_wr = {}
    for regime in ["low", "normal", "high", "extreme"]:
        rt = [t for t in train_trades if t["volatility_regime"] == regime]
        if len(rt) >= 5:
            regime_wr[regime] = sum(1 for t in rt if t["win"]) / len(rt)
        else:
            regime_wr[regime] = 0.5  # neutral if not enough data

    print(f"  Training regime WRs: {', '.join(f'{k}={v*100:.1f}%' for k, v in sorted(regime_wr.items()))}")
    good_regimes = {k for k, v in regime_wr.items() if v >= 0.50}
    print(f"  'Good' regimes (WR >= 50%): {good_regimes}")
    print(f"  Median spread threshold: {median_spread:.4f}")
    print()

    def compute_confluence_score(t):
        score = 0
        signals = []

        # Signal 1: Spot velocity confirms
        if t["spot_change_bps"] is not None:
            spot_up = t["spot_change_bps"] > 0
            if (t["dir"] == "bull" and spot_up) or (t["dir"] == "bear" and not spot_up):
                score += 1
                signals.append("spot")

        # Signal 2: Tight spread
        if t["pm_spread_t0"] is not None and t["pm_spread_t0"] < median_spread:
            score += 1
            signals.append("spread")

        # Signal 3: Good volatility regime
        if t["volatility_regime"] in good_regimes:
            score += 1
            signals.append("vol")

        # Signal 4: PM momentum confirms
        if t["pm_momentum_0_5"] is not None:
            mom = t["pm_momentum_0_5"]
            if (t["dir"] == "bull" and mom > 0) or (t["dir"] == "bear" and mom < 0):
                score += 1
                signals.append("mom")

        return score, signals

    # Score all trades
    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        print(f"  {label}:")
        scored = [(t, *compute_confluence_score(t)) for t in trades]

        for min_score in range(5):
            filtered = [t for t, s, _ in scored if s >= min_score]
            m = calc_metrics(filtered)
            if m["n"] >= 3:
                print(f"    Score >= {min_score}: {fmt_metrics(m)}")
            else:
                print(f"    Score >= {min_score}: N={m['n']} (insufficient)")
        print()

    # Best confluence combos
    print("  --- Individual Signal Combinations (TEST) ---")
    signal_names = ["spot", "spread", "vol", "mom"]
    test_scored = [(t, *compute_confluence_score(t)) for t in test_trades]
    train_scored = [(t, *compute_confluence_score(t)) for t in train_trades]

    # Two-signal combos
    from itertools import combinations
    print("\n  Two-signal combinations (both must be present):")
    results_2sig = []
    for combo in combinations(signal_names, 2):
        combo_set = set(combo)
        tr_f = [t for t, _, sigs in train_scored if combo_set.issubset(set(sigs))]
        te_f = [t for t, _, sigs in test_scored if combo_set.issubset(set(sigs))]
        tr_m = calc_metrics(tr_f)
        te_m = calc_metrics(te_f)
        combo_name = "+".join(combo)
        results_2sig.append((combo_name, tr_m, te_m))
        both_ok = tr_m["pnl"] > 0 and te_m["pnl"] > 0 and te_m["n"] >= 5
        flag = " >>> BOTH PROFITABLE <<<" if both_ok else ""
        if te_m["n"] >= 3:
            print(f"    {combo_name:<16}  TRAIN: N={tr_m['n']:<4} WR={tr_m['wr']*100:>5.1f}% P&L=${tr_m['pnl']:>+7.2f}  |  "
                  f"TEST: N={te_m['n']:<4} WR={te_m['wr']*100:>5.1f}% P&L=${te_m['pnl']:>+7.2f}{flag}")

    # Three-signal combos
    print("\n  Three-signal combinations:")
    for combo in combinations(signal_names, 3):
        combo_set = set(combo)
        tr_f = [t for t, _, sigs in train_scored if combo_set.issubset(set(sigs))]
        te_f = [t for t, _, sigs in test_scored if combo_set.issubset(set(sigs))]
        tr_m = calc_metrics(tr_f)
        te_m = calc_metrics(te_f)
        combo_name = "+".join(combo)
        both_ok = tr_m["pnl"] > 0 and te_m["pnl"] > 0 and te_m["n"] >= 5
        flag = " >>> BOTH PROFITABLE <<<" if both_ok else ""
        if te_m["n"] >= 3 or tr_m["n"] >= 3:
            print(f"    {combo_name:<22}  TRAIN: N={tr_m['n']:<4} WR={tr_m['wr']*100:>5.1f}% P&L=${tr_m['pnl']:>+7.2f}  |  "
                  f"TEST: N={te_m['n']:<4} WR={te_m['wr']*100:>5.1f}% P&L=${te_m['pnl']:>+7.2f}{flag}")

    # Four-signal combo
    print("\n  Four-signal (all agree):")
    tr_f = [t for t, s, _ in train_scored if s == 4]
    te_f = [t for t, s, _ in test_scored if s == 4]
    tr_m = calc_metrics(tr_f)
    te_m = calc_metrics(te_f)
    both_ok = tr_m["pnl"] > 0 and te_m["pnl"] > 0 and te_m["n"] >= 5
    flag = " >>> BOTH PROFITABLE <<<" if both_ok else ""
    print(f"    all 4 signals       TRAIN: N={tr_m['n']:<4} WR={tr_m['wr']*100:>5.1f}% P&L=${tr_m['pnl']:>+7.2f}  |  "
          f"TEST: N={te_m['n']:<4} WR={te_m['wr']*100:>5.1f}% P&L=${te_m['pnl']:>+7.2f}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# BONUS: COMBINED BEST FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def bonus_combined_analysis(train, test, cutoff, full_cidx):
    print_header("BONUS: COMBINED BEST FILTERS ANALYSIS")
    print("  Testing promising combinations found above, all using contrarian consensus base")
    print()

    # Base contrarian trades with extra data
    train_trades = identify_contrarian_trades(train, 0.80, full_cidx, 3)
    test_trades = identify_contrarian_trades(test, 0.80, full_cidx, 3)

    # Get median spread from train
    train_spreads = sorted([t["pm_spread_t0"] for t in train_trades if t["pm_spread_t0"] is not None])
    median_spread = train_spreads[len(train_spreads) // 2] if train_spreads else 0.01

    filters = {
        "Base (no filter)": lambda t: True,
        "Tight spread only": lambda t: t["pm_spread_t0"] is not None and t["pm_spread_t0"] < median_spread,
        "Spot confirms only": lambda t: t["spot_change_bps"] is not None and (
            (t["dir"] == "bull" and t["spot_change_bps"] > 0) or
            (t["dir"] == "bear" and t["spot_change_bps"] < 0)
        ),
        "Low/normal vol only": lambda t: t["volatility_regime"] in ("low", "normal"),
        "Double contrarian only": lambda t: t["prev2_pm"] is not None and (
            (t["dir"] == "bear" and t["prev2_pm"] >= 0.80) or
            (t["dir"] == "bull" and t["prev2_pm"] <= 0.20)
        ),
        "Spot + tight spread": lambda t: (
            t["spot_change_bps"] is not None and t["pm_spread_t0"] is not None and
            t["pm_spread_t0"] < median_spread and (
                (t["dir"] == "bull" and t["spot_change_bps"] > 0) or
                (t["dir"] == "bear" and t["spot_change_bps"] < 0)
            )
        ),
        "Double + spot confirms": lambda t: (
            t["prev2_pm"] is not None and t["spot_change_bps"] is not None and
            ((t["dir"] == "bear" and t["prev2_pm"] >= 0.80 and t["spot_change_bps"] < 0) or
             (t["dir"] == "bull" and t["prev2_pm"] <= 0.20 and t["spot_change_bps"] > 0))
        ),
        "Double + tight spread": lambda t: (
            t["prev2_pm"] is not None and t["pm_spread_t0"] is not None and
            t["pm_spread_t0"] < median_spread and
            ((t["dir"] == "bear" and t["prev2_pm"] >= 0.80) or
             (t["dir"] == "bull" and t["prev2_pm"] <= 0.20))
        ),
        "Double + low/normal vol": lambda t: (
            t["prev2_pm"] is not None and t["volatility_regime"] in ("low", "normal") and
            ((t["dir"] == "bear" and t["prev2_pm"] >= 0.80) or
             (t["dir"] == "bull" and t["prev2_pm"] <= 0.20))
        ),
        "Triple: double + spot + spread": lambda t: (
            t["prev2_pm"] is not None and t["spot_change_bps"] is not None and
            t["pm_spread_t0"] is not None and t["pm_spread_t0"] < median_spread and
            ((t["dir"] == "bear" and t["prev2_pm"] >= 0.80 and t["spot_change_bps"] < 0) or
             (t["dir"] == "bull" and t["prev2_pm"] <= 0.20 and t["spot_change_bps"] > 0))
        ),
        "Extreme prev (>0.90) only": lambda t: (
            (t["dir"] == "bear" and t["prev_pm"] >= 0.90) or
            (t["dir"] == "bull" and t["prev_pm"] <= 0.10)
        ),
        "Extreme prev + spot confirms": lambda t: (
            t["spot_change_bps"] is not None and (
                (t["dir"] == "bear" and t["prev_pm"] >= 0.90 and t["spot_change_bps"] < 0) or
                (t["dir"] == "bull" and t["prev_pm"] <= 0.10 and t["spot_change_bps"] > 0)
            )
        ),
    }

    print(f"  {'Filter':<38} | {'Tr N':>5} {'Tr WR':>6} {'Tr P&L':>9} | {'Te N':>5} {'Te WR':>6} {'Te P&L':>9} {'Te DD':>7} {'PnL/DD':>7}")
    print("  " + "-" * 110)

    winners = []
    for fname, ffunc in filters.items():
        tr_f = [t for t in train_trades if ffunc(t)]
        te_f = [t for t in test_trades if ffunc(t)]
        tr_m = calc_metrics(tr_f)
        te_m = calc_metrics(te_f)
        both_ok = tr_m["pnl"] > 0 and te_m["pnl"] > 0 and te_m["n"] >= 5
        pnl_dd_s = f"{te_m['pnl_dd']:>7.2f}" if not math.isinf(te_m['pnl_dd']) else "   +inf"
        flag = " ***" if both_ok else ""
        print(f"  {fname:<38} | {tr_m['n']:>5} {tr_m['wr']*100:>5.1f}% ${tr_m['pnl']:>+7.2f} | "
              f"{te_m['n']:>5} {te_m['wr']*100:>5.1f}% ${te_m['pnl']:>+7.2f} ${te_m['max_dd']:>6.2f} {pnl_dd_s}{flag}")
        if both_ok:
            winners.append((fname, tr_m, te_m))

    print()
    if winners:
        print(f"  *** = Profitable on BOTH train and test (N >= 5)")
        print(f"\n  WINNERS ({len(winners)} filters):")
        winners.sort(key=lambda x: x[2]["pnl_dd"], reverse=True)
        for fname, tr_m, te_m in winners:
            pnl_dd_s = f"{te_m['pnl_dd']:.2f}" if not math.isinf(te_m['pnl_dd']) else "+inf"
            print(f"    {fname}")
            print(f"      Train: N={tr_m['n']}, WR={tr_m['wr']*100:.1f}%, P&L=${tr_m['pnl']:+.2f}")
            print(f"      Test:  N={te_m['n']}, WR={te_m['wr']*100:.1f}%, P&L=${te_m['pnl']:+.2f}, DD=${te_m['max_dd']:.2f}, PnL/DD={pnl_dd_s}")
    else:
        print("  No filters profitable on both train and test with N >= 5")


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL: SPOT RANGE (REALIZED VOL) as filter
# ══════════════════════════════════════════════════════════════════════════════

def test_spot_range_filter(train_trades, test_trades):
    print_header("ADDITIONAL: SPOT RANGE (REALIZED VOL) AS FILTER")
    print("  Does filtering by spot price range (intra-window volatility) improve signal?")
    print()

    # Compute median spot range from training
    train_ranges = sorted([t["spot_range_bps"] for t in train_trades if t["spot_range_bps"] is not None])
    if not train_ranges:
        print("  No spot range data!")
        return
    median_range = train_ranges[len(train_ranges) // 2]
    p25 = train_ranges[len(train_ranges) // 4]
    p75 = train_ranges[int(len(train_ranges) * 0.75)]
    print(f"  Spot range (bps) -- p25={p25:.1f}, median={median_range:.1f}, p75={p75:.1f}")
    print()

    for label, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        valid = [t for t in trades if t["spot_range_bps"] is not None]
        low_range = [t for t in valid if t["spot_range_bps"] < median_range]
        high_range = [t for t in valid if t["spot_range_bps"] >= median_range]
        print(f"  {label}:")
        print(f"    Low range (< {median_range:.1f} bps):  {fmt_metrics(calc_metrics(low_range))}")
        print(f"    High range (>= {median_range:.1f} bps): {fmt_metrics(calc_metrics(high_range))}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("  POLYNANCE CONFLUENCE ANALYSIS")
    print("  Finding additional signals to improve contrarian consensus strategy")
    print("  Direction-agnostic filters that work regardless of bull/bear regime")
    print("  P&L: $25 flat bet, early exit at t12.5, fees = 2 * 0.001 * $25 = $0.05")
    print("=" * 100)

    print("\nLoading data...")
    all_rows = load_all()
    print(f"  Total windows: {len(all_rows)}")
    print(f"  Date range: {all_rows[0]['window_start_utc'][:16]} to {all_rows[-1]['window_start_utc'][:16]}")
    for asset in ASSETS:
        n = sum(1 for r in all_rows if r["asset"] == asset)
        print(f"  {asset}: {n} windows")

    train, test, cutoff = temporal_split(all_rows, 0.70)
    print(f"\n  70/30 TEMPORAL SPLIT at {cutoff[:16]}")
    print(f"  Train: {len(train)} rows ({train[0]['window_start_utc'][:10]} to {train[-1]['window_start_utc'][:10]})")
    print(f"  Test:  {len(test)} rows ({test[0]['window_start_utc'][:10]} to {test[-1]['window_start_utc'][:10]})")

    # Build consensus index (using all data -- for identifying trades only, not for lookahead)
    # Same approach as fresh_sweep.py
    full_cidx = build_consensus_index(all_rows)

    # ── TEST 0: Baseline ─────────────────────────────────────────────────────
    train_trades, test_trades = test0_baseline(train, test, full_cidx)

    # ── TEST 1: Spot Velocity ─────────────────────────────────────────────────
    test1_spot_velocity(train_trades, test_trades)

    # ── TEST 2: PM Spread ─────────────────────────────────────────────────────
    test2_spread_filter(train_trades, test_trades)

    # ── TEST 3: PM Momentum ───────────────────────────────────────────────────
    test3_pm_momentum(train_trades, test_trades)

    # ── TEST 4: Volatility Regime ─────────────────────────────────────────────
    test4_volatility_regime(train_trades, test_trades)

    # ── TEST 5: Double Contrarian Strength ────────────────────────────────────
    test5_double_contrarian_strength(train_trades, test_trades)

    # ── TEST 6: Adaptive Direction ────────────────────────────────────────────
    test6_adaptive_direction(all_rows, cutoff, full_cidx)

    # ── TEST 7: Multi-Signal Confluence ───────────────────────────────────────
    train_spreads = [t["pm_spread_t0"] for t in train_trades if t["pm_spread_t0"] is not None]
    test7_confluence_score(train_trades, test_trades, train_spreads)

    # ── Additional: Spot Range ────────────────────────────────────────────────
    test_spot_range_filter(train_trades, test_trades)

    # ── BONUS: Combined Best Filters ──────────────────────────────────────────
    bonus_combined_analysis(train, test, cutoff, full_cidx)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print_header("FINAL SUMMARY & RECOMMENDATIONS")

    print("""
  KEY QUESTION: "What if the market flips back to bear-dominant?"

  The goal is direction-agnostic filters that improve the contrarian signal
  regardless of whether bull or bear trades dominate P&L.

  Look above for filters marked '>>> BOTH PROFITABLE <<<' or '***'.
  These are the ones that worked on both the 70% training data and
  the 30% out-of-sample test data.

  RECOMMENDED APPROACH:
  1. Start with the base contrarian consensus signal (prev >= 0.80, xa3)
  2. Layer on 1-2 confluence filters that showed robustness on BOTH sets
  3. Avoid filters that only worked on train (overfit)
  4. Monitor weekly WR and P&L -- if signal degrades, reduce sizing

  FILTER PRIORITY (check results above for which actually passed):
  - Double contrarian (prev2 also extreme) = structural, direction-agnostic
  - Spot velocity confirmation = confirms the fade is real
  - Tight spread = better liquidity, less noise
  - Volatility regime = avoid certain regimes if they consistently lose
  - PM momentum = late confirmation but may improve entry timing
  - Adaptive direction = promising but watch for lag
""")

    print("=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
