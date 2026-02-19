#!/usr/bin/env python3
"""
fresh_sweep.py — Comprehensive diagnosis of live strategy degradation + alternative sweep
Feb 19 2026

Analyzes why contrarian_consensus (prev=0.80, xa3, bull/bear=0.50, t0->t12.5) is losing money.
70/30 temporal split. No cherry-picking.
"""

import sqlite3
import math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
FEE_RATE = 0.001     # 0.1% per side (Polymarket)
SPREAD_COST = 0.005  # 0.5% slippage per side
BET_SIZE = 25.0

COLS = [
    "window_id", "asset", "window_start_utc", "window_time",
    "pm_yes_t0", "pm_yes_t2_5", "pm_yes_t5", "pm_yes_t7_5", "pm_yes_t10", "pm_yes_t12_5",
    "pm_spread_t0", "pm_spread_t5",
    "pm_price_momentum_0_to_5", "pm_price_momentum_5_to_10",
    "spot_open", "spot_close", "spot_change_bps", "spot_range_bps",
    "prev_pm_t12_5", "prev2_pm_t12_5",
    "volatility_regime",
    "outcome", "outcome_binary",
]


# ── Data loading ─────────────────────────────────────────────────────────────

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
            d = dict(r)
            rows.append(d)
        conn.close()
    rows.sort(key=lambda r: (r["window_start_utc"], r["asset"]))
    return rows


def temporal_split(rows, frac=0.70):
    """70/30 chronological split by unique timestamp."""
    ts = sorted(set(r["window_start_utc"] for r in rows))
    cut = ts[int(len(ts) * frac)]
    train = [r for r in rows if r["window_start_utc"] < cut]
    test  = [r for r in rows if r["window_start_utc"] >= cut]
    return train, test, cut


# ── Cross-asset consensus ────────────────────────────────────────────────────

def build_consensus_index(rows):
    """Build lookup: window_time -> {n_strong_up, n_strong_down, n_dbl_up, n_dbl_down}"""
    by_wt = defaultdict(list)
    for r in rows:
        wt = r.get("window_time")
        if wt is None:
            parts = r["window_id"].split("_")
            if len(parts) >= 3:
                wt = "_".join(parts[1:])
        if wt:
            by_wt[wt].append(r)

    index = {}
    for wt, wrows in by_wt.items():
        n_up = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] >= 0.75)
        n_down = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] <= 0.25)
        n_dbl_up = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r.get("prev2_pm_t12_5") is not None
                       and r["prev_pm_t12_5"] >= 0.75 and r["prev2_pm_t12_5"] >= 0.75)
        n_dbl_down = sum(1 for r in wrows if r.get("prev_pm_t12_5") is not None and r.get("prev2_pm_t12_5") is not None
                         and r["prev_pm_t12_5"] <= 0.25 and r["prev2_pm_t12_5"] <= 0.25)
        index[wt] = {"n_up": n_up, "n_down": n_down, "n_dbl_up": n_dbl_up, "n_dbl_down": n_dbl_down}
    return index


def get_wt(r):
    wt = r.get("window_time")
    if wt is None:
        parts = r["window_id"].split("_")
        if len(parts) >= 3:
            wt = "_".join(parts[1:])
    return wt


# ── P&L engine ───────────────────────────────────────────────────────────────

def pnl_early_exit(direction, entry_pm, exit_pm, bet=BET_SIZE):
    """Early exit P&L with fees and spread."""
    if direction == "bull":
        entry_c, exit_c = entry_pm, exit_pm
    else:
        entry_c, exit_c = 1.0 - entry_pm, 1.0 - exit_pm
    if entry_c <= 0.01:
        return 0.0
    n = bet / entry_c
    gross = n * (exit_c - entry_c)
    fees = entry_c * n * FEE_RATE + exit_c * n * FEE_RATE
    spread = SPREAD_COST * bet + SPREAD_COST * (n * exit_c)
    return gross - fees - spread


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(trades):
    """Calculate comprehensive metrics from list of trade dicts."""
    if len(trades) < 3:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "avg_pnl": 0.0, "max_dd": 0.0,
                "pnl_dd": 0.0, "sharpe": 0.0, "pf": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n
    total = sum(pnls)
    avg = total / n
    var_ = sum((p - avg)**2 for p in pnls) / max(n-1, 1)
    std = math.sqrt(var_) if var_ > 0 else 0.0
    sharpe = avg / std * math.sqrt(n) if std > 0 else 0.0

    gross_w = sum(p for p in pnls if p > 0)
    gross_l = sum(-p for p in pnls if p < 0)
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    # Max drawdown
    eq, pk, dd = 0.0, 0.0, 0.0
    for p in pnls:
        eq += p
        pk = max(pk, eq)
        dd = max(dd, pk - eq)

    pnl_dd = total / dd if dd > 0 else (float("inf") if total > 0 else 0.0)

    # Wilson 95% CI
    z = 1.96
    denom = 1 + z*z/n
    center = (wr + z*z/(2*n)) / denom
    delta = z * math.sqrt(wr*(1-wr)/n + z*z/(4*n*n)) / denom
    ci_lo = max(0.0, center - delta)
    ci_hi = min(1.0, center + delta)

    return {"n": n, "wr": wr, "pnl": total, "avg_pnl": avg, "max_dd": dd,
            "pnl_dd": pnl_dd, "sharpe": sharpe, "pf": pf, "ci_lo": ci_lo, "ci_hi": ci_hi}


def _week_label(ts_str):
    months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    try:
        dt = datetime.fromisoformat(ts_str.replace("+00:00","")).replace(tzinfo=timezone.utc)
        week_start = dt - timedelta(days=dt.weekday())
        return f"{months[week_start.month]}-{week_start.day:02d}"
    except Exception:
        return ts_str[:10]


def _day_label(ts_str):
    try:
        return ts_str[:10]  # YYYY-MM-DD
    except Exception:
        return ts_str


# ── Signal functions ─────────────────────────────────────────────────────────

def make_contrarian_consensus(prev_thresh, bull_th, bear_th, entry_col, exit_col, min_agree, consensus_idx):
    """Current live strategy: contrarian + cross-asset consensus."""
    def signal(r):
        prev = r.get("prev_pm_t12_5")
        ep = r.get(entry_col)
        xp = r.get(exit_col)
        if prev is None or ep is None or xp is None:
            return None
        wt = get_wt(r)
        if wt is None or wt not in consensus_idx:
            return None
        ci = consensus_idx[wt]
        # Bear: prev strong UP + consensus agrees
        if prev >= prev_thresh and ci["n_up"] >= min_agree and ep <= bear_th:
            return {"dir": "bear", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bear", ep, xp)}
        # Bull: prev strong DOWN + consensus agrees
        if prev <= (1.0 - prev_thresh) and ci["n_down"] >= min_agree and ep >= bull_th:
            return {"dir": "bull", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bull", ep, xp)}
        return None
    return signal


def make_simple_contrarian(prev_thresh, bull_th, bear_th, entry_col, exit_col):
    """Simple contrarian without consensus."""
    def signal(r):
        prev = r.get("prev_pm_t12_5")
        ep = r.get(entry_col)
        xp = r.get(exit_col)
        if prev is None or ep is None or xp is None:
            return None
        if prev >= prev_thresh and ep <= bear_th:
            return {"dir": "bear", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bear", ep, xp)}
        if prev <= (1.0 - prev_thresh) and ep >= bull_th:
            return {"dir": "bull", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bull", ep, xp)}
        return None
    return signal


def make_bull_only_contrarian(prev_thresh, bull_th, entry_col, exit_col):
    """Bull-only contrarian."""
    def signal(r):
        prev = r.get("prev_pm_t12_5")
        ep = r.get(entry_col)
        xp = r.get(exit_col)
        if prev is None or ep is None or xp is None:
            return None
        if prev <= (1.0 - prev_thresh) and ep >= bull_th:
            return {"dir": "bull", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bull", ep, xp)}
        return None
    return signal


def make_bear_only_contrarian(prev_thresh, bear_th, entry_col, exit_col):
    """Bear-only contrarian."""
    def signal(r):
        prev = r.get("prev_pm_t12_5")
        ep = r.get(entry_col)
        xp = r.get(exit_col)
        if prev is None or ep is None or xp is None:
            return None
        if prev >= prev_thresh and ep <= bear_th:
            return {"dir": "bear", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bear", ep, xp)}
        return None
    return signal


def make_consensus_contrarian(prev_thresh, bull_th, bear_th, entry_col, exit_col, min_agree, consensus_idx):
    """Consensus contrarian (same as live but parameterized)."""
    return make_contrarian_consensus(prev_thresh, bull_th, bear_th, entry_col, exit_col, min_agree, consensus_idx)


def make_bull_only_consensus(prev_thresh, bull_th, entry_col, exit_col, min_agree, consensus_idx):
    """Bull-only with consensus."""
    def signal(r):
        prev = r.get("prev_pm_t12_5")
        ep = r.get(entry_col)
        xp = r.get(exit_col)
        if prev is None or ep is None or xp is None:
            return None
        wt = get_wt(r)
        if wt is None or wt not in consensus_idx:
            return None
        ci = consensus_idx[wt]
        if prev <= (1.0 - prev_thresh) and ci["n_down"] >= min_agree and ep >= bull_th:
            return {"dir": "bull", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bull", ep, xp)}
        return None
    return signal


def make_bear_only_consensus(prev_thresh, bear_th, entry_col, exit_col, min_agree, consensus_idx):
    """Bear-only with consensus."""
    def signal(r):
        prev = r.get("prev_pm_t12_5")
        ep = r.get(entry_col)
        xp = r.get(exit_col)
        if prev is None or ep is None or xp is None:
            return None
        wt = get_wt(r)
        if wt is None or wt not in consensus_idx:
            return None
        ci = consensus_idx[wt]
        if prev >= prev_thresh and ci["n_up"] >= min_agree and ep <= bear_th:
            return {"dir": "bear", "ep": ep, "xp": xp, "pnl": pnl_early_exit("bear", ep, xp)}
        return None
    return signal


def make_asset_filtered(signal_fn, allowed_assets):
    """Wrap a signal function with asset filter."""
    def filtered(r):
        if r["asset"] not in allowed_assets:
            return None
        return signal_fn(r)
    return filtered


def run_signal(rows, signal_fn):
    """Run signal over rows, return list of trade dicts."""
    trades = []
    for r in rows:
        sig = signal_fn(r)
        if sig is None:
            continue
        if not (0.01 < sig["ep"] < 0.99 and 0.01 < sig["xp"] < 0.99):
            continue
        is_win = sig["pnl"] > 0
        trades.append({
            "pnl": sig["pnl"],
            "win": is_win,
            "dir": sig["dir"],
            "asset": r["asset"],
            "ts": r["window_start_utc"],
            "day": _day_label(r["window_start_utc"]),
            "week": _week_label(r["window_start_utc"]),
            "prev_pm": r.get("prev_pm_t12_5"),
            "entry_pm": sig["ep"],
            "outcome": r.get("outcome"),
        })
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: DIAGNOSE THE CURRENT STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def part1_diagnosis(train, test, train_cidx, test_cidx, cutoff):
    print("\n" + "=" * 100)
    print("PART 1: DIAGNOSIS OF CURRENT LIVE STRATEGY")
    print("  contrarian_consensus | prev=0.80 | xa3 | bull/bear=0.50 | entry=t0 | exit=t12.5")
    print("=" * 100)

    # Current live strategy
    sig_fn = make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, {**train_cidx, **test_cidx})

    train_trades = run_signal(train, sig_fn)
    test_trades = run_signal(test, sig_fn)
    all_trades = run_signal(train + test, sig_fn)

    tr_m = calc_metrics(train_trades)
    te_m = calc_metrics(test_trades)

    print(f"\n  TRAIN ({train[0]['window_start_utc'][:10]} to {cutoff[:10]})")
    print(f"    Trades: {tr_m['n']:<6}  WR: {tr_m['wr']*100:.1f}%  [{tr_m['ci_lo']*100:.0f}-{tr_m['ci_hi']*100:.0f}% CI]")
    print(f"    P&L: ${tr_m['pnl']:+.2f}   Avg: ${tr_m['avg_pnl']:+.3f}   MaxDD: ${tr_m['max_dd']:.2f}   PF: {tr_m['pf']:.2f}")

    print(f"\n  TEST ({cutoff[:10]} to {test[-1]['window_start_utc'][:10]})")
    print(f"    Trades: {te_m['n']:<6}  WR: {te_m['wr']*100:.1f}%  [{te_m['ci_lo']*100:.0f}-{te_m['ci_hi']*100:.0f}% CI]")
    print(f"    P&L: ${te_m['pnl']:+.2f}   Avg: ${te_m['avg_pnl']:+.3f}   MaxDD: ${te_m['max_dd']:.2f}   PF: {te_m['pf']:.2f}")

    # --- Breakdown by direction ---
    print("\n  --- By Direction (TEST) ---")
    for d in ["bull", "bear"]:
        dt = [t for t in test_trades if t["dir"] == d]
        if len(dt) >= 3:
            m = calc_metrics(dt)
            print(f"    {d.upper():<6}  N={m['n']:<5}  WR={m['wr']*100:.1f}%  P&L=${m['pnl']:+.2f}  Avg=${m['avg_pnl']:+.3f}")
        else:
            print(f"    {d.upper():<6}  N={len(dt)} (insufficient)")

    # --- Breakdown by asset ---
    print("\n  --- By Asset (TEST) ---")
    for asset in ASSETS:
        at = [t for t in test_trades if t["asset"] == asset]
        if len(at) >= 3:
            m = calc_metrics(at)
            print(f"    {asset:<6}  N={m['n']:<5}  WR={m['wr']*100:.1f}%  P&L=${m['pnl']:+.2f}  Avg=${m['avg_pnl']:+.3f}")
        else:
            print(f"    {asset:<6}  N={len(at)} (insufficient)")

    # --- Breakdown by day ---
    print("\n  --- By Day (TEST) ---")
    days = sorted(set(t["day"] for t in test_trades))
    print(f"    {'Day':<12}  {'N':>4}  {'WR':>7}  {'P&L':>9}  {'Cum':>9}")
    cum = 0.0
    for day in days:
        dt = [t for t in test_trades if t["day"] == day]
        if dt:
            m = calc_metrics(dt)
            cum += m["pnl"]
            flag = "  <<<" if m["pnl"] < -5 else ""
            print(f"    {day:<12}  {m['n']:>4}  {m['wr']*100:>5.1f}%  ${m['pnl']:>+8.2f}  ${cum:>+8.2f}{flag}")

    # --- Breakdown by prev_pm strength ---
    print("\n  --- By Prev PM Strength Buckets (TEST) ---")
    buckets = [
        ("0.80-0.85", 0.80, 0.85),
        ("0.85-0.90", 0.85, 0.90),
        ("0.90-0.95", 0.90, 0.95),
        ("0.95-1.00", 0.95, 1.01),
        # also the bull side
        ("0.00-0.05 (bull)", 0.00, 0.05),
        ("0.05-0.10 (bull)", 0.05, 0.10),
        ("0.10-0.15 (bull)", 0.10, 0.15),
        ("0.15-0.20 (bull)", 0.15, 0.20),
    ]
    for label, lo, hi in buckets:
        bt = [t for t in test_trades if t["prev_pm"] is not None and lo <= t["prev_pm"] < hi]
        if len(bt) >= 3:
            m = calc_metrics(bt)
            print(f"    prev [{label:<16}]  N={m['n']:<4}  WR={m['wr']*100:.1f}%  P&L=${m['pnl']:+.2f}")
        elif bt:
            print(f"    prev [{label:<16}]  N={len(bt)} (insufficient)")

    # --- Breakdown by direction + asset ---
    print("\n  --- By Direction + Asset (TEST) ---")
    for d in ["bull", "bear"]:
        for asset in ASSETS:
            dt = [t for t in test_trades if t["dir"] == d and t["asset"] == asset]
            if len(dt) >= 3:
                m = calc_metrics(dt)
                print(f"    {d.upper():<5} {asset:<5}  N={m['n']:<4}  WR={m['wr']*100:.1f}%  P&L=${m['pnl']:+.2f}")
            elif dt:
                wins = sum(1 for t in dt if t["win"])
                print(f"    {d.upper():<5} {asset:<5}  N={len(dt):<4}  W={wins}")

    return train_trades, test_trades


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: STRATEGY SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def part2_sweep(train, test, train_cidx, test_cidx):
    print("\n\n" + "=" * 100)
    print("PART 2: STRATEGY SWEEP — TRAIN vs TEST")
    print("=" * 100)

    full_cidx = {**train_cidx, **test_cidx}

    strategies = {}

    # ── BASELINE: Current live config ─────────────────────────────────────
    strategies["BASELINE: cc_p80_xa3_b50_t0->t12.5"] = \
        make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)

    # ── DIRECTION VARIANTS ────────────────────────────────────────────────
    strategies["BULL_ONLY: p80_b50_t0->t12.5"] = \
        make_bull_only_contrarian(0.80, 0.50, "pm_yes_t0", "pm_yes_t12_5")
    strategies["BEAR_ONLY: p80_r50_t0->t12.5"] = \
        make_bear_only_contrarian(0.80, 0.50, "pm_yes_t0", "pm_yes_t12_5")

    # ── HIGHER PREV THRESHOLD ─────────────────────────────────────────────
    for pt in [0.85, 0.90]:
        strategies[f"cc_p{int(pt*100)}_xa3_b50_t0->t12.5"] = \
            make_contrarian_consensus(pt, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)
        strategies[f"simple_p{int(pt*100)}_b50_t0->t12.5"] = \
            make_simple_contrarian(pt, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5")
        strategies[f"BULL_ONLY: p{int(pt*100)}_b50_t0->t12.5"] = \
            make_bull_only_contrarian(pt, 0.50, "pm_yes_t0", "pm_yes_t12_5")
        strategies[f"BEAR_ONLY: p{int(pt*100)}_r50_t0->t12.5"] = \
            make_bear_only_contrarian(pt, 0.50, "pm_yes_t0", "pm_yes_t12_5")

    # ── HIGHER BULL THRESHOLD ─────────────────────────────────────────────
    for bt in [0.55, 0.60]:
        strategies[f"cc_p80_xa3_b{int(bt*100)}_r50_t0->t12.5"] = \
            make_contrarian_consensus(0.80, bt, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)

    # ── HIGHER BEAR THRESHOLD (more selective bears) ──────────────────────
    for brt in [0.45, 0.40]:
        strategies[f"cc_p80_xa3_b50_r{int(brt*100)}_t0->t12.5"] = \
            make_contrarian_consensus(0.80, 0.50, brt, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)

    # ── DIFFERENT CONSENSUS REQUIREMENTS ──────────────────────────────────
    for xa in [2, 4]:
        strategies[f"cc_p80_xa{xa}_b50_t0->t12.5"] = \
            make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", xa, full_cidx)

    # ── DIFFERENT EXIT TIMES ──────────────────────────────────────────────
    for exit_col, exit_name in [("pm_yes_t10", "t10"), ("pm_yes_t12_5", "t12.5")]:
        if exit_name != "t12.5":  # t12.5 already covered
            strategies[f"cc_p80_xa3_b50_t0->{exit_name}"] = \
                make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", exit_col, 3, full_cidx)

    # ── DIFFERENT ENTRY TIMES ─────────────────────────────────────────────
    for entry_col, entry_name in [("pm_yes_t5", "t5"), ("pm_yes_t2_5", "t2.5")]:
        strategies[f"cc_p80_xa3_b50_{entry_name}->t12.5"] = \
            make_contrarian_consensus(0.80, 0.50, 0.50, entry_col, "pm_yes_t12_5", 3, full_cidx)
        strategies[f"simple_p80_b50_{entry_name}->t12.5"] = \
            make_simple_contrarian(0.80, 0.50, 0.50, entry_col, "pm_yes_t12_5")

    # ── COMBINATION: bull-only + higher prev ──────────────────────────────
    for pt in [0.85, 0.90]:
        strategies[f"BULL_ONLY+cc: p{int(pt*100)}_xa3_b50_t0->t12.5"] = \
            make_bull_only_consensus(pt, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)
        strategies[f"BEAR_ONLY+cc: p{int(pt*100)}_xa3_r50_t0->t12.5"] = \
            make_bear_only_consensus(pt, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)

    # ── BULL-ONLY + higher bull threshold ─────────────────────────────────
    strategies["BULL_ONLY: p80_b55_t0->t12.5"] = \
        make_bull_only_contrarian(0.80, 0.55, "pm_yes_t0", "pm_yes_t12_5")

    # ── ENTRY at t5 variants (more info available) ────────────────────────
    strategies["BEAR_ONLY: p80_r50_t5->t12.5"] = \
        make_bear_only_contrarian(0.80, 0.50, "pm_yes_t5", "pm_yes_t12_5")
    strategies["BULL_ONLY: p80_b50_t5->t12.5"] = \
        make_bull_only_contrarian(0.80, 0.50, "pm_yes_t5", "pm_yes_t12_5")
    strategies["BEAR_ONLY: p85_r50_t5->t12.5"] = \
        make_bear_only_contrarian(0.85, 0.50, "pm_yes_t5", "pm_yes_t12_5")
    strategies["BEAR_ONLY: p90_r50_t5->t12.5"] = \
        make_bear_only_contrarian(0.90, 0.50, "pm_yes_t5", "pm_yes_t12_5")

    # ── ASSET-FILTERED VARIANTS ───────────────────────────────────────────
    base_sig = make_simple_contrarian(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5")
    strategies["NO_BTC: simple_p80_b50_t0->t12.5"] = \
        make_asset_filtered(base_sig, {"ETH", "SOL", "XRP"})
    strategies["SOL+XRP: simple_p80_b50_t0->t12.5"] = \
        make_asset_filtered(base_sig, {"SOL", "XRP"})

    base_sig_cc = make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)
    strategies["NO_BTC: cc_p80_xa3_b50_t0->t12.5"] = \
        make_asset_filtered(base_sig_cc, {"ETH", "SOL", "XRP"})

    # Bear-only no BTC
    bear_sig = make_bear_only_contrarian(0.80, 0.50, "pm_yes_t5", "pm_yes_t12_5")
    strategies["BEAR_NO_BTC: p80_r50_t5->t12.5"] = \
        make_asset_filtered(bear_sig, {"ETH", "SOL", "XRP"})
    bear_sig_90 = make_bear_only_contrarian(0.90, 0.50, "pm_yes_t5", "pm_yes_t12_5")
    strategies["BEAR_NO_BTC: p90_r50_t5->t12.5"] = \
        make_asset_filtered(bear_sig_90, {"ETH", "SOL", "XRP"})

    # ── RESOLUTION-HOLD (no early exit) variants ──────────────────────────
    # Simulated: entry at t0, hold to resolution (use outcome instead of t12.5)
    def make_resolution_contrarian(prev_thresh, bull_th, bear_th, entry_col):
        """Hold to binary resolution instead of early exit."""
        def signal(r):
            prev = r.get("prev_pm_t12_5")
            ep = r.get(entry_col)
            outcome = r.get("outcome")
            if prev is None or ep is None or outcome is None:
                return None
            if prev >= prev_thresh and ep <= bear_th:
                direction = "bear"
                entry_c = 1.0 - ep
            elif prev <= (1.0 - prev_thresh) and ep >= bull_th:
                direction = "bull"
                entry_c = ep
            else:
                return None
            if entry_c <= 0.01:
                return None
            n = BET_SIZE / entry_c
            fee = entry_c * n * FEE_RATE
            spread = SPREAD_COST * BET_SIZE
            won = (direction == "bear" and outcome == "down") or (direction == "bull" and outcome == "up")
            if won:
                pnl_val = n * (1.0 - entry_c) - fee - spread
            else:
                pnl_val = -BET_SIZE - fee - spread
            return {"dir": direction, "ep": ep, "xp": ep, "pnl": pnl_val}
        return signal

    strategies["RESOLUTION: simple_p80_b50_t0"] = \
        make_resolution_contrarian(0.80, 0.50, 0.50, "pm_yes_t0")
    strategies["RESOLUTION: simple_p85_b50_t0"] = \
        make_resolution_contrarian(0.85, 0.50, 0.50, "pm_yes_t0")
    strategies["RESOLUTION: simple_p90_b50_t0"] = \
        make_resolution_contrarian(0.90, 0.50, 0.50, "pm_yes_t0")

    # ── Asymmetric thresholds ─────────────────────────────────────────────
    strategies["ASYM: p80_b55_r45_t0->t12.5"] = \
        make_simple_contrarian(0.80, 0.55, 0.45, "pm_yes_t0", "pm_yes_t12_5")
    strategies["ASYM: p80_b60_r40_t0->t12.5"] = \
        make_simple_contrarian(0.80, 0.60, 0.40, "pm_yes_t0", "pm_yes_t12_5")
    strategies["ASYM_cc: p80_xa3_b55_r45_t0->t12.5"] = \
        make_contrarian_consensus(0.80, 0.55, 0.45, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)
    strategies["ASYM_cc: p80_xa3_b60_r40_t0->t12.5"] = \
        make_contrarian_consensus(0.80, 0.60, 0.40, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)

    # ── t5 entry + consensus ──────────────────────────────────────────────
    for xa in [2, 3, 4]:
        strategies[f"cc_p80_xa{xa}_b50_t5->t12.5"] = \
            make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t5", "pm_yes_t12_5", xa, full_cidx)

    # ── xa4 (all 4 agree) variants ───────────────────────────────────────
    strategies["cc_p80_xa4_b50_t0->t12.5"] = \
        make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 4, full_cidx)
    strategies["cc_p75_xa4_b50_t0->t12.5"] = \
        make_contrarian_consensus(0.75, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 4, full_cidx)
    strategies["cc_p75_xa4_b50_t5->t12.5"] = \
        make_contrarian_consensus(0.75, 0.50, 0.50, "pm_yes_t5", "pm_yes_t12_5", 4, full_cidx)

    # ── Run all ───────────────────────────────────────────────────────────
    results = []
    for name, sig_fn in strategies.items():
        tr_trades = run_signal(train, sig_fn)
        te_trades = run_signal(test, sig_fn)
        tr_m = calc_metrics(tr_trades)
        te_m = calc_metrics(te_trades)
        results.append({
            "name": name,
            "tr": tr_m,
            "te": te_m,
        })

    # Sort by test PnL/DD ratio
    results.sort(key=lambda x: x["te"]["pnl_dd"], reverse=True)

    # Print table
    print(f"\n  {'Strategy':<48} |  {'Tr N':>5} {'Tr WR':>6} {'Tr P&L':>9} |  {'Te N':>5} {'Te WR':>6} {'Te P&L':>9} {'Te DD':>7} {'PnL/DD':>7} {'Te CI':>12}")
    print("  " + "-" * 140)

    for r in results:
        tr, te = r["tr"], r["te"]
        if te["n"] < 3 and tr["n"] < 3:
            continue
        pnl_dd_str = f"{te['pnl_dd']:>7.2f}" if not math.isinf(te["pnl_dd"]) else "   +inf"
        print(f"  {r['name']:<48} |  {tr['n']:>5} {tr['wr']*100:>5.1f}% ${tr['pnl']:>+8.2f} |  "
              f"{te['n']:>5} {te['wr']*100:>5.1f}% ${te['pnl']:>+8.2f} ${te['max_dd']:>6.2f} {pnl_dd_str} "
              f"[{te['ci_lo']*100:.0f}-{te['ci_hi']*100:.0f}%]")

    # Highlight strategies profitable on BOTH
    both_profit = [r for r in results if r["tr"]["pnl"] > 0 and r["te"]["pnl"] > 0 and r["te"]["n"] >= 5]
    both_profit.sort(key=lambda x: x["te"]["pnl_dd"], reverse=True)

    print(f"\n\n  STRATEGIES PROFITABLE ON BOTH TRAIN AND TEST: {len(both_profit)}")
    print(f"  {'Strategy':<48} |  {'Tr N':>5} {'Tr WR':>6} {'Tr P&L':>9} |  {'Te N':>5} {'Te WR':>6} {'Te P&L':>9} {'PnL/DD':>7}")
    print("  " + "-" * 110)
    for r in both_profit:
        tr, te = r["tr"], r["te"]
        pnl_dd_str = f"{te['pnl_dd']:>7.2f}" if not math.isinf(te["pnl_dd"]) else "   +inf"
        print(f"  {r['name']:<48} |  {tr['n']:>5} {tr['wr']*100:>5.1f}% ${tr['pnl']:>+8.2f} |  "
              f"{te['n']:>5} {te['wr']*100:>5.1f}% ${te['pnl']:>+8.2f} {pnl_dd_str}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: WHAT CHANGED?
# ══════════════════════════════════════════════════════════════════════════════

def part3_regime_analysis(train, test, cutoff):
    print("\n\n" + "=" * 100)
    print("PART 3: WHAT CHANGED? — Train vs Test Regime Analysis")
    print("=" * 100)

    # 3a: PM predictive power — correlation between t0 pm_yes and outcome
    print("\n  --- 3a: PM Predictive Power (t0 pm_yes vs outcome) ---")
    for label, data in [("TRAIN", train), ("TEST", test)]:
        # Compute correlation between pm_yes_t0 and outcome_binary
        pairs = [(r["pm_yes_t0"], r["outcome_binary"]) for r in data
                 if r.get("pm_yes_t0") is not None and r.get("outcome_binary") is not None]
        if len(pairs) < 10:
            print(f"    {label}: insufficient data")
            continue
        n = len(pairs)
        pm_vals = [p[0] for p in pairs]
        out_vals = [p[1] for p in pairs]
        mean_pm = sum(pm_vals) / n
        mean_out = sum(out_vals) / n
        cov = sum((pm_vals[i] - mean_pm) * (out_vals[i] - mean_out) for i in range(n)) / n
        std_pm = math.sqrt(sum((p - mean_pm)**2 for p in pm_vals) / n)
        std_out = math.sqrt(sum((o - mean_out)**2 for o in out_vals) / n)
        corr = cov / (std_pm * std_out) if std_pm > 0 and std_out > 0 else 0

        # Win rate when PM strongly disagrees with 50/50
        strong_up = [p for p in pairs if p[0] >= 0.65]
        strong_down = [p for p in pairs if p[0] <= 0.35]
        up_wr = sum(1 for p in strong_up if p[1] == 1) / len(strong_up) if strong_up else 0
        down_wr = sum(1 for p in strong_down if p[1] == 0) / len(strong_down) if strong_down else 0

        print(f"    {label}:  N={n:<5}  corr(pm_t0, outcome)={corr:+.4f}")
        print(f"           pm_t0>=0.65: N={len(strong_up)}, actually up {up_wr*100:.1f}%")
        print(f"           pm_t0<=0.35: N={len(strong_down)}, actually down {down_wr*100:.1f}%")

    # 3b: Prev PM patterns still working?
    print("\n  --- 3b: Prev PM Contrarian Signal Strength ---")
    for label, data in [("TRAIN", train), ("TEST", test)]:
        # When prev_pm >= 0.80, what fraction go DOWN?
        strong_prev_up = [r for r in data if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] >= 0.80
                          and r.get("outcome") is not None]
        strong_prev_down = [r for r in data if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] <= 0.20
                            and r.get("outcome") is not None]

        if strong_prev_up:
            fade_wr = sum(1 for r in strong_prev_up if r["outcome"] == "down") / len(strong_prev_up)
            print(f"    {label}: prev>=0.80 -> goes DOWN (contrarian correct): {fade_wr*100:.1f}%  (N={len(strong_prev_up)})")
        if strong_prev_down:
            fade_wr_bull = sum(1 for r in strong_prev_down if r["outcome"] == "up") / len(strong_prev_down)
            print(f"    {label}: prev<=0.20 -> goes UP   (contrarian correct): {fade_wr_bull*100:.1f}%  (N={len(strong_prev_down)})")

    # 3b2: Break down by asset
    print("\n  --- 3b2: Prev PM Signal by Asset (TEST only) ---")
    for asset in ASSETS:
        asset_data = [r for r in test if r["asset"] == asset]
        strong_prev_up = [r for r in asset_data if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] >= 0.80
                          and r.get("outcome") is not None]
        strong_prev_down = [r for r in asset_data if r.get("prev_pm_t12_5") is not None and r["prev_pm_t12_5"] <= 0.20
                            and r.get("outcome") is not None]
        parts = []
        if strong_prev_up:
            fade_wr = sum(1 for r in strong_prev_up if r["outcome"] == "down") / len(strong_prev_up)
            parts.append(f"bear fade {fade_wr*100:.0f}% (N={len(strong_prev_up)})")
        if strong_prev_down:
            fade_wr = sum(1 for r in strong_prev_down if r["outcome"] == "up") / len(strong_prev_down)
            parts.append(f"bull fade {fade_wr*100:.0f}% (N={len(strong_prev_down)})")
        if parts:
            print(f"    {asset}: {' | '.join(parts)}")

    # 3c: Volatility regime shift
    print("\n  --- 3c: Volatility Regime Distribution ---")
    for label, data in [("TRAIN", train), ("TEST", test)]:
        regimes = defaultdict(int)
        for r in data:
            regime = r.get("volatility_regime") or "unknown"
            regimes[regime] += 1
        total = sum(regimes.values())
        parts = [f"{k}: {v} ({v/total*100:.0f}%)" for k, v in sorted(regimes.items(), key=lambda x: str(x[0]))]
        print(f"    {label}: {', '.join(parts)}")

    # 3d: Outcome distribution
    print("\n  --- 3d: Outcome Distribution ---")
    for label, data in [("TRAIN", train), ("TEST", test)]:
        ups = sum(1 for r in data if r.get("outcome") == "up")
        downs = sum(1 for r in data if r.get("outcome") == "down")
        total = ups + downs
        if total > 0:
            print(f"    {label}: Up={ups} ({ups/total*100:.1f}%), Down={downs} ({downs/total*100:.1f}%)")

    # 3e: Spot volatility shift
    print("\n  --- 3e: Spot Price Volatility (avg range_bps) ---")
    for label, data in [("TRAIN", train), ("TEST", test)]:
        range_bps = [r["spot_range_bps"] for r in data if r.get("spot_range_bps") is not None]
        if range_bps:
            avg_range = sum(range_bps) / len(range_bps)
            median_range = sorted(range_bps)[len(range_bps)//2]
            print(f"    {label}: avg={avg_range:.1f} bps, median={median_range:.1f} bps")

    # 3f: PM spread (liquidity shift)
    print("\n  --- 3f: PM Spread at t0 (liquidity indicator) ---")
    for label, data in [("TRAIN", train), ("TEST", test)]:
        spreads = [r["pm_spread_t0"] for r in data if r.get("pm_spread_t0") is not None]
        if spreads:
            avg_sp = sum(spreads) / len(spreads)
            print(f"    {label}: avg spread={avg_sp:.4f}")

    # 3g: Weekly walk-forward of the baseline
    print("\n  --- 3g: Weekly Walk-Forward — Current Live Strategy ---")
    full_cidx = build_consensus_index(train + test)
    sig_fn = make_contrarian_consensus(0.80, 0.50, 0.50, "pm_yes_t0", "pm_yes_t12_5", 3, full_cidx)
    all_trades = run_signal(train + test, sig_fn)

    by_week = defaultdict(list)
    for t in all_trades:
        by_week[t["week"]].append(t)

    weeks_sorted = sorted(by_week.keys())
    print(f"    {'Week':<10}  {'N':>4}  {'WR':>7}  {'P&L':>9}  {'Cum':>9}")
    cum = 0.0
    for week in weeks_sorted:
        wt = by_week[week]
        m = calc_metrics(wt)
        cum += m["pnl"]
        flag = " <<<" if m["pnl"] < -5 else ""
        print(f"    {week:<10}  {m['n']:>4}  {m['wr']*100:>5.1f}%  ${m['pnl']:>+8.2f}  ${cum:>+8.2f}{flag}")

    # 3h: Correlation decay over time
    print("\n  --- 3h: PM t0 Predictive Power by Week ---")
    all_rows = train + test
    by_week_raw = defaultdict(list)
    for r in all_rows:
        wk = _week_label(r["window_start_utc"])
        if r.get("pm_yes_t0") is not None and r.get("outcome_binary") is not None:
            by_week_raw[wk].append(r)

    weeks = sorted(by_week_raw.keys())
    print(f"    {'Week':<10}  {'N':>5}  {'Corr(t0,out)':>13}  {'PM>0.65 up%':>12}  {'PM<0.35 dn%':>12}")
    for week in weeks:
        wrows = by_week_raw[week]
        n = len(wrows)
        pm_vals = [r["pm_yes_t0"] for r in wrows]
        out_vals = [r["outcome_binary"] for r in wrows]
        mean_pm = sum(pm_vals) / n
        mean_out = sum(out_vals) / n
        cov = sum((pm_vals[i] - mean_pm) * (out_vals[i] - mean_out) for i in range(n)) / n
        std_pm = math.sqrt(sum((p - mean_pm)**2 for p in pm_vals) / n) if n > 1 else 0
        std_out = math.sqrt(sum((o - mean_out)**2 for o in out_vals) / n) if n > 1 else 0
        corr = cov / (std_pm * std_out) if std_pm > 0 and std_out > 0 else 0

        strong_up = [r for r in wrows if r["pm_yes_t0"] >= 0.65]
        strong_down = [r for r in wrows if r["pm_yes_t0"] <= 0.35]
        up_pct = (sum(1 for r in strong_up if r["outcome_binary"] == 1) / len(strong_up) * 100) if strong_up else 0
        dn_pct = (sum(1 for r in strong_down if r["outcome_binary"] == 0) / len(strong_down) * 100) if strong_down else 0

        print(f"    {week:<10}  {n:>5}  {corr:>+13.4f}  {up_pct:>10.1f}%  {dn_pct:>10.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 100)
    print("POLYNANCE FRESH SWEEP — Feb 19, 2026")
    print("Diagnosing live strategy degradation + testing alternatives")
    print("=" * 100)

    print("\nLoading data from all 4 assets...")
    all_rows = load_all()
    print(f"  Total rows: {len(all_rows)}")
    print(f"  Date range: {all_rows[0]['window_start_utc'][:16]} to {all_rows[-1]['window_start_utc'][:16]}")

    for asset in ASSETS:
        n = sum(1 for r in all_rows if r["asset"] == asset)
        print(f"  {asset}: {n} windows")

    train, test, cutoff = temporal_split(all_rows, 0.70)
    print(f"\n  SPLIT at {cutoff[:16]}")
    print(f"  Train: {len(train)} rows  ({train[0]['window_start_utc'][:10]} to {train[-1]['window_start_utc'][:10]})")
    print(f"  Test:  {len(test)} rows  ({test[0]['window_start_utc'][:10]} to {test[-1]['window_start_utc'][:10]})")

    # Build consensus indexes per split (no lookahead)
    train_cidx = build_consensus_index(train)
    test_cidx = build_consensus_index(test)

    # Part 1: Diagnosis
    train_trades, test_trades = part1_diagnosis(train, test, train_cidx, test_cidx, cutoff)

    # Part 2: Sweep
    sweep_results = part2_sweep(train, test, train_cidx, test_cidx)

    # Part 3: What changed
    part3_regime_analysis(train, test, cutoff)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 100)

    both_profit = [r for r in sweep_results if r["tr"]["pnl"] > 0 and r["te"]["pnl"] > 0 and r["te"]["n"] >= 5]
    both_profit.sort(key=lambda x: x["te"]["pnl_dd"], reverse=True)

    if both_profit:
        print(f"\n  {len(both_profit)} strategies profitable on both train AND test.")
        print(f"\n  TOP 5 by test PnL/DD ratio:")
        for i, r in enumerate(both_profit[:5]):
            te = r["te"]
            pnl_dd = f"{te['pnl_dd']:.2f}" if not math.isinf(te["pnl_dd"]) else "+inf"
            print(f"    {i+1}. {r['name']}")
            print(f"       Test: N={te['n']}, WR={te['wr']*100:.1f}%, P&L=${te['pnl']:+.2f}, MaxDD=${te['max_dd']:.2f}, PnL/DD={pnl_dd}")
    else:
        print("\n  WARNING: No strategies profitable on both train AND test!")
        print("  The signal may be dead. Consider:")
        print("  - Reducing bet size dramatically")
        print("  - Stopping trading entirely until regime shifts")
        print("  - Fundamentally rethinking the approach")

    # Worst performers
    worst = sorted(sweep_results, key=lambda x: x["te"]["pnl"])[:3]
    print(f"\n  WORST 3 on test (avoid these!):")
    for r in worst:
        te = r["te"]
        print(f"    {r['name']}: Test P&L=${te['pnl']:+.2f}, WR={te['wr']*100:.1f}%")

    print("\n" + "=" * 100)
    print("SWEEP COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
