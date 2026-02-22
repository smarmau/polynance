#!/usr/bin/env python3
"""
strategy_lab.py — Comprehensive Strategy Lab
Feb 20, 2026

CLEAN backtest with NO lookahead bias.
Tests:
  A) Legitimate confluence filters on contrarian consensus base
  B) Entirely new strategy ideas
  C) All with 70/30 temporal train/test split

Legitimate signals available at t=0 (entry time):
  - prev_pm_t12_5: previous window's PM YES price at t=12.5 (known)
  - prev2_pm_t12_5: two windows ago PM YES price at t=12.5 (known)
  - pm_yes_t0: current window PM YES at t=0 (known at entry)
  - pm_spread_t0: current window spread at t=0 (known at entry)
  - prev_spot_open/close: previous window's spot data (known)
  - spot_open: current window's spot_open (known at t=0)
  - prev_volatility_regime: PREVIOUS window's vol regime (known)

NOT available at t=0 (LOOKAHEAD — DO NOT USE):
  - spot_change_bps, spot_close, spot_range_bps, volatility_regime (current window end data)
  - pm_yes_t5+ (future within window)
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
                # Cross-window spot momentum: (current_spot_open - prev_spot_open) / prev_spot_open
                if r["spot_open"] and prev["spot_open"] and prev["spot_open"] > 0:
                    r["xw_spot_momentum_bps"] = ((r["spot_open"] - prev["spot_open"]) / prev["spot_open"]) * 10000
                else:
                    r["xw_spot_momentum_bps"] = None

                # Previous window's volatility regime (known — already finalized)
                r["prev_vol_regime"] = prev["volatility_regime"]
                r["prev_spot_range_bps"] = prev["spot_range_bps"]
                r["prev_outcome_binary"] = prev["outcome_binary"]
                r["prev_spot_close"] = prev["spot_close"]
            else:
                r["xw_spot_momentum_bps"] = None
                r["prev_vol_regime"] = None
                r["prev_spot_range_bps"] = None
                r["prev_outcome_binary"] = None
                r["prev_spot_close"] = None

    return rows


def temporal_split(rows, frac=0.70):
    ts = sorted(set(r["window_start_utc"] for r in rows))
    cut = ts[int(len(ts) * frac)]
    train = [r for r in rows if r["window_start_utc"] < cut]
    test = [r for r in rows if r["window_start_utc"] >= cut]
    return train, test, cut


# ── P&L Engine ───────────────────────────────────────────────────────────────

def polymarket_crypto_fee(n_contracts, price):
    """Polymarket 15-min crypto: C × 0.25 × (p × (1-p))^2 per side."""
    raw = n_contracts * 0.25 * (price * (1.0 - price)) ** 2
    return max(round(raw, 4), 0.0001) if raw > 0 else 0.0

def pnl_trade(direction, entry_pm, exit_pm, bet=BET_SIZE):
    if direction == "bull":
        entry_c, exit_c = entry_pm, exit_pm
    else:
        entry_c, exit_c = 1.0 - entry_pm, 1.0 - exit_pm
    if entry_c <= 0.01 or entry_c >= 0.99:
        return 0.0
    contracts = bet / entry_c
    gross = contracts * (exit_c - entry_c)
    fees = polymarket_crypto_fee(contracts, entry_c) + polymarket_crypto_fee(contracts, exit_c)
    return gross - fees


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


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_contrarian(rows, cidx, params):
    """Contrarian consensus with parameterized filters."""
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    entry_t_key = params.get("entry_t", "pm_yes_t0")
    exit_t_key = params.get("exit_t", "pm_yes_t12_5")
    bull_only = params.get("bull_only", False)
    bear_only = params.get("bear_only", False)
    require_double = params.get("require_double", False)
    double_thresh = params.get("double_thresh", 0.80)
    prev_vol_filter = params.get("prev_vol_filter", None)
    spread_max = params.get("spread_max", None)
    xw_spot_confirms = params.get("xw_spot_confirms", False)
    xw_spot_min_bps = params.get("xw_spot_min_bps", 0)
    pm_t0_neutral_band = params.get("pm_t0_neutral_band", None)
    adaptive_n = params.get("adaptive_n", 0)
    all_trades_for_adaptive = []

    trades = []
    for r in rows:
        prev = r.get("prev_pm_t12_5")
        entry_pm = r.get(entry_t_key)
        exit_pm = r.get(exit_t_key)
        if prev is None or entry_pm is None or exit_pm is None:
            continue
        if not (0.01 < entry_pm < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        direction = None
        wt = get_wt(r)
        if cidx is not None:
            if wt is None or wt not in cidx:
                continue
            ci = cidx[wt]
            if prev >= prev_thresh and ci["n_bear_prev"] >= min_agree:
                direction = "bear"
            elif prev <= (1.0 - prev_thresh) and ci["n_bull_prev"] >= min_agree:
                direction = "bull"
        else:
            if prev >= prev_thresh:
                direction = "bear"
            elif prev <= (1.0 - prev_thresh):
                direction = "bull"

        if direction is None:
            continue
        if bull_only and direction != "bull":
            continue
        if bear_only and direction != "bear":
            continue

        if require_double:
            prev2 = r.get("prev2_pm_t12_5")
            if prev2 is None:
                continue
            if direction == "bear" and prev2 < double_thresh:
                continue
            if direction == "bull" and prev2 > (1.0 - double_thresh):
                continue

        if prev_vol_filter is not None:
            pv = r.get("prev_vol_regime")
            if pv not in prev_vol_filter:
                continue

        if spread_max is not None:
            sp = r.get("pm_spread_t0")
            if sp is None or sp > spread_max:
                continue

        if xw_spot_confirms:
            xw = r.get("xw_spot_momentum_bps")
            if xw is None:
                continue
            # Bull contrarian: prev was bear, spot reversing UP confirms
            if direction == "bull" and xw < xw_spot_min_bps:
                continue
            # Bear contrarian: prev was bull, spot reversing DOWN confirms
            if direction == "bear" and xw > -xw_spot_min_bps:
                continue

        if pm_t0_neutral_band is not None:
            pm0 = r.get("pm_yes_t0")
            if pm0 is None or abs(pm0 - 0.50) > pm_t0_neutral_band:
                continue

        p = pnl_trade(direction, entry_pm, exit_pm)
        trade = {"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                 "ts": r["window_start_utc"]}

        if adaptive_n > 0:
            all_trades_for_adaptive.append(trade)
        else:
            trades.append(trade)

    if adaptive_n > 0:
        for i, t in enumerate(all_trades_for_adaptive):
            if i < adaptive_n:
                continue
            history = all_trades_for_adaptive[i - adaptive_n:i]
            bull_h = [h for h in history if h["dir"] == "bull"]
            bear_h = [h for h in history if h["dir"] == "bear"]
            bull_wr = sum(1 for h in bull_h if h["win"]) / len(bull_h) if bull_h else 0.5
            bear_wr = sum(1 for h in bear_h if h["win"]) / len(bear_h) if bear_h else 0.5
            if (t["dir"] == "bull" and bull_wr >= bear_wr) or \
               (t["dir"] == "bear" and bear_wr > bull_wr):
                trades.append(t)

    return trades


def run_momentum_follow(rows, cidx=None, params=None):
    """MOMENTUM FOLLOW: if prev was strong X, bet SAME direction (trend continues)."""
    params = params or {}
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    xw_spot_confirms = params.get("xw_spot_confirms", False)

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
        if cidx is not None:
            wt = get_wt(r)
            if wt is None or wt not in cidx:
                continue
            ci = cidx[wt]
            if prev >= prev_thresh and ci["n_bear_prev"] >= min_agree:
                direction = "bull"  # prev strong bull → follow bull
            elif prev <= (1.0 - prev_thresh) and ci["n_bull_prev"] >= min_agree:
                direction = "bear"  # prev strong bear → follow bear
        else:
            if prev >= prev_thresh:
                direction = "bull"
            elif prev <= (1.0 - prev_thresh):
                direction = "bear"

        if direction is None:
            continue

        if xw_spot_confirms:
            xw = r.get("xw_spot_momentum_bps")
            if xw is None:
                continue
            if direction == "bull" and xw <= 0:
                continue
            if direction == "bear" and xw >= 0:
                continue

        p = pnl_trade(direction, entry_pm, exit_pm)
        trades.append({"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                       "ts": r["window_start_utc"]})
    return trades


def run_pm_mean_reversion(rows, params=None):
    """PM MEAN REVERSION: fade extreme PM prices at t=0 (no prev_pm dependency)."""
    params = params or {}
    pm_thresh = params.get("pm_thresh", 0.70)
    spread_max = params.get("spread_max", None)

    trades = []
    for r in rows:
        pm0 = r.get("pm_yes_t0")
        exit_pm = r.get("pm_yes_t12_5")
        if pm0 is None or exit_pm is None:
            continue
        if not (0.01 < pm0 < 0.99 and 0.01 < exit_pm < 0.99):
            continue
        if spread_max is not None:
            sp = r.get("pm_spread_t0")
            if sp is None or sp > spread_max:
                continue

        direction = None
        if pm0 >= pm_thresh:
            direction = "bear"
        elif pm0 <= (1.0 - pm_thresh):
            direction = "bull"
        if direction is None:
            continue

        p = pnl_trade(direction, pm0, exit_pm)
        trades.append({"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                       "ts": r["window_start_utc"]})
    return trades


def run_cross_window_spot_reversal(rows, params=None):
    """CROSS-WINDOW SPOT REVERSAL: pure spot mean reversion, no PM signal."""
    params = params or {}
    min_bps = params.get("min_bps", 20)

    trades = []
    for r in rows:
        xw = r.get("xw_spot_momentum_bps")
        pm0 = r.get("pm_yes_t0")
        exit_pm = r.get("pm_yes_t12_5")
        if xw is None or pm0 is None or exit_pm is None:
            continue
        if not (0.01 < pm0 < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        direction = None
        if xw >= min_bps:
            direction = "bear"
        elif xw <= -min_bps:
            direction = "bull"
        if direction is None:
            continue

        p = pnl_trade(direction, pm0, exit_pm)
        trades.append({"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                       "ts": r["window_start_utc"]})
    return trades


def run_outcome_streak(rows, params=None):
    """OUTCOME STREAK: bet against N consecutive same-direction outcomes."""
    params = params or {}
    streak_len = params.get("streak_len", 3)

    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset"]].append(r)
    for asset in by_asset:
        by_asset[asset].sort(key=lambda r: r["window_start_utc"])

    streak_lookup = {}
    for asset, asset_rows in by_asset.items():
        for i, r in enumerate(asset_rows):
            if i < streak_len:
                continue
            prev_outcomes = [asset_rows[i - j - 1].get("outcome_binary") for j in range(streak_len)]
            if None in prev_outcomes:
                continue
            streak_lookup[(asset, r["window_start_utc"])] = prev_outcomes

    trades = []
    for r in rows:
        key = (r["asset"], r["window_start_utc"])
        if key not in streak_lookup:
            continue
        prev_outcomes = streak_lookup[key]
        pm0 = r.get("pm_yes_t0")
        exit_pm = r.get("pm_yes_t12_5")
        if pm0 is None or exit_pm is None:
            continue
        if not (0.01 < pm0 < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        direction = None
        if all(o == 1 for o in prev_outcomes):
            direction = "bear"
        elif all(o == 0 for o in prev_outcomes):
            direction = "bull"
        if direction is None:
            continue

        p = pnl_trade(direction, pm0, exit_pm)
        trades.append({"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                       "ts": r["window_start_utc"]})
    return trades


def run_pm_open_vs_prev_close(rows, cidx=None, params=None):
    """PM OPEN vs PREV CLOSE DIVERGENCE.
    If pm_yes_t0 diverges significantly from prev_pm_t12_5, there's a gap.
    Bet that the gap closes (reversion to prev PM level).
    e.g., prev ended at 0.85 (strong bull), current opens at 0.55 → gap down in PM.
    Bet bull (expect PM to recover toward previous close).
    """
    params = params or {}
    min_gap = params.get("min_gap", 0.10)
    prev_thresh = params.get("prev_thresh", 0.70)

    trades = []
    for r in rows:
        prev = r.get("prev_pm_t12_5")
        pm0 = r.get("pm_yes_t0")
        exit_pm = r.get("pm_yes_t12_5")
        if prev is None or pm0 is None or exit_pm is None:
            continue
        if not (0.01 < pm0 < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        gap = pm0 - prev  # positive = PM opened higher than prev close
        direction = None

        # PM gapped down from strong prev (prev was high, pm0 dropped)
        if prev >= prev_thresh and gap <= -min_gap:
            direction = "bull"  # expect recovery toward prev level
        # PM gapped up from weak prev (prev was low, pm0 jumped)
        elif prev <= (1.0 - prev_thresh) and gap >= min_gap:
            direction = "bear"  # expect reversion back down

        if direction is None:
            continue

        p = pnl_trade(direction, pm0, exit_pm)
        trades.append({"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                       "ts": r["window_start_utc"]})
    return trades


def run_consensus_disagreement(rows, cidx, params=None):
    """CONSENSUS DISAGREEMENT: trade when most assets agree on prev direction
    but the current asset's PM is moving against it (single asset divergence).
    If 3/4 assets had strong bear prev (consensus=bear), but THIS asset's
    pm_yes_t0 is above 0.55 (leaning bull), bet bear (expect convergence).
    """
    params = params or {}
    prev_thresh = params.get("prev_thresh", 0.80)
    min_agree = params.get("min_agree", 3)
    divergence_thresh = params.get("divergence_thresh", 0.55)

    trades = []
    for r in rows:
        prev = r.get("prev_pm_t12_5")
        pm0 = r.get("pm_yes_t0")
        exit_pm = r.get("pm_yes_t12_5")
        if prev is None or pm0 is None or exit_pm is None:
            continue
        if not (0.01 < pm0 < 0.99 and 0.01 < exit_pm < 0.99):
            continue

        wt = get_wt(r)
        if wt is None or wt not in cidx:
            continue
        ci = cidx[wt]

        direction = None
        # Consensus says bear (most prev were strong bull), but this asset's PM is leaning bull
        if ci["n_bear_prev"] >= min_agree and pm0 >= divergence_thresh:
            direction = "bear"  # converge with consensus
        # Consensus says bull (most prev were strong bear), but this asset's PM is leaning bear
        elif ci["n_bull_prev"] >= min_agree and pm0 <= (1.0 - divergence_thresh):
            direction = "bull"

        if direction is None:
            continue

        p = pnl_trade(direction, pm0, exit_pm)
        trades.append({"pnl": p, "win": p > 0, "dir": direction, "asset": r["asset"],
                       "ts": r["window_start_utc"]})
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 115
    print(SEP)
    print("  POLYNANCE STRATEGY LAB — Feb 20, 2026")
    print("  Clean backtest: NO lookahead, 70/30 temporal split")
    print("  $25 flat bet, early exit at t12.5, fees = C * 0.25 * (p*(1-p))^2 per side")
    print(SEP)

    print("\nLoading data...")
    all_rows = load_all()
    print(f"  Total windows: {len(all_rows)}")
    print(f"  Date range: {all_rows[0]['window_start_utc'][:16]} to {all_rows[-1]['window_start_utc'][:16]}")

    print("\nEnriching cross-window features...")
    all_rows = enrich_cross_window(all_rows)
    xw_valid = sum(1 for r in all_rows if r.get("xw_spot_momentum_bps") is not None)
    print(f"  Cross-window spot momentum: {xw_valid}/{len(all_rows)} rows")

    train, test, cutoff = temporal_split(all_rows, 0.70)
    print(f"\n  70/30 SPLIT at {cutoff[:16]}")
    print(f"  Train: {len(train)} rows  Test: {len(test)} rows\n")

    full_cidx = build_consensus_index(all_rows)

    results = []

    def add(name, tr_trades, te_trades):
        tr_m = calc_metrics(tr_trades)
        te_m = calc_metrics(te_trades)
        results.append((name, tr_m, te_m))

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A: CONTRARIAN CONSENSUS VARIANTS
    # ══════════════════════════════════════════════════════════════════════════
    print("  Running Section A: Contrarian Consensus Variants...")

    strategies_a = [
        # Baselines
        ("BASELINE: cc_p80_xa3", {"prev_thresh": 0.80, "min_agree": 3}),
        ("simple_p80 (no consensus)", {"prev_thresh": 0.80, "min_agree": 0}),

        # Bull only
        ("BULL_ONLY: cc_p80_xa3", {"prev_thresh": 0.80, "min_agree": 3, "bull_only": True}),
        ("BULL_ONLY: cc_p85_xa3", {"prev_thresh": 0.85, "min_agree": 3, "bull_only": True}),
        ("BULL_ONLY: simple_p80", {"prev_thresh": 0.80, "bull_only": True}),

        # Stricter consensus
        ("cc_p75_xa4", {"prev_thresh": 0.75, "min_agree": 4}),
        ("cc_p80_xa4", {"prev_thresh": 0.80, "min_agree": 4}),

        # Double contrarian
        ("dbl_cc_p80_xa3", {"prev_thresh": 0.80, "min_agree": 3, "require_double": True}),
        ("dbl_cc_p75_xa3", {"prev_thresh": 0.75, "min_agree": 3, "require_double": True}),
        ("dbl_cc_p80_xa4", {"prev_thresh": 0.80, "min_agree": 4, "require_double": True}),

        # Prev vol regime filter (legitimate — uses PREVIOUS window's regime)
        ("cc_p80_xa3_prevVol=norm", {"prev_thresh": 0.80, "min_agree": 3, "prev_vol_filter": {"normal"}}),
        ("cc_p80_xa3_prevVol=norm+hi", {"prev_thresh": 0.80, "min_agree": 3, "prev_vol_filter": {"normal", "high"}}),
        ("cc_p80_xa3_prevVol≠extreme", {"prev_thresh": 0.80, "min_agree": 3, "prev_vol_filter": {"low", "normal", "high"}}),

        # Spread filter (known at t0)
        ("cc_p80_xa3_spread<0.010", {"prev_thresh": 0.80, "min_agree": 3, "spread_max": 0.010}),
        ("cc_p80_xa3_spread<0.008", {"prev_thresh": 0.80, "min_agree": 3, "spread_max": 0.008}),

        # Cross-window spot momentum (LEGITIMATE — prev→current open)
        ("cc_p80_xa3_xwSpot>0", {"prev_thresh": 0.80, "min_agree": 3, "xw_spot_confirms": True}),
        ("cc_p80_xa3_xwSpot>5bps", {"prev_thresh": 0.80, "min_agree": 3, "xw_spot_confirms": True, "xw_spot_min_bps": 5}),
        ("cc_p80_xa3_xwSpot>10bps", {"prev_thresh": 0.80, "min_agree": 3, "xw_spot_confirms": True, "xw_spot_min_bps": 10}),

        # PM t0 neutral band
        ("cc_p80_xa3_pm0±0.15", {"prev_thresh": 0.80, "min_agree": 3, "pm_t0_neutral_band": 0.15}),
        ("cc_p80_xa3_pm0±0.10", {"prev_thresh": 0.80, "min_agree": 3, "pm_t0_neutral_band": 0.10}),

        # Adaptive direction
        ("cc_p80_xa3_adapt25", {"prev_thresh": 0.80, "min_agree": 3, "adaptive_n": 25}),
        ("cc_p80_xa3_adapt50", {"prev_thresh": 0.80, "min_agree": 3, "adaptive_n": 50}),

        # Combined filters
        ("dbl_cc_p80_xa3_spread<0.01", {"prev_thresh": 0.80, "min_agree": 3, "require_double": True, "spread_max": 0.010}),
        ("BULL_cc_p80_xa3_prevVol≠ext", {"prev_thresh": 0.80, "min_agree": 3, "bull_only": True, "prev_vol_filter": {"low", "normal", "high"}}),
        ("cc_p80_xa4_xwSpot>0", {"prev_thresh": 0.80, "min_agree": 4, "xw_spot_confirms": True}),
        ("dbl_cc_p80_xa3_xwSpot>0", {"prev_thresh": 0.80, "min_agree": 3, "require_double": True, "xw_spot_confirms": True}),
        ("BULL_cc_p80_xa3_xwSpot>0", {"prev_thresh": 0.80, "min_agree": 3, "bull_only": True, "xw_spot_confirms": True}),
        ("cc_p80_xa3_spread<0.01_xwSpot>0", {"prev_thresh": 0.80, "min_agree": 3, "spread_max": 0.010, "xw_spot_confirms": True}),
        ("BULL_cc_p80_xa3_adapt25", {"prev_thresh": 0.80, "min_agree": 3, "bull_only": True, "adaptive_n": 25}),

        # No-consensus + filters
        ("simple_p80_xwSpot>0", {"prev_thresh": 0.80, "xw_spot_confirms": True}),
        ("simple_p80_spread<0.01", {"prev_thresh": 0.80, "spread_max": 0.010}),
    ]

    for name, params in strategies_a:
        # For strategies without consensus, pass None for cidx
        use_cidx = full_cidx if params.get("min_agree", 3) > 0 else None
        tr = run_contrarian(train, use_cidx, params)
        te = run_contrarian(test, use_cidx, params)
        add(name, tr, te)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B: NEW STRATEGY IDEAS
    # ══════════════════════════════════════════════════════════════════════════
    print("  Running Section B: New Strategy Ideas...")

    # B1: Momentum follow
    for name, params in [
        ("MOMENTUM: p80_xa3", {"prev_thresh": 0.80, "min_agree": 3}),
        ("MOMENTUM: p80_simple", {"prev_thresh": 0.80}),
        ("MOMENTUM: p80_xa3_xwSpot", {"prev_thresh": 0.80, "min_agree": 3, "xw_spot_confirms": True}),
        ("MOMENTUM: p90_xa3", {"prev_thresh": 0.90, "min_agree": 3}),
    ]:
        use_cidx = full_cidx if params.get("min_agree", 0) > 0 else None
        tr = run_momentum_follow(train, use_cidx, params)
        te = run_momentum_follow(test, use_cidx, params)
        add(name, tr, te)

    # B2: PM Mean Reversion
    for name, params in [
        ("PM_REVERT: pm>=0.65", {"pm_thresh": 0.65}),
        ("PM_REVERT: pm>=0.70", {"pm_thresh": 0.70}),
        ("PM_REVERT: pm>=0.75", {"pm_thresh": 0.75}),
        ("PM_REVERT: pm>=0.70_spread<0.01", {"pm_thresh": 0.70, "spread_max": 0.010}),
    ]:
        tr = run_pm_mean_reversion(train, params)
        te = run_pm_mean_reversion(test, params)
        add(name, tr, te)

    # B3: Cross-window spot reversal
    for name, params in [
        ("XW_SPOT_REVERSAL: >10bps", {"min_bps": 10}),
        ("XW_SPOT_REVERSAL: >20bps", {"min_bps": 20}),
        ("XW_SPOT_REVERSAL: >30bps", {"min_bps": 30}),
        ("XW_SPOT_REVERSAL: >50bps", {"min_bps": 50}),
    ]:
        tr = run_cross_window_spot_reversal(train, params)
        te = run_cross_window_spot_reversal(test, params)
        add(name, tr, te)

    # B4: Outcome streak fade
    for name, params in [
        ("STREAK_FADE: 3_consec", {"streak_len": 3}),
        ("STREAK_FADE: 4_consec", {"streak_len": 4}),
        ("STREAK_FADE: 5_consec", {"streak_len": 5}),
    ]:
        tr = run_outcome_streak(train, params)
        te = run_outcome_streak(test, params)
        add(name, tr, te)

    # B5: PM open vs prev close divergence
    for name, params in [
        ("PM_GAP: gap>=0.10_prev>=0.70", {"min_gap": 0.10, "prev_thresh": 0.70}),
        ("PM_GAP: gap>=0.15_prev>=0.70", {"min_gap": 0.15, "prev_thresh": 0.70}),
        ("PM_GAP: gap>=0.10_prev>=0.80", {"min_gap": 0.10, "prev_thresh": 0.80}),
        ("PM_GAP: gap>=0.20_prev>=0.80", {"min_gap": 0.20, "prev_thresh": 0.80}),
    ]:
        tr = run_pm_open_vs_prev_close(train, params=params)
        te = run_pm_open_vs_prev_close(test, params=params)
        add(name, tr, te)

    # B6: Consensus disagreement
    for name, params in [
        ("CONSENSUS_DIVERGE: xa3_div0.55", {"prev_thresh": 0.80, "min_agree": 3, "divergence_thresh": 0.55}),
        ("CONSENSUS_DIVERGE: xa3_div0.60", {"prev_thresh": 0.80, "min_agree": 3, "divergence_thresh": 0.60}),
        ("CONSENSUS_DIVERGE: xa4_div0.55", {"prev_thresh": 0.80, "min_agree": 4, "divergence_thresh": 0.55}),
    ]:
        tr = run_consensus_disagreement(train, full_cidx, params)
        te = run_consensus_disagreement(test, full_cidx, params)
        add(name, tr, te)

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    results.sort(key=lambda x: x[2]["pnl_dd"] if not math.isinf(x[2]["pnl_dd"]) else 999, reverse=True)

    print(f"\n\n{SEP}")
    print("  FULL RESULTS — sorted by Test PnL/DD")
    print(SEP)

    hdr = f"  {'Strategy':<48} | {'Tr N':>5} {'Tr WR':>6} {'Tr P&L':>10} | {'Te N':>5} {'Te WR':>6} {'Te P&L':>10} {'Te DD':>8} {'PnL/DD':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, tr_m, te_m in results:
        both_ok = tr_m["pnl"] > 0 and te_m["pnl"] > 0 and te_m["n"] >= 10
        pdd = f"{te_m['pnl_dd']:.2f}" if not math.isinf(te_m['pnl_dd']) else "+inf"
        flag = " ***" if both_ok else ""
        print(f"  {name:<48} | {tr_m['n']:>5} {tr_m['wr']*100:>5.1f}% ${tr_m['pnl']:>+9.2f} | "
              f"{te_m['n']:>5} {te_m['wr']*100:>5.1f}% ${te_m['pnl']:>+9.2f} ${te_m['max_dd']:>7.2f} {pdd:>7}{flag}")

    # ── Winners ──────────────────────────────────────────────────────────────
    winners = [(n, tr, te) for n, tr, te in results if tr["pnl"] > 0 and te["pnl"] > 0 and te["n"] >= 10]

    print(f"\n\n{SEP}")
    print(f"  WINNERS: Profitable on BOTH train AND test (N>=10) — {len(winners)}/{len(results)}")
    print(SEP)

    hdr2 = f"  {'#':>3} {'Strategy':<48} | {'Te N':>5} {'Te WR':>6} {'Te P&L':>10} {'Te DD':>8} {'PnL/DD':>7} | {'Tr WR':>6} {'Tr P&L':>10}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for i, (name, tr_m, te_m) in enumerate(winners, 1):
        pdd = f"{te_m['pnl_dd']:.2f}" if not math.isinf(te_m['pnl_dd']) else "+inf"
        print(f"  {i:>3} {name:<48} | {te_m['n']:>5} {te_m['wr']*100:>5.1f}% ${te_m['pnl']:>+9.2f} ${te_m['max_dd']:>7.2f} {pdd:>7} | {tr_m['wr']*100:>5.1f}% ${tr_m['pnl']:>+9.2f}")

    # ── Losers ───────────────────────────────────────────────────────────────
    losers = [(n, tr, te) for n, tr, te in results if te["pnl"] <= 0 and te["n"] >= 10]
    if losers:
        print(f"\n  LOSERS on test (avoid): {len(losers)}")
        for name, tr_m, te_m in losers[:10]:
            print(f"    {name}: Test P&L=${te_m['pnl']:+.2f}, WR={te_m['wr']*100:.1f}%")

    # ── XW Spot Momentum Stats ───────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("  CROSS-WINDOW SPOT MOMENTUM DISTRIBUTION")
    print(SEP)

    for label, rows_set in [("TRAIN", train), ("TEST", test)]:
        xw_vals = sorted([r["xw_spot_momentum_bps"] for r in rows_set if r.get("xw_spot_momentum_bps") is not None])
        if xw_vals:
            p10 = xw_vals[int(len(xw_vals)*0.10)]
            p25 = xw_vals[int(len(xw_vals)*0.25)]
            p50 = xw_vals[int(len(xw_vals)*0.50)]
            p75 = xw_vals[int(len(xw_vals)*0.75)]
            p90 = xw_vals[int(len(xw_vals)*0.90)]
            print(f"  {label}: N={len(xw_vals)}, p10={p10:.1f}, p25={p25:.1f}, p50={p50:.1f}, p75={p75:.1f}, p90={p90:.1f} bps")

    print(f"\n{SEP}")
    print("  STRATEGY LAB COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
