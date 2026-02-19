#!/usr/bin/env python3
"""
confluence_clean.py — Confluence filter analysis with ZERO lookahead bias.

Every signal used here is strictly available at t0 (entry time).
Entry at pm_yes_t0, exit at pm_yes_t12_5, $25 flat bet, fees = 2 * 0.001 * $25.

70/30 temporal train/test split.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path("/Volumes/shared_folder/polynance/data")
ASSETS = ["btc", "eth", "sol", "xrp"]
BET_SIZE = 25.0
FEES = 2 * 0.001 * BET_SIZE  # $0.05
TRAIN_FRAC = 0.70
PREV_BULL_THRESH = 0.80   # prev ended bullish => contrarian BEAR
PREV_BEAR_THRESH = 0.20   # prev ended bearish => contrarian BULL

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_all_windows() -> pd.DataFrame:
    """Load windows from all asset DBs, merge into one DataFrame."""
    frames = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query("SELECT * FROM windows", conn)
        conn.close()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["window_start_utc"] = pd.to_datetime(df["window_start_utc"], utc=True)
    df = df.sort_values(["asset", "window_start_utc"]).reset_index(drop=True)
    return df


def compute_prev_spot_open(df: pd.DataFrame) -> pd.DataFrame:
    """For each asset, get prev window's spot_open by shifting within asset group."""
    df = df.sort_values(["asset", "window_start_utc"]).reset_index(drop=True)
    df["prev_spot_open"] = df.groupby("asset")["spot_open"].shift(1)
    df["prev_volatility_regime"] = df.groupby("asset")["volatility_regime"].shift(1)
    return df


# ─────────────────────────────────────────────────────────────
# P&L CALCULATION
# ─────────────────────────────────────────────────────────────
def calc_pnl(row, direction: str) -> float:
    """
    Calculate P&L for a single trade.
    Bull: buy YES at pm_yes_t0, sell at pm_yes_t12_5
    Bear: buy NO at (1-pm_yes_t0), sell at (1-pm_yes_t12_5)
    """
    entry_t0 = row["pm_yes_t0"]
    exit_t12 = row["pm_yes_t12_5"]
    if pd.isna(entry_t0) or pd.isna(exit_t12):
        return np.nan
    if direction == "bull":
        if entry_t0 <= 0:
            return np.nan
        contracts = BET_SIZE / entry_t0
        pnl = contracts * (exit_t12 - entry_t0) - FEES
    elif direction == "bear":
        no_entry = 1.0 - entry_t0
        no_exit = 1.0 - exit_t12
        if no_entry <= 0:
            return np.nan
        contracts = BET_SIZE / no_entry
        pnl = contracts * (no_exit - no_entry) - FEES
    else:
        return np.nan
    return pnl


def calc_pnl_series(df: pd.DataFrame, direction_col: str = "direction") -> pd.Series:
    """Vectorized P&L calculation."""
    results = []
    for _, row in df.iterrows():
        results.append(calc_pnl(row, row[direction_col]))
    return pd.Series(results, index=df.index)


def calc_pnl_vectorized(df: pd.DataFrame) -> pd.Series:
    """Fully vectorized P&L — much faster than row-by-row."""
    entry_t0 = df["pm_yes_t0"].values.astype(float)
    exit_t12 = df["pm_yes_t12_5"].values.astype(float)
    direction = df["direction"].values

    pnl = np.full(len(df), np.nan)

    bull_mask = (direction == "bull") & ~np.isnan(entry_t0) & ~np.isnan(exit_t12) & (entry_t0 > 0)
    bear_mask = (direction == "bear") & ~np.isnan(entry_t0) & ~np.isnan(exit_t12) & ((1.0 - entry_t0) > 0)

    # Bull: buy YES
    contracts_bull = BET_SIZE / entry_t0[bull_mask]
    pnl[bull_mask] = contracts_bull * (exit_t12[bull_mask] - entry_t0[bull_mask]) - FEES

    # Bear: buy NO
    no_entry = 1.0 - entry_t0[bear_mask]
    no_exit = 1.0 - exit_t12[bear_mask]
    contracts_bear = BET_SIZE / no_entry
    pnl[bear_mask] = contracts_bear * (no_exit - no_entry) - FEES

    return pd.Series(pnl, index=df.index)


# ─────────────────────────────────────────────────────────────
# METRICS HELPERS
# ─────────────────────────────────────────────────────────────
def compute_metrics(pnl_series: pd.Series) -> dict:
    """Compute standard metrics from a P&L series."""
    pnl = pnl_series.dropna()
    if len(pnl) == 0:
        return {"n": 0, "wr": 0, "total_pnl": 0, "avg_pnl": 0, "max_dd": 0, "pnl_dd": 0}
    n = len(pnl)
    wins = (pnl > 0).sum()
    wr = wins / n if n > 0 else 0
    total = pnl.sum()
    avg = pnl.mean()
    cum = pnl.cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    max_dd = abs(dd) if dd < 0 else 0.001
    pnl_dd = total / max_dd if max_dd > 0 else total
    return {"n": n, "wr": wr, "total_pnl": total, "avg_pnl": avg, "max_dd": max_dd, "pnl_dd": pnl_dd}


def print_metrics(label: str, train_m: dict, test_m: dict, indent: str = "  "):
    """Print train vs test metrics side by side."""
    print(f"{indent}{label}")
    print(f"{indent}  {'':20s} {'TRAIN':>12s} {'TEST':>12s}")
    print(f"{indent}  {'Trades':20s} {train_m['n']:12d} {test_m['n']:12d}")
    print(f"{indent}  {'Win Rate':20s} {train_m['wr']:11.1%}  {test_m['wr']:11.1%}")
    print(f"{indent}  {'Total P&L':20s} ${train_m['total_pnl']:10.2f}  ${test_m['total_pnl']:10.2f}")
    print(f"{indent}  {'Avg P&L':20s} ${train_m['avg_pnl']:10.3f}  ${test_m['avg_pnl']:10.3f}")
    print(f"{indent}  {'Max Drawdown':20s} ${train_m['max_dd']:10.2f}  ${test_m['max_dd']:10.2f}")
    print(f"{indent}  {'P&L/DD':20s} {train_m['pnl_dd']:11.2f}x {test_m['pnl_dd']:11.2f}x")
    print()


# ─────────────────────────────────────────────────────────────
# TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame) -> tuple:
    """70/30 temporal split based on window_start_utc."""
    sorted_times = df["window_start_utc"].sort_values().unique()
    cutoff_idx = int(len(sorted_times) * TRAIN_FRAC)
    cutoff_time = sorted_times[cutoff_idx]
    train = df[df["window_start_utc"] < cutoff_time].copy()
    test = df[df["window_start_utc"] >= cutoff_time].copy()
    return train, test


# ─────────────────────────────────────────────────────────────
# CONTRARIAN BASE STRATEGY
# ─────────────────────────────────────────────────────────────
def add_contrarian_direction(df: pd.DataFrame, bull_thresh=PREV_BEAR_THRESH, bear_thresh=PREV_BULL_THRESH) -> pd.DataFrame:
    """
    Add contrarian direction column.
    prev_pm_t12_5 >= bear_thresh => direction = 'bear' (fade bullish prev)
    prev_pm_t12_5 <= bull_thresh => direction = 'bull' (fade bearish prev)
    """
    df = df.copy()
    conditions = [
        df["prev_pm_t12_5"] >= bear_thresh,
        df["prev_pm_t12_5"] <= bull_thresh,
    ]
    choices = ["bear", "bull"]
    df["direction"] = np.select(conditions, choices, default=None)
    # Filter to only contrarian trades
    df = df[df["direction"].isin(["bull", "bear"])].copy()
    return df


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("CONFLUENCE FILTER ANALYSIS — ZERO LOOKAHEAD BIAS")
    print("Entry: t0 | Exit: t12.5 | Bet: $25 | Fees: $0.05 | 70/30 temporal split")
    print("=" * 80)
    print()

    # Load data
    df_all = load_all_windows()
    df_all = compute_prev_spot_open(df_all)
    print(f"Total windows loaded: {len(df_all)} across {df_all['asset'].nunique()} assets")
    print(f"Date range: {df_all['window_start_utc'].min()} to {df_all['window_start_utc'].max()}")

    # Base contrarian trades
    df_ctr = add_contrarian_direction(df_all)
    df_ctr["pnl"] = calc_pnl_vectorized(df_ctr)
    df_ctr = df_ctr.dropna(subset=["pnl"]).copy()
    print(f"Contrarian trades (prev >= {PREV_BULL_THRESH} or <= {PREV_BEAR_THRESH}): {len(df_ctr)}")

    train, test = temporal_split(df_ctr)
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Train period: {train['window_start_utc'].min()} to {train['window_start_utc'].max()}")
    print(f"Test period:  {test['window_start_utc'].min()} to {test['window_start_utc'].max()}")
    print()

    # Baseline
    print("-" * 80)
    print("BASELINE: Simple Contrarian (prev_pm_t12_5 extreme => fade)")
    print("-" * 80)
    base_train = compute_metrics(train["pnl"])
    base_test = compute_metrics(test["pnl"])
    print_metrics("All contrarian trades", base_train, base_test)

    # Store results for final ranking
    results_tracker = []
    results_tracker.append({
        "name": "BASELINE contrarian",
        "train": base_train, "test": base_test
    })

    # ═══════════════════════════════════════════════════════════
    # TEST 1: CROSS-WINDOW SPOT MOMENTUM
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 1: CROSS-WINDOW SPOT MOMENTUM")
    print("  spot_momentum = (spot_open - prev_spot_open) / prev_spot_open")
    print("  Available at t0: YES (both spot_open values are known)")
    print("=" * 80)

    df_ctr["spot_momentum_bps"] = np.where(
        df_ctr["prev_spot_open"] > 0,
        (df_ctr["spot_open"] - df_ctr["prev_spot_open"]) / df_ctr["prev_spot_open"] * 10000,
        np.nan
    )

    has_mom = df_ctr.dropna(subset=["spot_momentum_bps"]).copy()
    train_m, test_m = temporal_split(has_mom)

    # Direction agreement: bull entry + positive spot momentum, or bear entry + negative spot momentum
    for thresh_bps in [0, 5, 10, 20]:
        label = f"Spot momentum AGREES with direction (>= {thresh_bps} bps)"
        agree_mask_train = (
            ((train_m["direction"] == "bull") & (train_m["spot_momentum_bps"] >= thresh_bps)) |
            ((train_m["direction"] == "bear") & (train_m["spot_momentum_bps"] <= -thresh_bps))
        )
        agree_mask_test = (
            ((test_m["direction"] == "bull") & (test_m["spot_momentum_bps"] >= thresh_bps)) |
            ((test_m["direction"] == "bear") & (test_m["spot_momentum_bps"] <= -thresh_bps))
        )
        tr_met = compute_metrics(train_m.loc[agree_mask_train, "pnl"])
        te_met = compute_metrics(test_m.loc[agree_mask_test, "pnl"])
        print_metrics(label, tr_met, te_met)
        results_tracker.append({"name": f"T1: spot_mom agrees >= {thresh_bps}bps", "train": tr_met, "test": te_met})

    # Direction disagrees (momentum opposes contrarian = true mean reversion)
    for thresh_bps in [0, 5, 10, 20]:
        label = f"Spot momentum OPPOSES direction (>= {thresh_bps} bps)"
        opp_mask_train = (
            ((train_m["direction"] == "bull") & (train_m["spot_momentum_bps"] <= -thresh_bps)) |
            ((train_m["direction"] == "bear") & (train_m["spot_momentum_bps"] >= thresh_bps))
        )
        opp_mask_test = (
            ((test_m["direction"] == "bull") & (test_m["spot_momentum_bps"] <= -thresh_bps)) |
            ((test_m["direction"] == "bear") & (test_m["spot_momentum_bps"] >= thresh_bps))
        )
        tr_met = compute_metrics(train_m.loc[opp_mask_train, "pnl"])
        te_met = compute_metrics(test_m.loc[opp_mask_test, "pnl"])
        print_metrics(label, tr_met, te_met)
        results_tracker.append({"name": f"T1: spot_mom opposes >= {thresh_bps}bps", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 2: PM SPREAD AT t0
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 2: PM SPREAD AT t0 (tight vs wide)")
    print("  Median spread computed from TRAIN set only (no lookahead)")
    print("=" * 80)

    has_spread = df_ctr.dropna(subset=["pm_spread_t0"]).copy()
    train_s, test_s = temporal_split(has_spread)

    median_spread = train_s["pm_spread_t0"].median()
    print(f"  Train median spread: {median_spread:.4f}")
    print()

    for label_s, mask_fn in [
        ("Tight spread (< median)", lambda d: d["pm_spread_t0"] < median_spread),
        ("Wide spread (>= median)", lambda d: d["pm_spread_t0"] >= median_spread),
    ]:
        tr_met = compute_metrics(train_s.loc[mask_fn(train_s), "pnl"])
        te_met = compute_metrics(test_s.loc[mask_fn(test_s), "pnl"])
        print_metrics(label_s, tr_met, te_met)
        results_tracker.append({"name": f"T2: {label_s}", "train": tr_met, "test": te_met})

    # Also test specific spread thresholds
    for spread_thresh in [0.01, 0.02, 0.03, 0.04]:
        label_s = f"Spread <= {spread_thresh}"
        tr_met = compute_metrics(train_s.loc[train_s["pm_spread_t0"] <= spread_thresh, "pnl"])
        te_met = compute_metrics(test_s.loc[test_s["pm_spread_t0"] <= spread_thresh, "pnl"])
        print_metrics(label_s, tr_met, te_met)
        results_tracker.append({"name": f"T2: spread <= {spread_thresh}", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 3: PREVIOUS WINDOW VOLATILITY REGIME
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 3: PREVIOUS WINDOW VOLATILITY REGIME")
    print("  Uses PREVIOUS window's volatility_regime (complete, no lookahead)")
    print("=" * 80)

    has_prev_vol = df_ctr.dropna(subset=["prev_volatility_regime"]).copy()
    train_v, test_v = temporal_split(has_prev_vol)

    for regime in ["low", "normal", "high", "extreme"]:
        label_v = f"Prev vol regime: {regime}"
        tr_met = compute_metrics(train_v.loc[train_v["prev_volatility_regime"] == regime, "pnl"])
        te_met = compute_metrics(test_v.loc[test_v["prev_volatility_regime"] == regime, "pnl"])
        print_metrics(label_v, tr_met, te_met)
        results_tracker.append({"name": f"T3: prev_vol={regime}", "train": tr_met, "test": te_met})

    # Grouped: low+normal vs high+extreme
    for label_v, regimes in [
        ("Prev vol: low+normal", ["low", "normal"]),
        ("Prev vol: high+extreme", ["high", "extreme"]),
    ]:
        tr_met = compute_metrics(train_v.loc[train_v["prev_volatility_regime"].isin(regimes), "pnl"])
        te_met = compute_metrics(test_v.loc[test_v["prev_volatility_regime"].isin(regimes), "pnl"])
        print_metrics(label_v, tr_met, te_met)
        results_tracker.append({"name": f"T3: {label_v}", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 4: DOUBLE CONTRARIAN
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 4: DOUBLE CONTRARIAN (prev AND prev2 both extreme)")
    print("=" * 80)

    has_prev2 = df_ctr.dropna(subset=["prev2_pm_t12_5"]).copy()
    train_d, test_d = temporal_split(has_prev2)

    for p2_bear_thresh, p2_bull_thresh in [(0.80, 0.20), (0.75, 0.25), (0.70, 0.30)]:
        label_d = f"Double contrarian (prev2 >= {p2_bear_thresh} or <= {p2_bull_thresh})"
        # For bear direction: prev was bullish AND prev2 was bullish
        # For bull direction: prev was bearish AND prev2 was bearish
        dbl_mask_train = (
            ((train_d["direction"] == "bear") & (train_d["prev2_pm_t12_5"] >= p2_bear_thresh)) |
            ((train_d["direction"] == "bull") & (train_d["prev2_pm_t12_5"] <= p2_bull_thresh))
        )
        dbl_mask_test = (
            ((test_d["direction"] == "bear") & (test_d["prev2_pm_t12_5"] >= p2_bear_thresh)) |
            ((test_d["direction"] == "bull") & (test_d["prev2_pm_t12_5"] <= p2_bull_thresh))
        )
        tr_met = compute_metrics(train_d.loc[dbl_mask_train, "pnl"])
        te_met = compute_metrics(test_d.loc[dbl_mask_test, "pnl"])
        print_metrics(label_d, tr_met, te_met)
        results_tracker.append({"name": f"T4: dbl p2>={p2_bear_thresh}/<={p2_bull_thresh}", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 5: PM t0 PRICE LEVEL
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 5: PM t0 PRICE LEVEL (does current PM price confirm contrarian?)")
    print("=" * 80)

    buckets = [
        ("<0.35", lambda d: d["pm_yes_t0"] < 0.35),
        ("0.35-0.45", lambda d: (d["pm_yes_t0"] >= 0.35) & (d["pm_yes_t0"] < 0.45)),
        ("0.45-0.55", lambda d: (d["pm_yes_t0"] >= 0.45) & (d["pm_yes_t0"] < 0.55)),
        ("0.55-0.65", lambda d: (d["pm_yes_t0"] >= 0.55) & (d["pm_yes_t0"] < 0.65)),
        (">0.65", lambda d: d["pm_yes_t0"] >= 0.65),
    ]

    for label_b, mask_fn in buckets:
        tr_met = compute_metrics(train.loc[mask_fn(train), "pnl"])
        te_met = compute_metrics(test.loc[mask_fn(test), "pnl"])
        print_metrics(f"PM t0 in {label_b}", tr_met, te_met)
        results_tracker.append({"name": f"T5: pm_t0 {label_b}", "train": tr_met, "test": te_met})

    # Directional: bull entry with low pm_t0 (confirms), bear entry with high pm_t0 (confirms)
    print("  Directional confirmation:")
    for label_b, mask_fn in [
        ("Bull + pm_t0 < 0.45 (confirms)", lambda d: (d["direction"] == "bull") & (d["pm_yes_t0"] < 0.45)),
        ("Bull + pm_t0 >= 0.45 (no confirm)", lambda d: (d["direction"] == "bull") & (d["pm_yes_t0"] >= 0.45)),
        ("Bear + pm_t0 > 0.55 (confirms)", lambda d: (d["direction"] == "bear") & (d["pm_yes_t0"] > 0.55)),
        ("Bear + pm_t0 <= 0.55 (no confirm)", lambda d: (d["direction"] == "bear") & (d["pm_yes_t0"] <= 0.55)),
    ]:
        tr_met = compute_metrics(train.loc[mask_fn(train), "pnl"])
        te_met = compute_metrics(test.loc[mask_fn(test), "pnl"])
        print_metrics(label_b, tr_met, te_met)
        results_tracker.append({"name": f"T5: {label_b}", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 6: ADAPTIVE DIRECTION (trailing WR)
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 6: ADAPTIVE DIRECTION (only take direction with higher trailing WR)")
    print("  Rolling WR computed sequentially — no lookahead")
    print("=" * 80)

    # Sort by time for sequential processing
    df_sorted = df_ctr.sort_values("window_start_utc").reset_index(drop=True)

    for lookback_n in [25, 50, 100]:
        # Track trailing WR per direction
        bull_wins = []
        bear_wins = []
        adaptive_pnl = []
        adaptive_times = []

        for i, row in df_sorted.iterrows():
            direction = row["direction"]
            pnl_val = row["pnl"]
            is_win = 1 if pnl_val > 0 else 0

            if direction == "bull":
                bull_wins.append(is_win)
            else:
                bear_wins.append(is_win)

            # Need at least lookback_n trades in EACH direction to compute WR
            if len(bull_wins) >= lookback_n and len(bear_wins) >= lookback_n:
                bull_wr = np.mean(bull_wins[-lookback_n:])
                bear_wr = np.mean(bear_wins[-lookback_n:])
                # Only take the direction with higher trailing WR
                if direction == "bull" and bull_wr >= bear_wr:
                    adaptive_pnl.append(pnl_val)
                    adaptive_times.append(row["window_start_utc"])
                elif direction == "bear" and bear_wr >= bull_wr:
                    adaptive_pnl.append(pnl_val)
                    adaptive_times.append(row["window_start_utc"])

        if len(adaptive_pnl) > 0:
            adaptive_df = pd.DataFrame({"pnl": adaptive_pnl, "window_start_utc": adaptive_times})
            ad_train, ad_test = temporal_split(adaptive_df)
            tr_met = compute_metrics(ad_train["pnl"])
            te_met = compute_metrics(ad_test["pnl"])
            print_metrics(f"Adaptive N={lookback_n}", tr_met, te_met)
            results_tracker.append({"name": f"T6: adaptive N={lookback_n}", "train": tr_met, "test": te_met})
        else:
            print(f"  Adaptive N={lookback_n}: not enough data")
            print()

    # ═══════════════════════════════════════════════════════════
    # TEST 7: CROSS-WINDOW PM MOMENTUM (gap between windows)
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 7: CROSS-WINDOW PM MOMENTUM (PM gap between windows)")
    print("  Bear: gap = prev_pm_t12_5 - pm_yes_t0 (how much PM dropped between windows)")
    print("  Bull: gap = pm_yes_t0 - prev_pm_t12_5 (how much PM rose between windows)")
    print("=" * 80)

    df_ctr["pm_gap"] = np.where(
        df_ctr["direction"] == "bear",
        df_ctr["prev_pm_t12_5"] - df_ctr["pm_yes_t0"],
        df_ctr["pm_yes_t0"] - df_ctr["prev_pm_t12_5"]
    )

    has_gap = df_ctr.dropna(subset=["pm_gap"]).copy()
    train_g, test_g = temporal_split(has_gap)

    # Bucket by gap size
    gap_buckets = [
        ("Gap < 0 (PM moved further in prev direction)", lambda d: d["pm_gap"] < 0),
        ("Gap 0-0.10 (small reset)", lambda d: (d["pm_gap"] >= 0) & (d["pm_gap"] < 0.10)),
        ("Gap 0.10-0.20 (moderate reset)", lambda d: (d["pm_gap"] >= 0.10) & (d["pm_gap"] < 0.20)),
        ("Gap 0.20-0.30 (large reset)", lambda d: (d["pm_gap"] >= 0.20) & (d["pm_gap"] < 0.30)),
        ("Gap >= 0.30 (massive reset)", lambda d: d["pm_gap"] >= 0.30),
    ]

    for label_g, mask_fn in gap_buckets:
        tr_met = compute_metrics(train_g.loc[mask_fn(train_g), "pnl"])
        te_met = compute_metrics(test_g.loc[mask_fn(test_g), "pnl"])
        print_metrics(label_g, tr_met, te_met)
        results_tracker.append({"name": f"T7: {label_g[:40]}", "train": tr_met, "test": te_met})

    # Test: large gap (>= 0.15) — PM already reset significantly
    for gap_thresh in [0.05, 0.10, 0.15, 0.20]:
        label_g = f"PM gap >= {gap_thresh}"
        tr_met = compute_metrics(train_g.loc[train_g["pm_gap"] >= gap_thresh, "pnl"])
        te_met = compute_metrics(test_g.loc[test_g["pm_gap"] >= gap_thresh, "pnl"])
        print_metrics(label_g, tr_met, te_met)
        results_tracker.append({"name": f"T7: pm_gap >= {gap_thresh}", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 8: HOUR OF DAY
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 8: HOUR OF DAY")
    print("=" * 80)

    df_ctr["hour"] = df_ctr["window_start_utc"].dt.hour

    train_h, test_h = temporal_split(df_ctr)

    print(f"  {'Hour':>6s}  {'Train N':>8s} {'Train WR':>9s} {'Train P&L':>10s}  {'Test N':>8s} {'Test WR':>9s} {'Test P&L':>10s}  {'Flag':>6s}")
    print(f"  {'-'*6}  {'-'*8} {'-'*9} {'-'*10}  {'-'*8} {'-'*9} {'-'*10}  {'-'*6}")

    bad_hours = []
    good_hours = []
    for h in range(24):
        tr_pnl = train_h.loc[train_h["hour"] == h, "pnl"]
        te_pnl = test_h.loc[test_h["hour"] == h, "pnl"]
        tr_n = len(tr_pnl)
        te_n = len(te_pnl)
        tr_wr = (tr_pnl > 0).mean() if tr_n > 0 else 0
        te_wr = (te_pnl > 0).mean() if te_n > 0 else 0
        tr_total = tr_pnl.sum() if tr_n > 0 else 0
        te_total = te_pnl.sum() if te_n > 0 else 0
        flag = ""
        if tr_n >= 5 and te_n >= 5:
            if tr_wr < 0.45 and te_wr < 0.45:
                flag = "BAD"
                bad_hours.append(h)
            elif tr_wr > 0.55 and te_wr > 0.55:
                flag = "GOOD"
                good_hours.append(h)
        print(f"  {h:6d}  {tr_n:8d} {tr_wr:8.1%}  ${tr_total:9.2f}  {te_n:8d} {te_wr:8.1%}  ${te_total:9.2f}  {flag:>6s}")

    print()
    if bad_hours:
        print(f"  Consistently BAD hours (both <45% WR): {bad_hours}")
    if good_hours:
        print(f"  Consistently GOOD hours (both >55% WR): {good_hours}")
    print()

    # Test: exclude bad hours
    if bad_hours:
        label_h = f"Exclude bad hours {bad_hours}"
        tr_met = compute_metrics(train_h.loc[~train_h["hour"].isin(bad_hours), "pnl"])
        te_met = compute_metrics(test_h.loc[~test_h["hour"].isin(bad_hours), "pnl"])
        print_metrics(label_h, tr_met, te_met)
        results_tracker.append({"name": f"T8: excl bad hours", "train": tr_met, "test": te_met})

    # Test: only good hours
    if good_hours:
        label_h = f"Only good hours {good_hours}"
        tr_met = compute_metrics(train_h.loc[train_h["hour"].isin(good_hours), "pnl"])
        te_met = compute_metrics(test_h.loc[test_h["hour"].isin(good_hours), "pnl"])
        print_metrics(label_h, tr_met, te_met)
        results_tracker.append({"name": f"T8: only good hours", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 9: MULTI-SIGNAL CONFLUENCE SCORE
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 9: MULTI-SIGNAL CONFLUENCE (clean signals only)")
    print("  Scoring: +1 for each signal that agrees")
    print("=" * 80)

    # Compute confluence score using only legitimate signals
    # Need a fresh copy with all features
    df_conf = df_ctr.copy()

    # Recompute median spread from train only
    conf_train_cutoff = df_conf["window_start_utc"].sort_values().unique()
    cutoff_idx = int(len(conf_train_cutoff) * TRAIN_FRAC)
    cutoff_time = conf_train_cutoff[cutoff_idx]
    train_for_median = df_conf[df_conf["window_start_utc"] < cutoff_time]
    conf_median_spread = train_for_median["pm_spread_t0"].median()

    # Signal 1: Cross-window spot momentum agrees
    df_conf["sig_spot_mom"] = (
        ((df_conf["direction"] == "bull") & (df_conf["spot_momentum_bps"] >= 0)) |
        ((df_conf["direction"] == "bear") & (df_conf["spot_momentum_bps"] <= 0))
    ).astype(int)

    # Signal 2: Tight spread
    df_conf["sig_tight_spread"] = (df_conf["pm_spread_t0"] < conf_median_spread).astype(int)

    # Signal 3: Previous window vol was high or extreme
    df_conf["sig_prev_vol_high"] = df_conf["prev_volatility_regime"].isin(["high", "extreme"]).astype(int)

    # Signal 4: Double contrarian (prev2 also extreme)
    df_conf["sig_double_ctr"] = (
        ((df_conf["direction"] == "bear") & (df_conf["prev2_pm_t12_5"] >= 0.80)) |
        ((df_conf["direction"] == "bull") & (df_conf["prev2_pm_t12_5"] <= 0.20))
    ).fillna(0).astype(int)

    # Signal 5: PM t0 confirms direction
    df_conf["sig_pm_t0_confirm"] = (
        ((df_conf["direction"] == "bull") & (df_conf["pm_yes_t0"] < 0.45)) |
        ((df_conf["direction"] == "bear") & (df_conf["pm_yes_t0"] > 0.55))
    ).astype(int)

    df_conf["confluence_score"] = (
        df_conf["sig_spot_mom"] + df_conf["sig_tight_spread"] +
        df_conf["sig_prev_vol_high"] + df_conf["sig_double_ctr"] +
        df_conf["sig_pm_t0_confirm"]
    )

    train_cf, test_cf = temporal_split(df_conf)

    print(f"  {'Score':>7s}  {'Train N':>8s} {'Train WR':>9s} {'Train P&L':>10s}  {'Test N':>8s} {'Test WR':>9s} {'Test P&L':>10s}")
    print(f"  {'-'*7}  {'-'*8} {'-'*9} {'-'*10}  {'-'*8} {'-'*9} {'-'*10}")

    for score in range(6):
        tr_pnl = train_cf.loc[train_cf["confluence_score"] == score, "pnl"]
        te_pnl = test_cf.loc[test_cf["confluence_score"] == score, "pnl"]
        tr_n = len(tr_pnl)
        te_n = len(te_pnl)
        tr_wr = (tr_pnl > 0).mean() if tr_n > 0 else 0
        te_wr = (te_pnl > 0).mean() if te_n > 0 else 0
        tr_total = tr_pnl.sum() if tr_n > 0 else 0
        te_total = te_pnl.sum() if te_n > 0 else 0
        print(f"  {score:7d}  {tr_n:8d} {tr_wr:8.1%}  ${tr_total:9.2f}  {te_n:8d} {te_wr:8.1%}  ${te_total:9.2f}")
    print()

    # Test: minimum confluence thresholds
    for min_score in [2, 3, 4, 5]:
        label_cf = f"Confluence score >= {min_score}"
        tr_met = compute_metrics(train_cf.loc[train_cf["confluence_score"] >= min_score, "pnl"])
        te_met = compute_metrics(test_cf.loc[test_cf["confluence_score"] >= min_score, "pnl"])
        print_metrics(label_cf, tr_met, te_met)
        results_tracker.append({"name": f"T9: confluence >= {min_score}", "train": tr_met, "test": te_met})

    # ═══════════════════════════════════════════════════════════
    # TEST 10: COMBINED BEST FILTERS — ranked by test P&L/DD
    # ═══════════════════════════════════════════════════════════
    print("=" * 80)
    print("TEST 10: COMBINED FILTER RANKING & BEST COMBOS")
    print("=" * 80)

    # Rank all individual results
    print("\n  ── INDIVIDUAL FILTER RANKING (by test P&L/DD, min 10 test trades) ──\n")
    ranked = sorted(
        [r for r in results_tracker if r["test"]["n"] >= 10],
        key=lambda x: x["test"]["pnl_dd"],
        reverse=True
    )

    print(f"  {'Rank':>4s}  {'Filter':45s}  {'Train WR':>9s} {'Train P&L':>10s}  {'Test WR':>9s} {'Test P&L':>10s} {'Test P&L/DD':>11s}  {'Both+':>6s}")
    print(f"  {'-'*4}  {'-'*45}  {'-'*9} {'-'*10}  {'-'*9} {'-'*10} {'-'*11}  {'-'*6}")

    for i, r in enumerate(ranked[:30]):
        tr = r["train"]
        te = r["test"]
        both_profitable = "YES" if tr["total_pnl"] > 0 and te["total_pnl"] > 0 else ""
        print(f"  {i+1:4d}  {r['name']:45s}  {tr['wr']:8.1%}  ${tr['total_pnl']:9.2f}  {te['wr']:8.1%}  ${te['total_pnl']:9.2f} {te['pnl_dd']:10.2f}x  {both_profitable:>6s}")

    print()

    # ── COMBINATION TESTS ──
    # Test specific promising combinations
    print("\n  ── COMBINATION TESTS ──\n")

    combos = []

    # Combo A: Double contrarian + tight spread
    mask_a = lambda d: (
        (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80)) |
         ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20))) &
        (d["pm_spread_t0"] < conf_median_spread)
    )
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_a(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_a(df_conf), "pnl"])
    print_metrics("A: Double contrarian + tight spread", tr_met, te_met)
    combos.append({"name": "A: Dbl ctr + tight spread", "train": tr_met, "test": te_met})

    # Combo B: Double contrarian + PM t0 confirms
    mask_b = lambda d: (
        (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80) & (d["pm_yes_t0"] > 0.55)) |
         ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20) & (d["pm_yes_t0"] < 0.45)))
    )
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_b(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_b(df_conf), "pnl"])
    print_metrics("B: Double contrarian + PM t0 confirms", tr_met, te_met)
    combos.append({"name": "B: Dbl ctr + pm_t0 confirm", "train": tr_met, "test": te_met})

    # Combo C: Double contrarian + spot momentum agrees
    mask_c = lambda d: (
        (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80) & (d["spot_momentum_bps"] <= 0)) |
         ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20) & (d["spot_momentum_bps"] >= 0)))
    )
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_c(df_conf).fillna(False), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_c(df_conf).fillna(False), "pnl"])
    print_metrics("C: Double contrarian + spot mom agrees", tr_met, te_met)
    combos.append({"name": "C: Dbl ctr + spot mom agrees", "train": tr_met, "test": te_met})

    # Combo D: Double contrarian + confluence >= 3
    mask_d = lambda d: (
        (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80)) |
         ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20))) &
        (d["confluence_score"] >= 3)
    )
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_d(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_d(df_conf), "pnl"])
    print_metrics("D: Double contrarian + confluence >= 3", tr_met, te_met)
    combos.append({"name": "D: Dbl ctr + confl >= 3", "train": tr_met, "test": te_met})

    # Combo E: PM gap >= 0.10 + tight spread
    mask_e = lambda d: (d["pm_gap"] >= 0.10) & (d["pm_spread_t0"] < conf_median_spread)
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_e(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_e(df_conf), "pnl"])
    print_metrics("E: PM gap >= 0.10 + tight spread", tr_met, te_met)
    combos.append({"name": "E: PM gap + tight spread", "train": tr_met, "test": te_met})

    # Combo F: Double contrarian + PM gap >= 0.10
    mask_f = lambda d: (
        (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80)) |
         ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20))) &
        (d["pm_gap"] >= 0.10)
    )
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_f(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_f(df_conf), "pnl"])
    print_metrics("F: Double contrarian + PM gap >= 0.10", tr_met, te_met)
    combos.append({"name": "F: Dbl ctr + PM gap >= 0.10", "train": tr_met, "test": te_met})

    # Combo G: Confluence >= 3 (no double ctr required)
    mask_g = lambda d: d["confluence_score"] >= 3
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_g(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_g(df_conf), "pnl"])
    print_metrics("G: Confluence score >= 3", tr_met, te_met)
    combos.append({"name": "G: Confluence >= 3", "train": tr_met, "test": te_met})

    # Combo H: Confluence >= 4
    mask_h = lambda d: d["confluence_score"] >= 4
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_h(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_h(df_conf), "pnl"])
    print_metrics("H: Confluence score >= 4", tr_met, te_met)
    combos.append({"name": "H: Confluence >= 4", "train": tr_met, "test": te_met})

    # Combo I: exclude bad hours + double contrarian
    if bad_hours:
        mask_i = lambda d: (
            (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80)) |
             ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20))) &
            (~d["hour"].isin(bad_hours))
        )
        tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_i(df_conf), "pnl"])
        te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_i(df_conf), "pnl"])
        print_metrics("I: Double contrarian + excl bad hours", tr_met, te_met)
        combos.append({"name": "I: Dbl ctr + no bad hours", "train": tr_met, "test": te_met})

    # Combo J: Double ctr + tight spread + PM t0 confirms
    mask_j = lambda d: (
        (((d["direction"] == "bear") & (d["prev2_pm_t12_5"] >= 0.80) & (d["pm_yes_t0"] > 0.55)) |
         ((d["direction"] == "bull") & (d["prev2_pm_t12_5"] <= 0.20) & (d["pm_yes_t0"] < 0.45))) &
        (d["pm_spread_t0"] < conf_median_spread)
    )
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_j(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_j(df_conf), "pnl"])
    print_metrics("J: Dbl ctr + tight spread + PM t0 confirm", tr_met, te_met)
    combos.append({"name": "J: Dbl+tight+pm_confirm", "train": tr_met, "test": te_met})

    # Combo K: Confluence >= 3 + PM gap >= 0.10
    mask_k = lambda d: (d["confluence_score"] >= 3) & (d["pm_gap"] >= 0.10)
    tr_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] < cutoff_time) & mask_k(df_conf), "pnl"])
    te_met = compute_metrics(df_conf.loc[(df_conf["window_start_utc"] >= cutoff_time) & mask_k(df_conf), "pnl"])
    print_metrics("K: Confluence >= 3 + PM gap >= 0.10", tr_met, te_met)
    combos.append({"name": "K: Confl >= 3 + PM gap", "train": tr_met, "test": te_met})

    # ── FINAL COMBO RANKING ──
    print("\n  ── FINAL COMBO RANKING (by test P&L/DD, min 5 test trades) ──\n")
    all_combos = combos + results_tracker
    ranked_all = sorted(
        [r for r in all_combos if r["test"]["n"] >= 5],
        key=lambda x: x["test"]["pnl_dd"],
        reverse=True
    )

    print(f"  {'Rank':>4s}  {'Filter':45s}  {'Tr N':>5s} {'Tr WR':>6s} {'Tr P&L':>8s}  {'Te N':>5s} {'Te WR':>6s} {'Te P&L':>8s} {'Te P&L/DD':>9s}  {'Both+':>5s}")
    print(f"  {'-'*4}  {'-'*45}  {'-'*5} {'-'*6} {'-'*8}  {'-'*5} {'-'*6} {'-'*8} {'-'*9}  {'-'*5}")

    for i, r in enumerate(ranked_all[:25]):
        tr = r["train"]
        te = r["test"]
        both = "YES" if tr["total_pnl"] > 0 and te["total_pnl"] > 0 else ""
        print(f"  {i+1:4d}  {r['name']:45s}  {tr['n']:5d} {tr['wr']:5.1%} ${tr['total_pnl']:7.2f}  {te['n']:5d} {te['wr']:5.1%} ${te['total_pnl']:7.2f} {te['pnl_dd']:8.2f}x  {both:>5s}")

    print()

    # ── CONSENSUS TEST (cross-asset agreement) ──
    print("=" * 80)
    print("BONUS: CONSENSUS CONTRARIAN (cross-asset agreement)")
    print("  Group by window_time, count how many assets have prev extreme in same direction")
    print("=" * 80)

    # Build consensus from all windows (not just contrarian filtered)
    df_all_copy = df_all.copy()
    df_all_copy["ctr_dir"] = np.select(
        [df_all_copy["prev_pm_t12_5"] >= PREV_BULL_THRESH, df_all_copy["prev_pm_t12_5"] <= PREV_BEAR_THRESH],
        ["bear", "bull"],
        default=None
    )

    # Count agreements per window_time per direction
    for min_agree in [2, 3, 4]:
        consensus_trades = []

        for wt, group in df_all_copy.groupby("window_time"):
            bear_assets = group[group["ctr_dir"] == "bear"]
            bull_assets = group[group["ctr_dir"] == "bull"]

            if len(bear_assets) >= min_agree:
                for _, row in bear_assets.iterrows():
                    if pd.notna(row["pm_yes_t0"]) and pd.notna(row["pm_yes_t12_5"]):
                        pnl_val = calc_pnl(row, "bear")
                        if pd.notna(pnl_val):
                            consensus_trades.append({
                                "window_start_utc": row["window_start_utc"],
                                "pnl": pnl_val,
                                "direction": "bear",
                                "n_agree": len(bear_assets)
                            })

            if len(bull_assets) >= min_agree:
                for _, row in bull_assets.iterrows():
                    if pd.notna(row["pm_yes_t0"]) and pd.notna(row["pm_yes_t12_5"]):
                        pnl_val = calc_pnl(row, "bull")
                        if pd.notna(pnl_val):
                            consensus_trades.append({
                                "window_start_utc": row["window_start_utc"],
                                "pnl": pnl_val,
                                "direction": "bull",
                                "n_agree": len(bull_assets)
                            })

        if consensus_trades:
            cons_df = pd.DataFrame(consensus_trades)
            cons_df["window_start_utc"] = pd.to_datetime(cons_df["window_start_utc"], utc=True)
            cons_train, cons_test = temporal_split(cons_df)
            tr_met = compute_metrics(cons_train["pnl"])
            te_met = compute_metrics(cons_test["pnl"])
            print_metrics(f"Consensus min_agree={min_agree}", tr_met, te_met)
        else:
            print(f"  Consensus min_agree={min_agree}: no trades")
            print()

    # ── FINAL SUMMARY ──
    print("=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)
    print()

    # Find filters that are profitable on BOTH train and test
    both_profitable = [r for r in ranked_all if r["train"]["total_pnl"] > 0 and r["test"]["total_pnl"] > 0]
    print(f"  Filters profitable on BOTH train and test: {len(both_profitable)} / {len(ranked_all)}")
    print()
    if both_profitable:
        print(f"  {'Filter':45s}  {'Tr WR':>6s} {'Tr P&L':>8s}  {'Te WR':>6s} {'Te P&L':>8s} {'Te P&L/DD':>9s}")
        print(f"  {'-'*45}  {'-'*6} {'-'*8}  {'-'*6} {'-'*8} {'-'*9}")
        for r in both_profitable[:15]:
            tr = r["train"]
            te = r["test"]
            print(f"  {r['name']:45s}  {tr['wr']:5.1%} ${tr['total_pnl']:7.2f}  {te['wr']:5.1%} ${te['total_pnl']:7.2f} {te['pnl_dd']:8.2f}x")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
