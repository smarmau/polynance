"""
Advanced Strategy Ideas Backtest

Tests:
1. Spot Velocity + PM Momentum filter on double contrarian (ideas #1 + #3)
2. "Fade the Fade" — triple contrarian (3 consecutive strong, reversal even more likely)
3. "Spread Squeeze" — enter when spread collapses after being wide (liquidity rush = conviction)
4. "Spot Divergence" — PM says one thing, spot moving the other way → bet with spot
5. "Momentum Exhaustion" — huge PM move t0→t5, but spot barely moved → PM overreacted, fade it
6. "Volatility Breakout" — high prev range + extreme PM → trend continuation (NOT contrarian)
7. "Dead Cat" — after massive spot drop (>50bps), PM still bearish at t5, but bet bull (oversold bounce)

All use 70/30 temporal train/test split and cross-spread slippage model.
"""

import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
ASSETS = ["BTC", "ETH", "SOL", "XRP"]
BET_SIZE = 50.0
FEE_RATE = 0.02
SPREAD_COST = 0.006


def load_all_data():
    """Load windows + samples data for all assets."""
    windows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset.lower()}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT w.*,
                   s0.pm_yes_price as pm_t0, s0.pm_spread as spread_t0,
                   s0.spot_price as spot_t0, s0.spot_price_change_from_open as spot_chg_t0,
                   s25.pm_yes_price as pm_t2_5, s25.spot_price_change_from_open as spot_chg_t2_5,
                   s25.spot_price as spot_t2_5,
                   s5.pm_yes_price as pm_t5, s5.pm_spread as spread_t5,
                   s5.spot_price as spot_t5, s5.spot_price_change_from_open as spot_chg_t5,
                   s75.pm_yes_price as pm_t7_5, s75.pm_spread as spread_t7_5,
                   s75.spot_price as spot_t7_5, s75.spot_price_change_from_open as spot_chg_t7_5,
                   s125.pm_yes_price as pm_t12_5, s125.pm_spread as spread_t12_5,
                   s125.spot_price as spot_t12_5, s125.spot_price_change_from_open as spot_chg_t12_5
            FROM windows w
            LEFT JOIN samples s0 ON s0.window_id = w.window_id AND s0.t_minutes = 0.0
            LEFT JOIN samples s25 ON s25.window_id = w.window_id AND s25.t_minutes = 2.5
            LEFT JOIN samples s5 ON s5.window_id = w.window_id AND s5.t_minutes = 5.0
            LEFT JOIN samples s75 ON s75.window_id = w.window_id AND s75.t_minutes = 7.5
            LEFT JOIN samples s125 ON s125.window_id = w.window_id AND s125.t_minutes = 12.5
        """).fetchall()
        for r in rows:
            windows.append(dict(r) | {"_asset": asset})
        conn.close()
    return windows


def split_train_test(windows, train_pct=0.70):
    """70/30 temporal split."""
    all_times = sorted(set(w["window_id"].split("_", 1)[1] for w in windows))
    split_idx = int(len(all_times) * train_pct)
    cutoff = all_times[split_idx]
    train = [w for w in windows if w["window_id"].split("_", 1)[1] < cutoff]
    test = [w for w in windows if w["window_id"].split("_", 1)[1] >= cutoff]
    return train, test, cutoff


def calc_pnl(direction, entry_pm, exit_pm, spread_entry=0.01, spread_exit=0.01):
    """Calculate P&L with cross-spread slippage."""
    if direction == "bull":
        entry_contract = entry_pm + spread_entry / 2
        exit_contract = exit_pm - spread_exit / 2
    else:
        entry_contract = (1.0 - entry_pm) + spread_entry / 2
        exit_contract = (1.0 - exit_pm) - spread_exit / 2

    entry_contract = max(0.01, min(0.99, entry_contract))
    exit_contract = max(0.01, min(0.99, exit_contract))

    n_contracts = BET_SIZE / entry_contract
    gross = n_contracts * (exit_contract - entry_contract)
    fee = max(0, gross) * FEE_RATE
    spread_fee = BET_SIZE * SPREAD_COST * 2
    return gross - fee - spread_fee


def get_asset_windows(windows):
    """Group and sort windows by asset."""
    by_asset = defaultdict(list)
    for w in windows:
        by_asset[w["_asset"]].append(w)
    for asset in by_asset:
        by_asset[asset].sort(key=lambda w: w["window_id"])
    return by_asset


def compute_metrics(trades, label=""):
    """Compute strategy metrics."""
    if not trades:
        return {"label": label, "n": 0, "pnl": 0, "wr": 0, "avg": 0,
                "sharpe": 0, "maxdd": 0, "avg_win": 0, "avg_loss": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    max_dd = (cumulative - peak).min()

    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0

    return {
        "label": label,
        "n": len(trades),
        "pnl": sum(pnls),
        "wr": len(wins) / len(trades),
        "avg": np.mean(pnls),
        "sharpe": sharpe,
        "maxdd": max_dd,
        "avg_win": np.mean(wins) if wins else 0,
        "avg_loss": np.mean(losses) if losses else 0,
    }


def print_results(results_list, title=""):
    """Print comparison table."""
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    print(f"  {'Strategy':<45} {'N':>5} {'Win%':>6} {'Total P&L':>10} {'Avg P&L':>9} "
          f"{'AvgWin':>8} {'AvgLos':>8} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─'*115}")
    for r in results_list:
        if r["n"] == 0:
            print(f"  {r['label']:<45} {'-- no trades --':>5}")
            continue
        print(f"  {r['label']:<45} {r['n']:>5} {r['wr']*100:>5.1f}% "
              f"${r['pnl']:>+8.2f} ${r['avg']:>+7.2f} "
              f"${r['avg_win']:>+6.2f} ${r['avg_loss']:>+6.2f} "
              f"{r['sharpe']:>6.2f} ${r['maxdd']:>+7.2f}")
    print(f"  {'─'*115}")


# =============================================================================
# STRATEGY 1: Spot Velocity + PM Momentum (Ideas #1 + #3 combined)
# Double contrarian + spot must confirm reversal + PM momentum must be strong
# =============================================================================

def strat_spot_velocity_pm_momentum(windows, prev_thresh=0.75,
                                     bull_thresh=0.55, bear_thresh=0.45,
                                     spot_confirm=True, pm_momentum_min=0.0):
    """
    Double contrarian with additional filters:
    - spot_confirm: spot_price_change_from_open at t5 must agree with reversal direction
    - pm_momentum_min: |pm_t5 - pm_t0| must exceed this (PM is moving, not stale)
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for i in range(2, len(aws)):
            w, p1, p2 = aws[i], aws[i-1], aws[i-2]

            prev2_pm = p2.get("pm_yes_t12_5")
            prev1_pm = p1.get("pm_yes_t12_5")
            pm_t0 = w.get("pm_t0")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spot_chg = w.get("spot_chg_t5")  # spot change from open at t5
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [prev2_pm, prev1_pm, pm_t0, pm_t5, pm_t12_5]):
                continue

            # Double contrarian
            direction = None
            if prev2_pm >= prev_thresh and prev1_pm >= prev_thresh:
                direction = "bear"
            elif prev2_pm <= (1 - prev_thresh) and prev1_pm <= (1 - prev_thresh):
                direction = "bull"
            if direction is None:
                continue

            # Confirmation
            if direction == "bull" and pm_t5 < bull_thresh:
                continue
            if direction == "bear" and pm_t5 > bear_thresh:
                continue

            # Filter: spot velocity confirmation
            if spot_confirm and spot_chg is not None:
                if direction == "bull" and spot_chg < 0:
                    continue  # spot dropping, don't go bull
                if direction == "bear" and spot_chg > 0:
                    continue  # spot rising, don't go bear

            # Filter: PM momentum minimum
            if pm_momentum_min > 0 and pm_t0 is not None:
                pm_move = abs(pm_t5 - pm_t0)
                if pm_move < pm_momentum_min:
                    continue

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset})

    return trades


# =============================================================================
# STRATEGY 2: Triple Contrarian — 3 consecutive strong same direction
# =============================================================================

def strat_triple_contrarian(windows, prev_thresh=0.75,
                            bull_thresh=0.55, bear_thresh=0.45):
    """Require THREE consecutive strong prev windows, not just two."""
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for i in range(3, len(aws)):
            w = aws[i]
            p1, p2, p3 = aws[i-1], aws[i-2], aws[i-3]

            pm3 = p3.get("pm_yes_t12_5")
            pm2 = p2.get("pm_yes_t12_5")
            pm1 = p1.get("pm_yes_t12_5")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [pm3, pm2, pm1, pm_t5, pm_t12_5]):
                continue

            direction = None
            if pm3 >= prev_thresh and pm2 >= prev_thresh and pm1 >= prev_thresh:
                direction = "bear"
            elif pm3 <= (1-prev_thresh) and pm2 <= (1-prev_thresh) and pm1 <= (1-prev_thresh):
                direction = "bull"
            if direction is None:
                continue

            if direction == "bull" and pm_t5 < bull_thresh:
                continue
            if direction == "bear" and pm_t5 > bear_thresh:
                continue

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset})

    return trades


# =============================================================================
# STRATEGY 3: Spread Squeeze — wide spread narrows = conviction entering
# =============================================================================

def strat_spread_squeeze(windows, spread_wide_thresh=0.02, spread_narrow_thresh=0.012,
                         bull_thresh=0.60, bear_thresh=0.40):
    """
    When spread at t0 is wide (uncertain market) but narrows by t5 (conviction arriving),
    AND PM has moved to an extreme by t5, enter in PM direction (NOT contrarian).
    Low frequency, high conviction.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for w in aws:
            spread_0 = w.get("spread_t0")
            spread_5 = w.get("spread_t5")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [spread_0, spread_5, pm_t5, pm_t12_5]):
                continue

            # Wide spread at t0, narrowing by t5
            if spread_0 < spread_wide_thresh:
                continue
            if spread_5 > spread_narrow_thresh:
                continue

            # PM has conviction by t5
            direction = None
            if pm_t5 >= bull_thresh:
                direction = "bull"
            elif pm_t5 <= bear_thresh:
                direction = "bear"
            if direction is None:
                continue

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_5, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset})

    return trades


# =============================================================================
# STRATEGY 4: Spot Divergence — PM and spot disagree, bet with spot
# =============================================================================

def strat_spot_divergence(windows, pm_neutral_band=0.10, spot_move_min=0.001):
    """
    PM at t5 is near neutral (0.45-0.55) but spot has moved significantly.
    Bet in the direction spot is moving — the market hasn't priced it in yet.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for w in aws:
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spot_chg = w.get("spot_chg_t5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [pm_t5, pm_t12_5, spot_chg]):
                continue

            # PM is near neutral
            if abs(pm_t5 - 0.50) > pm_neutral_band:
                continue

            # Spot has moved significantly
            if abs(spot_chg) < spot_move_min:
                continue

            # Bet with spot direction
            direction = "bull" if spot_chg > 0 else "bear"

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset,
                           "spot_chg": spot_chg})

    return trades


# =============================================================================
# STRATEGY 5: Momentum Exhaustion — PM overreacted relative to spot
# =============================================================================

def strat_momentum_exhaustion(windows, pm_move_min=0.15, spot_move_max_bps=10):
    """
    PM moved a lot t0→t5 but spot barely moved.
    PM overreacted → fade the PM move.
    Low W/R but large wins when PM reverts.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for w in aws:
            pm_t0 = w.get("pm_t0")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spot_t0 = w.get("spot_t0")
            spot_t5 = w.get("spot_t5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [pm_t0, pm_t5, pm_t12_5, spot_t0, spot_t5]):
                continue
            if spot_t0 == 0:
                continue

            pm_move = pm_t5 - pm_t0
            spot_move_bps = abs((spot_t5 - spot_t0) / spot_t0 * 10000)

            # PM moved big but spot didn't
            if abs(pm_move) < pm_move_min:
                continue
            if spot_move_bps > spot_move_max_bps:
                continue

            # Fade the PM move
            if pm_move > 0:
                direction = "bear"  # PM went up, fade it
            else:
                direction = "bull"  # PM went down, fade it

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset,
                           "pm_move": pm_move, "spot_bps": spot_move_bps})

    return trades


# =============================================================================
# STRATEGY 6: Volatility Continuation — high vol prev window + momentum = trend
# =============================================================================

def strat_vol_continuation(windows, vol_min_bps=50, pm_thresh=0.60):
    """
    After a high-volatility previous window, if PM at t5 shows strong direction,
    go WITH the PM (trend continuation, not contrarian).
    Anti-contrarian: volatile markets trend more than mean-revert.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for i in range(1, len(aws)):
            w, prev = aws[i], aws[i-1]

            prev_range = prev.get("spot_range_bps")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [prev_range, pm_t5, pm_t12_5]):
                continue

            if prev_range < vol_min_bps:
                continue

            direction = None
            if pm_t5 >= pm_thresh:
                direction = "bull"
            elif pm_t5 <= (1 - pm_thresh):
                direction = "bear"
            if direction is None:
                continue

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset})

    return trades


# =============================================================================
# STRATEGY 7: Dead Cat Bounce — massive spot drop, PM still bearish, bet bull
# =============================================================================

def strat_dead_cat(windows, spot_drop_min_bps=40, pm_bear_max=0.40):
    """
    Previous window had a massive spot drop. New window opens, PM at t5 still
    bearish (below pm_bear_max). But oversold conditions → bet bull reversal.
    Contrarian on the panic.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for i in range(1, len(aws)):
            w, prev = aws[i], aws[i-1]

            prev_change = prev.get("spot_change_bps")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [prev_change, pm_t5, pm_t12_5]):
                continue

            # Previous window was a big drop
            if prev_change > -spot_drop_min_bps:
                continue

            # Current window PM at t5 still bearish
            if pm_t5 > pm_bear_max:
                continue

            # Contrarian: bet bull (bounce)
            direction = "bull"

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset,
                           "prev_drop": prev_change, "pm_t5": pm_t5})

    return trades


# =============================================================================
# STRATEGY 8: "The Boring One" — only trade when EVERYTHING is boring
# Flat prev window + flat PM + tight spread → tiny edge compounds
# =============================================================================

def strat_boring(windows, prev_range_max_bps=15, pm_neutral_band=0.05,
                 spot_confirm_min=0.0003):
    """
    Previous window was boring (low range). Current window PM at t5 is near 0.50.
    Spot has drifted slightly in one direction by t5.
    Bet with the spot drift. Boring markets have less noise → small edges stick.
    Very high frequency, small edge per trade.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for i in range(1, len(aws)):
            w, prev = aws[i], aws[i-1]

            prev_range = prev.get("spot_range_bps")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spot_chg = w.get("spot_chg_t5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [prev_range, pm_t5, pm_t12_5, spot_chg]):
                continue

            # Previous window was boring
            if prev_range > prev_range_max_bps:
                continue

            # Current PM near neutral
            if abs(pm_t5 - 0.50) > pm_neutral_band:
                continue

            # Spot has drifted
            if abs(spot_chg) < spot_confirm_min:
                continue

            direction = "bull" if spot_chg > 0 else "bear"

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset})

    return trades


# =============================================================================
# STRATEGY 9: PM-Spot Convergence Lag
# When spot moves first and PM is slow to follow, the PM will catch up
# =============================================================================

def strat_convergence_lag(windows, spot_move_min_bps=15, pm_lag_max=0.08):
    """
    At t2.5, spot has moved significantly from open, but PM barely moved from t0.
    By t5, check if PM has started to follow. If so, PM will continue → ride it.
    If PM still hasn't moved by t5, the spot move may reverse.
    """
    by_asset = get_asset_windows(windows)
    trades = []

    for asset, aws in by_asset.items():
        for w in aws:
            pm_t0 = w.get("pm_t0")
            pm_t2_5 = w.get("pm_t2_5")
            pm_t5 = w.get("pm_t5")
            pm_t12_5 = w.get("pm_t12_5")
            spot_chg_t2_5 = w.get("spot_chg_t2_5")
            spot_t0 = w.get("spot_t0")
            spot_t2_5 = w.get("spot_t2_5")
            spread_e = w.get("spread_t5") or 0.01
            spread_x = w.get("spread_t12_5") or 0.01

            if any(v is None for v in [pm_t0, pm_t2_5, pm_t5, pm_t12_5, spot_t0, spot_t2_5]):
                continue
            if spot_t0 == 0:
                continue

            spot_move_bps = (spot_t2_5 - spot_t0) / spot_t0 * 10000
            pm_move_early = abs(pm_t2_5 - pm_t0)

            # Spot moved big at t2.5 but PM was slow
            if abs(spot_move_bps) < spot_move_min_bps:
                continue
            if pm_move_early > pm_lag_max:
                continue  # PM already reacted, no lag

            # By t5, has PM started catching up?
            pm_move_t5 = pm_t5 - pm_t0
            spot_direction = "bull" if spot_move_bps > 0 else "bear"

            # PM is now moving in spot direction → ride the catchup
            if spot_direction == "bull" and pm_move_t5 > 0.02:
                direction = "bull"
            elif spot_direction == "bear" and pm_move_t5 < -0.02:
                direction = "bear"
            else:
                continue  # PM still not following, skip

            pnl = calc_pnl(direction, pm_t5, pm_t12_5, spread_e, spread_x)
            trades.append({"pnl": pnl, "direction": direction, "asset": asset})

    return trades


def main():
    print("Loading data...")
    windows = load_all_data()
    train, test, cutoff = split_train_test(windows)
    print(f"Total: {len(windows)} windows | Train: {len(train)} | Test: {len(test)} | Cutoff: {cutoff}")

    # =========================================================================
    # Test all strategies on TRAIN first, then TEST
    # =========================================================================

    for dataset, label in [(train, "TRAIN"), (test, "TEST")]:
        results = []

        # --- Baseline: vanilla double contrarian ---
        t = strat_spot_velocity_pm_momentum(dataset, spot_confirm=False, pm_momentum_min=0)
        results.append(compute_metrics(t, "Baseline: DblContrarian"))

        # --- #1+#3: Spot velocity + PM momentum ---
        t = strat_spot_velocity_pm_momentum(dataset, spot_confirm=True, pm_momentum_min=0)
        results.append(compute_metrics(t, "#1+#3: +SpotVelocity"))

        t = strat_spot_velocity_pm_momentum(dataset, spot_confirm=False, pm_momentum_min=0.05)
        results.append(compute_metrics(t, "#1+#3: +PmMomentum>0.05"))

        t = strat_spot_velocity_pm_momentum(dataset, spot_confirm=True, pm_momentum_min=0.05)
        results.append(compute_metrics(t, "#1+#3: +Both (spot+pm>0.05)"))

        t = strat_spot_velocity_pm_momentum(dataset, spot_confirm=True, pm_momentum_min=0.03)
        results.append(compute_metrics(t, "#1+#3: +Both (spot+pm>0.03)"))

        t = strat_spot_velocity_pm_momentum(dataset, spot_confirm=True, pm_momentum_min=0.08)
        results.append(compute_metrics(t, "#1+#3: +Both (spot+pm>0.08)"))

        print_results(results, f"{label} SET — Ideas #1+#3: Spot Velocity + PM Momentum on DblContrarian")

        # --- Left-field strategies ---
        results2 = []

        t = strat_triple_contrarian(dataset)
        results2.append(compute_metrics(t, "#2: Triple Contrarian (3x strong prev)"))

        t = strat_triple_contrarian(dataset, prev_thresh=0.70)
        results2.append(compute_metrics(t, "#2: Triple Contrarian (thresh=0.70)"))

        t = strat_spread_squeeze(dataset)
        results2.append(compute_metrics(t, "#3: Spread Squeeze (wide→narrow)"))

        t = strat_spread_squeeze(dataset, spread_wide_thresh=0.015, spread_narrow_thresh=0.011)
        results2.append(compute_metrics(t, "#3: Spread Squeeze (relaxed)"))

        t = strat_spot_divergence(dataset, pm_neutral_band=0.10, spot_move_min=0.001)
        results2.append(compute_metrics(t, "#4: Spot Divergence (pm neutral, spot moved)"))

        t = strat_spot_divergence(dataset, pm_neutral_band=0.05, spot_move_min=0.0015)
        results2.append(compute_metrics(t, "#4: Spot Divergence (tight)"))

        t = strat_momentum_exhaustion(dataset, pm_move_min=0.15, spot_move_max_bps=10)
        results2.append(compute_metrics(t, "#5: Momentum Exhaustion (pm>0.15, spot<10bps)"))

        t = strat_momentum_exhaustion(dataset, pm_move_min=0.10, spot_move_max_bps=15)
        results2.append(compute_metrics(t, "#5: Momentum Exhaustion (relaxed)"))

        t = strat_vol_continuation(dataset, vol_min_bps=50, pm_thresh=0.60)
        results2.append(compute_metrics(t, "#6: Vol Continuation (>50bps, pm>0.60)"))

        t = strat_vol_continuation(dataset, vol_min_bps=80, pm_thresh=0.65)
        results2.append(compute_metrics(t, "#6: Vol Continuation (>80bps, pm>0.65)"))

        t = strat_dead_cat(dataset, spot_drop_min_bps=40, pm_bear_max=0.40)
        results2.append(compute_metrics(t, "#7: Dead Cat Bounce (drop>40bps, pm<0.40)"))

        t = strat_dead_cat(dataset, spot_drop_min_bps=30, pm_bear_max=0.45)
        results2.append(compute_metrics(t, "#7: Dead Cat Bounce (relaxed)"))

        t = strat_boring(dataset)
        results2.append(compute_metrics(t, "#8: Boring Market (low vol+neutral+spot drift)"))

        t = strat_boring(dataset, prev_range_max_bps=20, pm_neutral_band=0.08)
        results2.append(compute_metrics(t, "#8: Boring Market (relaxed)"))

        t = strat_convergence_lag(dataset)
        results2.append(compute_metrics(t, "#9: PM-Spot Convergence Lag"))

        t = strat_convergence_lag(dataset, spot_move_min_bps=10, pm_lag_max=0.06)
        results2.append(compute_metrics(t, "#9: Convergence Lag (relaxed)"))

        print_results(results2, f"{label} SET — Left-Field Strategy Ideas")


if __name__ == "__main__":
    main()
