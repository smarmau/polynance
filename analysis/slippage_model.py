"""
Slippage / Spread Impact Model for ACCEL_DBL and COMBO_DBL strategies.

Models how real execution costs (crossing the spread on entry AND exit)
eat into the backtest P&L that assumed midpoint execution.

Uses actual spread data from the databases at the relevant sample times.
"""

import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
ASSETS = ["BTC", "ETH", "SOL", "XRP"]

# Fee structure (matches config)
FEE_RATE = 0.02       # 2% on profits
SPREAD_COST = 0.006   # 0.6% spread cost already modeled in backtest
BET_SIZE = 50.0       # base bet


def load_all_windows():
    """Load all windows with spread data from all assets."""
    windows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset.lower()}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT w.*,
                   s5.pm_spread as spread_at_t5,
                   s5.pm_yes_bid as bid_at_t5,
                   s5.pm_yes_ask as ask_at_t5,
                   s5.pm_yes_price as pm_at_t5,
                   s75.pm_spread as spread_at_t7_5,
                   s75.pm_yes_bid as bid_at_t7_5,
                   s75.pm_yes_ask as ask_at_t7_5,
                   s75.pm_yes_price as pm_at_t7_5,
                   s125.pm_spread as spread_at_t12_5,
                   s125.pm_yes_bid as bid_at_t12_5,
                   s125.pm_yes_ask as ask_at_t12_5,
                   s125.pm_yes_price as pm_at_t12_5,
                   s0.pm_yes_price as pm_at_t0,
                   s0.pm_spread as spread_at_t0
            FROM windows w
            LEFT JOIN samples s5 ON s5.window_id = w.window_id AND s5.t_minutes = 5.0
            LEFT JOIN samples s75 ON s75.window_id = w.window_id AND s75.t_minutes = 7.5
            LEFT JOIN samples s125 ON s125.window_id = w.window_id AND s125.t_minutes = 12.5
            LEFT JOIN samples s0 ON s0.window_id = w.window_id AND s0.t_minutes = 0.0
        """).fetchall()
        for r in rows:
            windows.append(dict(r) | {"asset": asset})
        conn.close()
    return windows


def simulate_accel_dbl(windows, prev_thresh=0.75, neutral_band=0.15,
                       bull_thresh=0.55, bear_thresh=0.45,
                       slippage_mode="midpoint"):
    """
    Simulate ACCEL_DBL with different execution assumptions.

    slippage_mode:
      - "midpoint": execute at midpoint (backtest assumption)
      - "cross_spread": pay half-spread on entry AND exit (realistic)
      - "full_spread": pay full spread on entry AND exit (worst case)
    """
    # Group by asset, sort by window_id for temporal ordering
    by_asset = defaultdict(list)
    for w in windows:
        by_asset[w["asset"]].append(w)
    for asset in by_asset:
        by_asset[asset].sort(key=lambda w: w["window_id"])

    trades = []

    for asset, asset_windows in by_asset.items():
        for i in range(2, len(asset_windows)):
            w = asset_windows[i]
            prev1 = asset_windows[i-1]
            prev2 = asset_windows[i-2]

            # Check required fields
            pm_t12_5_prev2 = prev2.get("pm_yes_t12_5")
            pm_t12_5_prev1 = prev1.get("pm_yes_t12_5")
            pm_t0 = w.get("pm_at_t0")
            pm_t5 = w.get("pm_at_t5")
            pm_t12_5 = w.get("pm_at_t12_5")
            spread_entry = w.get("spread_at_t5") or 0.01
            spread_exit = w.get("spread_at_t12_5") or 0.01

            if any(v is None for v in [pm_t12_5_prev2, pm_t12_5_prev1, pm_t0, pm_t5, pm_t12_5]):
                continue

            # Double contrarian check
            direction = None
            if pm_t12_5_prev2 >= prev_thresh and pm_t12_5_prev1 >= prev_thresh:
                direction = "bear"
            elif pm_t12_5_prev2 <= (1 - prev_thresh) and pm_t12_5_prev1 <= (1 - prev_thresh):
                direction = "bull"
            if direction is None:
                continue

            # Acceleration filter: t0 near neutral
            if abs(pm_t0 - 0.50) > neutral_band:
                continue

            # Confirmation check
            if direction == "bull" and pm_t5 < bull_thresh:
                continue
            if direction == "bear" and pm_t5 > bear_thresh:
                continue

            # Calculate entry/exit prices under different slippage models
            if direction == "bull":
                # Buying YES: entry = ask side, exit = bid side
                if slippage_mode == "midpoint":
                    entry_contract = pm_t5
                    exit_contract = pm_t12_5
                elif slippage_mode == "cross_spread":
                    entry_contract = pm_t5 + spread_entry / 2   # pay half-spread to buy
                    exit_contract = pm_t12_5 - spread_exit / 2  # lose half-spread to sell
                elif slippage_mode == "full_spread":
                    entry_contract = pm_t5 + spread_entry       # worst case
                    exit_contract = pm_t12_5 - spread_exit
            else:
                # Buying NO (= selling YES): entry = 1 - ask, exit = 1 - bid
                if slippage_mode == "midpoint":
                    entry_contract = 1.0 - pm_t5
                    exit_contract = 1.0 - pm_t12_5
                elif slippage_mode == "cross_spread":
                    entry_contract = (1.0 - pm_t5) + spread_entry / 2
                    exit_contract = (1.0 - pm_t12_5) - spread_exit / 2
                elif slippage_mode == "full_spread":
                    entry_contract = (1.0 - pm_t5) + spread_entry
                    exit_contract = (1.0 - pm_t12_5) - spread_exit

            entry_contract = max(0.01, min(0.99, entry_contract))
            exit_contract = max(0.01, min(0.99, exit_contract))

            # P&L calculation (early exit model)
            n_contracts = BET_SIZE / entry_contract
            gross = n_contracts * (exit_contract - entry_contract)
            fee = max(0, gross) * FEE_RATE
            spread_fee = BET_SIZE * SPREAD_COST * 2  # entry + exit
            pnl = gross - fee - spread_fee

            trades.append({
                "asset": asset,
                "window_id": w["window_id"],
                "direction": direction,
                "entry_contract": entry_contract,
                "exit_contract": exit_contract,
                "spread_entry": spread_entry,
                "spread_exit": spread_exit,
                "pnl": pnl,
                "n_contracts": n_contracts,
            })

    return trades


def simulate_combo_dbl(windows, prev_thresh=0.75, bull_thresh=0.55,
                       bear_thresh=0.45, stop_delta=0.10, xasset_min=2,
                       slippage_mode="midpoint"):
    """
    Simulate COMBO_DBL with different execution assumptions.
    """
    # Group by asset
    by_asset = defaultdict(list)
    for w in windows:
        by_asset[w["asset"]].append(w)
    for asset in by_asset:
        by_asset[asset].sort(key=lambda w: w["window_id"])

    # Build time-indexed lookup for cross-asset check
    # Key: time portion of window_id → set of assets with double-strong + direction
    time_double_strong = defaultdict(lambda: {"up": set(), "down": set()})

    for asset, asset_windows in by_asset.items():
        for i in range(2, len(asset_windows)):
            w = asset_windows[i]
            prev1 = asset_windows[i-1]
            prev2 = asset_windows[i-2]

            pm_prev2 = prev2.get("pm_yes_t12_5")
            pm_prev1 = prev1.get("pm_yes_t12_5")
            if pm_prev2 is None or pm_prev1 is None:
                continue

            time_key = "_".join(w["window_id"].split("_")[1:])

            if pm_prev2 >= prev_thresh and pm_prev1 >= prev_thresh:
                time_double_strong[time_key]["up"].add(asset)
            elif pm_prev2 <= (1 - prev_thresh) and pm_prev1 <= (1 - prev_thresh):
                time_double_strong[time_key]["down"].add(asset)

    trades = []

    for asset, asset_windows in by_asset.items():
        for i in range(2, len(asset_windows)):
            w = asset_windows[i]
            prev1 = asset_windows[i-1]
            prev2 = asset_windows[i-2]

            pm_prev2 = prev2.get("pm_yes_t12_5")
            pm_prev1 = prev1.get("pm_yes_t12_5")
            pm_t5 = w.get("pm_at_t5")
            pm_t7_5 = w.get("pm_at_t7_5")
            pm_t12_5 = w.get("pm_at_t12_5")
            spread_entry = w.get("spread_at_t5") or 0.01
            spread_stop = w.get("spread_at_t7_5") or 0.01
            spread_exit = w.get("spread_at_t12_5") or 0.01

            if any(v is None for v in [pm_prev2, pm_prev1, pm_t5, pm_t7_5, pm_t12_5]):
                continue

            time_key = "_".join(w["window_id"].split("_")[1:])

            # Double contrarian
            direction = None
            if pm_prev2 >= prev_thresh and pm_prev1 >= prev_thresh:
                direction = "bear"
                others = len(time_double_strong[time_key]["up"]) - 1
            elif pm_prev2 <= (1 - prev_thresh) and pm_prev1 <= (1 - prev_thresh):
                direction = "bull"
                others = len(time_double_strong[time_key]["down"]) - 1
            if direction is None:
                continue

            # Cross-asset filter
            if others < xasset_min:
                continue

            # Confirmation
            if direction == "bull" and pm_t5 < bull_thresh:
                continue
            if direction == "bear" and pm_t5 > bear_thresh:
                continue

            # Check stop-loss at t7.5
            stopped = False
            if direction == "bull":
                if (pm_t5 - pm_t7_5) >= stop_delta:
                    stopped = True
            else:
                if (pm_t7_5 - pm_t5) >= stop_delta:
                    stopped = True

            # Determine exit point and spread
            if stopped:
                exit_pm = pm_t7_5
                exit_spread = spread_stop
            else:
                exit_pm = pm_t12_5
                exit_spread = spread_exit

            # Calculate entry/exit contracts
            if direction == "bull":
                if slippage_mode == "midpoint":
                    entry_contract = pm_t5
                    exit_contract = exit_pm
                elif slippage_mode == "cross_spread":
                    entry_contract = pm_t5 + spread_entry / 2
                    exit_contract = exit_pm - exit_spread / 2
                elif slippage_mode == "full_spread":
                    entry_contract = pm_t5 + spread_entry
                    exit_contract = exit_pm - exit_spread
            else:
                if slippage_mode == "midpoint":
                    entry_contract = 1.0 - pm_t5
                    exit_contract = 1.0 - exit_pm
                elif slippage_mode == "cross_spread":
                    entry_contract = (1.0 - pm_t5) + spread_entry / 2
                    exit_contract = (1.0 - exit_pm) - exit_spread / 2
                elif slippage_mode == "full_spread":
                    entry_contract = (1.0 - pm_t5) + spread_entry
                    exit_contract = (1.0 - exit_pm) - exit_spread

            entry_contract = max(0.01, min(0.99, entry_contract))
            exit_contract = max(0.01, min(0.99, exit_contract))

            n_contracts = BET_SIZE / entry_contract
            gross = n_contracts * (exit_contract - entry_contract)
            fee = max(0, gross) * FEE_RATE
            spread_fee = BET_SIZE * SPREAD_COST * 2
            pnl = gross - fee - spread_fee

            trades.append({
                "asset": asset,
                "window_id": w["window_id"],
                "direction": direction,
                "entry_contract": entry_contract,
                "exit_contract": exit_contract,
                "spread_entry": spread_entry,
                "spread_exit": exit_spread,
                "stopped": stopped,
                "pnl": pnl,
                "n_contracts": n_contracts,
            })

    return trades


def compute_metrics(trades, label=""):
    """Compute trading metrics from a list of trades."""
    if not trades:
        return {"label": label, "n_trades": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n_trades = len(trades)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_pnl = np.mean(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Sharpe (annualized from 15-min periods, ~35,000 periods/year)
    if np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(35000 / n_trades * n_trades)
        # Simplified: sharpe per trade * sqrt(n_trades annualized)
        sharpe_simple = np.mean(pnls) / np.std(pnls) * np.sqrt(n_trades)
    else:
        sharpe_simple = 0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = drawdown.min()

    # Spread stats
    entry_spreads = [t["spread_entry"] for t in trades]
    exit_spreads = [t["spread_exit"] for t in trades]

    return {
        "label": label,
        "n_trades": n_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe_simple,
        "max_dd": max_dd,
        "avg_entry_spread": np.mean(entry_spreads),
        "avg_exit_spread": np.mean(exit_spreads),
        "p95_entry_spread": np.percentile(entry_spreads, 95),
    }


def print_comparison(results_list):
    """Print a comparison table of metrics across slippage scenarios."""
    # Header
    print(f"\n{'─' * 100}")
    print(f"{'Scenario':<35} {'Trades':>6} {'Win%':>6} {'Total P&L':>10} {'Avg P&L':>9} "
          f"{'AvgWin':>8} {'AvgLoss':>8} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"{'─' * 100}")

    for r in results_list:
        if r["n_trades"] == 0:
            print(f"{r['label']:<35} {'NO TRADES':>6}")
            continue
        print(f"{r['label']:<35} {r['n_trades']:>6} {r['win_rate']*100:>5.1f}% "
              f"${r['total_pnl']:>+8.2f} ${r['avg_pnl']:>+7.2f} "
              f"${r['avg_win']:>+6.2f} ${r['avg_loss']:>+6.2f} "
              f"{r['sharpe']:>6.2f} ${r['max_dd']:>+7.2f}")
    print(f"{'─' * 100}")


def print_spread_analysis(trades, label=""):
    """Print detailed spread analysis for a set of trades."""
    if not trades:
        return

    entry_spreads = [t["spread_entry"] for t in trades]
    exit_spreads = [t["spread_exit"] for t in trades]

    # Cost of crossing spread per trade (half-spread each way)
    cross_costs = [(s_e / 2 + s_x / 2) * (BET_SIZE / t["entry_contract"])
                   for t, s_e, s_x in zip(trades, entry_spreads, exit_spreads)]

    print(f"\n  {label} — Spread Analysis:")
    print(f"    Entry spread:  avg={np.mean(entry_spreads):.4f}  "
          f"p50={np.median(entry_spreads):.4f}  "
          f"p95={np.percentile(entry_spreads, 95):.4f}")
    print(f"    Exit spread:   avg={np.mean(exit_spreads):.4f}  "
          f"p50={np.median(exit_spreads):.4f}  "
          f"p95={np.percentile(exit_spreads, 95):.4f}")
    print(f"    Round-trip cost (crossing half-spread each way):")
    print(f"      avg=${np.mean(cross_costs):.2f}/trade  "
          f"total=${sum(cross_costs):.2f}  "
          f"as % of P&L budget: {sum(cross_costs)/max(abs(sum(t['pnl'] for t in trades)),1)*100:.1f}%")


def main():
    print("=" * 100)
    print("SLIPPAGE / SPREAD IMPACT MODEL")
    print("=" * 100)

    # Load data
    windows = load_all_windows()
    print(f"\nLoaded {len(windows)} windows across {len(ASSETS)} assets")

    # Split into train/test (70/30 temporal)
    # Sort all windows by window_id to get temporal order
    all_window_ids = sorted(set(w["window_id"].split("_", 1)[1] for w in windows))
    split_idx = int(len(all_window_ids) * 0.7)
    train_cutoff = all_window_ids[split_idx]

    test_windows = [w for w in windows if w["window_id"].split("_", 1)[1] >= train_cutoff]
    print(f"Using TEST set: {len(test_windows)} windows (after {train_cutoff})")

    # =========================================================================
    # ACCEL_DBL
    # =========================================================================
    print(f"\n{'='*100}")
    print("STRATEGY 1: ACCEL_DBL (Double Contrarian + Acceleration)")
    print(f"  Params: prev_thresh=0.75, neutral_band=0.15, bull=0.55, bear=0.45")
    print(f"  Entry=t5, Exit=t12.5")
    print(f"{'='*100}")

    accel_midpoint = simulate_accel_dbl(test_windows, slippage_mode="midpoint")
    accel_cross = simulate_accel_dbl(test_windows, slippage_mode="cross_spread")
    accel_full = simulate_accel_dbl(test_windows, slippage_mode="full_spread")

    accel_results = [
        compute_metrics(accel_midpoint, "Midpoint (backtest)"),
        compute_metrics(accel_cross, "Cross half-spread (realistic)"),
        compute_metrics(accel_full, "Cross full spread (worst case)"),
    ]
    print_comparison(accel_results)

    if accel_cross:
        print_spread_analysis(accel_cross, "ACCEL_DBL realistic")

        # Show P&L erosion
        mid_pnl = sum(t["pnl"] for t in accel_midpoint)
        cross_pnl = sum(t["pnl"] for t in accel_cross)
        full_pnl = sum(t["pnl"] for t in accel_full)
        print(f"\n  P&L Erosion from slippage:")
        print(f"    Midpoint:     ${mid_pnl:+.2f}")
        print(f"    Half-spread:  ${cross_pnl:+.2f}  (−${mid_pnl - cross_pnl:.2f}, "
              f"{(mid_pnl - cross_pnl) / max(abs(mid_pnl), 1) * 100:.1f}% erosion)")
        print(f"    Full-spread:  ${full_pnl:+.2f}  (−${mid_pnl - full_pnl:.2f}, "
              f"{(mid_pnl - full_pnl) / max(abs(mid_pnl), 1) * 100:.1f}% erosion)")

    # =========================================================================
    # COMBO_DBL
    # =========================================================================
    print(f"\n{'='*100}")
    print("STRATEGY 2: COMBO_DBL (Double Contrarian + Stop-Loss + Cross-Asset)")
    print(f"  Params: prev_thresh=0.75, bull=0.55, bear=0.45, stop_delta=0.10, xasset_min=2")
    print(f"  Entry=t5, Stop=t7.5, Exit=t12.5")
    print(f"{'='*100}")

    combo_midpoint = simulate_combo_dbl(test_windows, slippage_mode="midpoint")
    combo_cross = simulate_combo_dbl(test_windows, slippage_mode="cross_spread")
    combo_full = simulate_combo_dbl(test_windows, slippage_mode="full_spread")

    combo_results = [
        compute_metrics(combo_midpoint, "Midpoint (backtest)"),
        compute_metrics(combo_cross, "Cross half-spread (realistic)"),
        compute_metrics(combo_full, "Cross full spread (worst case)"),
    ]
    print_comparison(combo_results)

    if combo_cross:
        print_spread_analysis(combo_cross, "COMBO_DBL realistic")

        mid_pnl = sum(t["pnl"] for t in combo_midpoint)
        cross_pnl = sum(t["pnl"] for t in combo_cross)
        full_pnl = sum(t["pnl"] for t in combo_full)
        print(f"\n  P&L Erosion from slippage:")
        print(f"    Midpoint:     ${mid_pnl:+.2f}")
        print(f"    Half-spread:  ${cross_pnl:+.2f}  (−${mid_pnl - cross_pnl:.2f}, "
              f"{(mid_pnl - cross_pnl) / max(abs(mid_pnl), 1) * 100:.1f}% erosion)")
        print(f"    Full-spread:  ${full_pnl:+.2f}  (−${mid_pnl - full_pnl:.2f}, "
              f"{(mid_pnl - full_pnl) / max(abs(mid_pnl), 1) * 100:.1f}% erosion)")

        # Stop-loss specific analysis
        stopped = [t for t in combo_cross if t.get("stopped")]
        held = [t for t in combo_cross if not t.get("stopped")]
        if stopped:
            print(f"\n  Stop-loss breakdown (realistic slippage):")
            print(f"    Stopped at t7.5: {len(stopped)} trades, "
                  f"avg P&L ${np.mean([t['pnl'] for t in stopped]):+.2f}")
            print(f"    Held to t12.5:   {len(held)} trades, "
                  f"avg P&L ${np.mean([t['pnl'] for t in held]):+.2f}")
            print(f"    Stop-loss spread (t7.5): avg={np.mean([t['spread_exit'] for t in stopped]):.4f}")

    # =========================================================================
    # SENSITIVITY: What if spreads widen?
    # =========================================================================
    print(f"\n{'='*100}")
    print("SENSITIVITY: What if spreads widen 2x or 3x?")
    print(f"{'='*100}")

    # Create modified windows with wider spreads
    for multiplier in [1.0, 1.5, 2.0, 3.0]:
        modified_windows = []
        for w in test_windows:
            wc = dict(w)
            for key in ["spread_at_t0", "spread_at_t5", "spread_at_t7_5", "spread_at_t12_5"]:
                if wc.get(key) is not None:
                    wc[key] = wc[key] * multiplier
            modified_windows.append(wc)

        accel = simulate_accel_dbl(modified_windows, slippage_mode="cross_spread")
        combo = simulate_combo_dbl(modified_windows, slippage_mode="cross_spread")

        accel_m = compute_metrics(accel, f"ACCEL spread×{multiplier:.1f}")
        combo_m = compute_metrics(combo, f"COMBO spread×{multiplier:.1f}")

        if multiplier == 1.0:
            sens_results = []
        sens_results.append(accel_m)
        sens_results.append(combo_m)

    print_comparison(sens_results)

    # =========================================================================
    # PER-ASSET SPREAD IMPACT
    # =========================================================================
    print(f"\n{'='*100}")
    print("PER-ASSET: Spread impact varies by asset liquidity")
    print(f"{'='*100}")

    for asset in ASSETS:
        asset_windows = [w for w in test_windows if w["asset"] == asset]
        accel = simulate_accel_dbl(asset_windows, slippage_mode="cross_spread")
        if accel:
            spreads = [t["spread_entry"] for t in accel]
            pnl = sum(t["pnl"] for t in accel)
            print(f"  {asset}: {len(accel)} trades, avg spread={np.mean(spreads):.4f}, "
                  f"total P&L=${pnl:+.2f}")

    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}")
    mid_accel = sum(t["pnl"] for t in accel_midpoint) if accel_midpoint else 0
    real_accel = sum(t["pnl"] for t in accel_cross) if accel_cross else 0
    mid_combo = sum(t["pnl"] for t in combo_midpoint) if combo_midpoint else 0
    real_combo = sum(t["pnl"] for t in combo_cross) if combo_cross else 0

    print(f"  ACCEL_DBL: Backtest ${mid_accel:+.2f} → Realistic ${real_accel:+.2f} "
          f"(−{(mid_accel-real_accel)/max(abs(mid_accel),1)*100:.1f}%)")
    print(f"  COMBO_DBL: Backtest ${mid_combo:+.2f} → Realistic ${real_combo:+.2f} "
          f"(−{(mid_combo-real_combo)/max(abs(mid_combo),1)*100:.1f}%)")
    print(f"\n  Typical spread: ~$0.01 (1 cent)")
    print(f"  Cost per round-trip (crossing half each way): ~$0.01 × contracts")
    print(f"  At $50 bet, ~100 contracts → ~$1.00 round-trip slippage cost")
    print(f"  This is ON TOP of the 0.6% spread_cost already modeled in backtest")


if __name__ == "__main__":
    main()
