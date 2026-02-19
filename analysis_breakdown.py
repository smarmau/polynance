#!/usr/bin/env python3
"""
Comprehensive breakdown analysis of 652 contrarian_consensus trades
from sim_trading.db (Feb 12-17, 2026). READ ONLY.
"""
import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

DB_PATH = "/Volumes/shared_folder/polynance/sim_trading.db"

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM sim_trades
        WHERE entry_mode = 'contrarian_consensus'
        ORDER BY entry_time
    """)
    rows = cur.fetchall()
    conn.close()

    trades = []
    for r in rows:
        et = r["entry_time"]
        # Parse ISO timestamp
        if "+" in et:
            dt = datetime.fromisoformat(et)
        else:
            dt = datetime.fromisoformat(et).replace(tzinfo=timezone.utc)
        trades.append({
            "dt": dt,
            "hour": dt.hour,
            "weekday": dt.strftime("%a"),
            "date_str": dt.strftime("%a %b %d"),
            "date_key": dt.strftime("%Y-%m-%d"),
            "direction": r["direction"],
            "entry_price": r["entry_price"],
            "exit_price": r["exit_price"],
            "outcome": r["outcome"],
            "net_pnl": r["net_pnl"] or 0.0,
            "prev_pm": r["prev_pm"],
            "prev2_pm": r["prev2_pm"],
            "bet_size": r["bet_size"],
            "asset": r["asset"],
            "bankroll_after": r["bankroll_after"] or 0.0,
        })
    return trades


def bucket_stats(trades_list):
    """Return (count, win_rate, total_pnl, avg_pnl)."""
    if not trades_list:
        return (0, 0.0, 0.0, 0.0)
    n = len(trades_list)
    wins = sum(1 for t in trades_list if t["outcome"] == "win")
    pnl = sum(t["net_pnl"] for t in trades_list)
    return (n, wins / n * 100, pnl, pnl / n)


def print_group_by_day(trades_list, label):
    """Print per-day breakdown for a group of trades, then overall."""
    by_date = defaultdict(list)
    for t in trades_list:
        by_date[t["date_key"]].append(t)

    print(f"  {label}:")
    for dk in sorted(by_date.keys()):
        day_trades = by_date[dk]
        n, wr, pnl, avg = bucket_stats(day_trades)
        date_label = day_trades[0]["date_str"]
        print(f"    {date_label}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")

    n, wr, pnl, avg = bucket_stats(trades_list)
    print(f"    {'Overall':>14s}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")
    print()


def section1(trades):
    """Hour performance broken down by each day."""
    print("=" * 80)
    print("1. HOUR PERFORMANCE BROKEN DOWN BY EACH DAY")
    print("=" * 80)

    by_hour = defaultdict(list)
    for t in trades:
        by_hour[t["hour"]].append(t)

    bad_hours = [5, 6, 8, 9, 16, 22]
    good_hours = [4, 12, 15, 23]

    print("\n--- BAD HOURS ---")
    for h in bad_hours:
        htrades = by_hour.get(h, [])
        if not htrades:
            print(f"\nHour {h} (UTC): No trades")
            continue
        print(f"\nHour {h} (UTC):")
        print_group_by_day(htrades, "")

    print("\n--- GOOD HOURS ---")
    for h in good_hours:
        htrades = by_hour.get(h, [])
        if not htrades:
            print(f"\nHour {h} (UTC): No trades")
            continue
        print(f"\nHour {h} (UTC):")
        print_group_by_day(htrades, "")

    # Also show ALL hours summary for context
    print("\n--- ALL HOURS SUMMARY (for context) ---")
    for h in sorted(by_hour.keys()):
        n, wr, pnl, avg = bucket_stats(by_hour[h])
        print(f"  Hour {h:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")
    print()


def section2(trades):
    """Saturday performance broken down by hour."""
    print("=" * 80)
    print("2. SATURDAY PERFORMANCE BROKEN DOWN BY HOUR")
    print("=" * 80)

    sat_trades = [t for t in trades if t["dt"].weekday() == 5]  # Saturday=5
    if not sat_trades:
        # Check Sunday too
        sun_trades = [t for t in trades if t["dt"].weekday() == 6]
        print(f"\nNo Saturday trades found. Sunday trades: {len(sun_trades)}")
        # Show weekend
        print("\nLet me show weekend (Sat+Sun) breakdown instead:")
        weekend = [t for t in trades if t["dt"].weekday() >= 5]
        if weekend:
            by_hour = defaultdict(list)
            for t in weekend:
                by_hour[t["hour"]].append(t)
            for h in sorted(by_hour.keys()):
                n, wr, pnl, avg = bucket_stats(by_hour[h])
                day_label = by_hour[h][0]["date_str"]
                print(f"  Hour {h:2d} ({day_label}): {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")
        return

    by_hour = defaultdict(list)
    for t in sat_trades:
        by_hour[t["hour"]].append(t)

    n_total, wr_total, pnl_total, avg_total = bucket_stats(sat_trades)
    print(f"\nSaturday overall: {n_total} trades, {wr_total:.1f}% WR, ${pnl_total:+.2f} PnL\n")

    for h in sorted(by_hour.keys()):
        htrades = by_hour[h]
        n, wr, pnl, avg = bucket_stats(htrades)
        print(f"  Hour {h:2d}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")
    print()


def section3(trades):
    """Day-of-week performance broken down by each individual date."""
    print("=" * 80)
    print("3. DAY-OF-WEEK: EACH INDIVIDUAL DATE")
    print("=" * 80)

    by_date = defaultdict(list)
    for t in trades:
        by_date[t["date_key"]].append(t)

    print()
    cum_pnl = 0
    for dk in sorted(by_date.keys()):
        day_trades = by_date[dk]
        n, wr, pnl, avg = bucket_stats(day_trades)
        cum_pnl += pnl
        date_label = day_trades[0]["date_str"]
        dow = day_trades[0]["dt"].strftime("%A")
        print(f"  {date_label} ({dow:>9s}): {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})  cumPnL: ${cum_pnl:+.2f}")

    # Group by day-of-week
    print("\n  --- Grouped by Day-of-Week ---")
    by_dow = defaultdict(list)
    for t in trades:
        by_dow[t["dt"].strftime("%A")].append(t)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for dow in dow_order:
        if dow in by_dow:
            n, wr, pnl, avg = bucket_stats(by_dow[dow])
            n_dates = len(set(t["date_key"] for t in by_dow[dow]))
            print(f"  {dow:>9s}: {n:3d} trades across {n_dates} date(s), {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")
    print()


def section4(trades):
    """Bull vs Bear improvement analysis."""
    print("=" * 80)
    print("4. BULL vs BEAR IMPROVEMENT ANALYSIS")
    print("=" * 80)

    bulls = [t for t in trades if t["direction"] == "bull"]
    bears = [t for t in trades if t["direction"] == "bear"]

    # --- BULLS ---
    print(f"\n{'='*40}")
    print(f"BULL TRADES: {len(bulls)} total")
    n, wr, pnl, avg = bucket_stats(bulls)
    print(f"Overall: {wr:.1f}% WR, ${pnl:+.2f} PnL, avg ${avg:+.2f}")
    print(f"{'='*40}")

    # Bull entry_price ranges
    bull_ep_buckets = [
        ("0.45-0.50", 0.45, 0.50),
        ("0.50-0.55", 0.50, 0.55),
        ("0.55-0.60", 0.55, 0.60),
        ("0.60-0.65", 0.60, 0.65),
        ("0.65+",     0.65, 1.00),
    ]
    print("\n  Bull by entry_price:")
    for label, lo, hi in bull_ep_buckets:
        bucket = [t for t in bulls if lo <= t["entry_price"] < hi]
        if not bucket:
            continue
        n, wr, pnl, avg = bucket_stats(bucket)
        print(f"    {label}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")

    # Bull prev_pm ranges
    bull_pm_buckets = [
        ("< 0.05",    0.00, 0.05),
        ("0.05-0.10", 0.05, 0.10),
        ("0.10-0.15", 0.10, 0.15),
        ("0.15-0.20", 0.15, 0.20),
        ("0.20-0.25", 0.20, 0.25),
        ("0.25+",     0.25, 1.00),
    ]
    print("\n  Bull by prev_pm:")
    for label, lo, hi in bull_pm_buckets:
        bucket = [t for t in bulls if t["prev_pm"] is not None and lo <= t["prev_pm"] < hi]
        if not bucket:
            continue
        n, wr, pnl, avg = bucket_stats(bucket)
        print(f"    {label}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")

    # Bull cross-tab: entry_price x prev_pm
    print("\n  Bull: entry_price x prev_pm (best combos):")
    combos = []
    for ep_label, ep_lo, ep_hi in bull_ep_buckets:
        for pm_label, pm_lo, pm_hi in bull_pm_buckets:
            bucket = [t for t in bulls
                      if ep_lo <= t["entry_price"] < ep_hi
                      and t["prev_pm"] is not None
                      and pm_lo <= t["prev_pm"] < pm_hi]
            if len(bucket) >= 3:  # minimum sample
                n, wr, pnl, avg = bucket_stats(bucket)
                combos.append((wr, n, pnl, avg, ep_label, pm_label))
    combos.sort(key=lambda x: (-x[0], -x[2]))
    for wr, n, pnl, avg, ep, pm in combos:
        marker = " ***" if wr >= 60 else ""
        print(f"    EP {ep} + PM {pm}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f}){marker}")

    # --- BEARS ---
    print(f"\n{'='*40}")
    print(f"BEAR TRADES: {len(bears)} total")
    n, wr, pnl, avg = bucket_stats(bears)
    print(f"Overall: {wr:.1f}% WR, ${pnl:+.2f} PnL, avg ${avg:+.2f}")
    print(f"{'='*40}")

    # Bear entry_price ranges (bears buy high, so entry near 1.0 is better for them...
    # Actually for bears: entry_price is the price paid for the NO contract or the YES price?
    # In Polymarket contrarian bear: entry_price is the YES price they're betting against
    # So a bear with entry_price=0.55 means YES is at 0.55, they bought NO at ~0.45
    # Lower entry_price = cheaper YES = more expensive NO = worse for bear
    # Higher entry_price = more expensive YES = cheaper NO = better for bear
    bear_ep_buckets = [
        ("0.35-0.40", 0.35, 0.40),
        ("0.40-0.45", 0.40, 0.45),
        ("0.45-0.50", 0.45, 0.50),
        ("0.50-0.55", 0.50, 0.55),
        ("0.55-0.60", 0.55, 0.60),
    ]
    print("\n  Bear by entry_price:")
    for label, lo, hi in bear_ep_buckets:
        bucket = [t for t in bears if lo <= t["entry_price"] < hi]
        if not bucket:
            continue
        n, wr, pnl, avg = bucket_stats(bucket)
        print(f"    {label}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")

    # Bear prev_pm ranges (prev_pm for bears = high prev YES price, meaning bearish signal)
    bear_pm_buckets = [
        ("0.75-0.80", 0.75, 0.80),
        ("0.80-0.85", 0.80, 0.85),
        ("0.85-0.90", 0.85, 0.90),
        ("0.90-0.95", 0.90, 0.95),
        ("0.95-1.00", 0.95, 1.01),
    ]
    print("\n  Bear by prev_pm:")
    for label, lo, hi in bear_pm_buckets:
        bucket = [t for t in bears if t["prev_pm"] is not None and lo <= t["prev_pm"] < hi]
        if not bucket:
            continue
        n, wr, pnl, avg = bucket_stats(bucket)
        print(f"    {label}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f})")

    # Bear cross-tab
    print("\n  Bear: entry_price x prev_pm (sorted by WR):")
    combos = []
    for ep_label, ep_lo, ep_hi in bear_ep_buckets:
        for pm_label, pm_lo, pm_hi in bear_pm_buckets:
            bucket = [t for t in bears
                      if ep_lo <= t["entry_price"] < ep_hi
                      and t["prev_pm"] is not None
                      and pm_lo <= t["prev_pm"] < pm_hi]
            if len(bucket) >= 3:
                n, wr, pnl, avg = bucket_stats(bucket)
                combos.append((wr, n, pnl, avg, ep_label, pm_label))
    combos.sort(key=lambda x: (-x[0], -x[2]))
    for wr, n, pnl, avg, ep, pm in combos:
        marker = " ***" if wr >= 60 else (" !!" if wr < 40 else "")
        print(f"    EP {ep} + PM {pm}: {n:3d} trades, {wr:5.1f}% WR, ${pnl:+8.2f} PnL (avg ${avg:+.2f}){marker}")
    print()


def section5(trades):
    """What if we only traded bulls?"""
    print("=" * 80)
    print("5. WHAT IF WE ONLY TRADED BULLS?")
    print("=" * 80)

    bulls = [t for t in trades if t["direction"] == "bull"]
    bears = [t for t in trades if t["direction"] == "bear"]

    # Simulate sequential PnL for bulls only
    starting_bankroll = 1044.76  # approximate starting from first trade's bankroll_after - net_pnl
    # Actually let's compute from data
    first_trade = trades[0]
    starting_bankroll = first_trade["bankroll_after"] - first_trade["net_pnl"]

    bankroll = starting_bankroll
    peak = bankroll
    max_dd = 0
    max_dd_pct = 0
    pnl_curve = []

    for t in trades:
        if t["direction"] != "bull":
            continue
        bankroll += t["net_pnl"]
        if bankroll > peak:
            peak = bankroll
        dd = peak - bankroll
        dd_pct = dd / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        pnl_curve.append(bankroll)

    n, wr, total_pnl, avg_pnl = bucket_stats(bulls)
    n_bear, _, bear_pnl, _ = bucket_stats(bears)

    print(f"\n  Starting bankroll: ${starting_bankroll:.2f}")
    print(f"  Bulls only: {n} trades, {wr:.1f}% WR")
    print(f"  Total PnL:  ${total_pnl:+.2f}")
    print(f"  Avg PnL:    ${avg_pnl:+.2f}")
    print(f"  Final BR:   ${bankroll:.2f}")
    print(f"  Max DD:     ${max_dd:.2f} ({max_dd_pct:.1f}%)")
    print(f"\n  Skipped bears: {n_bear} trades worth ${bear_pnl:+.2f}")

    # Compare to all trades
    all_n, all_wr, all_pnl, all_avg = bucket_stats(trades)
    # Compute all-trade max DD
    bankroll_all = starting_bankroll
    peak_all = bankroll_all
    max_dd_all = 0
    max_dd_pct_all = 0
    for t in trades:
        bankroll_all += t["net_pnl"]
        if bankroll_all > peak_all:
            peak_all = bankroll_all
        dd = peak_all - bankroll_all
        dd_pct = dd / peak_all * 100 if peak_all > 0 else 0
        if dd > max_dd_all:
            max_dd_all = dd
        if dd_pct > max_dd_pct_all:
            max_dd_pct_all = dd_pct

    print(f"\n  --- Comparison ---")
    print(f"  {'Metric':<20s} {'All Trades':>15s} {'Bulls Only':>15s} {'Delta':>15s}")
    print(f"  {'-'*65}")
    print(f"  {'Trades':<20s} {all_n:>15d} {n:>15d} {n - all_n:>+15d}")
    print(f"  {'Win Rate':<20s} {all_wr:>14.1f}% {wr:>14.1f}% {wr-all_wr:>+14.1f}%")
    print(f"  {'Total PnL':<20s} ${all_pnl:>13.2f} ${total_pnl:>13.2f} ${total_pnl-all_pnl:>+13.2f}")
    print(f"  {'Avg PnL/trade':<20s} ${all_avg:>13.2f} ${avg_pnl:>13.2f} ${avg_pnl-all_avg:>+13.2f}")
    print(f"  {'Max DD':<20s} ${max_dd_all:>13.2f} ${max_dd:>13.2f} ${max_dd-max_dd_all:>+13.2f}")
    print()


def section6(trades):
    """What if we required higher bear threshold?"""
    print("=" * 80)
    print("6. BEAR THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("  (Currently: bear trades fire when prev_pm >= 0.80)")
    print("  What if we raised the threshold?\n")

    bears = [t for t in trades if t["direction"] == "bear"]
    bulls = [t for t in trades if t["direction"] == "bull"]

    first_trade = trades[0]
    starting_bankroll = first_trade["bankroll_after"] - first_trade["net_pnl"]

    thresholds = [0.80, 0.85, 0.90, 0.95]

    print(f"  {'Threshold':<12s} {'Bears Kept':>12s} {'Bears Cut':>12s} {'Bear WR':>10s} {'Bear PnL':>12s} {'Total PnL':>12s} {'Total WR':>10s} {'Max DD':>12s}")
    print(f"  {'-'*92}")

    for thresh in thresholds:
        # Filter bears by threshold
        kept_bears = [t for t in bears if t["prev_pm"] is not None and t["prev_pm"] >= thresh]
        cut_bears = [t for t in bears if t["prev_pm"] is not None and t["prev_pm"] < thresh]

        # Simulate: all bulls + only kept bears, in original time order
        kept_set = set()
        for t in kept_bears:
            # Use entry_time as unique key since we don't have trade_id easily
            kept_set.add(id(t))
        for t in bulls:
            kept_set.add(id(t))

        sim_trades = [t for t in trades if id(t) in kept_set]
        sim_trades.sort(key=lambda x: x["dt"])

        bankroll = starting_bankroll
        peak = bankroll
        max_dd = 0
        for t in sim_trades:
            bankroll += t["net_pnl"]
            if bankroll > peak:
                peak = bankroll
            dd = peak - bankroll
            if dd > max_dd:
                max_dd = dd

        n_sim, wr_sim, pnl_sim, _ = bucket_stats(sim_trades)
        n_kb, wr_kb, pnl_kb, _ = bucket_stats(kept_bears)
        n_cut = len(cut_bears)
        cut_pnl = sum(t["net_pnl"] for t in cut_bears)

        print(f"  >= {thresh:<8.2f} {n_kb:>12d} {n_cut:>12d} {wr_kb:>9.1f}% ${pnl_kb:>10.2f} ${pnl_sim:>10.2f} {wr_sim:>9.1f}% ${max_dd:>10.2f}")

    # Detail: what do the cut bears look like at each threshold?
    print(f"\n  --- Bears that would be CUT at each threshold ---")
    for thresh in [0.85, 0.90, 0.95]:
        cut = [t for t in bears if t["prev_pm"] is not None and t["prev_pm"] < thresh]
        if cut:
            n, wr, pnl, avg = bucket_stats(cut)
            print(f"  prev_pm < {thresh}: {n} trades, {wr:.1f}% WR, ${pnl:+.2f} PnL (avg ${avg:+.2f})")
    print()

    # Also show: what if we ONLY took bears with prev_pm >= 0.95?
    print("  --- What if bears required prev_pm >= 0.95? ---")
    elite_bears = [t for t in bears if t["prev_pm"] is not None and t["prev_pm"] >= 0.95]
    if elite_bears:
        n, wr, pnl, avg = bucket_stats(elite_bears)
        print(f"  {n} trades, {wr:.1f}% WR, ${pnl:+.2f} PnL (avg ${avg:+.2f})")
        # Show by asset
        by_asset = defaultdict(list)
        for t in elite_bears:
            by_asset[t["asset"]].append(t)
        for asset in sorted(by_asset.keys()):
            n, wr, pnl, avg = bucket_stats(by_asset[asset])
            print(f"    {asset}: {n} trades, {wr:.1f}% WR, ${pnl:+.2f} PnL")
    else:
        print("  No bears with prev_pm >= 0.95")
    print()


def main():
    trades = load_trades()
    print(f"\nLoaded {len(trades)} contrarian_consensus trades")
    print(f"Date range: {trades[0]['date_str']} to {trades[-1]['date_str']}")
    bulls = [t for t in trades if t["direction"] == "bull"]
    bears = [t for t in trades if t["direction"] == "bear"]
    print(f"Bulls: {len(bulls)}, Bears: {len(bears)}")
    n, wr, pnl, avg = bucket_stats(trades)
    print(f"Overall: {wr:.1f}% WR, ${pnl:+.2f} PnL, avg ${avg:+.2f}/trade\n")

    section1(trades)
    section2(trades)
    section3(trades)
    section4(trades)
    section5(trades)
    section6(trades)


if __name__ == "__main__":
    main()
