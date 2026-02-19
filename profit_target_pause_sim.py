"""
Profit Target Pause Simulation
-------------------------------
Simulates a mechanism where trading pauses for Y hours after cumulative PnL
reaches +X% of the rolling bankroll target. Tests 8 combinations plus baseline.

Database: sim_trading.db (652 contrarian_consensus trades, Feb 12-17 2026)
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

DB_PATH = "/Volumes/shared_folder/polynance/sim_trading.db"

# ── Load trades ──────────────────────────────────────────────────────────────

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT entry_time, net_pnl, outcome, bet_size, asset, direction
        FROM sim_trades
        WHERE entry_mode = 'contrarian_consensus'
          AND outcome IN ('win', 'loss')
        ORDER BY entry_time ASC
    """)
    rows = cur.fetchall()
    conn.close()

    trades = []
    for r in rows:
        ts = r["entry_time"]
        # Parse ISO timestamp
        if ts.endswith("+00:00"):
            dt = datetime.fromisoformat(ts)
        else:
            dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        trades.append({
            "entry_time": dt,
            "net_pnl": r["net_pnl"],
            "outcome": r["outcome"],
            "bet_size": r["bet_size"],
            "asset": r["asset"],
            "direction": r["direction"],
        })
    return trades


# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    name: str
    starting_bankroll: float
    final_bankroll: float
    total_pnl: float
    trades_taken: int
    trades_skipped: int
    wins: int
    losses: int
    max_dd_dollar: float
    max_dd_pct: float
    times_target_hit: int
    skipped_pnl_list: list = field(default_factory=list)

    @property
    def win_rate(self):
        return self.wins / self.trades_taken * 100 if self.trades_taken else 0

    @property
    def avg_skipped_pnl(self):
        if not self.skipped_pnl_list:
            return 0.0
        return sum(self.skipped_pnl_list) / len(self.skipped_pnl_list)

    @property
    def pnl_dd_ratio(self):
        if self.max_dd_dollar == 0:
            return float("inf") if self.total_pnl > 0 else 0
        return self.total_pnl / self.max_dd_dollar


def simulate(trades, profit_target_pct=None, pause_hours=None, starting_bankroll=1000.0):
    """
    profit_target_pct: e.g. 0.10 for +10%. None = baseline (no pausing).
    pause_hours: hours to pause after target hit. None = baseline.
    """
    name = "Baseline (no pause)"
    if profit_target_pct is not None and pause_hours is not None:
        name = f"+{int(profit_target_pct*100)}% / {pause_hours}h pause"

    bankroll = starting_bankroll
    target_base = starting_bankroll
    peak_bankroll = starting_bankroll
    pause_end_time = None

    trades_taken = 0
    trades_skipped = 0
    wins = 0
    losses = 0
    max_dd_dollar = 0.0
    max_dd_pct = 0.0
    times_target_hit = 0
    skipped_pnl_list = []

    for t in trades:
        entry_time = t["entry_time"]

        # Check if we are in a pause period
        if pause_end_time is not None and entry_time < pause_end_time:
            trades_skipped += 1
            skipped_pnl_list.append(t["net_pnl"])
            continue

        # If we were paused but now the pause is over, reset target_base
        # (already set when pause was triggered)
        if pause_end_time is not None and entry_time >= pause_end_time:
            pause_end_time = None  # clear pause

        # Apply the trade
        bankroll += t["net_pnl"]
        trades_taken += 1
        if t["outcome"] == "win":
            wins += 1
        else:
            losses += 1

        # Track peak and drawdown
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd_dollar = peak_bankroll - bankroll
        dd_pct = dd_dollar / peak_bankroll * 100 if peak_bankroll > 0 else 0
        if dd_dollar > max_dd_dollar:
            max_dd_dollar = dd_dollar
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        # Check profit target
        if profit_target_pct is not None and pause_hours is not None:
            gain_pct = (bankroll - target_base) / target_base
            if gain_pct >= profit_target_pct:
                times_target_hit += 1
                pause_end_time = entry_time + timedelta(hours=pause_hours)
                target_base = bankroll  # reset rolling target

    return SimResult(
        name=name,
        starting_bankroll=starting_bankroll,
        final_bankroll=bankroll,
        total_pnl=bankroll - starting_bankroll,
        trades_taken=trades_taken,
        trades_skipped=trades_skipped,
        wins=wins,
        losses=losses,
        max_dd_dollar=max_dd_dollar,
        max_dd_pct=max_dd_pct,
        times_target_hit=times_target_hit,
        skipped_pnl_list=skipped_pnl_list,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    trades = load_trades()
    print(f"Loaded {len(trades)} contrarian_consensus trades")
    print(f"Date range: {trades[0]['entry_time'].strftime('%Y-%m-%d %H:%M')} -> "
          f"{trades[-1]['entry_time'].strftime('%Y-%m-%d %H:%M')}")
    print()

    # Quick data summary
    total_pnl_raw = sum(t["net_pnl"] for t in trades)
    total_wins_raw = sum(1 for t in trades if t["outcome"] == "win")
    print(f"Raw data: total PnL = ${total_pnl_raw:.2f}, "
          f"wins = {total_wins_raw}/{len(trades)} "
          f"({total_wins_raw/len(trades)*100:.1f}%)")
    print()

    # Test matrix
    profit_targets = [0.10, 0.25, 0.50, 1.00]
    pause_durations = [6, 12]

    results = []

    # Baseline
    results.append(simulate(trades))

    # All combos
    for pt in profit_targets:
        for pd in pause_durations:
            results.append(simulate(trades, profit_target_pct=pt, pause_hours=pd))

    # Sort by PnL/DD ratio descending
    results.sort(key=lambda r: r.pnl_dd_ratio, reverse=True)

    # ── Print table ──────────────────────────────────────────────────────
    print("=" * 160)
    print(f"{'Strategy':<22} {'Taken':>6} {'Skip':>5} {'Hits':>5} "
          f"{'Final $':>10} {'PnL $':>10} {'MaxDD $':>9} {'MaxDD%':>7} "
          f"{'PnL/DD':>7} {'WinR%':>6} {'Avg Skip PnL':>13}")
    print("-" * 160)

    for r in results:
        pnl_dd_str = f"{r.pnl_dd_ratio:.2f}" if r.max_dd_dollar > 0 else "inf"
        avg_skip = f"${r.avg_skipped_pnl:+.2f}" if r.skipped_pnl_list else "n/a"

        print(f"{r.name:<22} {r.trades_taken:>6} {r.trades_skipped:>5} "
              f"{r.times_target_hit:>5} "
              f"${r.final_bankroll:>9.2f} ${r.total_pnl:>9.2f} "
              f"${r.max_dd_dollar:>8.2f} {r.max_dd_pct:>6.2f}% "
              f"{pnl_dd_str:>7} {r.win_rate:>5.1f}% "
              f"{avg_skip:>13}")

    print("=" * 160)

    # ── Detailed skipped trade analysis ──────────────────────────────────
    print("\n\n--- Skipped Trade Analysis ---\n")
    for r in results:
        if not r.skipped_pnl_list:
            continue
        skip_wins = sum(1 for p in r.skipped_pnl_list if p > 0)
        skip_losses = sum(1 for p in r.skipped_pnl_list if p <= 0)
        skip_total = sum(r.skipped_pnl_list)
        print(f"{r.name:<22}  Skipped {len(r.skipped_pnl_list)} trades: "
              f"{skip_wins}W / {skip_losses}L  |  "
              f"Total skipped PnL: ${skip_total:+.2f}  |  "
              f"Avg: ${r.avg_skipped_pnl:+.2f}  |  "
              f"Skip WR: {skip_wins/len(r.skipped_pnl_list)*100:.1f}%")

    # ── Key observations ─────────────────────────────────────────────────
    print("\n\n--- Key Observations ---\n")
    baseline = [r for r in results if "Baseline" in r.name][0]
    print(f"Baseline: ${baseline.total_pnl:.2f} PnL, "
          f"${baseline.max_dd_dollar:.2f} max DD, "
          f"PnL/DD = {baseline.pnl_dd_ratio:.2f}")

    best_ratio = results[0]
    best_pnl = max(results, key=lambda r: r.total_pnl)
    lowest_dd = min(results, key=lambda r: r.max_dd_dollar)

    print(f"Best PnL/DD ratio:  {best_ratio.name} (ratio = {best_ratio.pnl_dd_ratio:.2f})")
    print(f"Best raw PnL:       {best_pnl.name} (${best_pnl.total_pnl:.2f})")
    print(f"Lowest max DD:      {lowest_dd.name} (${lowest_dd.max_dd_dollar:.2f})")

    # Compare each to baseline
    print("\n--- vs Baseline ---\n")
    for r in results:
        if "Baseline" in r.name:
            continue
        pnl_diff = r.total_pnl - baseline.total_pnl
        dd_diff = r.max_dd_dollar - baseline.max_dd_dollar
        ratio_diff = r.pnl_dd_ratio - baseline.pnl_dd_ratio
        print(f"{r.name:<22}  PnL diff: ${pnl_diff:+.2f}  |  "
              f"DD diff: ${dd_diff:+.2f}  |  "
              f"PnL/DD diff: {ratio_diff:+.2f}")


if __name__ == "__main__":
    main()
