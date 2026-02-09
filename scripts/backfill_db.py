"""Backfill new columns in existing databases.

Populates:
  windows table: prev_pm_t12_5, prev2_pm_t12_5, window_time, volatility_regime
  sim_trades table: entry_mode (marks existing trades as 'contrarian_consensus')

Also drops analysis_results table if present.
"""

import sqlite3
from pathlib import Path

DATA_DIR = Path("data")
ASSETS = ["btc", "eth", "sol", "xrp"]


def backfill_windows(db_path: Path, asset: str):
    """Backfill new columns for all windows in an asset database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if columns exist (migration may not have run yet)
    cursor = conn.execute("PRAGMA table_info(windows)")
    columns = [row[1] for row in cursor.fetchall()]

    new_cols = {
        "prev_pm_t12_5": "REAL",
        "prev2_pm_t12_5": "REAL",
        "window_time": "TEXT",
        "volatility_regime": "TEXT",
    }
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            conn.execute(f"ALTER TABLE windows ADD COLUMN {col_name} {col_type}")
            print(f"  Added column {col_name}")

    # Create index if missing
    conn.execute("CREATE INDEX IF NOT EXISTS idx_windows_time_key ON windows(window_time)")

    # Drop analysis_results if present
    conn.execute("DROP TABLE IF EXISTS analysis_results")

    # Fetch all windows sorted by time
    rows = conn.execute(
        "SELECT window_id, asset, spot_range_bps, pm_yes_t12_5 "
        "FROM windows ORDER BY window_start_utc ASC"
    ).fetchall()

    print(f"  {len(rows)} windows to backfill")

    prev_pm = None
    prev2_pm = None
    updated = 0

    for row in rows:
        wid = row["window_id"]
        range_bps = row["spot_range_bps"]
        pm_12_5 = row["pm_yes_t12_5"]

        # window_time: strip asset prefix
        parts = wid.split("_", 1)
        window_time = parts[1] if len(parts) > 1 else wid

        # volatility_regime
        regime = None
        if range_bps is not None:
            if range_bps < 15:
                regime = "low"
            elif range_bps < 40:
                regime = "normal"
            elif range_bps < 80:
                regime = "high"
            else:
                regime = "extreme"

        conn.execute(
            """UPDATE windows SET
                prev_pm_t12_5 = ?,
                prev2_pm_t12_5 = ?,
                window_time = ?,
                volatility_regime = ?
            WHERE window_id = ? AND asset = ?""",
            (prev_pm, prev2_pm, window_time, regime, wid, row["asset"]),
        )
        updated += 1

        # Shift prev tracking
        prev2_pm = prev_pm
        prev_pm = pm_12_5

    conn.commit()
    conn.close()
    print(f"  Updated {updated} windows")


def backfill_sim_trades():
    """Mark existing sim_trades with entry_mode if missing."""
    db_path = DATA_DIR / "sim_trading.db"
    if not db_path.exists():
        print("No sim_trading.db found, skipping")
        return

    conn = sqlite3.connect(db_path)

    # Check if column exists
    cursor = conn.execute("PRAGMA table_info(sim_trades)")
    columns = [row[1] for row in cursor.fetchall()]

    new_cols = {
        "entry_mode": "TEXT",
        "prev_pm": "REAL",
        "prev2_pm": "REAL",
        "spot_velocity": "REAL",
        "pm_momentum": "REAL",
    }
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            conn.execute(f"ALTER TABLE sim_trades ADD COLUMN {col_name} {col_type}")
            print(f"  Added column {col_name} to sim_trades")

    # Tag existing trades with entry_mode='contrarian_consensus' (what was running)
    result = conn.execute(
        "UPDATE sim_trades SET entry_mode = 'contrarian_consensus' WHERE entry_mode IS NULL"
    )
    print(f"  Tagged {result.rowcount} existing trades as 'contrarian_consensus'")

    conn.commit()
    conn.close()


def main():
    print("=" * 60)
    print("Database Backfill Script")
    print("=" * 60)

    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            print(f"\n{asset.upper()}: database not found, skipping")
            continue
        print(f"\n{asset.upper()}: {db_path}")
        backfill_windows(db_path, asset)

    print(f"\nSIM_TRADING:")
    backfill_sim_trades()

    print(f"\nDone!")


if __name__ == "__main__":
    main()
