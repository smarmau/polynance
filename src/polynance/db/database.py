"""SQLite database for polynance data storage."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Union

import aiosqlite

from .models import Sample, Window

logger = logging.getLogger(__name__)


SCHEMA = """
-- Main table: one row per sample point
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    window_start_utc TIMESTAMP NOT NULL,
    sample_time_utc TIMESTAMP NOT NULL,
    t_minutes REAL NOT NULL,

    -- Polymarket data
    pm_yes_price REAL,
    pm_no_price REAL,
    pm_yes_bid REAL,
    pm_yes_ask REAL,
    pm_spread REAL,
    pm_midpoint REAL,

    -- Spot price data
    spot_price REAL,
    spot_price_change_from_open REAL,

    -- Market metadata
    pm_market_id TEXT,
    pm_condition_id TEXT,

    UNIQUE(window_id, asset, t_minutes)
);

-- Window summary table: one row per completed window
CREATE TABLE IF NOT EXISTS windows (
    window_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    window_start_utc TIMESTAMP NOT NULL,
    window_end_utc TIMESTAMP NOT NULL,

    -- Outcome
    outcome TEXT,
    outcome_binary INTEGER,

    -- Spot price movement
    spot_open REAL,
    spot_close REAL,
    spot_change_pct REAL,
    spot_change_bps REAL,
    spot_high REAL,
    spot_low REAL,
    spot_range_bps REAL,

    -- Polymarket prices at key times
    pm_yes_t0 REAL,
    pm_yes_t2_5 REAL,
    pm_yes_t5 REAL,
    pm_yes_t7_5 REAL,
    pm_yes_t10 REAL,
    pm_yes_t12_5 REAL,
    pm_spread_t0 REAL,
    pm_spread_t5 REAL,

    -- Derived signals
    pm_price_momentum_0_to_5 REAL,
    pm_price_momentum_5_to_10 REAL,

    -- Cross-window references
    prev_pm_t12_5 REAL,
    prev2_pm_t12_5 REAL,

    -- Cross-asset key
    window_time TEXT,

    -- Regime classification
    volatility_regime TEXT,

    -- Resolution
    resolved_at_utc TIMESTAMP,
    resolution_source TEXT,

    PRIMARY KEY (window_id, asset)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_samples_window ON samples(window_id, asset);
CREATE INDEX IF NOT EXISTS idx_samples_time ON samples(sample_time_utc);
CREATE INDEX IF NOT EXISTS idx_samples_asset ON samples(asset);
CREATE INDEX IF NOT EXISTS idx_windows_asset ON windows(asset);
CREATE INDEX IF NOT EXISTS idx_windows_time ON windows(window_start_utc);
-- Note: idx_windows_time_key on window_time is created by _run_migrations()
-- after ensuring the column exists (for pre-migration databases)
"""


class Database:
    """Async SQLite database for storing polynance data."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Connect to the database and initialize schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # Initialize schema
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

        # Run migrations for existing databases
        await self._run_migrations()

        logger.info(f"Connected to database: {self.db_path}")

    async def _run_migrations(self):
        """Run database migrations for schema updates."""
        # Migration: Add new columns to windows table
        try:
            cursor = await self._conn.execute("PRAGMA table_info(windows)")
            columns = [row[1] for row in await cursor.fetchall()]

            new_cols = {
                "prev_pm_t12_5": "REAL",
                "prev2_pm_t12_5": "REAL",
                "window_time": "TEXT",
                "volatility_regime": "TEXT",
            }
            for col_name, col_type in new_cols.items():
                if col_name not in columns:
                    await self._conn.execute(
                        f"ALTER TABLE windows ADD COLUMN {col_name} {col_type}"
                    )
                    logger.info(f"Migration: Added {col_name} column to windows")
            await self._conn.commit()
        except Exception as e:
            logger.debug(f"Migration check (windows): {e}")

        # Migration: Drop analysis_results table if it exists
        try:
            await self._conn.execute("DROP TABLE IF EXISTS analysis_results")
            await self._conn.commit()
            logger.debug("Migration: Dropped analysis_results table")
        except Exception as e:
            logger.debug(f"Migration check (drop analysis_results): {e}")

        # Migration: Create window_time index if missing
        try:
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_windows_time_key ON windows(window_time)"
            )
            await self._conn.commit()
        except Exception as e:
            logger.debug(f"Migration check (window_time index): {e}")

    async def close(self):
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    # Sample operations

    async def insert_sample(self, sample: Sample) -> int:
        """Insert a sample into the database.

        Returns:
            The row ID of the inserted sample
        """
        sql = """
        INSERT OR REPLACE INTO samples (
            window_id, asset, window_start_utc, sample_time_utc, t_minutes,
            pm_yes_price, pm_no_price, pm_yes_bid, pm_yes_ask, pm_spread, pm_midpoint,
            spot_price, spot_price_change_from_open, pm_market_id, pm_condition_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor = await self.conn.execute(
            sql,
            (
                sample.window_id,
                sample.asset,
                sample.window_start_utc.isoformat(),
                sample.sample_time_utc.isoformat(),
                sample.t_minutes,
                sample.pm_yes_price,
                sample.pm_no_price,
                sample.pm_yes_bid,
                sample.pm_yes_ask,
                sample.pm_spread,
                sample.pm_midpoint,
                sample.spot_price,
                sample.spot_price_change_from_open,
                sample.pm_market_id,
                sample.pm_condition_id,
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def get_samples_for_window(self, window_id: str, asset: str) -> List[Sample]:
        """Get all samples for a specific window and asset."""
        sql = """
        SELECT * FROM samples
        WHERE window_id = ? AND asset = ?
        ORDER BY t_minutes
        """
        cursor = await self.conn.execute(sql, (window_id, asset))
        rows = await cursor.fetchall()

        return [self._row_to_sample(row) for row in rows]

    def _row_to_sample(self, row: aiosqlite.Row) -> Sample:
        """Convert a database row to a Sample object."""
        return Sample(
            id=row["id"],
            window_id=row["window_id"],
            asset=row["asset"],
            window_start_utc=datetime.fromisoformat(row["window_start_utc"]),
            sample_time_utc=datetime.fromisoformat(row["sample_time_utc"]),
            t_minutes=row["t_minutes"],
            pm_yes_price=row["pm_yes_price"],
            pm_no_price=row["pm_no_price"],
            pm_yes_bid=row["pm_yes_bid"],
            pm_yes_ask=row["pm_yes_ask"],
            pm_spread=row["pm_spread"],
            pm_midpoint=row["pm_midpoint"],
            spot_price=row["spot_price"],
            spot_price_change_from_open=row["spot_price_change_from_open"],
            pm_market_id=row["pm_market_id"],
            pm_condition_id=row["pm_condition_id"],
        )

    # Window operations

    async def insert_window(self, window: Window):
        """Insert or update a window summary."""
        sql = """
        INSERT OR REPLACE INTO windows (
            window_id, asset, window_start_utc, window_end_utc,
            outcome, outcome_binary,
            spot_open, spot_close, spot_change_pct, spot_change_bps,
            spot_high, spot_low, spot_range_bps,
            pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5,
            pm_spread_t0, pm_spread_t5,
            pm_price_momentum_0_to_5, pm_price_momentum_5_to_10,
            prev_pm_t12_5, prev2_pm_t12_5,
            window_time, volatility_regime,
            resolved_at_utc, resolution_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.conn.execute(
            sql,
            (
                window.window_id,
                window.asset,
                window.window_start_utc.isoformat(),
                window.window_end_utc.isoformat(),
                window.outcome,
                window.outcome_binary,
                window.spot_open,
                window.spot_close,
                window.spot_change_pct,
                window.spot_change_bps,
                window.spot_high,
                window.spot_low,
                window.spot_range_bps,
                window.pm_yes_t0,
                window.pm_yes_t2_5,
                window.pm_yes_t5,
                window.pm_yes_t7_5,
                window.pm_yes_t10,
                window.pm_yes_t12_5,
                window.pm_spread_t0,
                window.pm_spread_t5,
                window.pm_price_momentum_0_to_5,
                window.pm_price_momentum_5_to_10,
                window.prev_pm_t12_5,
                window.prev2_pm_t12_5,
                window.window_time,
                window.volatility_regime,
                window.resolved_at_utc.isoformat() if window.resolved_at_utc else None,
                window.resolution_source,
            ),
        )
        await self.conn.commit()

    async def get_window(self, window_id: str, asset: str) -> Optional[Window]:
        """Get a window summary by ID and asset."""
        sql = "SELECT * FROM windows WHERE window_id = ? AND asset = ?"
        cursor = await self.conn.execute(sql, (window_id, asset))
        row = await cursor.fetchone()

        if row:
            return self._row_to_window(row)
        return None

    async def get_recent_windows(
        self, asset: str, limit: int = 100, resolved_only: bool = True
    ) -> List[Window]:
        """Get recent windows for an asset."""
        if resolved_only:
            sql = """
            SELECT * FROM windows
            WHERE asset = ? AND outcome IS NOT NULL
            ORDER BY window_start_utc DESC
            LIMIT ?
            """
        else:
            sql = """
            SELECT * FROM windows
            WHERE asset = ?
            ORDER BY window_start_utc DESC
            LIMIT ?
            """
        cursor = await self.conn.execute(sql, (asset, limit))
        rows = await cursor.fetchall()

        return [self._row_to_window(row) for row in rows]

    async def get_window_count(self, asset: str, resolved_only: bool = True) -> int:
        """Get the total count of windows for an asset."""
        if resolved_only:
            sql = "SELECT COUNT(*) FROM windows WHERE asset = ? AND outcome IS NOT NULL"
        else:
            sql = "SELECT COUNT(*) FROM windows WHERE asset = ?"
        cursor = await self.conn.execute(sql, (asset,))
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_all_windows(self, asset: str, resolved_only: bool = True) -> List[Window]:
        """Get all windows for an asset."""
        if resolved_only:
            sql = """
            SELECT * FROM windows
            WHERE asset = ? AND outcome IS NOT NULL
            ORDER BY window_start_utc ASC
            """
        else:
            sql = """
            SELECT * FROM windows
            WHERE asset = ?
            ORDER BY window_start_utc ASC
            """
        cursor = await self.conn.execute(sql, (asset,))
        rows = await cursor.fetchall()

        return [self._row_to_window(row) for row in rows]

    def _row_to_window(self, row: aiosqlite.Row) -> Window:
        """Convert a database row to a Window object."""
        keys = row.keys()
        return Window(
            window_id=row["window_id"],
            asset=row["asset"],
            window_start_utc=datetime.fromisoformat(row["window_start_utc"]),
            window_end_utc=datetime.fromisoformat(row["window_end_utc"]),
            outcome=row["outcome"],
            outcome_binary=row["outcome_binary"],
            spot_open=row["spot_open"],
            spot_close=row["spot_close"],
            spot_change_pct=row["spot_change_pct"],
            spot_change_bps=row["spot_change_bps"],
            spot_high=row["spot_high"],
            spot_low=row["spot_low"],
            spot_range_bps=row["spot_range_bps"],
            pm_yes_t0=row["pm_yes_t0"],
            pm_yes_t2_5=row["pm_yes_t2_5"],
            pm_yes_t5=row["pm_yes_t5"],
            pm_yes_t7_5=row["pm_yes_t7_5"],
            pm_yes_t10=row["pm_yes_t10"],
            pm_yes_t12_5=row["pm_yes_t12_5"],
            pm_spread_t0=row["pm_spread_t0"],
            pm_spread_t5=row["pm_spread_t5"],
            pm_price_momentum_0_to_5=row["pm_price_momentum_0_to_5"],
            pm_price_momentum_5_to_10=row["pm_price_momentum_5_to_10"],
            prev_pm_t12_5=row["prev_pm_t12_5"] if "prev_pm_t12_5" in keys else None,
            prev2_pm_t12_5=row["prev2_pm_t12_5"] if "prev2_pm_t12_5" in keys else None,
            window_time=row["window_time"] if "window_time" in keys else None,
            volatility_regime=row["volatility_regime"] if "volatility_regime" in keys else None,
            resolved_at_utc=(
                datetime.fromisoformat(row["resolved_at_utc"])
                if row["resolved_at_utc"]
                else None
            ),
            resolution_source=row["resolution_source"],
        )

    # Statistics

    async def get_stats(self, asset: str) -> dict:
        """Get summary statistics for an asset."""
        # Count windows
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM windows WHERE asset = ? AND outcome IS NOT NULL",
            (asset,),
        )
        row = await cursor.fetchone()
        total_windows = row[0]

        # Count samples
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM samples WHERE asset = ?",
            (asset,),
        )
        row = await cursor.fetchone()
        total_samples = row[0]

        # Win rate (up outcomes)
        cursor = await self.conn.execute(
            """
            SELECT
                COUNT(CASE WHEN outcome = 'up' THEN 1 END) as up_count,
                COUNT(CASE WHEN outcome = 'down' THEN 1 END) as down_count
            FROM windows
            WHERE asset = ? AND outcome IS NOT NULL
            """,
            (asset,),
        )
        row = await cursor.fetchone()
        up_count = row[0] or 0
        down_count = row[1] or 0

        # Average move size
        cursor = await self.conn.execute(
            """
            SELECT AVG(ABS(spot_change_bps)) as avg_move_bps
            FROM windows
            WHERE asset = ? AND outcome IS NOT NULL
            """,
            (asset,),
        )
        row = await cursor.fetchone()
        avg_move_bps = row[0]

        return {
            "total_windows": total_windows,
            "total_samples": total_samples,
            "up_count": up_count,
            "down_count": down_count,
            "up_rate": up_count / total_windows if total_windows > 0 else 0,
            "avg_move_bps": avg_move_bps,
        }
