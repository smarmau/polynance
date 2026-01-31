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

    -- Resolution
    resolved_at_utc TIMESTAMP,
    resolution_source TEXT,

    PRIMARY KEY (window_id, asset)
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_time TIMESTAMP NOT NULL,
    asset TEXT NOT NULL,
    window_count INTEGER,

    -- Correlation coefficients
    corr_yes_t5_vs_outcome REAL,
    corr_yes_t10_vs_outcome REAL,
    corr_momentum_vs_outcome REAL,
    corr_signal_strength_vs_magnitude REAL,

    -- Signal performance
    accuracy_yes_gt_55 REAL,
    accuracy_yes_gt_60 REAL,
    accuracy_yes_lt_45 REAL,
    accuracy_yes_lt_40 REAL,

    -- Expected values
    ev_yes_gt_55_bps REAL,
    ev_yes_gt_60_bps REAL,

    -- Calibration
    calibration_error REAL,

    -- JSON blob for detailed results
    raw_data TEXT
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_samples_window ON samples(window_id, asset);
CREATE INDEX IF NOT EXISTS idx_samples_time ON samples(sample_time_utc);
CREATE INDEX IF NOT EXISTS idx_samples_asset ON samples(asset);
CREATE INDEX IF NOT EXISTS idx_windows_asset ON windows(asset);
CREATE INDEX IF NOT EXISTS idx_windows_time ON windows(window_start_utc);
CREATE INDEX IF NOT EXISTS idx_analysis_asset ON analysis_results(asset);
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

        logger.info(f"Connected to database: {self.db_path}")

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
            resolved_at_utc, resolution_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            resolved_at_utc=(
                datetime.fromisoformat(row["resolved_at_utc"])
                if row["resolved_at_utc"]
                else None
            ),
            resolution_source=row["resolution_source"],
        )

    # Analysis operations

    async def insert_analysis_result(
        self,
        asset: str,
        window_count: int,
        correlations: dict,
        accuracies: dict,
        expected_values: dict,
        calibration_error: float,
        raw_data: dict,
    ):
        """Insert an analysis result."""
        import json

        sql = """
        INSERT INTO analysis_results (
            analysis_time, asset, window_count,
            corr_yes_t5_vs_outcome, corr_yes_t10_vs_outcome,
            corr_momentum_vs_outcome, corr_signal_strength_vs_magnitude,
            accuracy_yes_gt_55, accuracy_yes_gt_60,
            accuracy_yes_lt_45, accuracy_yes_lt_40,
            ev_yes_gt_55_bps, ev_yes_gt_60_bps,
            calibration_error, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.conn.execute(
            sql,
            (
                datetime.now(timezone.utc).isoformat(),
                asset,
                window_count,
                correlations.get("yes_t5_vs_outcome"),
                correlations.get("yes_t10_vs_outcome"),
                correlations.get("momentum_vs_outcome"),
                correlations.get("signal_strength_vs_magnitude"),
                accuracies.get("yes_gt_55"),
                accuracies.get("yes_gt_60"),
                accuracies.get("yes_lt_45"),
                accuracies.get("yes_lt_40"),
                expected_values.get("yes_gt_55_bps"),
                expected_values.get("yes_gt_60_bps"),
                calibration_error,
                json.dumps(raw_data),
            ),
        )
        await self.conn.commit()

    async def get_latest_analysis(self, asset: str) -> Optional[dict]:
        """Get the most recent analysis result for an asset."""
        import json

        sql = """
        SELECT * FROM analysis_results
        WHERE asset = ?
        ORDER BY analysis_time DESC
        LIMIT 1
        """
        cursor = await self.conn.execute(sql, (asset,))
        row = await cursor.fetchone()

        if row:
            return {
                "analysis_time": datetime.fromisoformat(row["analysis_time"]),
                "asset": row["asset"],
                "window_count": row["window_count"],
                "correlations": {
                    "yes_t5_vs_outcome": row["corr_yes_t5_vs_outcome"],
                    "yes_t10_vs_outcome": row["corr_yes_t10_vs_outcome"],
                    "momentum_vs_outcome": row["corr_momentum_vs_outcome"],
                    "signal_strength_vs_magnitude": row["corr_signal_strength_vs_magnitude"],
                },
                "accuracies": {
                    "yes_gt_55": row["accuracy_yes_gt_55"],
                    "yes_gt_60": row["accuracy_yes_gt_60"],
                    "yes_lt_45": row["accuracy_yes_lt_45"],
                    "yes_lt_40": row["accuracy_yes_lt_40"],
                },
                "expected_values": {
                    "yes_gt_55_bps": row["ev_yes_gt_55_bps"],
                    "yes_gt_60_bps": row["ev_yes_gt_60_bps"],
                },
                "calibration_error": row["calibration_error"],
                "raw_data": json.loads(row["raw_data"]) if row["raw_data"] else None,
            }
        return None

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
