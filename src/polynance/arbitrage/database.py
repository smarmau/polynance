"""Separate database for arbitrage/lock strategy tracking.

High-frequency data storage optimized for:
- 30-second tick data
- Lock opportunity signals
- Position tracking simulation
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Union

import aiosqlite

logger = logging.getLogger(__name__)


SCHEMA = """
-- High-frequency tick data (30-second samples)
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    window_id TEXT NOT NULL,
    t_seconds REAL NOT NULL,  -- Seconds into the 15-min window

    -- Polymarket prices
    yes_price REAL,
    no_price REAL,
    yes_bid REAL,
    yes_ask REAL,
    no_bid REAL,
    no_ask REAL,
    spread REAL,

    -- Spot price (for reference)
    spot_price REAL,

    -- Computed signals (updated each tick)
    rhr REAL,  -- Rolling High Range
    obi REAL,  -- Order Book Imbalance
    pulse REAL,  -- Price velocity
    delta_from_fair REAL,  -- Distance from 0.50

    UNIQUE(asset, window_id, t_seconds)
);

-- Window-level summary for arbitrage
CREATE TABLE IF NOT EXISTS arb_windows (
    window_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    window_start_utc TIMESTAMP NOT NULL,
    window_end_utc TIMESTAMP,

    -- Opening conditions
    yes_open REAL,
    opening_bias TEXT,  -- 'strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'

    -- Price action
    yes_high REAL,
    yes_low REAL,
    yes_close REAL,
    yes_range REAL,

    -- Signals at key times
    rhr_max REAL,
    obi_min REAL,
    obi_max REAL,
    flip_count INTEGER,  -- Times price crossed 0.50

    -- Lock analysis
    lock_achievable BOOLEAN,
    lock_achieved_at_seconds REAL,  -- When lock was first achievable
    estimated_lock_profit REAL,

    -- Pattern
    pattern TEXT,  -- 'trending', 'reversal', 'choppy', 'flat'

    -- Outcome
    outcome TEXT,  -- 'up', 'down'

    PRIMARY KEY (window_id, asset)
);

-- Simulated positions (for backtesting lock strategy)
CREATE TABLE IF NOT EXISTS sim_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL,
    action TEXT NOT NULL,  -- 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
    shares REAL NOT NULL,
    price REAL NOT NULL,
    cost REAL NOT NULL,

    -- Running totals after this trade
    yes_shares REAL,
    no_shares REAL,
    total_cost REAL,

    -- Lock status
    is_locked BOOLEAN,
    profit_if_yes REAL,
    profit_if_no REAL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ticks_window ON ticks(asset, window_id);
CREATE INDEX IF NOT EXISTS idx_ticks_time ON ticks(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_arb_windows_asset ON arb_windows(asset);
CREATE INDEX IF NOT EXISTS idx_arb_windows_time ON arb_windows(window_start_utc);
"""


class ArbitrageDatabase:
    """Database for arbitrage tracking - separate from main polynance DB."""

    def __init__(self, db_path: Union[str, Path] = "data/arbitrage.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Connect and initialize schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info(f"Connected to arbitrage database: {self.db_path}")

    async def close(self):
        """Close connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    # =========================================================================
    # Tick operations
    # =========================================================================

    async def insert_tick(
        self,
        asset: str,
        timestamp: datetime,
        window_id: str,
        t_seconds: float,
        yes_price: float,
        no_price: float,
        yes_bid: float,
        yes_ask: float,
        spread: float,
        spot_price: Optional[float] = None,
        rhr: Optional[float] = None,
        obi: Optional[float] = None,
        pulse: Optional[float] = None,
    ):
        """Insert a tick record."""
        delta_from_fair = abs(yes_price - 0.5) if yes_price else None
        no_bid = 1 - yes_ask if yes_ask else None
        no_ask = 1 - yes_bid if yes_bid else None

        sql = """
        INSERT OR REPLACE INTO ticks (
            asset, timestamp_utc, window_id, t_seconds,
            yes_price, no_price, yes_bid, yes_ask, no_bid, no_ask, spread,
            spot_price, rhr, obi, pulse, delta_from_fair
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.conn.execute(sql, (
            asset, timestamp.isoformat(), window_id, t_seconds,
            yes_price, no_price, yes_bid, yes_ask, no_bid, no_ask, spread,
            spot_price, rhr, obi, pulse, delta_from_fair
        ))
        await self.conn.commit()

    async def get_window_ticks(self, asset: str, window_id: str) -> List[Dict]:
        """Get all ticks for a window."""
        sql = """
        SELECT * FROM ticks
        WHERE asset = ? AND window_id = ?
        ORDER BY t_seconds
        """
        cursor = await self.conn.execute(sql, (asset, window_id))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_recent_ticks(self, asset: str, limit: int = 100) -> List[Dict]:
        """Get recent ticks for an asset."""
        sql = """
        SELECT * FROM ticks
        WHERE asset = ?
        ORDER BY timestamp_utc DESC
        LIMIT ?
        """
        cursor = await self.conn.execute(sql, (asset, limit))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Window operations
    # =========================================================================

    async def upsert_window(
        self,
        window_id: str,
        asset: str,
        window_start: datetime,
        yes_open: float,
        **kwargs
    ):
        """Insert or update window summary."""
        # Determine opening bias
        if yes_open >= 0.65:
            opening_bias = 'strong_bull'
        elif yes_open >= 0.55:
            opening_bias = 'bull'
        elif yes_open >= 0.45:
            opening_bias = 'neutral'
        elif yes_open >= 0.35:
            opening_bias = 'bear'
        else:
            opening_bias = 'strong_bear'

        # Build dynamic SQL for optional fields
        fields = ['window_id', 'asset', 'window_start_utc', 'yes_open', 'opening_bias']
        values = [window_id, asset, window_start.isoformat(), yes_open, opening_bias]

        for key, val in kwargs.items():
            if val is not None:
                fields.append(key)
                if isinstance(val, datetime):
                    values.append(val.isoformat())
                else:
                    values.append(val)

        placeholders = ', '.join(['?'] * len(values))
        field_names = ', '.join(fields)

        sql = f"""
        INSERT OR REPLACE INTO arb_windows ({field_names})
        VALUES ({placeholders})
        """
        await self.conn.execute(sql, values)
        await self.conn.commit()

    async def get_window(self, asset: str, window_id: str) -> Optional[Dict]:
        """Get window summary."""
        sql = "SELECT * FROM arb_windows WHERE asset = ? AND window_id = ?"
        cursor = await self.conn.execute(sql, (asset, window_id))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_recent_windows(self, asset: str, limit: int = 50) -> List[Dict]:
        """Get recent windows."""
        sql = """
        SELECT * FROM arb_windows
        WHERE asset = ?
        ORDER BY window_start_utc DESC
        LIMIT ?
        """
        cursor = await self.conn.execute(sql, (asset, limit))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_lockable_windows(self, asset: Optional[str] = None) -> List[Dict]:
        """Get windows where lock was achievable."""
        if asset:
            sql = """
            SELECT * FROM arb_windows
            WHERE asset = ? AND lock_achievable = 1
            ORDER BY window_start_utc DESC
            """
            cursor = await self.conn.execute(sql, (asset,))
        else:
            sql = """
            SELECT * FROM arb_windows
            WHERE lock_achievable = 1
            ORDER BY window_start_utc DESC
            """
            cursor = await self.conn.execute(sql)

        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # =========================================================================
    # Stats
    # =========================================================================

    async def get_stats(self, asset: Optional[str] = None) -> Dict:
        """Get aggregate statistics."""
        if asset:
            tick_sql = "SELECT COUNT(*) FROM ticks WHERE asset = ?"
            window_sql = "SELECT COUNT(*) FROM arb_windows WHERE asset = ?"
            lockable_sql = "SELECT COUNT(*) FROM arb_windows WHERE asset = ? AND lock_achievable = 1"
            params = (asset,)
        else:
            tick_sql = "SELECT COUNT(*) FROM ticks"
            window_sql = "SELECT COUNT(*) FROM arb_windows"
            lockable_sql = "SELECT COUNT(*) FROM arb_windows WHERE lock_achievable = 1"
            params = ()

        tick_cursor = await self.conn.execute(tick_sql, params)
        tick_count = (await tick_cursor.fetchone())[0]

        window_cursor = await self.conn.execute(window_sql, params)
        window_count = (await window_cursor.fetchone())[0]

        lockable_cursor = await self.conn.execute(lockable_sql, params)
        lockable_count = (await lockable_cursor.fetchone())[0]

        return {
            'tick_count': tick_count,
            'window_count': window_count,
            'lockable_count': lockable_count,
            'lockable_pct': lockable_count / window_count * 100 if window_count > 0 else 0,
        }
