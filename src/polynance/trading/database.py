"""Database operations for simulated trading."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Union

import aiosqlite

from .models import SimulatedTrade, TradingState

logger = logging.getLogger(__name__)


SCHEMA = """
-- Global trading state (single row enforced by CHECK constraint)
CREATE TABLE IF NOT EXISTS sim_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    timestamp TIMESTAMP NOT NULL,

    -- Portfolio state
    current_bankroll REAL NOT NULL,
    current_bet_size REAL NOT NULL,
    initial_bankroll REAL NOT NULL,
    base_bet_size REAL NOT NULL,

    -- Performance metrics
    total_trades INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_losses INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,

    -- Drawdown tracking
    peak_bankroll REAL NOT NULL,
    max_drawdown REAL DEFAULT 0,
    max_drawdown_pct REAL DEFAULT 0,

    -- Streaks
    current_win_streak INTEGER DEFAULT 0,
    current_loss_streak INTEGER DEFAULT 0,
    max_win_streak INTEGER DEFAULT 0,
    max_loss_streak INTEGER DEFAULT 0,

    -- System state
    last_window_id TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,

    -- Pause-after-loss tracking
    pause_windows_remaining INTEGER DEFAULT 0
);

-- Trade records
CREATE TABLE IF NOT EXISTS sim_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,

    -- Window reference
    window_id TEXT NOT NULL,
    asset TEXT NOT NULL,

    -- Trade details
    direction TEXT NOT NULL CHECK (direction IN ('bull', 'bear')),
    entry_time TIMESTAMP NOT NULL,
    entry_price REAL NOT NULL,
    bet_size REAL NOT NULL,

    -- Exit details (NULL until resolved)
    exit_time TIMESTAMP,
    exit_price REAL,
    outcome TEXT CHECK (outcome IN ('win', 'loss', 'pending')),

    -- P&L calculation
    gross_pnl REAL,
    fee_paid REAL,
    spread_cost REAL,
    net_pnl REAL,

    -- State after trade
    bankroll_after REAL,
    drawdown REAL,
    drawdown_pct REAL,

    -- Metadata
    created_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_trades_window ON sim_trades(window_id, asset);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON sim_trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_time ON sim_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON sim_trades(outcome);
"""


class TradingDatabase:
    """Async SQLite database for simulated trading state and trades."""

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

        logger.info(f"Connected to trading database: {self.db_path}")

    async def _run_migrations(self):
        """Run database migrations for schema updates."""
        # Migration: Add pause_windows_remaining column if it doesn't exist
        try:
            cursor = await self._conn.execute("PRAGMA table_info(sim_state)")
            columns = [row[1] for row in await cursor.fetchall()]

            if "pause_windows_remaining" not in columns:
                await self._conn.execute(
                    "ALTER TABLE sim_state ADD COLUMN pause_windows_remaining INTEGER DEFAULT 0"
                )
                await self._conn.commit()
                logger.info("Migration: Added pause_windows_remaining column to sim_state")
        except Exception as e:
            logger.debug(f"Migration check: {e}")

    async def close(self):
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Trading database not connected")
        return self._conn

    # =========================================================================
    # State Operations
    # =========================================================================

    async def load_state(self) -> Optional[TradingState]:
        """Load trading state from database."""
        sql = "SELECT * FROM sim_state WHERE id = 1"
        cursor = await self.conn.execute(sql)
        row = await cursor.fetchone()

        if row is None:
            return None

        return TradingState(
            current_bankroll=row["current_bankroll"],
            current_bet_size=row["current_bet_size"],
            initial_bankroll=row["initial_bankroll"],
            base_bet_size=row["base_bet_size"],
            total_trades=row["total_trades"],
            total_wins=row["total_wins"],
            total_losses=row["total_losses"],
            total_pnl=row["total_pnl"],
            peak_bankroll=row["peak_bankroll"],
            max_drawdown=row["max_drawdown"],
            max_drawdown_pct=row["max_drawdown_pct"],
            current_win_streak=row["current_win_streak"],
            current_loss_streak=row["current_loss_streak"],
            max_win_streak=row["max_win_streak"],
            max_loss_streak=row["max_loss_streak"],
            last_window_id=row["last_window_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            pause_windows_remaining=row["pause_windows_remaining"] if "pause_windows_remaining" in row.keys() else 0,
        )

    async def save_state(self, state: TradingState):
        """Save trading state to database (upsert)."""
        now = datetime.now(timezone.utc)

        sql = """
        INSERT OR REPLACE INTO sim_state (
            id, timestamp,
            current_bankroll, current_bet_size, initial_bankroll, base_bet_size,
            total_trades, total_wins, total_losses, total_pnl,
            peak_bankroll, max_drawdown, max_drawdown_pct,
            current_win_streak, current_loss_streak, max_win_streak, max_loss_streak,
            last_window_id, created_at, updated_at, pause_windows_remaining
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        await self.conn.execute(
            sql,
            (
                now.isoformat(),
                state.current_bankroll,
                state.current_bet_size,
                state.initial_bankroll,
                state.base_bet_size,
                state.total_trades,
                state.total_wins,
                state.total_losses,
                state.total_pnl,
                state.peak_bankroll,
                state.max_drawdown,
                state.max_drawdown_pct,
                state.current_win_streak,
                state.current_loss_streak,
                state.max_win_streak,
                state.max_loss_streak,
                state.last_window_id,
                state.created_at.isoformat() if state.created_at else now.isoformat(),
                now.isoformat(),
                state.pause_windows_remaining,
            ),
        )
        await self.conn.commit()

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def insert_trade(self, trade: SimulatedTrade) -> int:
        """Insert a new trade record."""
        sql = """
        INSERT INTO sim_trades (
            trade_id, window_id, asset, direction,
            entry_time, entry_price, bet_size,
            exit_time, exit_price, outcome,
            gross_pnl, fee_paid, spread_cost, net_pnl,
            bankroll_after, drawdown, drawdown_pct,
            created_at, resolved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor = await self.conn.execute(
            sql,
            (
                trade.trade_id,
                trade.window_id,
                trade.asset,
                trade.direction,
                trade.entry_time.isoformat() if trade.entry_time else None,
                trade.entry_price,
                trade.bet_size,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price,
                trade.outcome,
                trade.gross_pnl,
                trade.fee_paid,
                trade.spread_cost,
                trade.net_pnl,
                trade.bankroll_after,
                trade.drawdown,
                trade.drawdown_pct,
                trade.created_at.isoformat() if trade.created_at else None,
                trade.resolved_at.isoformat() if trade.resolved_at else None,
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def update_trade(self, trade: SimulatedTrade):
        """Update an existing trade record."""
        sql = """
        UPDATE sim_trades SET
            exit_time = ?,
            exit_price = ?,
            outcome = ?,
            gross_pnl = ?,
            fee_paid = ?,
            spread_cost = ?,
            net_pnl = ?,
            bankroll_after = ?,
            drawdown = ?,
            drawdown_pct = ?,
            resolved_at = ?
        WHERE trade_id = ?
        """

        await self.conn.execute(
            sql,
            (
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price,
                trade.outcome,
                trade.gross_pnl,
                trade.fee_paid,
                trade.spread_cost,
                trade.net_pnl,
                trade.bankroll_after,
                trade.drawdown,
                trade.drawdown_pct,
                trade.resolved_at.isoformat() if trade.resolved_at else None,
                trade.trade_id,
            ),
        )
        await self.conn.commit()

    async def get_pending_trades(self) -> List[SimulatedTrade]:
        """Get all pending (unresolved) trades."""
        sql = """
        SELECT * FROM sim_trades
        WHERE outcome = 'pending'
        ORDER BY entry_time ASC
        """
        cursor = await self.conn.execute(sql)
        rows = await cursor.fetchall()

        return [self._row_to_trade(row) for row in rows]

    async def get_recent_trades(self, limit: int = 10) -> List[SimulatedTrade]:
        """Get most recent resolved trades."""
        sql = """
        SELECT * FROM sim_trades
        WHERE outcome != 'pending'
        ORDER BY resolved_at DESC
        LIMIT ?
        """
        cursor = await self.conn.execute(sql, (limit,))
        rows = await cursor.fetchall()

        return [self._row_to_trade(row) for row in rows]

    async def get_trades_by_asset(self, asset: str, limit: int = 100) -> List[SimulatedTrade]:
        """Get trades for a specific asset."""
        sql = """
        SELECT * FROM sim_trades
        WHERE asset = ?
        ORDER BY entry_time DESC
        LIMIT ?
        """
        cursor = await self.conn.execute(sql, (asset, limit))
        rows = await cursor.fetchall()

        return [self._row_to_trade(row) for row in rows]

    async def get_all_trades(self, resolved_only: bool = True) -> List[SimulatedTrade]:
        """Get all trades."""
        if resolved_only:
            sql = """
            SELECT * FROM sim_trades
            WHERE outcome != 'pending'
            ORDER BY entry_time ASC
            """
        else:
            sql = """
            SELECT * FROM sim_trades
            ORDER BY entry_time ASC
            """
        cursor = await self.conn.execute(sql)
        rows = await cursor.fetchall()

        return [self._row_to_trade(row) for row in rows]

    async def get_trade_by_window(self, window_id: str, asset: str) -> Optional[SimulatedTrade]:
        """Get trade for a specific window and asset."""
        sql = """
        SELECT * FROM sim_trades
        WHERE window_id = ? AND asset = ?
        """
        cursor = await self.conn.execute(sql, (window_id, asset))
        row = await cursor.fetchone()

        if row:
            return self._row_to_trade(row)
        return None

    def _row_to_trade(self, row: aiosqlite.Row) -> SimulatedTrade:
        """Convert a database row to a SimulatedTrade object."""
        return SimulatedTrade(
            trade_id=row["trade_id"],
            window_id=row["window_id"],
            asset=row["asset"],
            direction=row["direction"],
            entry_time=(
                datetime.fromisoformat(row["entry_time"])
                if row["entry_time"]
                else None
            ),
            entry_price=row["entry_price"],
            bet_size=row["bet_size"],
            exit_time=(
                datetime.fromisoformat(row["exit_time"])
                if row["exit_time"]
                else None
            ),
            exit_price=row["exit_price"],
            outcome=row["outcome"],
            gross_pnl=row["gross_pnl"],
            fee_paid=row["fee_paid"],
            spread_cost=row["spread_cost"],
            net_pnl=row["net_pnl"],
            bankroll_after=row["bankroll_after"],
            drawdown=row["drawdown"],
            drawdown_pct=row["drawdown_pct"],
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row["created_at"]
                else None
            ),
            resolved_at=(
                datetime.fromisoformat(row["resolved_at"])
                if row["resolved_at"]
                else None
            ),
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_asset_stats(self, asset: str) -> dict:
        """Get statistics for a specific asset."""
        sql = """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome != 'pending' THEN net_pnl ELSE 0 END) as total_pnl
        FROM sim_trades
        WHERE asset = ?
        """
        cursor = await self.conn.execute(sql, (asset,))
        row = await cursor.fetchone()

        if row:
            total = row["total_trades"] or 0
            wins = row["wins"] or 0
            return {
                "trades": total,
                "wins": wins,
                "losses": row["losses"] or 0,
                "total_pnl": row["total_pnl"] or 0.0,
                "win_rate": wins / total if total > 0 else 0.0,
            }
        return {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0, "win_rate": 0.0}

    async def get_today_stats(self) -> dict:
        """Get today's trading statistics."""
        today = datetime.now(timezone.utc).date().isoformat()

        sql = """
        SELECT
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome != 'pending' THEN net_pnl ELSE 0 END) as pnl
        FROM sim_trades
        WHERE date(entry_time) = ?
        """
        cursor = await self.conn.execute(sql, (today,))
        row = await cursor.fetchone()

        if row:
            return {
                "trades": row["trades"] or 0,
                "wins": row["wins"] or 0,
                "pnl": row["pnl"] or 0.0,
            }
        return {"trades": 0, "wins": 0, "pnl": 0.0}

    async def reset_state(self):
        """Reset all trading state (for testing/fresh start)."""
        await self.conn.execute("DELETE FROM sim_trades")
        await self.conn.execute("DELETE FROM sim_state")
        await self.conn.commit()
        logger.warning("Trading state has been reset")
