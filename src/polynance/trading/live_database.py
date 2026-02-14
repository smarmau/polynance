"""Database operations for live trading.

Separate from the simulated trading database — stores actual exchange orders,
fill prices, and fees. This is the source of truth for P&L when live_trading=True.

The sim_trades database continues to run in parallel with estimated values.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Union

import aiosqlite

from .models import LiveTrade

logger = logging.getLogger(__name__)


LIVE_SCHEMA = """
-- Live trade records (actual exchange orders)
CREATE TABLE IF NOT EXISTS live_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    sim_trade_id TEXT,  -- links to sim_trades.trade_id

    -- Window reference
    window_id TEXT NOT NULL,
    asset TEXT NOT NULL,

    -- Trade details
    direction TEXT NOT NULL CHECK (direction IN ('bull', 'bear')),
    entry_mode TEXT,
    bet_size REAL NOT NULL,

    -- Entry order
    entry_order_id TEXT,
    entry_time TIMESTAMP,
    entry_price_requested REAL,
    entry_fill_price REAL,
    entry_contracts REAL,
    entry_fee REAL,

    -- Exit order
    exit_order_id TEXT,
    exit_time TIMESTAMP,
    exit_price_requested REAL,
    exit_fill_price REAL,
    exit_contracts REAL,
    exit_fee REAL,

    -- P&L (from actual fills)
    gross_pnl REAL,
    total_fees REAL,
    net_pnl REAL,
    outcome TEXT CHECK (outcome IN ('win', 'loss', 'pending')),

    -- Status
    status TEXT NOT NULL DEFAULT 'entry_placed',
    exchange TEXT DEFAULT 'polymarket',

    -- Metadata
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_live_window ON live_trades(window_id, asset);
CREATE INDEX IF NOT EXISTS idx_live_asset ON live_trades(asset);
CREATE INDEX IF NOT EXISTS idx_live_status ON live_trades(status);
CREATE INDEX IF NOT EXISTS idx_live_sim ON live_trades(sim_trade_id);
CREATE INDEX IF NOT EXISTS idx_live_entry_order ON live_trades(entry_order_id);

-- Live trading state (tracks actual exchange balance, P&L)
CREATE TABLE IF NOT EXISTS live_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    timestamp TIMESTAMP NOT NULL,

    -- Balance from exchange
    exchange_balance REAL,
    initial_deposit REAL,

    -- Performance (from actual fills)
    total_trades INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_losses INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    total_fees REAL DEFAULT 0,

    -- Metadata
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
"""


class LiveTradingDatabase:
    """Async SQLite database for live trading records.

    Stored in a separate DB file (e.g., data/live_trading.db) from the
    simulated trading database.
    """

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

        await self._conn.executescript(LIVE_SCHEMA)
        await self._conn.commit()

        logger.info(f"Connected to live trading database: {self.db_path}")

    async def close(self):
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Live trading database not connected")
        return self._conn

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def insert_trade(self, trade: LiveTrade) -> int:
        """Insert a new live trade record (entry order placed)."""
        sql = """
        INSERT INTO live_trades (
            trade_id, sim_trade_id, window_id, asset,
            direction, entry_mode, bet_size,
            entry_order_id, entry_time, entry_price_requested,
            entry_fill_price, entry_contracts, entry_fee,
            exit_order_id, exit_time, exit_price_requested,
            exit_fill_price, exit_contracts, exit_fee,
            gross_pnl, total_fees, net_pnl, outcome,
            status, exchange, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        now = datetime.now(timezone.utc)
        cursor = await self.conn.execute(
            sql,
            (
                trade.trade_id,
                trade.sim_trade_id,
                trade.window_id,
                trade.asset,
                trade.direction,
                trade.entry_mode,
                trade.bet_size,
                trade.entry_order_id,
                trade.entry_time.isoformat() if trade.entry_time else None,
                trade.entry_price_requested,
                trade.entry_fill_price,
                trade.entry_contracts,
                trade.entry_fee,
                trade.exit_order_id,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price_requested,
                trade.exit_fill_price,
                trade.exit_contracts,
                trade.exit_fee,
                trade.gross_pnl,
                trade.total_fees,
                trade.net_pnl,
                trade.outcome,
                trade.status,
                trade.exchange,
                trade.created_at.isoformat() if trade.created_at else now.isoformat(),
                now.isoformat(),
            ),
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def update_trade(self, trade: LiveTrade):
        """Update a live trade record (fill received, exit placed, etc.)."""
        now = datetime.now(timezone.utc)
        sql = """
        UPDATE live_trades SET
            entry_fill_price = ?,
            entry_contracts = ?,
            entry_fee = ?,
            exit_order_id = ?,
            exit_time = ?,
            exit_price_requested = ?,
            exit_fill_price = ?,
            exit_contracts = ?,
            exit_fee = ?,
            gross_pnl = ?,
            total_fees = ?,
            net_pnl = ?,
            outcome = ?,
            status = ?,
            updated_at = ?
        WHERE trade_id = ?
        """

        await self.conn.execute(
            sql,
            (
                trade.entry_fill_price,
                trade.entry_contracts,
                trade.entry_fee,
                trade.exit_order_id,
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price_requested,
                trade.exit_fill_price,
                trade.exit_contracts,
                trade.exit_fee,
                trade.gross_pnl,
                trade.total_fees,
                trade.net_pnl,
                trade.outcome,
                trade.status,
                now.isoformat(),
                trade.trade_id,
            ),
        )
        await self.conn.commit()

    async def get_open_trade(self, asset: str) -> Optional[LiveTrade]:
        """Get the open live trade for an asset (if any)."""
        sql = """
        SELECT * FROM live_trades
        WHERE asset = ? AND status IN ('entry_placed', 'open')
        ORDER BY created_at DESC
        LIMIT 1
        """
        cursor = await self.conn.execute(sql, (asset,))
        row = await cursor.fetchone()
        return self._row_to_trade(row) if row else None

    async def get_trade_by_order_id(self, order_id: str) -> Optional[LiveTrade]:
        """Look up a trade by its entry or exit order ID."""
        sql = """
        SELECT * FROM live_trades
        WHERE entry_order_id = ? OR exit_order_id = ?
        LIMIT 1
        """
        cursor = await self.conn.execute(sql, (order_id, order_id))
        row = await cursor.fetchone()
        return self._row_to_trade(row) if row else None

    async def get_trade_by_sim_id(self, sim_trade_id: str) -> Optional[LiveTrade]:
        """Look up a live trade by its linked sim_trade_id."""
        sql = "SELECT * FROM live_trades WHERE sim_trade_id = ? LIMIT 1"
        cursor = await self.conn.execute(sql, (sim_trade_id,))
        row = await cursor.fetchone()
        return self._row_to_trade(row) if row else None

    async def get_recent_trades(self, limit: int = 20) -> List[LiveTrade]:
        """Get most recent live trades."""
        sql = """
        SELECT * FROM live_trades
        ORDER BY created_at DESC
        LIMIT ?
        """
        cursor = await self.conn.execute(sql, (limit,))
        rows = await cursor.fetchall()
        return [self._row_to_trade(row) for row in rows]

    async def get_all_closed_trades(self) -> List[LiveTrade]:
        """Get all closed/settled live trades."""
        sql = """
        SELECT * FROM live_trades
        WHERE status IN ('closed', 'settled')
        ORDER BY created_at ASC
        """
        cursor = await self.conn.execute(sql)
        rows = await cursor.fetchall()
        return [self._row_to_trade(row) for row in rows]

    async def get_live_pnl_summary(self) -> dict:
        """Get aggregate P&L from live trades."""
        sql = """
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(COALESCE(net_pnl, 0)) as total_pnl,
            SUM(COALESCE(total_fees, 0)) as total_fees,
            SUM(COALESCE(gross_pnl, 0)) as total_gross
        FROM live_trades
        WHERE status IN ('closed', 'settled')
        """
        cursor = await self.conn.execute(sql)
        row = await cursor.fetchone()

        if row:
            total = row["total_trades"] or 0
            wins = row["wins"] or 0
            return {
                "total_trades": total,
                "wins": wins,
                "losses": row["losses"] or 0,
                "total_pnl": row["total_pnl"] or 0.0,
                "total_fees": row["total_fees"] or 0.0,
                "total_gross": row["total_gross"] or 0.0,
                "win_rate": wins / total if total > 0 else 0.0,
            }
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "total_fees": 0.0, "total_gross": 0.0,
            "win_rate": 0.0,
        }

    def _row_to_trade(self, row: aiosqlite.Row) -> LiveTrade:
        """Convert a database row to a LiveTrade object."""
        return LiveTrade(
            trade_id=row["trade_id"],
            sim_trade_id=row["sim_trade_id"] or "",
            window_id=row["window_id"],
            asset=row["asset"],
            direction=row["direction"],
            entry_mode=row["entry_mode"],
            bet_size=row["bet_size"],
            entry_order_id=row["entry_order_id"],
            entry_time=(
                datetime.fromisoformat(row["entry_time"])
                if row["entry_time"] else None
            ),
            entry_price_requested=row["entry_price_requested"] or 0.0,
            entry_fill_price=row["entry_fill_price"],
            entry_contracts=row["entry_contracts"],
            entry_fee=row["entry_fee"],
            exit_order_id=row["exit_order_id"],
            exit_time=(
                datetime.fromisoformat(row["exit_time"])
                if row["exit_time"] else None
            ),
            exit_price_requested=row["exit_price_requested"],
            exit_fill_price=row["exit_fill_price"],
            exit_contracts=row["exit_contracts"],
            exit_fee=row["exit_fee"],
            gross_pnl=row["gross_pnl"],
            total_fees=row["total_fees"],
            net_pnl=row["net_pnl"],
            outcome=row["outcome"] or "pending",
            status=row["status"],
            exchange=row["exchange"] or "polymarket",
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row["created_at"] else None
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"])
                if row["updated_at"] else None
            ),
        )
