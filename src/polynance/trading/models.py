"""Data models for simulated and live trading."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal
import uuid


@dataclass
class SimulatedTrade:
    """Single simulated trade record."""

    # Trade identification
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_id: str = ""
    asset: str = ""

    # Trade details
    direction: Literal["bull", "bear"] = "bull"
    entry_time: Optional[datetime] = None
    entry_price: float = 0.0
    bet_size: float = 0.0

    # Exit details (null until resolved)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Literal["win", "loss", "pending"] = "pending"

    # P&L breakdown
    gross_pnl: Optional[float] = None
    fee_paid: Optional[float] = None
    spread_cost: Optional[float] = None
    net_pnl: Optional[float] = None

    # State tracking
    bankroll_after: Optional[float] = None
    drawdown: Optional[float] = None
    drawdown_pct: Optional[float] = None

    # Strategy metadata
    entry_mode: Optional[str] = None  # e.g. 'contrarian', 'contrarian_consensus', 'accel_dbl', 'combo_dbl'
    prev_pm: Optional[float] = None  # prev window pm_yes@t12.5
    prev2_pm: Optional[float] = None  # prev2 window pm_yes@t12.5
    spot_velocity: Optional[float] = None  # spot_price_change_from_open at entry
    pm_momentum: Optional[float] = None  # |pm_t_entry - pm_t0| at entry

    # Live exchange data (populated only when live_trading=True)
    live_order_id: Optional[str] = None          # Entry order ID from exchange
    live_entry_fill_price: Optional[float] = None  # Actual fill price from exchange
    live_entry_contracts: Optional[float] = None   # Actual contracts filled
    live_entry_fee: Optional[float] = None         # Fee charged on entry
    live_exit_order_id: Optional[str] = None       # Exit order ID from exchange
    live_exit_fill_price: Optional[float] = None   # Actual exit fill price
    live_exit_contracts: Optional[float] = None    # Actual contracts sold
    live_exit_fee: Optional[float] = None          # Fee charged on exit
    live_status: Optional[str] = None              # open, closed, settled, failed
    exchange: Optional[str] = None                 # e.g. "polymarket"

    # Metadata
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None

    @property
    def has_live_data(self) -> bool:
        """Whether this trade has actual exchange fill data."""
        return self.live_entry_fill_price is not None

    @property
    def is_pending(self) -> bool:
        """Check if trade is still pending resolution."""
        return self.outcome == "pending"

    @property
    def is_win(self) -> bool:
        """Check if trade was a win."""
        return self.outcome == "win"

    @property
    def is_loss(self) -> bool:
        """Check if trade was a loss."""
        return self.outcome == "loss"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "window_id": self.window_id,
            "asset": self.asset,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "bet_size": self.bet_size,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "outcome": self.outcome,
            "gross_pnl": self.gross_pnl,
            "fee_paid": self.fee_paid,
            "spread_cost": self.spread_cost,
            "net_pnl": self.net_pnl,
            "bankroll_after": self.bankroll_after,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
            "entry_mode": self.entry_mode,
            "prev_pm": self.prev_pm,
            "prev2_pm": self.prev2_pm,
            "spot_velocity": self.spot_velocity,
            "pm_momentum": self.pm_momentum,
            "live_order_id": self.live_order_id,
            "live_entry_fill_price": self.live_entry_fill_price,
            "live_entry_contracts": self.live_entry_contracts,
            "live_entry_fee": self.live_entry_fee,
            "live_exit_order_id": self.live_exit_order_id,
            "live_exit_fill_price": self.live_exit_fill_price,
            "live_exit_contracts": self.live_exit_contracts,
            "live_exit_fee": self.live_exit_fee,
            "live_status": self.live_status,
            "exchange": self.exchange,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class TradingState:
    """Global trading state."""

    # Portfolio
    current_bankroll: float
    current_bet_size: float
    initial_bankroll: float
    base_bet_size: float

    # Performance metrics
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl: float = 0.0

    # Drawdown tracking
    peak_bankroll: Optional[float] = None
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Streak tracking
    current_win_streak: int = 0
    current_loss_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # System state
    last_window_id: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Pause-after-loss: skip N windows after any loss to avoid clustering
    pause_windows_remaining: int = 0

    def __post_init__(self):
        """Initialize peak bankroll if not set."""
        if self.peak_bankroll is None:
            self.peak_bankroll = self.current_bankroll

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades

    @property
    def return_pct(self) -> float:
        """Calculate total return percentage."""
        if self.initial_bankroll == 0:
            return 0.0
        return ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100

    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_bankroll is None or self.peak_bankroll == 0:
            return 0.0
        return self.current_bankroll - self.peak_bankroll

    @property
    def current_drawdown_pct(self) -> float:
        """Calculate current drawdown percentage from peak."""
        if self.peak_bankroll is None or self.peak_bankroll == 0:
            return 0.0
        return ((self.current_bankroll - self.peak_bankroll) / self.peak_bankroll) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_bankroll": self.current_bankroll,
            "current_bet_size": self.current_bet_size,
            "initial_bankroll": self.initial_bankroll,
            "base_bet_size": self.base_bet_size,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_pnl": self.total_pnl,
            "peak_bankroll": self.peak_bankroll,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_win_streak": self.current_win_streak,
            "current_loss_streak": self.current_loss_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "last_window_id": self.last_window_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "win_rate": self.win_rate,
            "return_pct": self.return_pct,
            "current_drawdown": self.current_drawdown,
            "current_drawdown_pct": self.current_drawdown_pct,
            "pause_windows_remaining": self.pause_windows_remaining,
        }


@dataclass
class PerAssetStats:
    """Per-asset trading statistics."""

    asset: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    last_signal: Optional[str] = None  # 'bull', 'bear', or None

    @property
    def win_rate(self) -> float:
        """Calculate win rate for this asset."""
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades


@dataclass
class LiveTrade:
    """DEPRECATED: Live trade data is now stored directly on SimulatedTrade
    via the live_* fields. This class is retained for backward compatibility
    with existing live_trading.db files only.

    Use SimulatedTrade.live_order_id, .live_entry_fill_price, etc. instead.
    """

    # Trade identification
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sim_trade_id: str = ""  # Links to the parallel SimulatedTrade
    window_id: str = ""
    asset: str = ""

    # Trade direction and sizing
    direction: Literal["bull", "bear"] = "bull"
    entry_mode: Optional[str] = None
    bet_size: float = 0.0  # Dollar amount

    # Entry order (from exchange)
    entry_order_id: Optional[str] = None
    entry_time: Optional[datetime] = None
    entry_price_requested: float = 0.0  # Limit price we asked for
    entry_fill_price: Optional[float] = None  # Actual fill price from exchange
    entry_contracts: Optional[float] = None  # Actual contracts filled
    entry_fee: Optional[float] = None  # Actual fee charged by exchange

    # Exit order (from exchange)
    exit_order_id: Optional[str] = None
    exit_time: Optional[datetime] = None
    exit_price_requested: Optional[float] = None  # Limit price we asked for
    exit_fill_price: Optional[float] = None  # Actual fill price from exchange
    exit_contracts: Optional[float] = None  # Actual contracts sold
    exit_fee: Optional[float] = None  # Actual fee charged by exchange

    # P&L (calculated from actual fills)
    gross_pnl: Optional[float] = None
    total_fees: Optional[float] = None  # entry_fee + exit_fee
    net_pnl: Optional[float] = None
    outcome: Literal["win", "loss", "pending"] = "pending"

    # Status tracking
    status: str = "entry_placed"  # entry_placed, open, exit_placed, closed, settled, failed

    # Metadata
    exchange: str = "polymarket"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        return self.status in ("entry_placed", "open")

    @property
    def is_closed(self) -> bool:
        return self.status in ("closed", "settled")

    def calculate_pnl(self):
        """Calculate P&L from actual fill data.

        For early exit (contrarian-family): P&L = contracts * (exit - entry) - fees
        For binary resolution: P&L = contracts * (payout - entry) - fees
        """
        if self.entry_fill_price is None or self.entry_contracts is None:
            return  # Can't calculate without entry fill

        entry_fee = self.entry_fee or 0.0
        exit_fee = self.exit_fee or 0.0
        self.total_fees = entry_fee + exit_fee

        if self.exit_fill_price is not None:
            # Early exit: sold before resolution
            self.gross_pnl = self.entry_contracts * (self.exit_fill_price - self.entry_fill_price)
        elif self.status == "settled":
            # Binary resolution: payout is 1.0 (win) or 0.0 (loss)
            payout = 1.0 if self.outcome == "win" else 0.0
            self.gross_pnl = self.entry_contracts * (payout - self.entry_fill_price)
        else:
            return  # Can't calculate yet

        self.net_pnl = self.gross_pnl - self.total_fees
        self.outcome = "win" if self.net_pnl > 0 else "loss"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "sim_trade_id": self.sim_trade_id,
            "window_id": self.window_id,
            "asset": self.asset,
            "direction": self.direction,
            "entry_mode": self.entry_mode,
            "bet_size": self.bet_size,
            "entry_order_id": self.entry_order_id,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price_requested": self.entry_price_requested,
            "entry_fill_price": self.entry_fill_price,
            "entry_contracts": self.entry_contracts,
            "entry_fee": self.entry_fee,
            "exit_order_id": self.exit_order_id,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price_requested": self.exit_price_requested,
            "exit_fill_price": self.exit_fill_price,
            "exit_contracts": self.exit_contracts,
            "exit_fee": self.exit_fee,
            "gross_pnl": self.gross_pnl,
            "total_fees": self.total_fees,
            "net_pnl": self.net_pnl,
            "outcome": self.outcome,
            "status": self.status,
            "exchange": self.exchange,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
