"""Data models for simulated trading."""

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

    # Metadata
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None

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
