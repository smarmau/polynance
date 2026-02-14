"""Exchange abstraction layer for prediction market clients.

Defines the ExchangeClient ABC and shared data structures (MarketInfo, MarketPrice,
OrderResult) used by both PolymarketAdapter and KalshiAdapter. The pm_ prefix in
downstream code stands for "prediction market" (exchange-agnostic).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Literal

logger = logging.getLogger(__name__)


@dataclass
class MarketInfo:
    """Information about a 15-min prediction market."""

    condition_id: str
    question: str
    asset: str  # BTC, ETH, etc.
    yes_token_id: str
    no_token_id: str
    end_date: Optional[datetime] = None
    active: bool = True


@dataclass
class MarketPrice:
    """Current market price data."""

    timestamp: datetime
    yes_price: float
    no_price: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    spread: float  # yes_ask - yes_bid
    midpoint: float  # (yes_bid + yes_ask) / 2


@dataclass
class OrderResult:
    """Result of placing or cancelling an order on an exchange.

    Exchange-agnostic representation returned by place_order / cancel_order.
    """

    order_id: str
    market_id: str
    outcome_id: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    amount: float  # number of contracts
    price: Optional[float] = None  # limit price (0-1)
    status: str = "open"  # open, filled, cancelled, etc.
    filled: float = 0.0
    remaining: float = 0.0
    fee: Optional[float] = None
    timestamp: Optional[int] = None
    raw: Optional[dict] = field(default=None, repr=False)  # raw exchange response


@dataclass
class PositionInfo:
    """Current position in a market outcome."""

    market_id: str
    outcome_id: str
    outcome_label: str  # "Yes", "No", "Up", "Down"
    size: float  # number of contracts held
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: Optional[float] = None


@dataclass
class BalanceInfo:
    """Account balance on an exchange."""

    currency: str  # e.g. "USDC"
    total: float
    available: float
    locked: float  # in open orders


class ExchangeClient(ABC):
    """Abstract base class for prediction market exchange clients.

    All exchange adapters must implement this interface. The sampler and
    main application depend only on this ABC, not on any specific exchange.

    Trading methods (place_order, cancel_order, etc.) have default implementations
    that raise NotImplementedError — only adapters that support live trading need
    to override them.
    """

    # --- Market data (required) ---

    @abstractmethod
    async def connect(self):
        """Initialize the client (open sessions, authenticate, etc.)."""
        ...

    @abstractmethod
    async def close(self):
        """Clean up resources (close sessions, etc.)."""
        ...

    @abstractmethod
    async def find_active_15min_markets(self, assets: List[str]) -> List[MarketInfo]:
        """Find active 15-minute crypto prediction markets.

        Args:
            assets: List of assets to find markets for (e.g., ["BTC", "ETH"]).

        Returns:
            List of MarketInfo objects for active markets.
        """
        ...

    @abstractmethod
    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get comprehensive price data for a market.

        Args:
            market: MarketInfo object identifying the market.

        Returns:
            MarketPrice with bid/ask/midpoint/spread, or None on error.
        """
        ...

    @abstractmethod
    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get a cached market for an asset (no API call)."""
        ...

    # --- Trading (optional — default raises NotImplementedError) ---

    @property
    def supports_trading(self) -> bool:
        """Whether this adapter supports live order placement."""
        return False

    async def place_order(
        self,
        market: MarketInfo,
        side: Literal["buy", "sell"],
        outcome: Literal["yes", "no"],
        amount: float,
        price: Optional[float] = None,
        order_type: Literal["market", "limit"] = "limit",
    ) -> OrderResult:
        """Place an order on the exchange.

        Args:
            market: MarketInfo identifying the market.
            side: "buy" or "sell".
            outcome: "yes" or "no" (which outcome token to trade).
            amount: Number of contracts.
            price: Limit price (0-1). Required for limit orders.
            order_type: "market" or "limit" (default: "limit").

        Returns:
            OrderResult with order details.

        Raises:
            NotImplementedError: If adapter doesn't support trading.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order.

        Args:
            order_id: The order ID to cancel.

        Returns:
            OrderResult with cancellation details.

        Raises:
            NotImplementedError: If adapter doesn't support trading.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")

    async def fetch_open_orders(self, market: Optional[MarketInfo] = None) -> List[OrderResult]:
        """Get all open orders, optionally filtered by market.

        Args:
            market: Optional MarketInfo to filter by.

        Returns:
            List of OrderResult for open orders.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")

    async def fetch_positions(self) -> List[PositionInfo]:
        """Get current positions across all markets.

        Returns:
            List of PositionInfo objects.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")

    async def fetch_balance(self) -> List[BalanceInfo]:
        """Get account balance.

        Returns:
            List of BalanceInfo (one per currency).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")

    async def fetch_order(self, order_id: str) -> OrderResult:
        """Get details of a specific order by ID.

        Retrieves the full order state including actual fill price, filled amount,
        and fee. Essential for getting real fill data after order placement.

        Args:
            order_id: The exchange order ID.

        Returns:
            OrderResult with current order state (filled, price, fee, etc.).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")

    async def fetch_trades(
        self,
        outcome_id: str,
        limit: Optional[int] = None,
        since: Optional[int] = None,
    ) -> List[dict]:
        """Get trade history for a specific outcome token.

        Returns fills/trades for an outcome, useful for historical P&L
        reconstruction.

        Args:
            outcome_id: The outcome token ID to query trades for.
            limit: Maximum number of trades to return.
            since: Return trades since this Unix timestamp (milliseconds).

        Returns:
            List of trade dicts with id, timestamp, price, amount, side.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support live trading")


def create_exchange(name: str, live_trading: bool = False, **kwargs) -> ExchangeClient:
    """Factory function to create an exchange client by name.

    Args:
        name: Exchange name ("polymarket" or "kalshi").
        live_trading: If True, initialize with trading credentials.
        **kwargs: Additional arguments passed to the adapter (e.g., private_key).

    Returns:
        An ExchangeClient instance.
    """
    if name == "polymarket":
        from .polymarket_adapter import PolymarketAdapter
        return PolymarketAdapter(live_trading=live_trading, **kwargs)
    elif name == "kalshi":
        from .kalshi_adapter import KalshiAdapter
        return KalshiAdapter()
    else:
        raise ValueError(f"Unknown exchange: {name}. Supported: polymarket, kalshi")
