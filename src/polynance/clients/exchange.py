"""Exchange abstraction layer for prediction market clients.

Defines the ExchangeClient ABC and shared data structures (MarketInfo, MarketPrice)
used by both PolymarketAdapter and KalshiAdapter. The pm_ prefix in downstream code
stands for "prediction market" (exchange-agnostic).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

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


class ExchangeClient(ABC):
    """Abstract base class for prediction market exchange clients.

    All exchange adapters must implement this interface. The sampler and
    main application depend only on this ABC, not on any specific exchange.
    """

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


def create_exchange(name: str) -> ExchangeClient:
    """Factory function to create an exchange client by name.

    Args:
        name: Exchange name ("polymarket" or "kalshi").

    Returns:
        An ExchangeClient instance.
    """
    if name == "polymarket":
        from .polymarket_adapter import PolymarketAdapter
        return PolymarketAdapter()
    elif name == "kalshi":
        from .kalshi_adapter import KalshiAdapter
        return KalshiAdapter()
    else:
        raise ValueError(f"Unknown exchange: {name}. Supported: polymarket, kalshi")
