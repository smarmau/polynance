"""Polymarket adapter implementing the ExchangeClient interface.

Thin wrapper around the existing PolymarketClient, delegating all calls.
"""

import logging
from typing import Optional, List

from .exchange import ExchangeClient, MarketInfo, MarketPrice
from .polymarket import PolymarketClient

logger = logging.getLogger(__name__)


class PolymarketAdapter(ExchangeClient):
    """Wraps the existing PolymarketClient to implement ExchangeClient."""

    def __init__(self):
        self._client = PolymarketClient()

    async def connect(self):
        """Open the aiohttp session."""
        await self._client.__aenter__()
        logger.info("Polymarket exchange client connected")

    async def close(self):
        """Close the aiohttp session."""
        await self._client.__aexit__(None, None, None)

    async def find_active_15min_markets(self, assets: List[str]) -> List[MarketInfo]:
        """Find active 15-min markets via the Gamma API."""
        poly_markets = await self._client.find_active_15min_markets(assets)
        # PolymarketClient already returns our MarketInfo type
        return poly_markets

    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get market price via the CLOB order book."""
        return await self._client.get_market_price(market)

    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get cached market info."""
        return self._client.get_cached_market(asset)
