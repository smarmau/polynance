"""Kalshi adapter implementing the ExchangeClient interface.

Uses pmxt.Kalshi for market discovery and price fetching. Credentials
are read from environment variables KALSHI_API_KEY and KALSHI_PRIVATE_KEY.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict

from .exchange import ExchangeClient, MarketInfo, MarketPrice

logger = logging.getLogger(__name__)

# Kalshi 15-min crypto market search terms
KALSHI_15MIN_SEARCH = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "XRP": "XRP",
}


class KalshiAdapter(ExchangeClient):
    """Wraps pmxt.Kalshi to implement ExchangeClient.

    Discovers 15-minute crypto binary markets on Kalshi and maps
    them to the shared MarketInfo/MarketPrice types.
    """

    def __init__(self):
        self._exchange = None
        self._market_cache: Dict[str, MarketInfo] = {}

    async def connect(self):
        """Initialize the pmxt Kalshi client."""
        import pmxt

        api_key = os.environ.get("KALSHI_API_KEY")
        private_key = os.environ.get("KALSHI_PRIVATE_KEY")

        if not api_key or not private_key:
            logger.warning(
                "Kalshi credentials not set (KALSHI_API_KEY / KALSHI_PRIVATE_KEY). "
                "Running in dry-run mode — market discovery and pricing will be unavailable."
            )
            self._exchange = None
            return

        self._exchange = pmxt.Kalshi(
            api_key=api_key,
            private_key=private_key,
        )
        logger.info("Kalshi exchange client connected via pmxt")

    async def close(self):
        """Close the pmxt client."""
        if self._exchange:
            try:
                self._exchange.close()
            except Exception as e:
                logger.warning(f"Error closing Kalshi client: {e}")
        self._exchange = None

    async def find_active_15min_markets(self, assets: List[str]) -> List[MarketInfo]:
        """Find active 15-minute crypto markets on Kalshi.

        Uses pmxt.fetch_markets() to discover markets, then filters for
        active 15-minute binary up/down markets matching the requested assets.
        """
        if not self._exchange:
            logger.warning("Kalshi client not connected (no credentials). Skipping market discovery.")
            return []

        markets_found = []

        for asset in assets:
            try:
                market_info = await self._find_asset_market(asset)
                if market_info:
                    markets_found.append(market_info)
                    self._market_cache[asset] = market_info
                    logger.info(f"Found Kalshi {asset} 15-min market: {market_info.question[:50]}...")
                else:
                    logger.warning(f"No active 15-min market found for {asset} on Kalshi")
            except Exception as e:
                logger.error(f"Error finding Kalshi market for {asset}: {e}")

        return markets_found

    async def _find_asset_market(self, asset: str) -> Optional[MarketInfo]:
        """Find the current 15-min market for a specific asset."""
        search_term = KALSHI_15MIN_SEARCH.get(asset)
        if not search_term:
            return None

        # Fetch markets matching the asset
        all_markets = self._exchange.fetch_markets(query=f"{search_term} 15")

        # Filter for active 15-minute binary markets
        for market in all_markets:
            title_lower = market.title.lower() if market.title else ""

            # Look for 15-minute up/down markets
            is_15min = "15" in title_lower and ("min" in title_lower or "minute" in title_lower)
            is_crypto = asset.lower() in title_lower or search_term.lower() in title_lower
            has_outcomes = market.outcomes and len(market.outcomes) >= 2

            if is_15min and is_crypto and has_outcomes:
                # Check if market has up/down or yes/no outcomes
                yes_outcome = market.yes or market.up
                no_outcome = market.no or market.down

                if not yes_outcome or not no_outcome:
                    # Try finding from outcomes list
                    for outcome in market.outcomes:
                        label_lower = outcome.label.lower()
                        if label_lower in ("yes", "up"):
                            yes_outcome = outcome
                        elif label_lower in ("no", "down"):
                            no_outcome = outcome

                if yes_outcome and no_outcome:
                    return MarketInfo(
                        condition_id=market.market_id,
                        question=market.title,
                        asset=asset,
                        yes_token_id=yes_outcome.outcome_id,
                        no_token_id=no_outcome.outcome_id,
                        end_date=market.resolution_date,
                        active=True,
                    )

        return None

    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get price data for a market via pmxt order book.

        Fetches the order book for the YES outcome and derives all prices.
        """
        if not self._exchange:
            return None

        try:
            # Fetch order book for YES/UP outcome
            yes_book = self._exchange.fetch_order_book(market.yes_token_id)

            if not yes_book:
                return None

            # Extract bid/ask from order book
            yes_bid = yes_book.bids[0].price if yes_book.bids else 0.0
            yes_ask = yes_book.asks[0].price if yes_book.asks else 1.0

            # Calculate derived prices
            midpoint = (yes_bid + yes_ask) / 2
            spread = yes_ask - yes_bid

            # NO prices are complement of YES
            no_bid = 1.0 - yes_ask
            no_ask = 1.0 - yes_bid

            return MarketPrice(
                timestamp=datetime.now(timezone.utc),
                yes_price=midpoint,
                no_price=1.0 - midpoint,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                spread=spread,
                midpoint=midpoint,
            )

        except Exception as e:
            logger.error(f"Error getting Kalshi market price for {market.asset}: {e}")
            return None

    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get cached market info."""
        return self._market_cache.get(asset)
