"""Kalshi adapter implementing the ExchangeClient interface.

Uses the public Kalshi REST API (no authentication required) for market
discovery and price fetching. The 15-minute crypto markets use series
tickers like KXBTC15M, KXETH15M, KXSOL15M, KXXRP15M.

API docs: https://docs.kalshi.com/getting_started/quick_start_market_data
"""

import logging
import aiohttp
from datetime import datetime, timezone
from typing import Optional, List, Dict

from .exchange import ExchangeClient, MarketInfo, MarketPrice

logger = logging.getLogger(__name__)

# Kalshi public API base URL (no auth required for market data)
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Series tickers for 15-min crypto binary markets
KALSHI_15MIN_SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}


class KalshiAdapter(ExchangeClient):
    """Uses the public Kalshi REST API to implement ExchangeClient.

    Discovers 15-minute crypto binary markets on Kalshi and maps
    them to the shared MarketInfo/MarketPrice types. No API key needed.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._market_cache: Dict[str, MarketInfo] = {}

    async def connect(self):
        """Initialize the HTTP session for Kalshi public API."""
        self._session = aiohttp.ClientSession(
            headers={"Accept": "application/json"},
        )
        logger.info("Kalshi exchange client connected (public API, no auth)")

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
        self._session = None

    async def _api_get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a GET request to the Kalshi public API."""
        if not self._session:
            return None
        url = f"{KALSHI_API_BASE}{path}"
        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"Kalshi API {path} returned {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Kalshi API error ({path}): {e}")
            return None

    async def find_active_15min_markets(self, assets: List[str]) -> List[MarketInfo]:
        """Find active 15-minute crypto markets on Kalshi.

        Queries /markets?series_ticker=KXBTC15M&status=open for each asset
        and picks the soonest-expiring open market.
        """
        if not self._session:
            logger.warning("Kalshi client not connected. Call connect() first.")
            return []

        markets_found = []

        for asset in assets:
            try:
                market_info = await self._find_asset_market(asset)
                if market_info:
                    markets_found.append(market_info)
                    self._market_cache[asset] = market_info
                    logger.info(
                        f"Found Kalshi {asset} 15-min market: "
                        f"{market_info.condition_id} - {market_info.question[:60]}"
                    )
                else:
                    logger.warning(f"No active 15-min market found for {asset} on Kalshi")
            except Exception as e:
                logger.error(f"Error finding Kalshi market for {asset}: {e}")

        return markets_found

    async def _find_asset_market(self, asset: str) -> Optional[MarketInfo]:
        """Find the current open 15-min market for a specific asset."""
        series = KALSHI_15MIN_SERIES.get(asset)
        if not series:
            return None

        data = await self._api_get("/markets", params={
            "series_ticker": series,
            "status": "open",
            "limit": "10",
        })

        if not data or "markets" not in data:
            return None

        markets = data["markets"]
        if not markets:
            return None

        # Pick the first open market (soonest expiring)
        # Kalshi returns them in chronological order
        market = markets[0]

        ticker = market.get("ticker", "")
        title = market.get("title", "") or market.get("subtitle", "") or ticker
        event_ticker = market.get("event_ticker", "")
        close_time = market.get("close_time") or market.get("latest_expiration_time")

        return MarketInfo(
            condition_id=ticker,
            question=title,
            asset=asset,
            yes_token_id=ticker,   # Kalshi uses the market ticker for orderbook
            no_token_id=ticker,    # Same ticker, NO is complement
            end_date=close_time,
            active=True,
        )

    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get price data for a market via the public orderbook endpoint.

        GET /markets/{ticker}/orderbook returns YES/NO bids and asks.
        """
        if not self._session:
            return None

        try:
            # Try orderbook first for best bid/ask
            data = await self._api_get(f"/markets/{market.condition_id}/orderbook")

            if data:
                return self._parse_orderbook(data)

            # Fallback: get market snapshot for last_price
            data = await self._api_get(f"/markets/{market.condition_id}")
            if data and "market" in data:
                return self._parse_market_snapshot(data["market"])

            return None

        except Exception as e:
            logger.error(f"Error getting Kalshi market price for {market.asset}: {e}")
            return None

    def _parse_orderbook(self, data: dict) -> Optional[MarketPrice]:
        """Parse Kalshi orderbook response into MarketPrice.

        Kalshi orderbook format: {"orderbook": {"yes": [[price_cents, qty], ...], "no": [...]}}
        Prices are in cents (1-99).
        """
        book = data.get("orderbook", {})
        yes_levels = book.get("yes", [])
        no_levels = book.get("no", [])

        # Levels are ascending by price. Best bid = last (highest) level.
        # Best yes bid = highest price someone will pay for YES
        # Best yes ask = 100 - highest NO bid (complement)
        yes_bid_cents = yes_levels[-1][0] if yes_levels else 0
        no_bid_cents = no_levels[-1][0] if no_levels else 0

        # Convert cents to 0-1 range
        yes_bid = yes_bid_cents / 100.0
        yes_ask = (100 - no_bid_cents) / 100.0 if no_bid_cents else 1.0

        # Ensure ask > bid
        if yes_ask <= yes_bid:
            yes_ask = min(yes_bid + 0.01, 1.0)

        midpoint = (yes_bid + yes_ask) / 2
        spread = yes_ask - yes_bid

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

    def _parse_market_snapshot(self, market: dict) -> Optional[MarketPrice]:
        """Parse a market snapshot (fallback when orderbook is empty)."""
        # Kalshi returns prices in dollar format (0.01 - 0.99)
        yes_bid = market.get("yes_bid_dollars") or market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask", 0)
        last = market.get("last_price_dollars") or market.get("last_price", 0)

        # Use last price if bid/ask unavailable
        if not yes_bid and not yes_ask and last:
            yes_bid = max(last - 0.01, 0.01)
            yes_ask = min(last + 0.01, 0.99)

        if not yes_bid and not yes_ask:
            return None

        midpoint = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else (last or 0.5)
        spread = (yes_ask - yes_bid) if yes_bid and yes_ask else 0.02

        return MarketPrice(
            timestamp=datetime.now(timezone.utc),
            yes_price=midpoint,
            no_price=1.0 - midpoint,
            yes_bid=yes_bid or midpoint - 0.01,
            yes_ask=yes_ask or midpoint + 0.01,
            no_bid=1.0 - (yes_ask or midpoint + 0.01),
            no_ask=1.0 - (yes_bid or midpoint - 0.01),
            spread=spread,
            midpoint=midpoint,
        )

    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get cached market info."""
        return self._market_cache.get(asset)
