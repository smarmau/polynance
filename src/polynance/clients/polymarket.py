"""Polymarket API client for 15-minute crypto prediction markets."""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict

import aiohttp

logger = logging.getLogger(__name__)

# Asset slug prefixes for 15-min markets (used to construct epoch-based URLs)
CRYPTO_15MIN_SLUGS = {
    "BTC": "btc-updown-15m",
    "ETH": "eth-updown-15m",
    "SOL": "sol-updown-15m",
    "XRP": "xrp-updown-15m",
}


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
class OrderBookLevel:
    """Single level in the order book."""

    price: float
    size: float


@dataclass
class OrderBook:
    """Order book for a market."""

    timestamp: datetime
    bids: List[OrderBookLevel]  # Buy orders (highest first)
    asks: List[OrderBookLevel]  # Sell orders (lowest first)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2


class PolymarketClient:
    """Client for interacting with Polymarket CLOB API."""

    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

    # Default headers to avoid brotli compression issues
    DEFAULT_HEADERS = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",  # Exclude 'br' (brotli) since aiohttp may not have it
        "User-Agent": "polynance/1.0",
    }

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None
        self._market_cache: Dict[str, MarketInfo] = {}
        # Connection settings
        self._timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self._max_retries = 3

    async def __aenter__(self):
        if self._session is None:
            # Create connector with connection pooling limits
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection limit
                limit_per_host=30,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
            )
            self._session = aiohttp.ClientSession(
                headers=self.DEFAULT_HEADERS,
                timeout=self._timeout,
                connector=connector,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session and self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("Client not initialized. Use async with or call __aenter__")
        return self._session

    async def find_active_15min_markets(self, assets: Optional[List[str]] = None) -> List[MarketInfo]:
        """Find active 15-minute crypto prediction markets.

        Uses the epoch-based URL pattern: /events/slug/{asset}-updown-15m-{epoch}
        where epoch is the unix timestamp of the current 15-min window start.

        Args:
            assets: List of assets to find markets for (e.g., ["BTC", "ETH"]).
                   If None, finds all available 15-min crypto markets.

        Returns:
            List of MarketInfo objects for active markets.
        """
        if assets is None:
            assets = list(CRYPTO_15MIN_SLUGS.keys())

        markets = []

        # Calculate current 15-min window epoch
        now = datetime.now(timezone.utc)
        minute = (now.minute // 15) * 15
        window_start = now.replace(minute=minute, second=0, microsecond=0)
        epoch = int(window_start.timestamp())

        # Query each asset's market using the epoch-based URL
        for asset in assets:
            slug_prefix = CRYPTO_15MIN_SLUGS.get(asset)
            if not slug_prefix:
                logger.warning(f"No known slug prefix for asset: {asset}")
                continue

            market = await self._get_market_by_epoch(slug_prefix, epoch, asset)
            if market:
                markets.append(market)
                self._market_cache[asset] = market
                logger.info(f"Found {asset} 15-min market: {market.question[:50]}...")

        return markets

    async def _get_market_by_epoch(self, slug_prefix: str, epoch: int, asset: str) -> Optional[MarketInfo]:
        """Get market info by querying the Gamma API with epoch-based slug.

        Args:
            slug_prefix: The slug prefix (e.g., "btc-updown-15m")
            epoch: The unix timestamp of the 15-min window start
            asset: The asset symbol (BTC, ETH, etc.)

        Returns:
            MarketInfo object or None if not found
        """
        try:
            slug = f"{slug_prefix}-{epoch}"
            url = f"{self.GAMMA_BASE_URL}/events/slug/{slug}"

            logger.debug(f"Fetching market from: {url}")
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Get the first market from the event
                    markets = data.get("markets", [])
                    if not markets:
                        logger.warning(f"No markets found in event {slug}")
                        return None

                    market_data = markets[0]

                    # Extract token IDs from clobTokenIds (JSON string array)
                    clob_token_ids = market_data.get("clobTokenIds", "[]")
                    if isinstance(clob_token_ids, str):
                        import json
                        clob_token_ids = json.loads(clob_token_ids)

                    if len(clob_token_ids) < 2:
                        logger.warning(f"Not enough token IDs for {asset}")
                        return None

                    # First token is "Up", second is "Down"
                    yes_token = clob_token_ids[0]
                    no_token = clob_token_ids[1]

                    # Extract current prices
                    outcome_prices = market_data.get("outcomePrices", "[]")
                    if isinstance(outcome_prices, str):
                        import json
                        outcome_prices = json.loads(outcome_prices)

                    return MarketInfo(
                        condition_id=market_data.get("conditionId", ""),
                        question=market_data.get("question", f"{asset} 15-min market"),
                        asset=asset,
                        yes_token_id=yes_token,
                        no_token_id=no_token,
                        active=market_data.get("active", True),
                    )
                elif resp.status == 404:
                    logger.warning(f"Market not found for {slug} - may not exist yet")
                else:
                    text = await resp.text()
                    logger.warning(f"Gamma API returned {resp.status} for {slug}: {text[:100]}")

        except Exception as e:
            logger.error(f"Error fetching market for {asset}: {e}")

        return None

    async def _get_market_by_condition_id(self, condition_id: str, asset: str) -> Optional[MarketInfo]:
        """Get market info by querying the CLOB API with condition ID."""
        try:
            url = f"{self.CLOB_BASE_URL}/markets/{condition_id}"

            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Extract token IDs from the tokens array
                    tokens = data.get("tokens", [])
                    yes_token = None
                    no_token = None

                    for token in tokens:
                        outcome = token.get("outcome", "").lower()
                        token_id = token.get("token_id", "")

                        if outcome in ["up", "yes"]:
                            yes_token = token_id
                        elif outcome in ["down", "no"]:
                            no_token = token_id

                    if yes_token and no_token:
                        return MarketInfo(
                            condition_id=condition_id,
                            question=data.get("question", f"{asset} 15-min market"),
                            asset=asset,
                            yes_token_id=yes_token,
                            no_token_id=no_token,
                            active=data.get("active", True),
                        )
                    else:
                        logger.warning(f"Could not extract token IDs for {asset}")
                else:
                    logger.warning(f"CLOB API returned {resp.status} for {condition_id}")

        except Exception as e:
            logger.error(f"Error fetching market for {asset}: {e}")

        return None

    async def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        """Get the current price for a token.

        Args:
            token_id: The token ID to get price for
            side: 'buy' or 'sell'

        Returns:
            Price as float (0-1) or None if error
        """
        try:
            url = f"{self.CLOB_BASE_URL}/price"
            params = {"token_id": token_id, "side": side}

            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("price", 0))
        except Exception as e:
            logger.error(f"Error getting price for {token_id}: {e}")

        return None

    async def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Get the order book for a token with retry logic.

        Args:
            token_id: The token ID to get order book for

        Returns:
            OrderBook object or None if error
        """
        url = f"{self.CLOB_BASE_URL}/book"
        params = {"token_id": token_id}

        for attempt in range(self._max_retries):
            try:
                logger.debug(f"Fetching order book for token: {token_id[:20]}... (attempt {attempt + 1}/{self._max_retries})")
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.debug(f"Order book response: bids={len(data.get('bids', []))}, asks={len(data.get('asks', []))}")

                        bids = [
                            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
                            for b in data.get("bids", [])
                        ]
                        asks = [
                            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
                            for a in data.get("asks", [])
                        ]

                        # Sort: bids descending, asks ascending
                        bids.sort(key=lambda x: x.price, reverse=True)
                        asks.sort(key=lambda x: x.price)

                        return OrderBook(
                            timestamp=datetime.now(timezone.utc),
                            bids=bids,
                            asks=asks,
                        )
                    elif resp.status == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        text = await resp.text()
                        logger.error(f"Order book API returned {resp.status}: {text[:200]}")
                        return None

            except asyncio.TimeoutError:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Timeout fetching order book, retrying ({attempt + 1}/{self._max_retries})")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Timeout fetching order book after {self._max_retries} attempts")
            except Exception as e:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Error getting order book (attempt {attempt + 1}/{self._max_retries}): {e}")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Error getting order book for {token_id} after {self._max_retries} attempts: {e}")

        return None

    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get comprehensive price data for a market.

        Args:
            market: MarketInfo object

        Returns:
            MarketPrice object or None if error
        """
        try:
            # Get order books for both yes and no tokens
            yes_book, no_book = await asyncio.gather(
                self.get_order_book(market.yes_token_id),
                self.get_order_book(market.no_token_id),
            )

            if yes_book is None:
                return None

            # Calculate prices
            yes_bid = yes_book.best_bid
            yes_ask = yes_book.best_ask
            no_bid = no_book.best_bid if no_book else 1 - yes_ask
            no_ask = no_book.best_ask if no_book else 1 - yes_bid

            return MarketPrice(
                timestamp=datetime.now(timezone.utc),
                yes_price=yes_book.midpoint,
                no_price=1 - yes_book.midpoint,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                spread=yes_book.spread,
                midpoint=yes_book.midpoint,
            )

        except Exception as e:
            logger.error(f"Error getting market price for {market.asset}: {e}")
            return None

    async def get_price_history(
        self, token_id: str, fidelity: int = 1
    ) -> List[tuple]:
        """Get historical prices for a token.

        Args:
            token_id: The token ID
            fidelity: Time resolution in minutes

        Returns:
            List of (timestamp, price) tuples
        """
        try:
            url = f"{self.CLOB_BASE_URL}/prices-history"
            params = {"market": token_id, "fidelity": fidelity}

            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    history = []

                    for point in data.get("history", []):
                        ts = datetime.fromtimestamp(point["t"], tz=timezone.utc)
                        price = float(point["p"])
                        history.append((ts, price))

                    return history
        except Exception as e:
            logger.error(f"Error getting price history for {token_id}: {e}")

        return []

    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get a cached market for an asset."""
        return self._market_cache.get(asset)


class PolymarketWebSocket:
    """WebSocket client for real-time Polymarket updates."""

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, token_ids: List[str]):
        self.token_ids = token_ids
        self._ws = None
        self._running = False

    async def connect(self):
        """Connect to the WebSocket."""
        import websockets

        self._ws = await websockets.connect(self.WS_URL)
        self._running = True

        # Subscribe to markets
        subscribe_msg = {"assets_ids": self.token_ids, "type": "market"}
        await self._ws.send(json.dumps(subscribe_msg))

        logger.info(f"Connected to Polymarket WebSocket, subscribed to {len(self.token_ids)} tokens")

    async def disconnect(self):
        """Disconnect from the WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def listen(self):
        """Listen for updates from the WebSocket.

        Yields:
            dict: Price update messages
        """
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        while self._running:
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=30)
                data = json.loads(message)

                if data.get("type") == "price_update":
                    yield data

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await self._ws.ping()
                except Exception:
                    logger.warning("WebSocket ping failed, reconnecting...")
                    await self.connect()

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(1)
                    await self.connect()
