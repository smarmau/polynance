"""Binance API client for crypto spot prices."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict

import aiohttp

logger = logging.getLogger(__name__)

# Mapping from our asset symbols to Binance symbols
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


@dataclass
class SpotPrice:
    """Current spot price for a crypto asset."""

    symbol: str  # Our symbol (BTC, ETH, etc.)
    price: float
    timestamp: datetime


@dataclass
class OHLCV:
    """OHLCV candle data."""

    symbol: str
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime


class BinanceClient:
    """Client for interacting with Binance public API."""

    BASE_URL = "https://api.binance.com/api/v3"

    # Default headers to avoid compression issues
    DEFAULT_HEADERS = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",  # Exclude 'br' (brotli)
        "User-Agent": "polynance/1.0",
    }

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None
        self._price_cache: Dict[str, SpotPrice] = {}
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

    def _get_binance_symbol(self, asset: str) -> str:
        """Convert our asset symbol to Binance symbol."""
        return BINANCE_SYMBOLS.get(asset.upper(), f"{asset.upper()}USDT")

    async def get_price(self, asset: str) -> Optional[SpotPrice]:
        """Get current spot price for an asset with retry logic.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            SpotPrice object or None if error
        """
        symbol = self._get_binance_symbol(asset)
        url = f"{self.BASE_URL}/ticker/price"
        params = {"symbol": symbol}

        for attempt in range(self._max_retries):
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = SpotPrice(
                            symbol=asset,
                            price=float(data["price"]),
                            timestamp=datetime.now(timezone.utc),
                        )
                        self._price_cache[asset] = price
                        return price
                    elif resp.status == 429:  # Rate limit
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Binance rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Binance API error: {resp.status}")
                        return None

            except asyncio.TimeoutError:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Timeout getting price for {asset}, retrying ({attempt + 1}/{self._max_retries})")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Timeout getting price for {asset} after {self._max_retries} attempts")
            except Exception as e:
                if attempt < self._max_retries - 1:
                    logger.warning(f"Error getting price for {asset} (attempt {attempt + 1}/{self._max_retries}): {e}")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Error getting price for {asset} after {self._max_retries} attempts: {e}")

        return None

    async def get_prices(self, assets: List[str]) -> Dict[str, SpotPrice]:
        """Get current spot prices for multiple assets.

        Args:
            assets: List of asset symbols

        Returns:
            Dict mapping asset symbol to SpotPrice
        """
        # Fetch all prices concurrently
        tasks = [self.get_price(asset) for asset in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for asset, result in zip(assets, results):
            if isinstance(result, SpotPrice):
                prices[asset] = result
            elif isinstance(result, Exception):
                logger.error(f"Error fetching {asset}: {result}")

        return prices

    async def get_klines(
        self,
        asset: str,
        interval: str = "1m",
        limit: int = 15,
    ) -> List[OHLCV]:
        """Get kline/candlestick data.

        Args:
            asset: Asset symbol
            interval: Kline interval (1m, 5m, 15m, etc.)
            limit: Number of candles to fetch

        Returns:
            List of OHLCV objects
        """
        symbol = self._get_binance_symbol(asset)

        try:
            url = f"{self.BASE_URL}/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}

            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    klines = []

                    for k in data:
                        kline = OHLCV(
                            symbol=asset,
                            open_time=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                            open=float(k[1]),
                            high=float(k[2]),
                            low=float(k[3]),
                            close=float(k[4]),
                            volume=float(k[5]),
                            close_time=datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
                        )
                        klines.append(kline)

                    return klines

        except Exception as e:
            logger.error(f"Error getting klines for {asset}: {e}")

        return []

    async def get_24h_stats(self, asset: str) -> Optional[dict]:
        """Get 24-hour price statistics.

        Args:
            asset: Asset symbol

        Returns:
            Dict with 24h stats or None if error
        """
        symbol = self._get_binance_symbol(asset)

        try:
            url = f"{self.BASE_URL}/ticker/24hr"
            params = {"symbol": symbol}

            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "price_change": float(data["priceChange"]),
                        "price_change_percent": float(data["priceChangePercent"]),
                        "high": float(data["highPrice"]),
                        "low": float(data["lowPrice"]),
                        "volume": float(data["volume"]),
                        "quote_volume": float(data["quoteVolume"]),
                    }

        except Exception as e:
            logger.error(f"Error getting 24h stats for {asset}: {e}")

        return None

    def get_cached_price(self, asset: str) -> Optional[SpotPrice]:
        """Get the most recently cached price for an asset."""
        return self._price_cache.get(asset)


class BinanceWebSocket:
    """WebSocket client for real-time Binance price updates."""

    WS_BASE = "wss://stream.binance.com:9443/ws"

    def __init__(self, assets: List[str]):
        self.assets = assets
        self._ws = None
        self._running = False

    def _get_stream_name(self, asset: str) -> str:
        """Get the stream name for an asset."""
        symbol = BINANCE_SYMBOLS.get(asset.upper(), f"{asset.upper()}USDT")
        return f"{symbol.lower()}@trade"

    async def connect(self):
        """Connect to the WebSocket."""
        import websockets

        streams = "/".join(self._get_stream_name(a) for a in self.assets)
        url = f"{self.WS_BASE}/{streams}"

        self._ws = await websockets.connect(url)
        self._running = True

        logger.info(f"Connected to Binance WebSocket for {self.assets}")

    async def disconnect(self):
        """Disconnect from the WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def listen(self):
        """Listen for trade updates.

        Yields:
            SpotPrice: Price update
        """
        import json

        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        while self._running:
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=30)
                data = json.loads(message)

                if "s" in data and "p" in data:
                    # Extract asset from symbol
                    symbol = data["s"]
                    for asset, binance_symbol in BINANCE_SYMBOLS.items():
                        if symbol == binance_symbol:
                            yield SpotPrice(
                                symbol=asset,
                                price=float(data["p"]),
                                timestamp=datetime.fromtimestamp(
                                    data["T"] / 1000, tz=timezone.utc
                                ),
                            )
                            break

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await self._ws.ping()
                except Exception:
                    logger.warning("Binance WebSocket ping failed, reconnecting...")
                    await self.connect()

            except Exception as e:
                logger.error(f"Binance WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(1)
                    await self.connect()
