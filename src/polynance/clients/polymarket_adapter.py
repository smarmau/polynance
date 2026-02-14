"""Polymarket adapter implementing the ExchangeClient interface.

Thin wrapper around the existing PolymarketClient for market data.
When live_trading=True, also initializes pmxt.Polymarket for order placement.
"""

import asyncio
import logging
import os
from typing import Optional, List, Literal

from .exchange import (
    ExchangeClient,
    MarketInfo,
    MarketPrice,
    OrderResult,
    PositionInfo,
    BalanceInfo,
)
from .polymarket import PolymarketClient

logger = logging.getLogger(__name__)


class PolymarketAdapter(ExchangeClient):
    """Wraps the existing PolymarketClient to implement ExchangeClient.

    Market data always uses the direct aiohttp-based PolymarketClient.
    When live_trading=True, order placement uses pmxt.Polymarket which
    handles Polygon signing, the CLOB API, and order lifecycle.

    Args:
        live_trading: Enable live order placement via pmxt.
        private_key: Polygon private key for signing (or POLYMARKET_PRIVATE_KEY env var).
        proxy_address: Optional Polymarket proxy/smart-wallet address
            (or POLYMARKET_PROXY_ADDRESS env var).
    """

    def __init__(
        self,
        live_trading: bool = False,
        private_key: Optional[str] = None,
        proxy_address: Optional[str] = None,
    ):
        self._client = PolymarketClient()
        self._live_trading = live_trading
        self._pmxt = None  # pmxt.Polymarket instance (lazy-initialized)
        self._private_key = private_key
        self._proxy_address = proxy_address

    async def connect(self):
        """Open the aiohttp session and optionally init pmxt for trading."""
        await self._client.__aenter__()
        logger.info("Polymarket exchange client connected")

        if self._live_trading:
            self._init_pmxt()

    def _init_pmxt(self):
        """Initialize the pmxt Polymarket trading client.

        Reads credentials from constructor args or environment variables:
          - POLYMARKET_PRIVATE_KEY: Polygon private key (required)
          - POLYMARKET_PROXY_ADDRESS: Proxy/smart-wallet address (optional)
        """
        try:
            import pmxt
        except ImportError:
            logger.error(
                "pmxt is required for live trading. Install with: pip install pmxt"
            )
            self._live_trading = False
            return

        pk = self._private_key or os.getenv("POLYMARKET_PRIVATE_KEY", "")
        proxy = self._proxy_address or os.getenv("POLYMARKET_PROXY_ADDRESS")

        if not pk:
            logger.warning(
                "POLYMARKET_PRIVATE_KEY not set — live trading disabled. "
                "Set this env var or pass private_key= to enable."
            )
            self._live_trading = False
            return

        try:
            self._pmxt = pmxt.Polymarket(
                private_key=pk,
                proxy_address=proxy,
                signature_type="gnosis-safe",
            )
            logger.info("pmxt Polymarket trading client initialized (live trading enabled)")
        except Exception as e:
            logger.error(f"Failed to initialize pmxt trading client: {e}")
            self._live_trading = False

    async def close(self):
        """Close the aiohttp session and pmxt client."""
        await self._client.__aexit__(None, None, None)
        if self._pmxt:
            try:
                self._pmxt.close()
            except Exception:
                pass

    # --- Market data (delegates to PolymarketClient) ---

    async def find_active_15min_markets(self, assets: List[str]) -> List[MarketInfo]:
        """Find active 15-min markets via the Gamma API."""
        poly_markets = await self._client.find_active_15min_markets(assets)
        return poly_markets

    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get market price via the CLOB order book."""
        return await self._client.get_market_price(market)

    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get cached market info."""
        return self._client.get_cached_market(asset)

    # --- Trading (via pmxt) ---

    @property
    def supports_trading(self) -> bool:
        """Whether live trading is enabled and pmxt is initialized."""
        return self._live_trading and self._pmxt is not None

    async def place_order(
        self,
        market: MarketInfo,
        side: Literal["buy", "sell"],
        outcome: Literal["yes", "no"],
        amount: float,
        price: Optional[float] = None,
        order_type: Literal["market", "limit"] = "limit",
    ) -> OrderResult:
        """Place an order on Polymarket via pmxt.

        Args:
            market: MarketInfo for the target market.
            side: "buy" or "sell".
            outcome: "yes" or "no" — maps to yes_token_id / no_token_id.
            amount: Number of contracts to trade.
            price: Limit price (0-1). Required for limit orders.
            order_type: "market" or "limit".

        Returns:
            OrderResult with order details from the exchange.
        """
        if not self.supports_trading:
            raise NotImplementedError(
                "Live trading not enabled. Set live_trading=True and provide "
                "POLYMARKET_PRIVATE_KEY to enable."
            )

        # Map outcome to token ID
        outcome_id = (
            market.yes_token_id if outcome == "yes" else market.no_token_id
        )

        logger.info(
            f"[Polymarket] Placing {side} {order_type} order: "
            f"{amount} contracts @ {price} for {outcome.upper()} "
            f"(market={market.condition_id}, asset={market.asset})"
        )

        try:
            order = await asyncio.to_thread(
                self._pmxt.create_order,
                market_id=market.condition_id,
                outcome_id=outcome_id,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
            )

            result = OrderResult(
                order_id=order.id,
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                side=order.side,
                order_type=order.type,
                amount=order.amount,
                price=order.price,
                status=order.status,
                filled=order.filled,
                remaining=order.remaining,
                fee=order.fee,
                timestamp=order.timestamp,
                raw={
                    "id": order.id,
                    "status": order.status,
                    "filled": order.filled,
                },
            )

            logger.info(
                f"[Polymarket] Order placed: {result.order_id} "
                f"status={result.status} filled={result.filled}/{result.amount}"
            )
            return result

        except Exception as e:
            logger.error(f"[Polymarket] Order placement failed: {e}")
            raise

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order on Polymarket via pmxt."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.info(f"[Polymarket] Cancelling order: {order_id}")

        try:
            order = await asyncio.to_thread(self._pmxt.cancel_order, order_id)

            result = OrderResult(
                order_id=order.id,
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                side=order.side,
                order_type=order.type,
                amount=order.amount,
                price=order.price,
                status=order.status,
                filled=order.filled,
                remaining=order.remaining,
                fee=order.fee,
                timestamp=order.timestamp,
            )

            logger.info(f"[Polymarket] Order cancelled: {result.order_id}")
            return result

        except Exception as e:
            logger.error(f"[Polymarket] Cancel failed: {e}")
            raise

    async def fetch_open_orders(
        self, market: Optional[MarketInfo] = None
    ) -> List[OrderResult]:
        """Get open orders from Polymarket via pmxt."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        market_id = market.condition_id if market else None
        orders = await asyncio.to_thread(
            self._pmxt.fetch_open_orders, market_id=market_id
        )

        return [
            OrderResult(
                order_id=o.id,
                market_id=o.market_id,
                outcome_id=o.outcome_id,
                side=o.side,
                order_type=o.type,
                amount=o.amount,
                price=o.price,
                status=o.status,
                filled=o.filled,
                remaining=o.remaining,
                fee=o.fee,
                timestamp=o.timestamp,
            )
            for o in orders
        ]

    async def fetch_positions(self) -> List[PositionInfo]:
        """Get current positions from Polymarket via pmxt."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        positions = await asyncio.to_thread(self._pmxt.fetch_positions)

        return [
            PositionInfo(
                market_id=p.market_id,
                outcome_id=p.outcome_id,
                outcome_label=p.outcome_label,
                size=p.size,
                entry_price=p.entry_price,
                current_price=p.current_price,
                unrealized_pnl=p.unrealized_pnl,
                realized_pnl=p.realized_pnl,
            )
            for p in positions
        ]

    async def fetch_balance(self) -> List[BalanceInfo]:
        """Get account balance from Polymarket via pmxt."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        balances = await asyncio.to_thread(self._pmxt.fetch_balance)

        return [
            BalanceInfo(
                currency=b.currency,
                total=b.total,
                available=b.available,
                locked=b.locked,
            )
            for b in balances
        ]

    async def fetch_order(self, order_id: str) -> OrderResult:
        """Get full details of a specific order from Polymarket via pmxt.

        Retrieves actual fill price, filled amount, and fee — critical for
        accurate P&L after order fills.
        """
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.debug(f"[Polymarket] Fetching order: {order_id}")

        try:
            order = await asyncio.to_thread(self._pmxt.fetch_order, order_id)

            return OrderResult(
                order_id=order.id,
                market_id=order.market_id,
                outcome_id=order.outcome_id,
                side=order.side,
                order_type=order.type,
                amount=order.amount,
                price=order.price,
                status=order.status,
                filled=order.filled,
                remaining=order.remaining,
                fee=order.fee,
                timestamp=order.timestamp,
            )

        except Exception as e:
            logger.error(f"[Polymarket] fetch_order failed for {order_id}: {e}")
            raise

    async def fetch_trades(
        self,
        outcome_id: str,
        limit: Optional[int] = None,
        since: Optional[int] = None,
    ) -> List[dict]:
        """Get trade history for an outcome token from Polymarket via pmxt.

        Returns fill records for historical P&L reconstruction.
        """
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.debug(f"[Polymarket] Fetching trades for outcome: {outcome_id}")

        try:
            trades = await asyncio.to_thread(
                self._pmxt.fetch_trades,
                outcome_id, limit=limit, since=since
            )

            return [
                {
                    "id": t.id,
                    "timestamp": t.timestamp,
                    "price": t.price,
                    "amount": t.amount,
                    "side": t.side,
                }
                for t in trades
            ]

        except Exception as e:
            logger.error(
                f"[Polymarket] fetch_trades failed for {outcome_id}: {e}"
            )
            raise
