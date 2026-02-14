"""Polymarket adapter implementing the ExchangeClient interface.

Thin wrapper around the existing PolymarketClient for market data.
When live_trading=True, initializes py-clob-client's ClobClient for
order placement via Polymarket's CLOB API.
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

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID_POLYGON = 137


class PolymarketAdapter(ExchangeClient):
    """Wraps the existing PolymarketClient to implement ExchangeClient.

    Market data always uses the direct aiohttp-based PolymarketClient.
    When live_trading=True, order placement uses py-clob-client's ClobClient
    which handles EIP-712 signing, the CLOB API, and order lifecycle.

    Args:
        live_trading: Enable live order placement via py-clob-client.
        private_key: Polygon private key for signing (or POLYMARKET_PRIVATE_KEY env var).
        funder: Optional funder address (or POLYMARKET_FUNDER_ADDRESS env var).
    """

    def __init__(
        self,
        live_trading: bool = False,
        private_key: Optional[str] = None,
        funder: Optional[str] = None,
    ):
        self._client = PolymarketClient()
        self._live_trading = live_trading
        self._clob: Optional[object] = None  # ClobClient instance (lazy-initialized)
        self._private_key = private_key
        self._funder = funder

    async def connect(self):
        """Open the aiohttp session and optionally init ClobClient for trading."""
        await self._client.__aenter__()
        logger.info("Polymarket exchange client connected")

        if self._live_trading:
            await self._init_clob()

    async def _init_clob(self):
        """Initialize the py-clob-client ClobClient for live trading.

        Reads credentials from constructor args or environment variables:
          - POLYMARKET_PRIVATE_KEY: Polygon private key (required)
          - POLYMARKET_FUNDER_ADDRESS: Funder address (optional)
          - CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE: API creds (optional,
            will be derived from private key if not set)
        """
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError:
            logger.error(
                "py-clob-client is required for live trading. "
                "Install with: pip install py-clob-client"
            )
            self._live_trading = False
            return

        pk = self._private_key or os.getenv("POLYMARKET_PRIVATE_KEY", "")
        funder = self._funder or os.getenv("POLYMARKET_FUNDER_ADDRESS")

        if not pk:
            logger.warning(
                "POLYMARKET_PRIVATE_KEY not set — live trading disabled. "
                "Set this env var or pass private_key= to enable."
            )
            self._live_trading = False
            return

        try:
            # Build kwargs for ClobClient
            clob_kwargs = dict(
                host=CLOB_HOST,
                key=pk,
                chain_id=CHAIN_ID_POLYGON,
            )
            if funder:
                clob_kwargs["funder"] = funder

            # Check for pre-set API credentials
            api_key = os.getenv("CLOB_API_KEY", "")
            api_secret = os.getenv("CLOB_SECRET", "")
            api_passphrase = os.getenv("CLOB_PASS_PHRASE", "")

            if api_key and api_secret and api_passphrase:
                clob_kwargs["creds"] = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )

            self._clob = ClobClient(**clob_kwargs)

            # If no API creds were provided, derive them from the private key
            if "creds" not in clob_kwargs:
                logger.info("No CLOB API creds found, deriving from private key...")
                creds = await asyncio.to_thread(
                    self._clob.create_or_derive_api_creds
                )
                self._clob.set_api_creds(creds)
                logger.info("CLOB API credentials derived successfully")

            logger.info(
                "py-clob-client ClobClient initialized (live trading enabled)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ClobClient: {e}")
            self._live_trading = False

    async def close(self):
        """Close the aiohttp session."""
        await self._client.__aexit__(None, None, None)

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

    # --- Trading (via py-clob-client) ---

    @property
    def supports_trading(self) -> bool:
        """Whether live trading is enabled and ClobClient is initialized."""
        return self._live_trading and self._clob is not None

    async def place_order(
        self,
        market: MarketInfo,
        side: Literal["buy", "sell"],
        outcome: Literal["yes", "no"],
        amount: float,
        price: Optional[float] = None,
        order_type: Literal["market", "limit"] = "limit",
    ) -> OrderResult:
        """Place an order on Polymarket via py-clob-client.

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

        from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        # Map outcome to token ID
        token_id = (
            market.yes_token_id if outcome == "yes" else market.no_token_id
        )
        clob_side = BUY if side == "buy" else SELL

        logger.info(
            f"[Polymarket] Placing {side} {order_type} order: "
            f"{amount} contracts @ {price} for {outcome.upper()} "
            f"(market={market.condition_id}, asset={market.asset})"
        )

        try:
            if order_type == "market":
                # Market order: amount is dollar value (USDC)
                order_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=amount * (price or 0.5),  # approximate USDC value
                    side=clob_side,
                )
                signed_order = await asyncio.to_thread(
                    self._clob.create_market_order, order_args
                )
                resp = await asyncio.to_thread(
                    self._clob.post_order, signed_order, OrderType.FOK
                )
            else:
                # Limit order (GTC)
                order_args = OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=amount,
                    side=clob_side,
                )
                signed_order = await asyncio.to_thread(
                    self._clob.create_order, order_args
                )
                resp = await asyncio.to_thread(
                    self._clob.post_order, signed_order, OrderType.GTC
                )

            # py-clob-client returns a dict with order details
            order_id = resp.get("orderID", resp.get("id", ""))
            status = resp.get("status", "open")

            result = OrderResult(
                order_id=order_id,
                market_id=market.condition_id,
                outcome_id=token_id,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                status=status,
                filled=0.0,
                remaining=amount,
                raw=resp,
            )

            logger.info(
                f"[Polymarket] Order placed: {result.order_id} "
                f"status={result.status}"
            )
            return result

        except Exception as e:
            logger.error(f"[Polymarket] Order placement failed: {e}")
            raise

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order on Polymarket via py-clob-client."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.info(f"[Polymarket] Cancelling order: {order_id}")

        try:
            resp = await asyncio.to_thread(
                self._clob.cancel, order_id=order_id
            )

            result = OrderResult(
                order_id=order_id,
                market_id="",
                outcome_id="",
                side="buy",
                order_type="limit",
                amount=0.0,
                status="cancelled",
                raw=resp if isinstance(resp, dict) else {"response": str(resp)},
            )

            logger.info(f"[Polymarket] Order cancelled: {order_id}")
            return result

        except Exception as e:
            logger.error(f"[Polymarket] Cancel failed: {e}")
            raise

    async def fetch_open_orders(
        self, market: Optional[MarketInfo] = None
    ) -> List[OrderResult]:
        """Get open orders from Polymarket via py-clob-client."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        from py_clob_client.clob_types import OpenOrderParams

        try:
            if market:
                params = OpenOrderParams(market=market.condition_id)
                resp = await asyncio.to_thread(self._clob.get_orders, params)
            else:
                resp = await asyncio.to_thread(self._clob.get_orders)

            # resp may be a list or a paginated dict
            orders_data = resp if isinstance(resp, list) else resp.get("data", [])

            return [
                OrderResult(
                    order_id=o.get("id", o.get("orderID", "")),
                    market_id=o.get("market", o.get("conditionId", "")),
                    outcome_id=o.get("asset_id", o.get("tokenID", "")),
                    side="buy" if o.get("side", "").upper() == "BUY" else "sell",
                    order_type="limit",
                    amount=float(o.get("original_size", o.get("size", 0))),
                    price=float(o.get("price", 0)),
                    status=o.get("status", "open"),
                    filled=float(o.get("size_matched", 0)),
                    remaining=float(o.get("original_size", 0)) - float(o.get("size_matched", 0)),
                )
                for o in orders_data
            ]

        except Exception as e:
            logger.error(f"[Polymarket] fetch_open_orders failed: {e}")
            raise

    async def fetch_positions(self) -> List[PositionInfo]:
        """Get current positions from Polymarket.

        Note: py-clob-client doesn't have a direct positions endpoint.
        Uses balance allowance queries for conditional tokens instead.
        """
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        # py-clob-client doesn't provide a unified positions endpoint.
        # Return empty list — position tracking is handled by the trading engine.
        logger.debug("[Polymarket] fetch_positions: not directly supported by py-clob-client")
        return []

    async def fetch_balance(self) -> List[BalanceInfo]:
        """Get USDC collateral balance from Polymarket via py-clob-client."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            resp = await asyncio.to_thread(
                self._clob.get_balance_allowance,
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )

            balance = float(resp.get("balance", 0)) if isinstance(resp, dict) else 0.0
            allowance = float(resp.get("allowance", 0)) if isinstance(resp, dict) else 0.0

            return [
                BalanceInfo(
                    currency="USDC",
                    total=balance,
                    available=balance,
                    locked=0.0,
                )
            ]

        except Exception as e:
            logger.error(f"[Polymarket] fetch_balance failed: {e}")
            raise

    async def fetch_order(self, order_id: str) -> OrderResult:
        """Get full details of a specific order from Polymarket.

        Retrieves actual fill price, filled amount, and fee — critical for
        accurate P&L after order fills.
        """
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.debug(f"[Polymarket] Fetching order: {order_id}")

        try:
            resp = await asyncio.to_thread(self._clob.get_order, order_id)

            if not isinstance(resp, dict):
                resp = {"id": order_id}

            # Map CLOB API response fields
            size = float(resp.get("original_size", resp.get("size", 0)))
            filled = float(resp.get("size_matched", 0))
            price = float(resp.get("price", 0))
            status = resp.get("status", "unknown")

            # Map status strings from CLOB API
            status_map = {
                "MATCHED": "filled",
                "LIVE": "open",
                "CANCELLED": "cancelled",
                "DELAYED": "open",
            }
            mapped_status = status_map.get(status.upper(), status.lower())

            return OrderResult(
                order_id=resp.get("id", order_id),
                market_id=resp.get("market", resp.get("conditionId", "")),
                outcome_id=resp.get("asset_id", resp.get("tokenID", "")),
                side="buy" if resp.get("side", "").upper() == "BUY" else "sell",
                order_type="limit",
                amount=size,
                price=price,
                status=mapped_status,
                filled=filled,
                remaining=size - filled,
                fee=float(resp.get("fee", 0)) if resp.get("fee") else None,
                timestamp=resp.get("timestamp"),
                raw=resp,
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
        """Get trade history for an outcome token from Polymarket.

        Returns fill records for historical P&L reconstruction.
        """
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.debug(f"[Polymarket] Fetching trades for outcome: {outcome_id}")

        try:
            from py_clob_client.clob_types import TradeParams

            params = TradeParams(
                asset_id=outcome_id,
                maker_address=self._clob.get_address(),
            )
            if since:
                params.after = str(since)

            resp = await asyncio.to_thread(self._clob.get_trades, params)

            # resp may be a list or paginated dict
            trades_data = resp if isinstance(resp, list) else resp.get("data", [])

            results = []
            for t in trades_data:
                results.append({
                    "id": t.get("id", ""),
                    "timestamp": t.get("timestamp", t.get("match_time", "")),
                    "price": float(t.get("price", 0)),
                    "amount": float(t.get("size", 0)),
                    "side": t.get("side", "").lower(),
                })
                if limit and len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(
                f"[Polymarket] fetch_trades failed for {outcome_id}: {e}"
            )
            raise
