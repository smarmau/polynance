"""Polymarket adapter implementing the ExchangeClient interface.

Market data uses the aiohttp-based PolymarketClient.
When live_trading=True, all trading operations use the polymarket CLI
(~/.local/bin/polymarket) which handles EIP-712 signing, CLOB API, and
order lifecycle. This replaces the previous py-clob-client SDK approach.
"""

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
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
DATA_API_URL = "https://data-api.polymarket.com"
MIN_ORDER_SIZE = 5  # Polymarket 15-min crypto minimum BUY contracts
MIN_SELL_ORDER_SIZE = 1  # Minimum SELL size (closing existing position)

# Int → CLI string for POLYMARKET_SIGNATURE_TYPE env var
SIG_INT_TO_STR = {"0": "eoa", "1": "proxy", "2": "gnosis-safe"}


def _find_polymarket_cli() -> str:
    """Return the polymarket CLI executable path, checking common locations."""
    found = shutil.which("polymarket")
    if found:
        return found
    for candidate in [
        Path.home() / ".local" / "bin" / "polymarket",
        Path("/usr/local/bin/polymarket"),
        Path("/usr/bin/polymarket"),
    ]:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "polymarket CLI not found.\n"
        "Install: curl -sSL https://raw.githubusercontent.com/Polymarket/"
        "polymarket-cli/main/install.sh | sh\n"
        "Then add ~/.local/bin to your PATH if needed."
    )


class PolymarketAdapter(ExchangeClient):
    """Wraps PolymarketClient (market data) + polymarket CLI (trading).

    Market data always uses the direct aiohttp-based PolymarketClient.
    When live_trading=True, all order/balance/position operations go through
    the polymarket CLI binary, which handles signing and CLOB API communication.

    Args:
        live_trading: Enable live order placement via polymarket CLI.
        private_key: Polygon private key (or POLYMARKET_PRIVATE_KEY env var).
        funder: Optional funder address (or POLYMARKET_FUNDER_ADDRESS env var).
        signature_type: Wallet type. 0=EOA, 1=proxy wallet, 2=Gnosis Safe.
            Can also be set via POLYMARKET_SIGNATURE_TYPE env var.
    """

    def __init__(
        self,
        live_trading: bool = False,
        private_key: Optional[str] = None,
        funder: Optional[str] = None,
        signature_type: Optional[int] = None,
    ):
        self._client = PolymarketClient()
        self._live_trading = live_trading
        self._private_key = private_key
        self._funder = funder
        self._signature_type = signature_type

        # CLI path and auth args (set in connect())
        self._cli_path: Optional[str] = None
        self._auth_args: list[str] = []
        self._wallet_address: str = ""

    # ── CLI wrapper ──────────────────────────────────────────────────────────

    def _poly(self, *args, check: bool = True, auth: bool = True):
        """Run: polymarket <args> [auth-flags] --output json → dict/list/str.

        Raises RuntimeError on non-zero exit when check=True.
        Set auth=False for commands that don't need --private-key.
        """
        cmd = [self._cli_path] + list(str(a) for a in args)
        if auth and self._auth_args:
            cmd += self._auth_args
        cmd += ["--output", "json"]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"polymarket {' '.join(str(a) for a in args)}: timed out after 30s")
        except FileNotFoundError:
            raise RuntimeError("polymarket CLI not found")

        if check and r.returncode != 0:
            err = (r.stderr.strip() or r.stdout.strip())[:500]
            raise RuntimeError(f"polymarket {' '.join(str(a) for a in args)}: {err}")

        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return r.stdout.strip()

    @staticmethod
    def _parse_balance(raw) -> float:
        """Parse USDC balance from CLI response.

        CLI returns balance as a float string (e.g. "84.418887"), not micro-USDC.
        """
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, dict):
            for key in ("balance", "amount", "usdc"):
                if key in raw:
                    try:
                        return float(raw[key])
                    except (ValueError, TypeError):
                        pass
        if isinstance(raw, str):
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return 0.0

    @staticmethod
    def _parse_order_id(resp) -> str:
        """Extract order ID from create-order response."""
        if isinstance(resp, str):
            return resp.strip()
        if isinstance(resp, dict):
            for key in ("orderID", "order_id", "id", "orderId"):
                if resp.get(key):
                    return str(resp[key])
        return ""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self):
        """Open the aiohttp session and optionally init CLI for trading."""
        await self._client.__aenter__()
        logger.info("Polymarket exchange client connected (market data)")

        if self._live_trading:
            await self._init_cli()

    async def _init_cli(self):
        """Initialize the polymarket CLI for live trading.

        Reads credentials from constructor args or environment variables:
          - POLYMARKET_PRIVATE_KEY: Polygon private key (required)
          - POLYMARKET_FUNDER_ADDRESS: Funder address (optional)
          - POLYMARKET_SIGNATURE_TYPE: 0/1/2 or eoa/proxy/gnosis-safe
        """
        pk = self._private_key or os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            logger.warning(
                "POLYMARKET_PRIVATE_KEY not set — live trading disabled. "
                "Set this env var or pass private_key= to enable."
            )
            self._live_trading = False
            return

        # Find CLI binary
        try:
            self._cli_path = _find_polymarket_cli()
        except RuntimeError as e:
            logger.error(str(e))
            self._live_trading = False
            return

        # Resolve signature_type: constructor arg > env var > default 1 (proxy)
        sig_type = self._signature_type
        if sig_type is None:
            sig_type_env = os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")
            sig_type = int(sig_type_env) if sig_type_env.isdigit() else None

        # Convert int to CLI string: "1" → "proxy"
        if sig_type is not None:
            sig_str = SIG_INT_TO_STR.get(str(sig_type), str(sig_type))
        else:
            sig_str = os.getenv("POLYMARKET_SIGNATURE_TYPE", "proxy")

        # Build auth args — injected into every _poly() call
        self._auth_args = ["--private-key", pk, "--signature-type", sig_str]

        # Store funder address for position queries
        self._funder = self._funder or os.getenv("POLYMARKET_FUNDER_ADDRESS", "")

        # Verify CLI works
        try:
            wallet_resp = await asyncio.to_thread(self._poly, "wallet", "address")
            self._wallet_address = (
                wallet_resp if isinstance(wallet_resp, str)
                else wallet_resp.get("address", str(wallet_resp))
            )
            logger.info(
                f"Polymarket CLI initialized (live trading enabled) "
                f"signer={self._wallet_address}, "
                f"funder={'set' if self._funder else 'not set'}, "
                f"sig_type={sig_str}"
            )
        except Exception as e:
            logger.error(f"CLI verification failed: {e}")
            self._live_trading = False

    async def close(self):
        """Close the aiohttp session."""
        await self._client.__aexit__(None, None, None)

    # ── Market data (delegates to PolymarketClient) ──────────────────────────

    async def find_active_15min_markets(self, assets: List[str]) -> List[MarketInfo]:
        """Find active 15-min markets via the Gamma API."""
        return await self._client.find_active_15min_markets(assets)

    async def get_market_price(self, market: MarketInfo) -> Optional[MarketPrice]:
        """Get market price via the CLOB order book."""
        return await self._client.get_market_price(market)

    def get_cached_market(self, asset: str) -> Optional[MarketInfo]:
        """Get cached market info."""
        return self._client.get_cached_market(asset)

    # ── Trading (via polymarket CLI) ─────────────────────────────────────────

    @property
    def supports_trading(self) -> bool:
        """Whether live trading is enabled and CLI is initialized."""
        return self._live_trading and self._cli_path is not None

    async def place_order(
        self,
        market: MarketInfo,
        side: Literal["buy", "sell"],
        outcome: Literal["yes", "no"],
        amount: float,
        price: Optional[float] = None,
        order_type: Literal["market", "limit"] = "limit",
    ) -> OrderResult:
        """Place an order on Polymarket via CLI.

        Args:
            market: MarketInfo for the target market.
            side: "buy" or "sell".
            outcome: "yes" or "no" — maps to yes_token_id / no_token_id.
            amount: Number of contracts to trade.
            price: Limit price (0-1). Required for limit orders.
            order_type: "market" → FAK (Fill-And-Kill), "limit" → GTC.

        Returns:
            OrderResult with order details from the exchange.
        """
        if not self.supports_trading:
            raise NotImplementedError(
                "Live trading not enabled. Set live_trading=True and provide "
                "POLYMARKET_PRIVATE_KEY to enable."
            )

        # Map outcome to token ID
        token_id = (
            market.yes_token_id if outcome == "yes" else market.no_token_id
        )

        # Enforce exchange minimum order size for buys
        min_size = MIN_ORDER_SIZE if side == "buy" else MIN_SELL_ORDER_SIZE
        if amount < min_size:
            logger.warning(
                f"[Polymarket] Order size {amount:.1f} below minimum "
                f"{min_size}. Bumping to {min_size}."
            )
            amount = float(min_size)

        logger.info(
            f"[Polymarket] Placing {side} {order_type} order: "
            f"{amount:.2f} contracts @ {price} for {outcome.upper()} "
            f"(asset={market.asset}, token={token_id[:16]}...)"
        )

        try:
            cmd_args = [
                "clob", "create-order",
                "--token", token_id,
                "--side", side,
                "--price", f"{price:.2f}",
                "--size", f"{amount:.2f}",
            ]

            # Market orders use FAK (Fill-And-Kill)
            if order_type == "market":
                cmd_args += ["--order-type", "FAK"]

            resp = await asyncio.to_thread(self._poly, *cmd_args)

            logger.info(f"[Polymarket] create-order response: {resp}")

            order_id = self._parse_order_id(resp)
            status = resp.get("status", "open") if isinstance(resp, dict) else "open"

            # For FAK orders, check if it filled immediately
            if order_type == "market" and isinstance(resp, dict):
                usdc_recvd = float(resp.get("taking_amount", 0))
                shares = float(resp.get("making_amount", 0))
                if shares > 0 or usdc_recvd > 0:
                    status = "filled"

            if not order_id:
                logger.warning(f"[Polymarket] No orderID in response: {resp}")

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
                raw=resp if isinstance(resp, dict) else {"response": str(resp)},
            )

            logger.info(
                f"[Polymarket] Order placed: {result.order_id} "
                f"status={result.status}"
            )
            return result

        except Exception as e:
            logger.error(
                f"[Polymarket] Order placement failed: {e}", exc_info=True
            )
            raise

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order on Polymarket via CLI."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.info(f"[Polymarket] Cancelling order: {order_id}")

        try:
            resp = await asyncio.to_thread(
                self._poly, "clob", "cancel", order_id, check=False
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
        """Get open orders from Polymarket via CLI."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        try:
            resp = await asyncio.to_thread(
                self._poly, "clob", "orders", check=False
            )

            if not isinstance(resp, list):
                if isinstance(resp, dict):
                    resp = resp.get("data", [])
                else:
                    return []

            results = []
            for o in resp:
                if not isinstance(o, dict):
                    continue
                # Filter by market if specified
                if market and o.get("market", "") != market.condition_id:
                    continue
                results.append(OrderResult(
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
                ))

            return results

        except Exception as e:
            logger.error(f"[Polymarket] fetch_open_orders failed: {e}")
            raise

    async def fetch_positions(self) -> List[PositionInfo]:
        """Get current positions from Polymarket data API."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        wallet = self._funder or self._wallet_address
        if not wallet:
            logger.debug("[Polymarket] No wallet address for position query")
            return []

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{DATA_API_URL}/positions",
                    params={"user": wallet.lower()},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    data = await r.json()
                    positions = data if isinstance(data, list) else data.get("data", [])

            results = []
            for p in positions:
                if not isinstance(p, dict):
                    continue
                size = float(p.get("size", 0))
                if size <= 0:
                    continue
                results.append(PositionInfo(
                    market_id=p.get("conditionId", ""),
                    outcome_id=p.get("asset", ""),
                    outcome_label=p.get("outcome", ""),
                    size=size,
                    entry_price=float(p.get("avgPrice", 0)),
                    current_price=float(p.get("curPrice", 0)),
                    unrealized_pnl=float(p.get("currentValue", 0)) - size * float(p.get("avgPrice", 0)),
                ))

            return results

        except Exception as e:
            logger.error(f"[Polymarket] fetch_positions failed: {e}")
            return []

    async def fetch_balance(self) -> List[BalanceInfo]:
        """Get USDC collateral balance from Polymarket via CLI."""
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        try:
            resp = await asyncio.to_thread(
                self._poly, "clob", "balance", "--asset-type", "collateral"
            )

            balance = self._parse_balance(resp)

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
        """Get full details of a specific order from Polymarket via CLI.

        Retrieves actual fill price, filled amount, and fee — critical for
        accurate P&L after order fills.
        """
        if not self.supports_trading:
            raise NotImplementedError("Live trading not enabled.")

        logger.debug(f"[Polymarket] Fetching order: {order_id}")

        try:
            resp = await asyncio.to_thread(
                self._poly, "clob", "order", order_id
            )

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
                "CLOSED": "filled",
                "LIVE": "open",
                "CANCELLED": "cancelled",
                "CANCELED": "cancelled",
                "EXPIRED": "expired",
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

        Note: CLI doesn't have a direct trades endpoint. Returns empty list.
        Use fetch_order() for fill data on specific orders.
        """
        logger.debug(
            f"[Polymarket] fetch_trades not supported via CLI for {outcome_id}"
        )
        return []

    # ── Polymarket-specific methods ──────────────────────────────────────────

    async def fetch_conditional_balance(self, token_id: str) -> float:
        """Get conditional token balance (number of shares held) via CLI.

        Args:
            token_id: The outcome token ID to check balance for.

        Returns:
            Number of shares held (float), or 0.0 if none.
        """
        if not self.supports_trading:
            return 0.0

        try:
            resp = await asyncio.to_thread(
                self._poly, "clob", "balance",
                "--asset-type", "conditional",
                "--token", token_id,
            )
            return self._parse_balance(resp)
        except Exception as e:
            logger.warning(f"[Polymarket] fetch_conditional_balance failed: {e}")
            return 0.0

    async def sell_position_fak(
        self, token_id: str, max_attempts: int = 5
    ) -> float:
        """Exit a position using repeated FAK (Fill-And-Kill) orders.

        FAK fills whatever bids exist at the moment and cancels the rest
        instantly. We loop up to max_attempts times to drain the position.

        Args:
            token_id: The outcome token to sell.
            max_attempts: Maximum FAK attempts (default: 5).

        Returns:
            Total USDC received across all fills.
        """
        if not self.supports_trading:
            return 0.0

        total_usdc = 0.0

        for attempt in range(1, max_attempts + 1):
            # Re-query live balance — previous FAK may have partially filled
            try:
                remaining = await self.fetch_conditional_balance(token_id)
            except Exception as e:
                logger.warning(f"  Balance query failed: {e}")
                break

            # Floor to 2 dp so we never submit more than we hold
            remaining = math.floor(remaining * 100) / 100

            if remaining < MIN_SELL_ORDER_SIZE:
                if attempt > 1:
                    logger.info(
                        f"  Remaining {remaining:.2f} shares below min — done"
                    )
                break

            logger.info(
                f"  [{attempt}/{max_attempts}] FAK sell "
                f"{remaining:.2f} shares..."
            )

            try:
                resp = await asyncio.to_thread(
                    self._poly,
                    "clob", "create-order",
                    "--token", token_id,
                    "--side", "sell",
                    "--price", "0.01",  # floor price — fills at actual best bid
                    "--size", f"{remaining:.2f}",
                    "--order-type", "FAK",
                )
            except RuntimeError as e:
                err_str = str(e)
                # FAK with zero liquidity returns a 400 — not a crash
                if any(s in err_str.lower() for s in (
                    "no liquidity", "couldn't be", "400"
                )):
                    logger.warning(
                        f"  No liquidity — {remaining:.2f} shares remain unsold"
                    )
                else:
                    logger.error(f"  FAK sell attempt {attempt} failed: {e}")
                break

            # Parse fill info
            if isinstance(resp, dict):
                usdc_recvd = float(resp.get("taking_amount", 0))
                shares_sold = float(resp.get("making_amount", 0))
                status = resp.get("status", "")
            else:
                usdc_recvd = 0.0
                shares_sold = 0.0
                status = ""

            total_usdc += usdc_recvd

            if shares_sold > 0:
                logger.info(
                    f"  Filled {shares_sold:.2f} shares "
                    f"→ ${usdc_recvd:.4f} USDC"
                )
            elif status in ("CANCELLED", "CANCELED") or usdc_recvd == 0:
                logger.warning(
                    f"  No fill (no bids at ≥0.01) — "
                    f"{remaining:.2f} shares remain unsold"
                )
                break
            else:
                logger.info(
                    f"  status={status} usdc={usdc_recvd:.4f}"
                )

        if total_usdc > 0:
            logger.info(f"  FAK sell total received: ${total_usdc:.4f} USDC")

        return total_usdc

    def get_funder_address(self) -> str:
        """Get the funder/proxy wallet address for position queries."""
        return self._funder or self._wallet_address
