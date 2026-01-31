#!/usr/bin/env python3
"""Diagnostic script to test API connectivity and identify crash causes."""

import asyncio
import logging
from datetime import datetime, timezone

from src.polynance.clients.polymarket import PolymarketClient
from src.polynance.clients.binance import BinanceClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def test_apis():
    """Test API connectivity and response times."""

    assets = ["BTC", "ETH", "SOL", "XRP"]

    # Test Polymarket
    logger.info("Testing Polymarket API...")
    async with PolymarketClient() as pm:
        try:
            markets = await pm.find_active_15min_markets(assets)
            logger.info(f"Found {len(markets)} Polymarket markets")

            for market in markets:
                logger.info(f"  {market.asset}: {market.condition_id[:20]}...")

                # Test getting price
                start = datetime.now(timezone.utc)
                price = await pm.get_market_price(market)
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()

                if price:
                    logger.info(f"    ✓ Price: YES={price.yes_price:.3f}, took {elapsed:.2f}s")
                else:
                    logger.error(f"    ✗ Failed to get price")

        except Exception as e:
            logger.error(f"Polymarket error: {e}", exc_info=True)

    # Test Binance
    logger.info("\nTesting Binance API...")
    async with BinanceClient() as binance:
        try:
            for asset in assets:
                start = datetime.now(timezone.utc)
                price = await binance.get_price(asset)
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()

                if price:
                    logger.info(f"  ✓ {asset}: ${price.price:.2f}, took {elapsed:.2f}s")
                else:
                    logger.error(f"  ✗ {asset}: Failed to get price")

        except Exception as e:
            logger.error(f"Binance error: {e}", exc_info=True)

    # Test concurrent load (simulating actual sampling)
    logger.info("\nTesting concurrent load (4 assets simultaneously)...")
    async with PolymarketClient() as pm, BinanceClient() as binance:
        markets = await pm.find_active_15min_markets(assets)

        for i in range(3):
            logger.info(f"\n--- Round {i+1}/3 ---")
            start = datetime.now(timezone.utc)

            tasks = []
            for market in markets:
                tasks.append(pm.get_market_price(market))
                tasks.append(binance.get_price(market.asset))

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()

                successes = sum(1 for r in results if not isinstance(r, Exception) and r is not None)
                failures = len(results) - successes

                logger.info(f"  Completed in {elapsed:.2f}s: {successes} successes, {failures} failures")

                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"  Task {j} failed: {result}")
                    elif result is None:
                        logger.warning(f"  Task {j} returned None")

            except Exception as e:
                logger.error(f"Concurrent test error: {e}", exc_info=True)

            if i < 2:
                logger.info("  Waiting 30s before next round...")
                await asyncio.sleep(30)

    logger.info("\n=== Diagnostics Complete ===")


if __name__ == "__main__":
    asyncio.run(test_apis())
