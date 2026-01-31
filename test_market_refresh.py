#!/usr/bin/env python3
"""Test that markets are refreshed on window transitions."""

import asyncio
import logging
from datetime import datetime, timezone

from src.polynance.clients.polymarket import PolymarketClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def test_market_refresh():
    """Test fetching markets for different windows."""

    assets = ["BTC"]

    async with PolymarketClient() as pm:
        # Fetch market for current window
        logger.info("=== Fetching market for CURRENT window ===")
        markets1 = await pm.find_active_15min_markets(assets)

        if markets1:
            market1 = markets1[0]
            logger.info(f"Current market: {market1.condition_id}")
            logger.info(f"Question: {market1.question}")

        # Wait a moment
        await asyncio.sleep(2)

        # Fetch again (should get same market if still in same window)
        logger.info("\n=== Fetching market again (same window) ===")
        markets2 = await pm.find_active_15min_markets(assets)

        if markets2:
            market2 = markets2[0]
            logger.info(f"Second fetch: {market2.condition_id}")

            if market1.condition_id == market2.condition_id:
                logger.info("✓ Same window = same market (expected)")
            else:
                logger.warning("✗ Different market in same window (unexpected!)")

        # Try to get price from market
        logger.info("\n=== Testing price fetch ===")
        if markets1:
            price = await pm.get_market_price(markets1[0])
            if price:
                logger.info(f"✓ Got price: YES={price.yes_price:.3f}")
            else:
                logger.error("✗ Failed to get price")


if __name__ == "__main__":
    asyncio.run(test_market_refresh())
