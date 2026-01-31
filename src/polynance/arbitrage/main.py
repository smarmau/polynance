#!/usr/bin/env python3
"""Main entry point for arbitrage tracker.

Runs high-frequency sampler and dashboard together.
Separate from main polynance - uses its own database.
"""

import asyncio
import argparse
import logging
import signal
from pathlib import Path

from ..clients.polymarket import PolymarketClient
from ..clients.binance import BinanceClient
from .database import ArbitrageDatabase
from .sampler import ArbitrageSampler
from .dashboard import ArbitrageDashboard

logger = logging.getLogger(__name__)


class ArbitrageTracker:
    """Main arbitrage tracking application."""

    def __init__(
        self,
        assets: list[str],
        data_dir: Path,
        sample_interval: int = 30,
        headless: bool = False,
    ):
        self.assets = assets
        self.data_dir = data_dir
        self.sample_interval = sample_interval
        self.headless = headless

        # Components
        self.db: ArbitrageDatabase | None = None
        self.polymarket: PolymarketClient | None = None
        self.binance: BinanceClient | None = None
        self.sampler: ArbitrageSampler | None = None
        self.dashboard: ArbitrageDashboard | None = None

        # State
        self._running = False

    async def start(self):
        """Start the tracker."""
        logger.info("Starting Arbitrage Tracker")
        logger.info(f"Assets: {self.assets}")
        logger.info(f"Sample interval: {self.sample_interval}s")
        logger.info(f"Data dir: {self.data_dir}")

        # Initialize database
        db_path = self.data_dir / "arbitrage.db"
        self.db = ArbitrageDatabase(db_path)
        await self.db.connect()

        # Initialize clients
        self.polymarket = PolymarketClient()
        self.binance = BinanceClient()
        await self.polymarket.connect()
        await self.binance.connect()

        # Initialize sampler
        self.sampler = ArbitrageSampler(
            db=self.db,
            polymarket=self.polymarket,
            binance=self.binance,
            assets=self.assets,
            sample_interval=self.sample_interval,
        )
        await self.sampler.initialize()

        # Initialize dashboard (if not headless)
        if not self.headless:
            self.dashboard = ArbitrageDashboard(
                db=self.db,
                assets=self.assets,
            )
            # Connect sampler signals to dashboard
            self.sampler.on_signal = self.dashboard.update_signal

        self._running = True

    async def run(self):
        """Run the main loop."""
        tasks = []

        # Sampler task
        sampler_task = asyncio.create_task(self.sampler.run())
        tasks.append(sampler_task)

        # Dashboard task (if not headless)
        if self.dashboard:
            dashboard_task = asyncio.create_task(self.dashboard.run())
            tasks.append(dashboard_task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """Stop the tracker."""
        logger.info("Stopping Arbitrage Tracker")
        self._running = False

        if self.sampler:
            self.sampler.stop()
        if self.dashboard:
            self.dashboard.stop()

        # Close clients
        if self.polymarket:
            await self.polymarket.close()
        if self.binance:
            await self.binance.close()

        # Close database
        if self.db:
            await self.db.close()

        logger.info("Arbitrage Tracker stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Arbitrage Tracker - High-frequency Polymarket signal monitor"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["BTC", "ETH", "SOL", "XRP"],
        help="Assets to track",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Sample interval in seconds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without dashboard",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Create tracker
    tracker = ArbitrageTracker(
        assets=args.assets,
        data_dir=args.data_dir,
        sample_interval=args.interval,
        headless=args.headless,
    )

    # Handle signals
    loop = asyncio.get_event_loop()

    def handle_signal():
        asyncio.create_task(tracker.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)

    try:
        await tracker.start()
        await tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        await tracker.stop()


if __name__ == "__main__":
    asyncio.run(main())
