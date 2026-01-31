"""High-frequency sampler for arbitrage tracking.

Samples Polymarket prices every 30 seconds (vs 2.5 minutes in main sampler).
Designed to capture the rapid price movements needed for lock strategy analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Callable

from ..clients.polymarket import PolymarketClient, MarketInfo, MarketPrice
from ..clients.binance import BinanceClient
from .database import ArbitrageDatabase
from .signals import SignalCalculator, MultiAssetSignalCalculator, Tick, SignalState

logger = logging.getLogger(__name__)


def get_window_info(dt: datetime) -> tuple:
    """Get window ID and start time for a given datetime."""
    minute = dt.minute
    window_minute = (minute // 15) * 15
    window_start = dt.replace(minute=window_minute, second=0, microsecond=0)
    window_id = f"{window_start.strftime('%Y%m%d_%H%M')}"
    return window_id, window_start


class ArbitrageSampler:
    """High-frequency sampler for arbitrage signal tracking."""

    def __init__(
        self,
        db: ArbitrageDatabase,
        polymarket: PolymarketClient,
        binance: BinanceClient,
        assets: List[str],
        sample_interval: int = 30,  # 30 seconds
        on_signal: Optional[Callable[[SignalState], None]] = None,
    ):
        self.db = db
        self.polymarket = polymarket
        self.binance = binance
        self.assets = assets
        self.sample_interval = sample_interval
        self.on_signal = on_signal

        # Signal calculators
        self.signals = MultiAssetSignalCalculator(assets)

        # Market info cache
        self.markets: dict[str, MarketInfo] = {}

        # Current window tracking
        self.current_windows: dict[str, str] = {}  # asset -> window_id

        # Running state
        self._running = False

        # Stats
        self.tick_count = 0
        self.error_count = 0

    async def initialize(self):
        """Initialize by finding active markets."""
        logger.info(f"Initializing arbitrage sampler for: {self.assets}")

        markets = await self.polymarket.find_active_15min_markets(self.assets)

        for market in markets:
            self.markets[market.asset] = market
            logger.info(f"Found market for {market.asset}: {market.condition_id[:20]}...")

        missing = [a for a in self.assets if a not in self.markets]
        if missing:
            logger.warning(f"No markets found for: {missing}")

    async def run(self):
        """Run the high-frequency sampling loop."""
        self._running = True
        logger.info(f"Starting arbitrage sampler (interval: {self.sample_interval}s)")

        while self._running:
            try:
                await self._sample_all()
                await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Sample error: {e}")
                await asyncio.sleep(self.sample_interval)

        logger.info("Arbitrage sampler stopped")

    def stop(self):
        """Stop the sampler."""
        self._running = False

    async def _sample_all(self):
        """Sample all assets."""
        now = datetime.now(timezone.utc)
        base_window_id, window_start = get_window_info(now)

        # Check for window transitions
        for asset in self.assets:
            window_id = f"{asset}_{base_window_id}"

            if self.current_windows.get(asset) != window_id:
                # New window - finalize old one and start new
                if self.current_windows.get(asset):
                    await self._finalize_window(asset, self.current_windows[asset])

                self.current_windows[asset] = window_id
                self.signals.new_window(asset, window_id, window_start)
                logger.debug(f"[{asset}] New window: {window_id}")

        # Sample each asset
        tasks = [self._sample_asset(asset, now) for asset in self.assets if asset in self.markets]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _sample_asset(self, asset: str, timestamp: datetime):
        """Sample a single asset."""
        market = self.markets.get(asset)
        if not market:
            return

        try:
            # Get Polymarket price
            price_data = await self.polymarket.get_market_price(market)
            if not price_data:
                return

            # Get spot price (optional, for reference)
            spot_price = None
            try:
                spot_data = await self.binance.get_spot_price(f"{asset}USDT")
                if spot_data:
                    spot_price = spot_data.price
            except Exception:
                pass  # Spot price is optional

            # Create tick
            tick = Tick(
                timestamp=timestamp,
                yes_price=price_data.yes_price,
                yes_bid=price_data.yes_bid,
                yes_ask=price_data.yes_ask,
                spread=price_data.spread,
                spot_price=spot_price,
            )

            # Process through signal calculator
            window_id = self.current_windows.get(asset, "")
            signal_state = self.signals.add_tick(asset, tick)

            # Store tick
            await self.db.insert_tick(
                asset=asset,
                timestamp=timestamp,
                window_id=window_id,
                t_seconds=signal_state.t_seconds,
                yes_price=tick.yes_price,
                no_price=price_data.no_price,
                yes_bid=tick.yes_bid,
                yes_ask=tick.yes_ask,
                spread=tick.spread,
                spot_price=spot_price,
                rhr=signal_state.rhr,
                obi=signal_state.obi,
                pulse=signal_state.pulse,
            )

            self.tick_count += 1

            # Callback if provided
            if self.on_signal:
                self.on_signal(signal_state)

            # Log significant events
            if signal_state.rhr_guard_triggered:
                logger.info(f"[{asset}] RHR Guard triggered: {signal_state.rhr:.3f}")
            if signal_state.flip_doom:
                logger.info(f"[{asset}] Flip Doom: {signal_state.flip_count} flips")

        except Exception as e:
            self.error_count += 1
            logger.error(f"[{asset}] Sample error: {e}")

    async def _finalize_window(self, asset: str, window_id: str):
        """Finalize a completed window."""
        calc = self.signals.get_calculator(asset)
        summary = calc.get_summary()

        if not summary:
            return

        # Get first tick for opening price
        ticks = await self.db.get_window_ticks(asset, window_id)
        if not ticks:
            return

        yes_open = ticks[0].get('yes_price', 0.5)
        yes_close = ticks[-1].get('yes_price', 0.5)

        # Determine outcome (did YES win?)
        # Note: We're estimating - actual outcome comes from resolution
        outcome = 'up' if yes_close > 0.90 else ('down' if yes_close < 0.10 else 'unknown')

        # Get max RHR, OBI range from ticks
        rhr_max = max((t.get('rhr', 0) or 0 for t in ticks), default=0)
        obi_vals = [t.get('obi', 0) or 0 for t in ticks]
        obi_min = min(obi_vals) if obi_vals else 0
        obi_max = max(obi_vals) if obi_vals else 0

        # Parse window start from ID
        try:
            # window_id format: "BTC_20260129_0815"
            parts = window_id.split('_')
            date_str = parts[1]
            time_str = parts[2]
            window_start = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M")
            window_start = window_start.replace(tzinfo=timezone.utc)
        except Exception:
            window_start = datetime.now(timezone.utc)

        await self.db.upsert_window(
            window_id=window_id,
            asset=asset,
            window_start=window_start,
            yes_open=yes_open,
            yes_high=summary.get('yes_high'),
            yes_low=summary.get('yes_low'),
            yes_close=yes_close,
            yes_range=summary.get('yes_range'),
            rhr_max=rhr_max,
            obi_min=obi_min,
            obi_max=obi_max,
            flip_count=summary.get('flip_count', 0),
            lock_achievable=summary.get('yes_range', 0) >= 0.05,
            pattern=summary.get('pattern'),
            outcome=outcome,
        )

        logger.info(
            f"[{asset}] Window {window_id} finalized: "
            f"range={summary.get('yes_range', 0)*100:.1f}%, "
            f"flips={summary.get('flip_count', 0)}, "
            f"pattern={summary.get('pattern')}"
        )


async def run_arbitrage_sampler(
    assets: List[str] = None,
    sample_interval: int = 30,
    data_dir: str = "data",
):
    """Run standalone arbitrage sampler."""
    from pathlib import Path

    if assets is None:
        assets = ['BTC', 'ETH', 'SOL', 'XRP']

    db_path = Path(data_dir) / "arbitrage.db"

    async with ArbitrageDatabase(db_path) as db:
        polymarket = PolymarketClient()
        binance = BinanceClient()

        await polymarket.connect()
        await binance.connect()

        try:
            sampler = ArbitrageSampler(
                db=db,
                polymarket=polymarket,
                binance=binance,
                assets=assets,
                sample_interval=sample_interval,
            )

            await sampler.initialize()
            await sampler.run()

        finally:
            await polymarket.close()
            await binance.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run arbitrage sampler")
    parser.add_argument("--interval", type=int, default=30, help="Sample interval in seconds")
    parser.add_argument("--assets", nargs="+", default=['BTC', 'ETH', 'SOL', 'XRP'])
    args = parser.parse_args()

    asyncio.run(run_arbitrage_sampler(
        assets=args.assets,
        sample_interval=args.interval,
    ))
