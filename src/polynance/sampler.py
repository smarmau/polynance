"""Main sampling loop for collecting Polymarket and spot price data."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable, List, Dict, Tuple

from .clients.polymarket import PolymarketClient, MarketInfo, MarketPrice
from .clients.binance import BinanceClient, SpotPrice
from .db.database import Database
from .db.models import Sample, Window

logger = logging.getLogger(__name__)

# Sample points within each 15-minute window (in minutes)
SAMPLE_POINTS = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]


@dataclass
class AssetState:
    """Current state for a single asset."""

    asset: str
    market: Optional[MarketInfo] = None
    current_window_id: Optional[str] = None
    window_start: Optional[datetime] = None
    samples: List[Sample] = None
    spot_open: Optional[float] = None
    spot_high: Optional[float] = None
    spot_low: Optional[float] = None

    def __post_init__(self):
        if self.samples is None:
            self.samples = []


def get_window_boundaries(dt: datetime) -> Tuple[datetime, datetime]:
    """Get the start and end times for the 15-minute window containing dt.

    Windows are on fixed 15-minute boundaries: :00, :15, :30, :45
    """
    minute = dt.minute
    window_minute = (minute // 15) * 15

    window_start = dt.replace(minute=window_minute, second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=15)

    return window_start, window_end


def get_window_id(window_start: datetime, asset: str) -> str:
    """Generate a unique window ID."""
    return f"{asset}_{window_start.strftime('%Y%m%d_%H%M')}"


def get_t_minutes(sample_time: datetime, window_start: datetime) -> float:
    """Calculate t_minutes (time into window) for a sample."""
    delta = sample_time - window_start
    return delta.total_seconds() / 60


def snap_to_sample_point(t_minutes: float) -> float:
    """Snap a t_minutes value to the nearest sample point.

    Always snaps to the closest defined sample point for consistency.
    """
    closest = min(SAMPLE_POINTS, key=lambda p: abs(t_minutes - p))
    return closest


class Sampler:
    """Main sampling engine for multi-asset data collection."""

    def __init__(
        self,
        db: Database,
        polymarket: PolymarketClient,
        binance: BinanceClient,
        assets: List[str],
        on_window_complete: Optional[Callable] = None,
        on_sample_collected: Optional[Callable] = None,
    ):
        self.db = db
        self.polymarket = polymarket
        self.binance = binance
        self.assets = assets
        self.on_window_complete = on_window_complete
        self.on_sample_collected = on_sample_collected  # Called after each sample

        # State per asset
        self.states: Dict[str, AssetState] = {}
        for asset in assets:
            self.states[asset] = AssetState(asset=asset)

        # Running flag
        self._running = False

        # Sampling interval (seconds)
        self.sample_interval = 30  # Check every 30 seconds

        # Health check tracking
        self._last_successful_sample_time: Dict[str, datetime] = {}
        self._iteration_count = 0
        self._total_samples_collected = 0
        self._consecutive_failures = 0

    async def initialize(self):
        """Initialize the sampler by finding active markets."""
        logger.info(f"Initializing sampler for assets: {self.assets}")

        markets = await self.polymarket.find_active_15min_markets(self.assets)

        for market in markets:
            if market.asset in self.states:
                self.states[market.asset].market = market
                logger.info(f"Found market for {market.asset}: {market.condition_id}")

        # Check which assets we couldn't find markets for
        missing = [a for a in self.assets if self.states[a].market is None]
        if missing:
            logger.warning(f"No markets found for: {missing}")

    async def run(self):
        """Main sampling loop."""
        self._running = True
        logger.info("Starting sampling loop...")

        while self._running:
            try:
                now = datetime.now(timezone.utc)
                self._iteration_count += 1

                logger.debug(f"Sampling loop iteration {self._iteration_count} at {now.strftime('%H:%M:%S')}")

                # Check if we need to transition windows
                await self._check_window_transitions(now)

                # Collect samples for all assets
                samples_collected = await self._collect_samples(now)

                # Health check logging (every 10 iterations = ~5 minutes)
                if self._iteration_count % 10 == 0:
                    self._log_health_check(now)

                # Check for stalled sampling
                if samples_collected == 0:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 5:
                        logger.error(
                            f"⚠️  SAMPLING STALLED: No samples collected for {self._consecutive_failures} consecutive iterations "
                            f"({self._consecutive_failures * self.sample_interval}s). Check API connectivity!"
                        )
                else:
                    self._consecutive_failures = 0

                # Wait for next sample
                logger.debug(f"Sleeping for {self.sample_interval}s...")
                await asyncio.sleep(self.sample_interval)

            except asyncio.CancelledError:
                logger.info("Sampling loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sampling loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    def stop(self):
        """Stop the sampling loop."""
        self._running = False

    async def _check_window_transitions(self, now: datetime):
        """Check if any assets need to transition to a new window."""
        window_start, window_end = get_window_boundaries(now)

        for asset, state in self.states.items():
            # Check if we're in a new window
            if state.window_start is None or window_start != state.window_start:
                # Finalize the old window if it exists
                if state.window_start is not None:
                    await self._finalize_window(state)

                # IMPORTANT: Fetch new market for this window!
                # Each 15-min window has a unique market with different epoch
                logger.info(f"[{asset}] Fetching market for new window starting at {window_start.strftime('%H:%M')}")
                markets = await self.polymarket.find_active_15min_markets([asset])

                if markets and len(markets) > 0:
                    state.market = markets[0]
                    logger.info(f"[{asset}] Found market: {state.market.condition_id[:20]}...")
                else:
                    logger.warning(f"[{asset}] No market found for this window - will retry next iteration")
                    state.market = None

                # Start new window
                state.window_start = window_start
                state.current_window_id = get_window_id(window_start, asset)
                state.samples = []
                state.spot_open = None
                state.spot_high = None
                state.spot_low = None

                logger.info(f"[{asset}] Started new window: {state.current_window_id}")

    async def _collect_samples(self, now: datetime) -> int:
        """Collect samples for all assets.

        Returns:
            Number of samples successfully collected
        """
        # Get current window boundaries
        window_start, window_end = get_window_boundaries(now)
        t_minutes = get_t_minutes(now, window_start)
        sample_point = snap_to_sample_point(t_minutes)

        logger.debug(f"_collect_samples: t_minutes={t_minutes:.2f}, sample_point={sample_point}")

        # Collect data for each asset
        tasks = []
        for asset, state in self.states.items():
            if state.market is not None:
                tasks.append(self._collect_asset_sample(asset, state, now, sample_point))
            else:
                logger.warning(f"[{asset}] No market available, skipping sample collection")

        samples_collected = 0

        if tasks:
            logger.debug(f"Collecting samples for {len(tasks)} assets")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Log any exceptions and count successes
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Sample collection exception: {result}", exc_info=True)
                elif result is True:  # Sample was collected
                    samples_collected += 1
        else:
            logger.warning("No tasks to collect samples")

        return samples_collected

    async def _collect_asset_sample(
        self, asset: str, state: AssetState, now: datetime, sample_point: float
    ) -> bool:
        """Collect a sample for a single asset.

        Returns:
            True if sample was collected successfully, False otherwise
        """
        try:
            # Check if we already have this sample point
            existing_points = [s.t_minutes for s in state.samples]
            if sample_point in existing_points:
                logger.debug(f"[{asset}] Already have sample at t={sample_point}")
                return False  # Already have this sample

            logger.info(f"[{asset}] Collecting sample at t={sample_point}...")

            # Get Polymarket price
            logger.debug(f"[{asset}] Fetching Polymarket price...")
            pm_price = await self.polymarket.get_market_price(state.market)
            if pm_price is None:
                logger.warning(f"[{asset}] Failed to get Polymarket price")
                return False
            logger.debug(f"[{asset}] Got Polymarket price: YES={pm_price.yes_price:.3f}")

            # Get spot price
            logger.debug(f"[{asset}] Fetching Binance spot price...")
            spot = await self.binance.get_price(asset)
            if spot is None:
                logger.warning(f"[{asset}] Failed to get spot price")
                return False
            logger.debug(f"[{asset}] Got Binance price: ${spot.price:.2f}")

            # Update high/low tracking
            if state.spot_open is None:
                state.spot_open = spot.price
            if state.spot_high is None or spot.price > state.spot_high:
                state.spot_high = spot.price
            if state.spot_low is None or spot.price < state.spot_low:
                state.spot_low = spot.price

            # Calculate price change from open
            change_from_open = None
            if state.spot_open and state.spot_open > 0:
                change_from_open = ((spot.price - state.spot_open) / state.spot_open) * 100

            # Create sample
            sample = Sample(
                window_id=state.current_window_id,
                asset=asset,
                window_start_utc=state.window_start,
                sample_time_utc=now,
                t_minutes=sample_point,
                pm_yes_price=pm_price.yes_price,
                pm_no_price=pm_price.no_price,
                pm_yes_bid=pm_price.yes_bid,
                pm_yes_ask=pm_price.yes_ask,
                pm_spread=pm_price.spread,
                pm_midpoint=pm_price.midpoint,
                spot_price=spot.price,
                spot_price_change_from_open=change_from_open,
                pm_market_id=state.market.yes_token_id,
                pm_condition_id=state.market.condition_id,
            )

            # Store in database
            await self.db.insert_sample(sample)
            state.samples.append(sample)

            # Update health tracking
            self._last_successful_sample_time[asset] = now
            self._total_samples_collected += 1

            logger.info(
                f"[{asset}] t={sample_point}: YES={pm_price.yes_price:.3f}, "
                f"spot=${spot.price:.2f} ({change_from_open:+.3f}% from open)"
            )

            # Callback for sample collection (e.g., for trading at t=7.5)
            if self.on_sample_collected:
                try:
                    await self.on_sample_collected(asset, sample, state)
                except Exception as e:
                    logger.error(f"Error in sample collected callback: {e}")

            return True

        except Exception as e:
            logger.error(f"[{asset}] Error collecting sample: {e}", exc_info=True)
            return False

    async def _finalize_window(self, state: AssetState):
        """Finalize a completed window and store summary."""
        if not state.samples:
            return

        asset = state.asset
        logger.info(f"[{asset}] Finalizing window: {state.current_window_id}")

        # Get samples at each time point
        samples_by_t = {s.t_minutes: s for s in state.samples}

        # Calculate outcome
        first_sample = samples_by_t.get(0.0)
        last_sample = samples_by_t.get(15.0) or samples_by_t.get(max(samples_by_t.keys()))

        spot_open = first_sample.spot_price if first_sample else state.spot_open
        spot_close = last_sample.spot_price if last_sample else None

        outcome = None
        outcome_binary = None
        spot_change_pct = None
        spot_change_bps = None

        if spot_open and spot_close:
            spot_change_pct = ((spot_close - spot_open) / spot_open) * 100
            spot_change_bps = spot_change_pct * 100  # Convert to basis points
            outcome = "up" if spot_close > spot_open else "down"
            outcome_binary = 1 if outcome == "up" else 0

        # Calculate range
        spot_range_bps = None
        if state.spot_high and state.spot_low and spot_open:
            spot_range_bps = ((state.spot_high - state.spot_low) / spot_open) * 10000

        # Extract PM prices at key times
        pm_yes_t0 = samples_by_t.get(0.0, samples_by_t.get(min(samples_by_t.keys())))
        pm_yes_t0 = pm_yes_t0.pm_yes_price if pm_yes_t0 else None

        pm_yes_t2_5 = samples_by_t.get(2.5)
        pm_yes_t2_5 = pm_yes_t2_5.pm_yes_price if pm_yes_t2_5 else None

        pm_yes_t5 = samples_by_t.get(5.0)
        pm_yes_t5 = pm_yes_t5.pm_yes_price if pm_yes_t5 else None

        pm_yes_t7_5 = samples_by_t.get(7.5)
        pm_yes_t7_5 = pm_yes_t7_5.pm_yes_price if pm_yes_t7_5 else None

        pm_yes_t10 = samples_by_t.get(10.0)
        pm_yes_t10 = pm_yes_t10.pm_yes_price if pm_yes_t10 else None

        pm_yes_t12_5 = samples_by_t.get(12.5)
        pm_yes_t12_5 = pm_yes_t12_5.pm_yes_price if pm_yes_t12_5 else None

        # Spreads
        pm_spread_t0 = samples_by_t.get(0.0)
        pm_spread_t0 = pm_spread_t0.pm_spread if pm_spread_t0 else None

        pm_spread_t5 = samples_by_t.get(5.0)
        pm_spread_t5 = pm_spread_t5.pm_spread if pm_spread_t5 else None

        # Momentum signals
        pm_momentum_0_to_5 = None
        if pm_yes_t0 is not None and pm_yes_t5 is not None:
            pm_momentum_0_to_5 = pm_yes_t5 - pm_yes_t0

        pm_momentum_5_to_10 = None
        if pm_yes_t5 is not None and pm_yes_t10 is not None:
            pm_momentum_5_to_10 = pm_yes_t10 - pm_yes_t5

        # Create window record
        window = Window(
            window_id=state.current_window_id,
            asset=asset,
            window_start_utc=state.window_start,
            window_end_utc=state.window_start + timedelta(minutes=15),
            outcome=outcome,
            outcome_binary=outcome_binary,
            spot_open=spot_open,
            spot_close=spot_close,
            spot_change_pct=spot_change_pct,
            spot_change_bps=spot_change_bps,
            spot_high=state.spot_high,
            spot_low=state.spot_low,
            spot_range_bps=spot_range_bps,
            pm_yes_t0=pm_yes_t0,
            pm_yes_t2_5=pm_yes_t2_5,
            pm_yes_t5=pm_yes_t5,
            pm_yes_t7_5=pm_yes_t7_5,
            pm_yes_t10=pm_yes_t10,
            pm_yes_t12_5=pm_yes_t12_5,
            pm_spread_t0=pm_spread_t0,
            pm_spread_t5=pm_spread_t5,
            pm_price_momentum_0_to_5=pm_momentum_0_to_5,
            pm_price_momentum_5_to_10=pm_momentum_5_to_10,
            resolved_at_utc=datetime.now(timezone.utc),
            resolution_source="spot_price_comparison",
        )

        # Store in database
        await self.db.insert_window(window)

        logger.info(
            f"[{asset}] Window complete: {outcome} ({spot_change_bps:+.1f} bps), "
            f"PM@t5={pm_yes_t5:.3f}" if pm_yes_t5 else f"[{asset}] Window complete"
        )

        # Callback for analysis
        if self.on_window_complete:
            try:
                await self.on_window_complete(asset, window)
            except Exception as e:
                logger.error(f"Error in window complete callback: {e}")

    def _log_health_check(self, now: datetime):
        """Log health check information."""
        logger.info("=" * 60)
        logger.info(f"Health Check - Iteration {self._iteration_count}")
        logger.info(f"Total samples collected: {self._total_samples_collected}")
        logger.info(f"Consecutive failures: {self._consecutive_failures}")

        for asset, last_time in self._last_successful_sample_time.items():
            time_since = (now - last_time).total_seconds() / 60
            logger.info(f"  {asset}: Last sample {time_since:.1f}m ago")

        for asset, state in self.states.items():
            if asset not in self._last_successful_sample_time:
                logger.warning(f"  {asset}: No samples collected yet!")

        logger.info("=" * 60)

    def get_current_state(self, asset: str) -> Optional[AssetState]:
        """Get the current state for an asset."""
        return self.states.get(asset)

    def get_all_states(self) -> Dict[str, AssetState]:
        """Get all asset states."""
        return self.states
