"""Signal calculations for arbitrage/lock strategy.

Implements the key signals used in the Incremental Pair strategy:
- RHR (Rolling High Range) - price volatility measure
- OBI (Order Book Imbalance) - buy/sell pressure
- Pulse - price velocity/momentum
- Flip detection - crossing 0.50 threshold
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Single price tick."""
    timestamp: datetime
    yes_price: float
    yes_bid: float
    yes_ask: float
    spread: float
    spot_price: Optional[float] = None


@dataclass
class SignalState:
    """Current state of all signals for an asset."""
    asset: str
    window_id: str
    t_seconds: float

    # Current prices
    yes_price: float
    yes_bid: float
    yes_ask: float
    spread: float

    # Computed signals
    rhr: float  # Rolling High Range
    obi: float  # Order Book Imbalance
    pulse: float  # Price velocity
    delta_from_fair: float  # Distance from 0.50

    # Window stats
    yes_high: float
    yes_low: float
    flip_count: int
    pattern: str  # 'trending', 'reversal', 'choppy', 'flat'

    # Lock assessment
    lock_achievable: bool
    estimated_lock_profit: float

    # Guards (from your bot)
    rhr_guard_triggered: bool
    obi_guard_triggered: bool
    flip_doom: bool
    ptb_doom: bool


class SignalCalculator:
    """Calculate arbitrage signals from tick stream."""

    # Configuration
    RHR_WINDOW_SECONDS = 120  # 2-minute rolling window for RHR
    RHR_THRESHOLD = 0.03  # 3% range triggers RHR guard
    OBI_THRESHOLD = -0.30  # OBI below this triggers guard
    FLIP_THRESHOLD = 3  # More than 3 flips = Flip Doom
    MIN_RANGE_FOR_LOCK = 0.05  # 5% range needed for lock
    LOCK_COST_ESTIMATE = 0.04  # 4% estimated cost to achieve lock

    def __init__(self, asset: str):
        self.asset = asset
        self.current_window_id: Optional[str] = None
        self.window_start: Optional[datetime] = None

        # Tick history for this window
        self.ticks: Deque[Tick] = deque(maxlen=500)  # ~250 minutes at 30s

        # Window tracking
        self.yes_high: float = 0
        self.yes_low: float = 1
        self.flip_count: int = 0
        self.last_side: Optional[str] = None  # 'yes' or 'no' (above/below 0.50)

        # Direction tracking for pattern detection
        self.direction_changes: List[float] = []
        self.last_direction: Optional[str] = None

    def new_window(self, window_id: str, window_start: datetime):
        """Reset for a new window."""
        self.current_window_id = window_id
        self.window_start = window_start
        self.ticks.clear()
        self.yes_high = 0
        self.yes_low = 1
        self.flip_count = 0
        self.last_side = None
        self.direction_changes = []
        self.last_direction = None

    def add_tick(self, tick: Tick) -> SignalState:
        """Process a new tick and return current signal state."""
        self.ticks.append(tick)

        # Update high/low
        self.yes_high = max(self.yes_high, tick.yes_price)
        self.yes_low = min(self.yes_low, tick.yes_price)

        # Detect flip (crossing 0.50)
        current_side = 'yes' if tick.yes_price >= 0.50 else 'no'
        if self.last_side is not None and current_side != self.last_side:
            self.flip_count += 1
        self.last_side = current_side

        # Track direction changes for pattern detection
        if len(self.ticks) >= 2:
            price_change = tick.yes_price - self.ticks[-2].yes_price
            if abs(price_change) > 0.01:  # Ignore noise
                direction = 'up' if price_change > 0 else 'down'
                if self.last_direction is not None and direction != self.last_direction:
                    self.direction_changes.append(tick.timestamp.timestamp())
                self.last_direction = direction

        # Calculate signals
        rhr = self._calculate_rhr()
        obi = self._calculate_obi(tick)
        pulse = self._calculate_pulse()
        delta_from_fair = abs(tick.yes_price - 0.50)

        # Calculate t_seconds
        t_seconds = 0
        if self.window_start:
            t_seconds = (tick.timestamp - self.window_start).total_seconds()

        # Determine pattern
        pattern = self._detect_pattern()

        # Lock assessment
        yes_range = self.yes_high - self.yes_low
        lock_achievable = yes_range >= self.MIN_RANGE_FOR_LOCK
        estimated_lock_profit = max(0, yes_range * 0.3 - self.LOCK_COST_ESTIMATE) if lock_achievable else 0

        # Guards
        rhr_guard = rhr >= self.RHR_THRESHOLD
        obi_guard = obi <= self.OBI_THRESHOLD
        flip_doom = self.flip_count >= self.FLIP_THRESHOLD
        ptb_doom = False  # Would need oracle price to calculate

        return SignalState(
            asset=self.asset,
            window_id=self.current_window_id or "",
            t_seconds=t_seconds,
            yes_price=tick.yes_price,
            yes_bid=tick.yes_bid,
            yes_ask=tick.yes_ask,
            spread=tick.spread,
            rhr=rhr,
            obi=obi,
            pulse=pulse,
            delta_from_fair=delta_from_fair,
            yes_high=self.yes_high,
            yes_low=self.yes_low,
            flip_count=self.flip_count,
            pattern=pattern,
            lock_achievable=lock_achievable,
            estimated_lock_profit=estimated_lock_profit,
            rhr_guard_triggered=rhr_guard,
            obi_guard_triggered=obi_guard,
            flip_doom=flip_doom,
            ptb_doom=ptb_doom,
        )

    def _calculate_rhr(self) -> float:
        """Calculate Rolling High Range over last N seconds."""
        if len(self.ticks) < 2:
            return 0

        # Get ticks within RHR window
        cutoff = self.ticks[-1].timestamp - timedelta(seconds=self.RHR_WINDOW_SECONDS)
        recent = [t for t in self.ticks if t.timestamp >= cutoff]

        if len(recent) < 2:
            return 0

        prices = [t.yes_price for t in recent]
        return max(prices) - min(prices)

    def _calculate_obi(self, tick: Tick) -> float:
        """Calculate Order Book Imbalance.

        OBI = (bid_size - ask_size) / (bid_size + ask_size)

        Since we don't have actual sizes, we estimate from spread position:
        If yes_price is closer to bid -> negative OBI (selling pressure)
        If yes_price is closer to ask -> positive OBI (buying pressure)
        """
        if tick.spread <= 0:
            return 0

        # Estimate OBI from price position within spread
        mid = (tick.yes_bid + tick.yes_ask) / 2
        if tick.yes_price < mid:
            # Price below mid -> selling pressure
            obi = -abs(mid - tick.yes_price) / (tick.spread / 2)
        else:
            # Price above mid -> buying pressure
            obi = abs(tick.yes_price - mid) / (tick.spread / 2)

        return max(-1, min(1, obi))  # Clamp to [-1, 1]

    def _calculate_pulse(self) -> float:
        """Calculate price velocity (cents per second)."""
        if len(self.ticks) < 3:
            return 0

        # Use last 5 ticks or available
        recent = list(self.ticks)[-5:]
        if len(recent) < 2:
            return 0

        time_diff = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
        if time_diff <= 0:
            return 0

        price_diff = recent[-1].yes_price - recent[0].yes_price
        return price_diff / time_diff

    def _detect_pattern(self) -> str:
        """Detect price pattern: trending, reversal, choppy, or flat."""
        if len(self.ticks) < 5:
            return 'flat'

        yes_range = self.yes_high - self.yes_low

        # Flat if range is tiny
        if yes_range < 0.03:
            return 'flat'

        # Count direction changes
        if len(self.direction_changes) >= 4:
            return 'choppy'
        elif len(self.direction_changes) == 0:
            return 'trending'
        elif len(self.direction_changes) <= 2:
            # Check if it's a clean reversal
            prices = [t.yes_price for t in self.ticks]
            first_half = prices[:len(prices)//2]
            second_half = prices[len(prices)//2:]

            if len(first_half) > 1 and len(second_half) > 1:
                first_dir = first_half[-1] - first_half[0]
                second_dir = second_half[-1] - second_half[0]

                if np.sign(first_dir) != np.sign(second_dir) and abs(first_dir) > 0.02:
                    return 'reversal'

            return 'trending'

        return 'choppy'

    def get_summary(self) -> dict:
        """Get current window summary."""
        if not self.ticks:
            return {}

        return {
            'asset': self.asset,
            'window_id': self.current_window_id,
            'tick_count': len(self.ticks),
            'yes_high': self.yes_high,
            'yes_low': self.yes_low,
            'yes_range': self.yes_high - self.yes_low,
            'flip_count': self.flip_count,
            'pattern': self._detect_pattern(),
            'current_price': self.ticks[-1].yes_price if self.ticks else None,
        }


class MultiAssetSignalCalculator:
    """Manage signal calculators for multiple assets."""

    def __init__(self, assets: List[str]):
        self.calculators = {asset: SignalCalculator(asset) for asset in assets}

    def get_calculator(self, asset: str) -> SignalCalculator:
        """Get calculator for an asset."""
        if asset not in self.calculators:
            self.calculators[asset] = SignalCalculator(asset)
        return self.calculators[asset]

    def new_window(self, asset: str, window_id: str, window_start: datetime):
        """Start new window for an asset."""
        self.get_calculator(asset).new_window(window_id, window_start)

    def add_tick(self, asset: str, tick: Tick) -> SignalState:
        """Add tick for an asset."""
        return self.get_calculator(asset).add_tick(tick)

    def get_all_summaries(self) -> dict:
        """Get summaries for all assets."""
        return {
            asset: calc.get_summary()
            for asset, calc in self.calculators.items()
        }
