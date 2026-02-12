"""Core simulated trading engine."""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List

from ..db.database import Database
from ..db.models import Window
from .models import SimulatedTrade, TradingState, PerAssetStats
from .database import TradingDatabase
from .bet_sizing import SlowGrowthSizer

logger = logging.getLogger(__name__)


class SimulatedTrader:
    """Simulated trading engine for dry-run mode.

    Evaluates trading signals from completed windows and tracks
    simulated positions, P&L, and portfolio performance.

    Strategy (two-stage, default):
    - At t=7.5 min: If pm_yes >= 0.70 → pending BULL signal
    - At t=7.5 min: If pm_yes <= 0.30 → pending BEAR signal
    - At t=10 min: If pm_yes >= 0.85 → CONFIRM BULL, enter trade at t=10 price
    - At t=10 min: If pm_yes <= 0.15 → CONFIRM BEAR, enter trade at t=10 price
    - If signal not confirmed at t=10 → NO TRADE (fader filtered out)

    Payout mechanics (matches backtest_suite.py):
    - Buy YES at price P: Win → profit = (1-P), Lose → lose P
    - Buy NO at price (1-P): Win → profit = P, Lose → lose (1-P)
    - Fee on contract premium, spread cost on all trades
    """

    # Default configuration
    DEFAULT_INITIAL_BANKROLL = 1000.0
    DEFAULT_BASE_BET = 25.0
    DEFAULT_FEE_RATE = 0.02  # 2% on profits
    DEFAULT_SPREAD_COST = 0.006  # 0.6%
    DEFAULT_BULL_THRESHOLD = 0.80
    DEFAULT_BEAR_THRESHOLD = 0.20
    DEFAULT_MAX_BET_PCT = 0.05  # 5% of bankroll
    DEFAULT_PAUSE_WINDOWS_AFTER_LOSS = 2  # Skip N windows after any loss
    DEFAULT_GROWTH_PER_WIN = 0.10  # 10% growth per consecutive win
    DEFAULT_MAX_BET_MULTIPLIER = 2.0  # Cap bet at 2x base
    DEFAULT_MIN_TRAJECTORY = 0.20  # Min PM price movement from t=0 to entry

    # Two-stage defaults
    DEFAULT_ENTRY_MODE = "two_stage"
    DEFAULT_SIGNAL_THRESHOLD_BULL = 0.70
    DEFAULT_SIGNAL_THRESHOLD_BEAR = 0.30
    DEFAULT_CONFIRM_THRESHOLD_BULL = 0.85
    DEFAULT_CONFIRM_THRESHOLD_BEAR = 0.15

    # Contrarian defaults
    DEFAULT_CONTRARIAN_PREV_THRESH = 0.75
    DEFAULT_CONTRARIAN_BULL_THRESH = 0.60
    DEFAULT_CONTRARIAN_BEAR_THRESH = 0.40
    DEFAULT_CONTRARIAN_ENTRY_TIME = "t0"
    DEFAULT_CONTRARIAN_EXIT_TIME = "t12.5"

    # Consensus defaults
    DEFAULT_CONSENSUS_MIN_AGREE = 3
    DEFAULT_CONSENSUS_ENTRY_TIME = "t5"
    DEFAULT_CONSENSUS_EXIT_TIME = "t12.5"

    # Accel_dbl defaults
    DEFAULT_ACCEL_NEUTRAL_BAND = 0.15
    DEFAULT_ACCEL_PREV_THRESH = 0.75
    DEFAULT_ACCEL_BULL_THRESH = 0.55
    DEFAULT_ACCEL_BEAR_THRESH = 0.45
    DEFAULT_ACCEL_ENTRY_TIME = "t5"
    DEFAULT_ACCEL_EXIT_TIME = "t12.5"

    # Combo_dbl defaults
    DEFAULT_COMBO_PREV_THRESH = 0.75
    DEFAULT_COMBO_BULL_THRESH = 0.55
    DEFAULT_COMBO_BEAR_THRESH = 0.45
    DEFAULT_COMBO_ENTRY_TIME = "t5"
    DEFAULT_COMBO_EXIT_TIME = "t12.5"
    DEFAULT_COMBO_STOP_TIME = "t7.5"
    DEFAULT_COMBO_STOP_DELTA = 0.10
    DEFAULT_COMBO_XASSET_MIN = 2

    # Triple filter defaults
    # Double contrarian + cross-asset consensus (N dbl-strong) + PM t0 confirmation
    DEFAULT_TRIPLE_PREV_THRESH = 0.70
    DEFAULT_TRIPLE_BULL_THRESH = 0.55
    DEFAULT_TRIPLE_BEAR_THRESH = 0.45
    DEFAULT_TRIPLE_ENTRY_TIME = "t5"
    DEFAULT_TRIPLE_EXIT_TIME = "t12.5"
    DEFAULT_TRIPLE_XASSET_MIN = 3         # min assets with double-strong prev
    DEFAULT_TRIPLE_PM0_BULL_MIN = 0.50    # pm t0 must be >= this for bull entry
    DEFAULT_TRIPLE_PM0_BEAR_MAX = 0.50    # pm t0 must be <= this for bear entry

    def __init__(
        self,
        trading_db: TradingDatabase,
        asset_databases: Optional[Dict[str, Database]] = None,
        initial_bankroll: float = DEFAULT_INITIAL_BANKROLL,
        base_bet: float = DEFAULT_BASE_BET,
        fee_rate: float = DEFAULT_FEE_RATE,
        spread_cost: float = DEFAULT_SPREAD_COST,
        bull_threshold: float = DEFAULT_BULL_THRESHOLD,
        bear_threshold: float = DEFAULT_BEAR_THRESHOLD,
        max_bet_pct: float = DEFAULT_MAX_BET_PCT,
        pause_windows_after_loss: int = DEFAULT_PAUSE_WINDOWS_AFTER_LOSS,
        growth_per_win: float = DEFAULT_GROWTH_PER_WIN,
        max_bet_multiplier: float = DEFAULT_MAX_BET_MULTIPLIER,
        min_trajectory: float = DEFAULT_MIN_TRAJECTORY,
        entry_mode: str = DEFAULT_ENTRY_MODE,
        signal_threshold_bull: float = DEFAULT_SIGNAL_THRESHOLD_BULL,
        signal_threshold_bear: float = DEFAULT_SIGNAL_THRESHOLD_BEAR,
        confirm_threshold_bull: float = DEFAULT_CONFIRM_THRESHOLD_BULL,
        confirm_threshold_bear: float = DEFAULT_CONFIRM_THRESHOLD_BEAR,
        contrarian_prev_thresh: float = DEFAULT_CONTRARIAN_PREV_THRESH,
        contrarian_bull_thresh: float = DEFAULT_CONTRARIAN_BULL_THRESH,
        contrarian_bear_thresh: float = DEFAULT_CONTRARIAN_BEAR_THRESH,
        contrarian_entry_time: str = DEFAULT_CONTRARIAN_ENTRY_TIME,
        contrarian_exit_time: str = DEFAULT_CONTRARIAN_EXIT_TIME,
        consensus_min_agree: int = DEFAULT_CONSENSUS_MIN_AGREE,
        consensus_entry_time: str = DEFAULT_CONSENSUS_ENTRY_TIME,
        consensus_exit_time: str = DEFAULT_CONSENSUS_EXIT_TIME,
        accel_neutral_band: float = DEFAULT_ACCEL_NEUTRAL_BAND,
        accel_prev_thresh: float = DEFAULT_ACCEL_PREV_THRESH,
        accel_bull_thresh: float = DEFAULT_ACCEL_BULL_THRESH,
        accel_bear_thresh: float = DEFAULT_ACCEL_BEAR_THRESH,
        accel_entry_time: str = DEFAULT_ACCEL_ENTRY_TIME,
        accel_exit_time: str = DEFAULT_ACCEL_EXIT_TIME,
        combo_prev_thresh: float = DEFAULT_COMBO_PREV_THRESH,
        combo_bull_thresh: float = DEFAULT_COMBO_BULL_THRESH,
        combo_bear_thresh: float = DEFAULT_COMBO_BEAR_THRESH,
        combo_entry_time: str = DEFAULT_COMBO_ENTRY_TIME,
        combo_exit_time: str = DEFAULT_COMBO_EXIT_TIME,
        combo_stop_time: str = DEFAULT_COMBO_STOP_TIME,
        combo_stop_delta: float = DEFAULT_COMBO_STOP_DELTA,
        combo_xasset_min: int = DEFAULT_COMBO_XASSET_MIN,
        # Triple filter
        triple_prev_thresh: float = DEFAULT_TRIPLE_PREV_THRESH,
        triple_bull_thresh: float = DEFAULT_TRIPLE_BULL_THRESH,
        triple_bear_thresh: float = DEFAULT_TRIPLE_BEAR_THRESH,
        triple_entry_time: str = DEFAULT_TRIPLE_ENTRY_TIME,
        triple_exit_time: str = DEFAULT_TRIPLE_EXIT_TIME,
        triple_xasset_min: int = DEFAULT_TRIPLE_XASSET_MIN,
        triple_pm0_bull_min: float = DEFAULT_TRIPLE_PM0_BULL_MIN,
        triple_pm0_bear_max: float = DEFAULT_TRIPLE_PM0_BEAR_MAX,
    ):
        """Initialize the simulated trader.

        Args:
            trading_db: Database for trading state and trades
            asset_databases: Optional dict of per-asset databases
            initial_bankroll: Starting bankroll (default: $1000)
            base_bet: Base bet size (default: $25)
            fee_rate: Fee rate on profits (default: 0.02 = 2%)
            spread_cost: Spread cost on all trades (default: 0.006 = 0.6%)
            bull_threshold: YES price threshold for bull signal in single mode (default: 0.80)
            bear_threshold: YES price threshold for bear signal in single mode (default: 0.20)
            max_bet_pct: Maximum bet as % of bankroll (default: 0.05 = 5%)
            pause_windows_after_loss: Skip N windows after any loss (default: 2)
            growth_per_win: Bet growth per consecutive win (default: 0.10 = 10%)
            max_bet_multiplier: Maximum bet multiplier on base (default: 2.0)
            min_trajectory: Min PM price movement from t=0 to entry (default: 0.20)
            entry_mode: "two_stage" or "single" (default: "two_stage")
            signal_threshold_bull: t=7.5 bull signal threshold (default: 0.70)
            signal_threshold_bear: t=7.5 bear signal threshold (default: 0.30)
            confirm_threshold_bull: t=10 bull confirm threshold (default: 0.85)
            confirm_threshold_bear: t=10 bear confirm threshold (default: 0.15)
        """
        self.trading_db = trading_db
        self.asset_databases = asset_databases or {}

        # Configuration
        self.initial_bankroll = initial_bankroll
        self.base_bet = base_bet
        self.fee_rate = fee_rate
        self.spread_cost = spread_cost
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.max_bet_pct = max_bet_pct
        self.pause_windows_after_loss = pause_windows_after_loss
        self.growth_per_win = growth_per_win
        self.max_bet_multiplier = max_bet_multiplier
        self.min_trajectory = min_trajectory

        # Two-stage configuration
        self.entry_mode = entry_mode
        self.signal_threshold_bull = signal_threshold_bull
        self.signal_threshold_bear = signal_threshold_bear
        self.confirm_threshold_bull = confirm_threshold_bull
        self.confirm_threshold_bear = confirm_threshold_bear

        # Contrarian configuration
        self.contrarian_prev_thresh = contrarian_prev_thresh
        self.contrarian_bull_thresh = contrarian_bull_thresh
        self.contrarian_bear_thresh = contrarian_bear_thresh
        self.contrarian_entry_time = contrarian_entry_time
        self.contrarian_exit_time = contrarian_exit_time

        # Consensus configuration
        self.consensus_min_agree = consensus_min_agree
        self.consensus_entry_time = consensus_entry_time
        self.consensus_exit_time = consensus_exit_time

        # Accel_dbl configuration
        self.accel_neutral_band = accel_neutral_band
        self.accel_prev_thresh = accel_prev_thresh
        self.accel_bull_thresh = accel_bull_thresh
        self.accel_bear_thresh = accel_bear_thresh
        self.accel_entry_time = accel_entry_time
        self.accel_exit_time = accel_exit_time

        # Combo_dbl configuration
        self.combo_prev_thresh = combo_prev_thresh
        self.combo_bull_thresh = combo_bull_thresh
        self.combo_bear_thresh = combo_bear_thresh
        self.combo_entry_time = combo_entry_time
        self.combo_exit_time = combo_exit_time
        self.combo_stop_time = combo_stop_time
        self.combo_stop_delta = combo_stop_delta
        self.combo_xasset_min = combo_xasset_min

        # Triple filter configuration
        self.triple_prev_thresh = triple_prev_thresh
        self.triple_bull_thresh = triple_bull_thresh
        self.triple_bear_thresh = triple_bear_thresh
        self.triple_entry_time = triple_entry_time
        self.triple_exit_time = triple_exit_time
        self.triple_xasset_min = triple_xasset_min
        self.triple_pm0_bull_min = triple_pm0_bull_min
        self.triple_pm0_bear_max = triple_pm0_bear_max

        # Map time labels to t_minutes values for contrarian
        self._time_to_minutes = {
            "t0": 0.0, "t2.5": 2.5, "t5": 5.0,
            "t7.5": 7.5, "t10": 10.0, "t12.5": 12.5,
        }

        # Bet sizing - use SlowGrowthSizer instead of AntiMartingale
        self.sizer = SlowGrowthSizer(
            base_bet=base_bet,
            growth_per_win=growth_per_win,
            max_multiplier=max_bet_multiplier,
            max_bet_pct=max_bet_pct,
        )

        # State (loaded from DB or initialized)
        self.state: Optional[TradingState] = None

        # Track open positions per asset
        self.open_trades: Dict[str, SimulatedTrade] = {}

        # Pending signals for two-stage entry (per-asset)
        # Maps asset -> {"direction": "bull"/"bear", "signal_price": float, "window_id": str}
        self._pending_signals: Dict[str, dict] = {}

        # Per-asset statistics cache
        self._asset_stats: Dict[str, PerAssetStats] = {}

        # Last signals for dashboard display
        self._last_signals: Dict[str, Optional[str]] = {}

        # Contrarian: track previous window's pm_yes at t=12.5 per asset
        # This is updated each time a window completes
        self._prev_window_pm: Dict[str, float] = {}

        # Double contrarian: track the window BEFORE prev (prev2)
        # Updated by shifting _prev_window_pm → _prev2_window_pm on window complete
        self._prev2_window_pm: Dict[str, float] = {}

        # Consensus: buffer samples at consensus entry time to evaluate cross-asset
        # Maps time_key (e.g. "20260206_0445") -> {asset: {"pm_yes": float, "sample": sample, "state": state}}
        # Uses time-based key (not asset-specific window_id) so all 4 assets group together
        self._consensus_buffer: Dict[str, Dict[str, dict]] = {}
        # Track which time_keys we've already evaluated consensus for
        self._consensus_evaluated: set = set()

        # Accel_dbl: store t0 pm_yes for acceleration check
        # Maps asset -> pm_yes at t0 of current window
        self._accel_t0_pm: Dict[str, float] = {}

        # Combo_dbl: buffer for cross-asset evaluation at entry time
        # Maps time_key -> {asset: {"pm_yes": float, "sample": sample, "state": state}}
        self._combo_buffer: Dict[str, Dict[str, dict]] = {}
        self._combo_evaluated: set = set()

        # Combo_dbl: store entry pm_yes for stop-loss delta check
        # Maps asset -> pm_yes at entry time
        self._combo_entry_pm: Dict[str, float] = {}

        # Triple filter: buffer for cross-asset evaluation at entry time
        # Maps time_key -> {asset: {"pm_yes": float, "sample": sample, "state": state}}
        self._triple_buffer: Dict[str, Dict[str, dict]] = {}
        self._triple_evaluated: set = set()
        # Triple filter: store t0 pm_yes for PM0 confirmation check
        self._triple_t0_pm: Dict[str, float] = {}

    async def initialize(self):
        """Load or initialize trading state from database."""
        logger.info("Initializing simulated trader...")

        # Try to load existing state
        self.state = await self.trading_db.load_state()

        if self.state is None:
            # First run - create initial state
            logger.info(f"Creating new trading state with bankroll=${self.initial_bankroll:.2f}")
            self.state = TradingState(
                current_bankroll=self.initial_bankroll,
                current_bet_size=self.base_bet,
                initial_bankroll=self.initial_bankroll,
                base_bet_size=self.base_bet,
                peak_bankroll=self.initial_bankroll,
            )
            await self.trading_db.save_state(self.state)
        else:
            # Recalculate bet size from win streak (handles sizer changes)
            old_bet = self.state.current_bet_size
            if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
                # Fixed sizing for contrarian-family modes
                self.state.current_bet_size = self.base_bet
            else:
                self.state.current_bet_size = self.sizer.calculate_bet_for_streak(
                    self.state.current_win_streak,
                    self.state.current_bankroll,
                )
            if abs(old_bet - self.state.current_bet_size) > 0.01:
                logger.info(
                    f"Recalculated bet size: ${old_bet:.2f} -> ${self.state.current_bet_size:.2f} "
                    f"(streak={self.state.current_win_streak})"
                )

            logger.info(
                f"Recovered trading state: "
                f"Bankroll=${self.state.current_bankroll:.2f}, "
                f"Trades={self.state.total_trades}, "
                f"Win Rate={self.state.win_rate:.1%}, "
                f"P&L=${self.state.total_pnl:.2f}"
            )

        # Load any pending trades (for restart recovery)
        pending = await self.trading_db.get_pending_trades()

        if pending:
            logger.info(f"Found {len(pending)} pending trade(s), checking for resolution...")

            for trade in pending:
                # Check if the window already resolved while we were down
                resolved = await self._try_resolve_pending_trade(trade)

                if not resolved:
                    # Window hasn't resolved yet - keep as open trade
                    self.open_trades[trade.asset] = trade
                    logger.info(
                        f"Recovered pending trade: {trade.asset} {trade.direction} "
                        f"at ${trade.entry_price:.3f} (window still open)"
                    )

            resolved_count = len(pending) - len(self.open_trades)
            if resolved_count > 0:
                logger.info(f"Resolved {resolved_count} trade(s) that completed while offline")
            if self.open_trades:
                logger.info(f"Keeping {len(self.open_trades)} trade(s) as open positions")

    async def on_sample_at_signal(self, asset: str, sample, state):
        """Handle sample at t=7.5 - check for initial signal (two-stage) or direct entry (single).

        In two_stage mode: checks for initial signal and stores as pending.
        In single mode: checks for signal and enters trade immediately.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            sample: The sample collected at t=7.5
            state: The asset's current sampler state
        """
        if self.entry_mode == "two_stage":
            await self._check_signal_two_stage(asset, sample, state)
        else:
            await self._enter_single_mode(asset, sample, state)

    async def on_sample_at_confirm(self, asset: str, sample, state):
        """Handle sample at t=10 - confirm pending signal and enter trade (two-stage only).

        Called at t=10 minutes. Checks if there's a pending signal for this asset
        and whether the PM price still confirms the direction.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            sample: The sample collected at t=10
            state: The asset's current sampler state
        """
        if self.entry_mode != "two_stage":
            return

        try:
            # Check if we have a pending signal for this asset
            if asset not in self._pending_signals:
                return

            pending = self._pending_signals[asset]
            direction = pending["direction"]
            signal_price = pending["signal_price"]

            # Don't open if we already have a position for this asset
            if asset in self.open_trades:
                logger.debug(f"[{asset}] Already have open position, clearing pending signal")
                del self._pending_signals[asset]
                return

            pm_yes = sample.pm_yes_price

            if pm_yes is None:
                logger.info(f"[{asset}] No pm_yes_price at t=10, dropping pending {direction.upper()} signal")
                self._last_signals[asset] = f"{direction}-no-data"
                del self._pending_signals[asset]
                return

            # Check confirmation thresholds
            confirmed = False
            entry_price = pm_yes

            if direction == "bull" and pm_yes >= self.confirm_threshold_bull:
                confirmed = True
                logger.info(
                    f"[{asset}] BULL CONFIRMED at t=10: pm_yes={pm_yes:.3f} >= {self.confirm_threshold_bull} "
                    f"(signal was {signal_price:.3f} at t=7.5)"
                )
            elif direction == "bear" and pm_yes <= self.confirm_threshold_bear:
                confirmed = True
                entry_price = 1 - pm_yes  # NO price
                logger.info(
                    f"[{asset}] BEAR CONFIRMED at t=10: pm_yes={pm_yes:.3f} <= {self.confirm_threshold_bear} "
                    f"(signal was {signal_price:.3f} at t=7.5)"
                )
            else:
                # Signal faded - this is exactly what we're filtering
                logger.info(
                    f"[{asset}] FADER FILTERED: {direction.upper()} signal at t=7.5 ({signal_price:.3f}) "
                    f"NOT confirmed at t=10 (pm_yes={pm_yes:.3f}, "
                    f"needed {'>='+str(self.confirm_threshold_bull) if direction == 'bull' else '<='+str(self.confirm_threshold_bear)})"
                )
                self._last_signals[asset] = f"{direction}-faded"
                del self._pending_signals[asset]
                return

            if not confirmed:
                del self._pending_signals[asset]
                return

            # Confirmed! Enter trade at t=10 price
            self._last_signals[asset] = direction

            # Fixed bet size
            bet_size = min(self.base_bet, self.state.current_bankroll * self.max_bet_pct)

            # Compute signal metadata
            _spot_vel = sample.spot_price_change_from_open
            _pm_t0 = None
            for s in state.samples:
                if s.t_minutes == 0.0:
                    _pm_t0 = s.pm_yes_price
                    break
            _pm_mom = abs(pm_yes - _pm_t0) if _pm_t0 is not None else None

            # Create trade
            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="two_stage",
                prev_pm=self._prev_window_pm.get(asset),
                prev2_pm=self._prev2_window_pm.get(asset),
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            # Store in database
            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade

            # Clean up pending signal
            del self._pending_signals[asset]

            # Save state
            await self.trading_db.save_state(self.state)

            logger.info(
                f"[{asset}] OPENED {direction.upper()} position (two-stage confirmed): "
                f"entry={entry_price:.3f}, bet=${bet_size:.2f}, window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in trade confirmation: {e}", exc_info=True)

    async def _check_signal_two_stage(self, asset: str, sample, state):
        """Check for initial signal at t=7.5 in two-stage mode.

        Stores pending signal but does NOT enter trade yet.
        """
        try:
            # Don't check if we already have a position for this asset
            if asset in self.open_trades:
                logger.debug(f"[{asset}] Already have open position, skipping signal check")
                return

            # Check if we're in a pause period after a loss
            if self.state.pause_windows_remaining > 0:
                logger.info(
                    f"[{asset}] PAUSED: Skipping signal ({self.state.pause_windows_remaining} windows remaining after loss)"
                )
                self._last_signals[asset] = None
                return

            pm_yes = sample.pm_yes_price

            if pm_yes is None:
                logger.debug(f"[{asset}] No pm_yes_price data at t=7.5")
                self._last_signals[asset] = None
                return

            # Check for initial signal using signal thresholds (wider than confirm)
            direction = None

            if pm_yes >= self.signal_threshold_bull:
                direction = "bull"
                logger.info(
                    f"[{asset}] BULL SIGNAL at t=7.5: pm_yes={pm_yes:.3f} >= {self.signal_threshold_bull} "
                    f"(awaiting confirmation at t=10 >= {self.confirm_threshold_bull})"
                )
            elif pm_yes <= self.signal_threshold_bear:
                direction = "bear"
                logger.info(
                    f"[{asset}] BEAR SIGNAL at t=7.5: pm_yes={pm_yes:.3f} <= {self.signal_threshold_bear} "
                    f"(awaiting confirmation at t=10 <= {self.confirm_threshold_bear})"
                )
            else:
                logger.debug(
                    f"[{asset}] No signal at t=7.5: pm_yes={pm_yes:.3f} "
                    f"(not >= {self.signal_threshold_bull} or <= {self.signal_threshold_bear})"
                )
                self._last_signals[asset] = None
                # Clear any stale pending signal
                self._pending_signals.pop(asset, None)
                return

            # Store pending signal for confirmation at t=10
            self._pending_signals[asset] = {
                "direction": direction,
                "signal_price": pm_yes,
                "window_id": sample.window_id,
            }
            self._last_signals[asset] = f"{direction}-pending"

        except Exception as e:
            logger.error(f"[{asset}] Error in signal check: {e}", exc_info=True)

    async def _enter_single_mode(self, asset: str, sample, state):
        """Enter trade immediately at t=7.5 (single/legacy mode)."""
        try:
            # Don't open if we already have a position for this asset
            if asset in self.open_trades:
                logger.debug(f"[{asset}] Already have open position, skipping entry check")
                return

            # Check if we're in a pause period after a loss
            if self.state.pause_windows_remaining > 0:
                logger.info(
                    f"[{asset}] PAUSED: Skipping entry ({self.state.pause_windows_remaining} windows remaining after loss)"
                )
                self._last_signals[asset] = None
                return

            pm_yes = sample.pm_yes_price

            if pm_yes is None:
                logger.debug(f"[{asset}] No pm_yes_price data at t=7.5")
                self._last_signals[asset] = None
                return

            # Check for signal
            direction = None
            entry_price = pm_yes

            if pm_yes >= self.bull_threshold:
                direction = "bull"
                logger.info(
                    f"[{asset}] BULL SIGNAL at t=7.5: pm_yes={pm_yes:.3f} >= {self.bull_threshold}"
                )
            elif pm_yes <= self.bear_threshold:
                direction = "bear"
                entry_price = 1 - pm_yes  # NO price
                logger.info(
                    f"[{asset}] BEAR SIGNAL at t=7.5: pm_yes={pm_yes:.3f} <= {self.bear_threshold}"
                )
            else:
                logger.debug(
                    f"[{asset}] No signal at t=7.5: pm_yes={pm_yes:.3f} "
                    f"(not >= {self.bull_threshold} or <= {self.bear_threshold})"
                )
                self._last_signals[asset] = None
                return

            # Trajectory filter: check PM price movement from t=0 to now
            if self.min_trajectory > 0:
                samples_by_t = {s.t_minutes: s for s in state.samples}
                t0_sample = samples_by_t.get(0.0)

                if t0_sample and t0_sample.pm_yes_price is not None:
                    trajectory = abs(pm_yes - t0_sample.pm_yes_price)
                    if trajectory < self.min_trajectory:
                        logger.info(
                            f"[{asset}] SKIPPED {direction.upper()}: trajectory {trajectory:.3f} "
                            f"< {self.min_trajectory} (t0={t0_sample.pm_yes_price:.3f}, t7.5={pm_yes:.3f})"
                        )
                        self._last_signals[asset] = f"{direction}-filtered"
                        return

            # Store last signal for dashboard
            self._last_signals[asset] = direction

            # Fixed bet size
            bet_size = min(self.base_bet, self.state.current_bankroll * self.max_bet_pct)

            # Compute signal metadata
            _spot_vel = sample.spot_price_change_from_open
            _pm_t0 = None
            for s in state.samples:
                if s.t_minutes == 0.0:
                    _pm_t0 = s.pm_yes_price
                    break
            _pm_mom = abs(pm_yes - _pm_t0) if _pm_t0 is not None else None

            # Create trade
            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="single",
                prev_pm=self._prev_window_pm.get(asset),
                prev2_pm=self._prev2_window_pm.get(asset),
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            # Store in database
            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade

            # Save state
            await self.trading_db.save_state(self.state)

            logger.info(
                f"[{asset}] OPENED {direction.upper()} position: "
                f"entry={entry_price:.3f}, bet=${bet_size:.2f}, window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in trade entry: {e}", exc_info=True)

    # =========================================================================
    # Contrarian Strategy Methods
    # =========================================================================

    async def on_sample_at_contrarian_entry(self, asset: str, sample, state):
        """Handle sample at contrarian entry time (default t=0).

        Contrarian strategy:
        1. Check if previous window was strong (pm@t12.5 >= thresh or <= 1-thresh)
        2. Check if current sample confirms reversal direction
        3. Enter trade if both conditions met

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            sample: The sample collected at entry time
            state: The asset's current sampler state
        """
        try:
            # Don't open if we already have a position for this asset
            if asset in self.open_trades:
                logger.debug(f"[{asset}] Already have open position, skipping contrarian entry")
                return

            # Check pause period
            if self.state.pause_windows_remaining > 0:
                logger.info(
                    f"[{asset}] PAUSED: Skipping contrarian entry "
                    f"({self.state.pause_windows_remaining} windows remaining)"
                )
                self._last_signals[asset] = None
                return

            pm_yes = sample.pm_yes_price
            if pm_yes is None:
                logger.debug(f"[{asset}] No pm_yes_price at contrarian entry time")
                self._last_signals[asset] = None
                return

            # Check previous window state
            prev_pm = self._prev_window_pm.get(asset)
            if prev_pm is None:
                logger.debug(f"[{asset}] No previous window data yet, skipping contrarian")
                self._last_signals[asset] = None
                return

            # Determine contrarian direction
            direction = None

            # After strong UP (prev pm >= thresh) → expect reversal DOWN → enter BEAR
            if prev_pm >= self.contrarian_prev_thresh:
                if pm_yes <= self.contrarian_bear_thresh:
                    direction = "bear"
                    logger.info(
                        f"[{asset}] CONTRARIAN BEAR: prev_pm={prev_pm:.3f} >= {self.contrarian_prev_thresh} "
                        f"(strong UP), current pm={pm_yes:.3f} <= {self.contrarian_bear_thresh} → reversal"
                    )
                else:
                    logger.debug(
                        f"[{asset}] Contrarian: prev strong UP (pm={prev_pm:.3f}) but current "
                        f"pm={pm_yes:.3f} not <= {self.contrarian_bear_thresh}, no entry"
                    )
                    self._last_signals[asset] = "bear-no-confirm"
                    return

            # After strong DOWN (prev pm <= 1-thresh) → expect reversal UP → enter BULL
            elif prev_pm <= (1.0 - self.contrarian_prev_thresh):
                if pm_yes >= self.contrarian_bull_thresh:
                    direction = "bull"
                    logger.info(
                        f"[{asset}] CONTRARIAN BULL: prev_pm={prev_pm:.3f} <= {1.0-self.contrarian_prev_thresh:.2f} "
                        f"(strong DOWN), current pm={pm_yes:.3f} >= {self.contrarian_bull_thresh} → reversal"
                    )
                else:
                    logger.debug(
                        f"[{asset}] Contrarian: prev strong DOWN (pm={prev_pm:.3f}) but current "
                        f"pm={pm_yes:.3f} not >= {self.contrarian_bull_thresh}, no entry"
                    )
                    self._last_signals[asset] = "bull-no-confirm"
                    return
            else:
                # Previous window wasn't strong enough
                logger.debug(
                    f"[{asset}] Contrarian: prev_pm={prev_pm:.3f} not strong enough "
                    f"(need >= {self.contrarian_prev_thresh} or <= {1.0-self.contrarian_prev_thresh:.2f})"
                )
                self._last_signals[asset] = None
                return

            if direction is None:
                return

            self._last_signals[asset] = f"{direction}-contrarian"

            # Calculate entry contract price
            if direction == "bull":
                entry_price = pm_yes  # buying YES
            else:
                entry_price = 1.0 - pm_yes  # buying NO

            # Fixed bet size
            bet_size = min(self.base_bet, self.state.current_bankroll * self.max_bet_pct)

            # Compute signal metadata
            _spot_vel = sample.spot_price_change_from_open
            _pm_t0 = None
            for s in state.samples:
                if s.t_minutes == 0.0:
                    _pm_t0 = s.pm_yes_price
                    break
            _pm_mom = abs(pm_yes - _pm_t0) if _pm_t0 is not None else None

            # Create trade
            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="contrarian",
                prev_pm=prev_pm,
                prev2_pm=self._prev2_window_pm.get(asset),
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            # Store in database
            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade

            # Save state
            await self.trading_db.save_state(self.state)

            logger.info(
                f"[{asset}] OPENED CONTRARIAN {direction.upper()} position: "
                f"entry_contract={entry_price:.3f} (pm_yes={pm_yes:.3f}), "
                f"bet=${bet_size:.2f}, window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in contrarian entry: {e}", exc_info=True)

    async def on_sample_at_contrarian_exit(self, asset: str, sample, state):
        """Handle sample at contrarian exit time (default t=12.5).

        Sell the position at current PM price. P&L is the price difference
        minus double fees (entry + exit).

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            sample: The sample collected at exit time
            state: The asset's current sampler state
        """
        try:
            if asset not in self.open_trades:
                return

            trade = self.open_trades[asset]
            pm_yes = sample.pm_yes_price

            if pm_yes is None:
                logger.warning(f"[{asset}] No pm_yes at exit time, cannot close position")
                return

            # Calculate exit contract price
            if trade.direction == "bull":
                exit_contract = pm_yes
                entry_contract = trade.entry_price  # was pm_yes at entry
            else:
                exit_contract = 1.0 - pm_yes
                entry_contract = trade.entry_price  # was (1 - pm_yes) at entry

            # Early exit P&L: sell position before resolution
            # n_contracts = bet_size / entry_contract
            # gross = n_contracts * (exit_contract - entry_contract)
            if entry_contract <= 0.001:
                logger.warning(f"[{asset}] Entry contract price too low, skipping exit")
                return

            n_contracts = trade.bet_size / entry_contract
            gross_pnl = n_contracts * (exit_contract - entry_contract)

            # Double fees: entry + exit
            entry_fee = entry_contract * n_contracts * self.fee_rate
            exit_fee = exit_contract * n_contracts * self.fee_rate
            entry_spread = self.spread_cost * trade.bet_size
            exit_spread = self.spread_cost * (n_contracts * exit_contract)
            total_fees = entry_fee + exit_fee + entry_spread + exit_spread

            net_pnl = gross_pnl - total_fees

            # Update portfolio state
            self.state.current_bankroll += net_pnl
            self.state.total_pnl += net_pnl
            self.state.total_trades += 1

            won = net_pnl > 0

            if won:
                self.state.total_wins += 1
                self.state.current_win_streak += 1
                self.state.current_loss_streak = 0
                self.state.max_win_streak = max(
                    self.state.max_win_streak,
                    self.state.current_win_streak,
                )
            else:
                self.state.total_losses += 1
                self.state.current_loss_streak += 1
                self.state.current_win_streak = 0
                self.state.max_loss_streak = max(
                    self.state.max_loss_streak,
                    self.state.current_loss_streak,
                )
                # Trigger pause after loss
                if self.pause_windows_after_loss > 0:
                    self.state.pause_windows_remaining = self.pause_windows_after_loss
                    logger.info(
                        f"[{asset}] LOSS DETECTED: Pausing for {self.pause_windows_after_loss} windows"
                    )

            # Update bet size (fixed for contrarian-family, dynamic for others)
            if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
                self.state.current_bet_size = self.base_bet
            else:
                self.state.current_bet_size = self.sizer.calculate_bet_for_streak(
                    self.state.current_win_streak,
                    self.state.current_bankroll,
                )

            # Update drawdown tracking
            self.state.peak_bankroll = max(
                self.state.peak_bankroll,
                self.state.current_bankroll,
            )
            drawdown = self.state.current_bankroll - self.state.peak_bankroll
            drawdown_pct = (
                (drawdown / self.state.peak_bankroll * 100)
                if self.state.peak_bankroll > 0
                else 0
            )
            if drawdown < self.state.max_drawdown:
                self.state.max_drawdown = drawdown
            if drawdown_pct < self.state.max_drawdown_pct:
                self.state.max_drawdown_pct = drawdown_pct

            # Update trade record
            trade.exit_time = sample.sample_time_utc
            trade.exit_price = pm_yes  # store the pm_yes at exit for reference
            trade.outcome = "win" if won else "loss"
            trade.gross_pnl = gross_pnl
            trade.fee_paid = entry_fee + exit_fee
            trade.spread_cost = entry_spread + exit_spread
            trade.net_pnl = net_pnl
            trade.bankroll_after = self.state.current_bankroll
            trade.drawdown = drawdown
            trade.drawdown_pct = drawdown_pct
            trade.resolved_at = datetime.now(timezone.utc)

            # Save to database
            await self.trading_db.update_trade(trade)

            # Remove from open trades
            del self.open_trades[asset]

            # Log result
            result_str = "WIN" if won else "LOSS"
            pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
            logger.info(
                f"[{asset}] CONTRARIAN EXIT: {result_str} | "
                f"P&L: {pnl_str} (gross: ${gross_pnl:.2f}, fees: ${total_fees:.2f}) | "
                f"entry={entry_contract:.3f} exit={exit_contract:.3f} | "
                f"Bankroll: ${self.state.current_bankroll:.2f} | "
                f"Win Rate: {self.state.win_rate:.1%}"
            )

            # Save state
            await self.trading_db.save_state(self.state)

        except Exception as e:
            logger.error(f"[{asset}] Error in contrarian exit: {e}", exc_info=True)

    # =========================================================================
    # Contrarian + Consensus Strategy Methods
    # =========================================================================

    async def on_sample_at_consensus_entry(self, asset: str, sample, state):
        """Buffer sample at consensus entry time. Once all assets arrive, evaluate consensus.

        Contrarian+Consensus strategy (two-phase cross-asset filter):
        Phase 1 - Previous window consensus: Count assets with strong prev pm@t12.5
                  Need >= min_agree assets with prev_pm >= thresh (strong UP) or <= 1-thresh (strong DOWN)
        Phase 2 - Current window consensus: Count assets confirming reversal direction
                  Need >= min_agree assets with current pm confirming the reversal
        Only trade assets that individually pass Phase 2 confirmation.

        Assets are always buffered even if pm_yes is None, so a single failed
        API call doesn't block consensus evaluation for the entire window.
        """
        try:
            pm_yes = sample.pm_yes_price

            # Buffer this asset's data for the current window
            # Always buffer — even None pm_yes — so we don't block waiting for it
            #
            # IMPORTANT: Use a time-based key, NOT sample.window_id.
            # window_id is asset-specific (e.g., "BTC_20260206_0445") so each asset
            # would go into a separate bucket and consensus would never fire.
            # Extract the time portion: "YYYYMMDD_HHMM" to group all assets together.
            time_key = "_".join(sample.window_id.split("_")[1:])

            if time_key not in self._consensus_buffer:
                self._consensus_buffer[time_key] = {}

            self._consensus_buffer[time_key][asset] = {
                "pm_yes": pm_yes,  # May be None — consensus eval handles this
                "sample": sample,
                "state": state,
            }

            if pm_yes is None:
                logger.debug(f"[{asset}] No pm_yes at consensus entry time (buffered as None)")
                self._last_signals[asset] = None

            # Check if all assets have reported for this window
            expected_assets = set(self.asset_databases.keys())
            arrived_assets = set(self._consensus_buffer[time_key].keys())

            if arrived_assets >= expected_assets and time_key not in self._consensus_evaluated:
                # All assets are in — evaluate consensus
                self._consensus_evaluated.add(time_key)
                await self._evaluate_consensus(time_key)

                # Clean up old buffers (keep only current window)
                old_keys = [
                    k for k in self._consensus_buffer
                    if k != time_key
                ]
                for k in old_keys:
                    del self._consensus_buffer[k]
                # Clean up old evaluated set
                self._consensus_evaluated -= set(old_keys)

        except Exception as e:
            logger.error(f"[{asset}] Error in consensus entry buffer: {e}", exc_info=True)

    async def _evaluate_consensus(self, time_key: str):
        """Evaluate cross-asset consensus for a window and enter trades.

        Phase 1: Count assets with strong previous window (contrarian setup)
        Phase 2: Count assets confirming reversal in current window
        Only trade if both phases pass min_agree threshold.

        Args:
            time_key: Time-based window key (e.g., "20260206_0445"), NOT asset-specific window_id.
        """
        buffer = self._consensus_buffer.get(time_key, {})
        if not buffer:
            return

        min_agree = self.consensus_min_agree
        thresh = self.contrarian_prev_thresh

        # Phase 1: Count assets with strong previous window
        prev_strong_up = 0   # prev pm >= thresh (strong UP → expect bear reversal)
        prev_strong_down = 0  # prev pm <= 1-thresh (strong DOWN → expect bull reversal)

        for asset, data in buffer.items():
            prev_pm = self._prev_window_pm.get(asset)
            if prev_pm is None:
                continue
            if prev_pm >= thresh:
                prev_strong_up += 1
            elif prev_pm <= (1.0 - thresh):
                prev_strong_down += 1

        # Determine consensus contrarian direction
        contrarian_dir = None
        if prev_strong_up >= min_agree:
            contrarian_dir = "bear"  # Strong UP → expect reversal DOWN
            logger.info(
                f"[CONSENSUS] Phase 1 PASS: {prev_strong_up}/{len(buffer)} assets had strong "
                f"prev UP (pm >= {thresh}) → expect BEAR reversal"
            )
        elif prev_strong_down >= min_agree:
            contrarian_dir = "bull"  # Strong DOWN → expect reversal UP
            logger.info(
                f"[CONSENSUS] Phase 1 PASS: {prev_strong_down}/{len(buffer)} assets had strong "
                f"prev DOWN (pm <= {1.0 - thresh:.2f}) → expect BULL reversal"
            )
        else:
            # Log prev PM values for visibility
            prev_vals = {
                a: f"{self._prev_window_pm.get(a, 'N/A')}"
                for a in buffer
            }
            logger.info(
                f"[CONSENSUS] Phase 1 FAIL: strong_up={prev_strong_up}, "
                f"strong_down={prev_strong_down}, need >= {min_agree}. "
                f"Prev PMs: {prev_vals}"
            )
            for asset in buffer:
                self._last_signals[asset] = None
            return

        # Phase 2: Count assets confirming reversal in CURRENT window
        confirming_assets = []
        for asset, data in buffer.items():
            pm_yes = data["pm_yes"]
            if pm_yes is None:
                continue  # Skip assets with no price data
            if contrarian_dir == "bear" and pm_yes <= self.contrarian_bear_thresh:
                confirming_assets.append(asset)
            elif contrarian_dir == "bull" and pm_yes >= self.contrarian_bull_thresh:
                confirming_assets.append(asset)

        if len(confirming_assets) < min_agree:
            logger.info(
                f"[CONSENSUS] Phase 2 FAIL: only {len(confirming_assets)}/{len(buffer)} assets "
                f"confirm {contrarian_dir.upper()} reversal (need >= {min_agree}). "
                f"Confirming: {confirming_assets}"
            )
            for asset in buffer:
                if asset in confirming_assets:
                    self._last_signals[asset] = f"{contrarian_dir}-no-consensus"
                else:
                    self._last_signals[asset] = f"{contrarian_dir}-no-confirm"
            return

        logger.info(
            f"[CONSENSUS] Phase 2 PASS: {len(confirming_assets)}/{len(buffer)} assets "
            f"confirm {contrarian_dir.upper()} reversal → ENTERING TRADES"
        )

        # Enter trades for all confirming assets
        for asset in confirming_assets:
            await self._enter_consensus_trade(
                asset, contrarian_dir, buffer[asset]
            )

        # Mark non-confirming assets
        for asset in buffer:
            if asset not in confirming_assets:
                self._last_signals[asset] = f"{contrarian_dir}-no-confirm"

    async def _enter_consensus_trade(
        self, asset: str, direction: str, data: dict
    ):
        """Enter a single consensus trade for an asset."""
        try:
            # Skip if already have open position or paused
            if asset in self.open_trades:
                logger.debug(f"[{asset}] Already have open position, skipping consensus entry")
                return

            if self.state.pause_windows_remaining > 0:
                logger.info(
                    f"[{asset}] PAUSED: Skipping consensus entry "
                    f"({self.state.pause_windows_remaining} windows remaining)"
                )
                self._last_signals[asset] = None
                return

            sample = data["sample"]
            pm_yes = data["pm_yes"]

            self._last_signals[asset] = f"{direction}-consensus"

            # Calculate entry contract price
            if direction == "bull":
                entry_price = pm_yes  # buying YES
            else:
                entry_price = 1.0 - pm_yes  # buying NO

            # Fixed bet size
            bet_size = min(
                self.base_bet,
                self.state.current_bankroll * self.max_bet_pct,
            )

            # Compute signal metadata
            _spot_vel = sample.spot_price_change_from_open
            _pm_mom = abs(pm_yes - 0.50)  # distance from neutral at entry

            # Create trade
            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="contrarian_consensus",
                prev_pm=self._prev_window_pm.get(asset),
                prev2_pm=self._prev2_window_pm.get(asset),
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            # Store in database
            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade

            # Save state
            await self.trading_db.save_state(self.state)

            logger.info(
                f"[{asset}] OPENED CONSENSUS {direction.upper()} position: "
                f"entry_contract={entry_price:.3f} (pm_yes={pm_yes:.3f}), "
                f"bet=${bet_size:.2f}, window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in consensus trade entry: {e}", exc_info=True)

    async def on_sample_at_consensus_exit(self, asset: str, sample, state):
        """Handle sample at consensus exit time — same P&L logic as contrarian exit."""
        # Reuse contrarian exit logic (identical early-exit P&L calculation)
        await self.on_sample_at_contrarian_exit(asset, sample, state)

    # =========================================================================
    # ACCEL_DBL Strategy Methods
    # Double contrarian + t0 near neutral (acceleration filter)
    # =========================================================================

    async def on_sample_at_accel_t0(self, asset: str, sample, state):
        """Record t0 pm_yes for acceleration filter check."""
        pm_yes = sample.pm_yes_price
        if pm_yes is not None:
            self._accel_t0_pm[asset] = pm_yes
        else:
            self._accel_t0_pm.pop(asset, None)

    async def on_sample_at_accel_entry(self, asset: str, sample, state):
        """Handle ACCEL_DBL entry at configured time (default t5).

        Requirements:
        1. prev2 AND prev1 must both be strong in same direction (double contrarian)
        2. t0 pm_yes must be near 0.50 (within accel_neutral_band)
        3. Current pm_yes must confirm reversal direction
        """
        try:
            if asset in self.open_trades:
                return

            if self.state.pause_windows_remaining > 0:
                self._last_signals[asset] = None
                return

            pm_yes = sample.pm_yes_price
            if pm_yes is None:
                self._last_signals[asset] = None
                return

            thresh = self.accel_prev_thresh

            # Check double contrarian: prev2 AND prev1 both strong same direction
            prev2 = self._prev2_window_pm.get(asset)
            prev1 = self._prev_window_pm.get(asset)

            if prev2 is None or prev1 is None:
                self._last_signals[asset] = None
                return

            # Determine direction from double strong prev
            direction = None
            if prev2 >= thresh and prev1 >= thresh:
                # Two consecutive strong UP → expect reversal DOWN → BEAR
                direction = "bear"
            elif prev2 <= (1.0 - thresh) and prev1 <= (1.0 - thresh):
                # Two consecutive strong DOWN → expect reversal UP → BULL
                direction = "bull"
            else:
                self._last_signals[asset] = None
                return

            # Check acceleration filter: t0 must be near neutral (0.50)
            t0_pm = self._accel_t0_pm.get(asset)
            if t0_pm is None:
                self._last_signals[asset] = "accel-no-t0"
                return

            if abs(t0_pm - 0.50) > self.accel_neutral_band:
                logger.debug(
                    f"[{asset}] ACCEL_DBL: t0={t0_pm:.3f} not near neutral "
                    f"(|{t0_pm:.3f} - 0.50| = {abs(t0_pm - 0.50):.3f} > {self.accel_neutral_band})"
                )
                self._last_signals[asset] = "accel-filtered"
                return

            # Check current pm confirms reversal
            if direction == "bear" and pm_yes > self.accel_bear_thresh:
                self._last_signals[asset] = "bear-no-confirm"
                return
            elif direction == "bull" and pm_yes < self.accel_bull_thresh:
                self._last_signals[asset] = "bull-no-confirm"
                return

            # All checks passed — enter trade
            self._last_signals[asset] = f"{direction}-accel"

            if direction == "bull":
                entry_price = pm_yes
            else:
                entry_price = 1.0 - pm_yes

            bet_size = min(self.base_bet, self.state.current_bankroll * self.max_bet_pct)

            # Signal metadata
            _spot_vel = sample.spot_price_change_from_open
            _pm_mom = abs(pm_yes - t0_pm) if t0_pm is not None else None

            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="accel_dbl",
                prev_pm=prev1,
                prev2_pm=prev2,
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade
            await self.trading_db.save_state(self.state)

            logger.info(
                f"[{asset}] ACCEL_DBL {direction.upper()}: "
                f"prev2={prev2:.3f} prev1={prev1:.3f} t0={t0_pm:.3f} "
                f"entry={entry_price:.3f} (pm={pm_yes:.3f}), "
                f"bet=${bet_size:.2f}, window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in accel_dbl entry: {e}", exc_info=True)

    async def on_sample_at_accel_exit(self, asset: str, sample, state):
        """Handle ACCEL_DBL exit — same early-exit P&L as contrarian."""
        await self.on_sample_at_contrarian_exit(asset, sample, state)

    # =========================================================================
    # COMBO_DBL Strategy Methods
    # Double contrarian + stop-loss at t7.5 + cross-asset filter
    # =========================================================================

    async def on_sample_at_combo_entry(self, asset: str, sample, state):
        """Buffer sample at combo entry time. Once all assets arrive, evaluate.

        Like consensus but checks for double-strong prev windows across assets.
        """
        try:
            pm_yes = sample.pm_yes_price
            time_key = "_".join(sample.window_id.split("_")[1:])

            if time_key not in self._combo_buffer:
                self._combo_buffer[time_key] = {}

            self._combo_buffer[time_key][asset] = {
                "pm_yes": pm_yes,
                "sample": sample,
                "state": state,
            }

            if pm_yes is None:
                self._last_signals[asset] = None

            expected_assets = set(self.asset_databases.keys())
            arrived_assets = set(self._combo_buffer[time_key].keys())

            if arrived_assets >= expected_assets and time_key not in self._combo_evaluated:
                self._combo_evaluated.add(time_key)
                await self._evaluate_combo(time_key)

                # Clean up old buffers
                old_keys = [k for k in self._combo_buffer if k != time_key]
                for k in old_keys:
                    del self._combo_buffer[k]
                self._combo_evaluated -= set(old_keys)

        except Exception as e:
            logger.error(f"[{asset}] Error in combo entry buffer: {e}", exc_info=True)

    async def _evaluate_combo(self, time_key: str):
        """Evaluate combo_dbl cross-asset filter and enter trades.

        Requirements:
        1. This asset has double-strong prev (prev2 AND prev1 both strong same direction)
        2. At least combo_xasset_min OTHER assets also have double-strong prev same direction
        3. Current pm confirms reversal
        """
        buffer = self._combo_buffer.get(time_key, {})
        if not buffer:
            return

        thresh = self.combo_prev_thresh

        # Count assets with double-strong prev in each direction
        dbl_strong_up = []   # assets with prev2 >= thresh AND prev1 >= thresh
        dbl_strong_down = []  # assets with prev2 <= 1-thresh AND prev1 <= 1-thresh

        for asset in buffer:
            prev2 = self._prev2_window_pm.get(asset)
            prev1 = self._prev_window_pm.get(asset)
            if prev2 is None or prev1 is None:
                continue
            if prev2 >= thresh and prev1 >= thresh:
                dbl_strong_up.append(asset)
            elif prev2 <= (1.0 - thresh) and prev1 <= (1.0 - thresh):
                dbl_strong_down.append(asset)

        # For each asset that qualifies, check if enough OTHER assets also qualify
        # Try bear direction (from double strong UP)
        for asset in list(dbl_strong_up):
            others_count = len(dbl_strong_up) - 1  # exclude self
            if others_count < self.combo_xasset_min:
                logger.debug(
                    f"[{asset}] COMBO: only {others_count} other assets double-strong UP "
                    f"(need {self.combo_xasset_min})"
                )
                self._last_signals[asset] = "combo-no-xasset"
                continue

            data = buffer[asset]
            pm_yes = data["pm_yes"]
            if pm_yes is None:
                self._last_signals[asset] = None
                continue

            if pm_yes <= self.combo_bear_thresh:
                await self._enter_combo_trade(asset, "bear", data)
            else:
                self._last_signals[asset] = "bear-no-confirm"

        # Try bull direction (from double strong DOWN)
        for asset in list(dbl_strong_down):
            others_count = len(dbl_strong_down) - 1
            if others_count < self.combo_xasset_min:
                logger.debug(
                    f"[{asset}] COMBO: only {others_count} other assets double-strong DOWN "
                    f"(need {self.combo_xasset_min})"
                )
                self._last_signals[asset] = "combo-no-xasset"
                continue

            data = buffer[asset]
            pm_yes = data["pm_yes"]
            if pm_yes is None:
                self._last_signals[asset] = None
                continue

            if pm_yes >= self.combo_bull_thresh:
                await self._enter_combo_trade(asset, "bull", data)
            else:
                self._last_signals[asset] = "bull-no-confirm"

        # Mark assets with no double-strong prev
        all_qualifying = set(dbl_strong_up) | set(dbl_strong_down)
        for asset in buffer:
            if asset not in all_qualifying:
                if self._last_signals.get(asset) is None:
                    self._last_signals[asset] = None

    async def _enter_combo_trade(self, asset: str, direction: str, data: dict):
        """Enter a single combo_dbl trade."""
        try:
            if asset in self.open_trades:
                return

            if self.state.pause_windows_remaining > 0:
                self._last_signals[asset] = None
                return

            sample = data["sample"]
            pm_yes = data["pm_yes"]

            self._last_signals[asset] = f"{direction}-combo"

            if direction == "bull":
                entry_price = pm_yes
            else:
                entry_price = 1.0 - pm_yes

            # Store entry pm_yes for stop-loss delta check
            self._combo_entry_pm[asset] = pm_yes

            bet_size = min(self.base_bet, self.state.current_bankroll * self.max_bet_pct)

            # Signal metadata
            prev2 = self._prev2_window_pm.get(asset, 0)
            prev1 = self._prev_window_pm.get(asset, 0)
            _spot_vel = sample.spot_price_change_from_open
            _pm_mom = abs(pm_yes - 0.50)

            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="combo_dbl",
                prev_pm=prev1,
                prev2_pm=prev2,
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade
            await self.trading_db.save_state(self.state)
            logger.info(
                f"[{asset}] COMBO_DBL {direction.upper()}: "
                f"prev2={prev2:.3f} prev1={prev1:.3f} "
                f"entry={entry_price:.3f} (pm={pm_yes:.3f}), "
                f"bet=${bet_size:.2f}, stop_delta={self.combo_stop_delta}, "
                f"window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in combo trade entry: {e}", exc_info=True)

    async def on_sample_at_combo_stop(self, asset: str, sample, state):
        """Handle combo_dbl stop-loss check at t7.5.

        If position has moved against us by combo_stop_delta, exit early.
        Otherwise, hold until normal exit time.
        """
        try:
            if asset not in self.open_trades:
                return

            trade = self.open_trades[asset]
            pm_yes = sample.pm_yes_price
            if pm_yes is None:
                return

            entry_pm = self._combo_entry_pm.get(asset)
            if entry_pm is None:
                return

            # Check if position has moved against us
            should_stop = False
            if trade.direction == "bull":
                # Bull position: bad if pm dropped (market says less likely UP)
                delta = entry_pm - pm_yes
                if delta >= self.combo_stop_delta:
                    should_stop = True
                    logger.info(
                        f"[{asset}] COMBO STOP-LOSS: BULL entry_pm={entry_pm:.3f} "
                        f"now={pm_yes:.3f}, delta={delta:.3f} >= {self.combo_stop_delta}"
                    )
            else:
                # Bear position: bad if pm rose (market says more likely UP)
                delta = pm_yes - entry_pm
                if delta >= self.combo_stop_delta:
                    should_stop = True
                    logger.info(
                        f"[{asset}] COMBO STOP-LOSS: BEAR entry_pm={entry_pm:.3f} "
                        f"now={pm_yes:.3f}, delta={delta:.3f} >= {self.combo_stop_delta}"
                    )

            if should_stop:
                # Exit early at t7.5 using contrarian exit logic
                await self.on_sample_at_contrarian_exit(asset, sample, state)
                # Clean up entry pm tracker
                self._combo_entry_pm.pop(asset, None)

        except Exception as e:
            logger.error(f"[{asset}] Error in combo stop-loss: {e}", exc_info=True)

    async def on_sample_at_combo_exit(self, asset: str, sample, state):
        """Handle combo_dbl normal exit — same as contrarian exit."""
        if asset in self.open_trades:
            await self.on_sample_at_contrarian_exit(asset, sample, state)
            self._combo_entry_pm.pop(asset, None)

    # =========================================================================
    # TRIPLE FILTER Strategy Methods
    # Double contrarian + cross-asset consensus (N dbl-strong) + PM t0 confirmation
    # =========================================================================

    async def on_sample_at_triple_t0(self, asset: str, sample, state):
        """Record t0 pm_yes for PM0 confirmation check."""
        pm_yes = sample.pm_yes_price
        if pm_yes is not None:
            self._triple_t0_pm[asset] = pm_yes
        else:
            self._triple_t0_pm.pop(asset, None)

    async def on_sample_at_triple_entry(self, asset: str, sample, state):
        """Buffer sample at triple entry time. Once all assets arrive, evaluate.

        Triple filter requires:
        1. This asset has double-strong prev (prev2 AND prev1 both strong)
        2. At least triple_xasset_min assets have double-strong prev same direction
        3. PM t0 confirms direction (pm0 <= bear_max for bear, >= bull_min for bull)
        4. Current pm confirms reversal (pm <= bear_thresh for bear, >= bull_thresh for bull)
        """
        try:
            pm_yes = sample.pm_yes_price
            time_key = "_".join(sample.window_id.split("_")[1:])

            if time_key not in self._triple_buffer:
                self._triple_buffer[time_key] = {}

            self._triple_buffer[time_key][asset] = {
                "pm_yes": pm_yes,
                "sample": sample,
                "state": state,
            }

            if pm_yes is None:
                self._last_signals[asset] = None

            expected_assets = set(self.asset_databases.keys())
            arrived_assets = set(self._triple_buffer[time_key].keys())

            if arrived_assets >= expected_assets and time_key not in self._triple_evaluated:
                self._triple_evaluated.add(time_key)
                await self._evaluate_triple(time_key)

                # Clean up old buffers
                old_keys = [k for k in self._triple_buffer if k != time_key]
                for k in old_keys:
                    del self._triple_buffer[k]
                self._triple_evaluated -= set(old_keys)

        except Exception as e:
            logger.error(f"[{asset}] Error in triple entry buffer: {e}", exc_info=True)

    async def _evaluate_triple(self, time_key: str):
        """Evaluate triple filter cross-asset and enter trades.

        Requirements for each traded asset:
        1. Double-strong prev: prev2 AND prev1 both >= thresh (UP) or <= 1-thresh (DOWN)
        2. Cross-asset consensus: at least triple_xasset_min assets have double-strong same dir
        3. PM t0 confirmation: pm0 <= triple_pm0_bear_max (bear) or >= triple_pm0_bull_min (bull)
        4. Current pm confirmation: pm <= triple_bear_thresh (bear) or >= triple_bull_thresh (bull)
        """
        buffer = self._triple_buffer.get(time_key, {})
        if not buffer:
            return

        thresh = self.triple_prev_thresh

        # Count assets with double-strong prev in each direction
        dbl_strong_up = []    # assets with prev2 >= thresh AND prev1 >= thresh
        dbl_strong_down = []  # assets with prev2 <= 1-thresh AND prev1 <= 1-thresh

        for asset in buffer:
            prev2 = self._prev2_window_pm.get(asset)
            prev1 = self._prev_window_pm.get(asset)
            if prev2 is None or prev1 is None:
                continue
            if prev2 >= thresh and prev1 >= thresh:
                dbl_strong_up.append(asset)
            elif prev2 <= (1.0 - thresh) and prev1 <= (1.0 - thresh):
                dbl_strong_down.append(asset)

        # Bear direction (from double strong UP)
        if len(dbl_strong_up) >= self.triple_xasset_min:
            logger.info(
                f"[TRIPLE] {len(dbl_strong_up)}/{len(buffer)} assets double-strong UP "
                f"(>= {self.triple_xasset_min}) → evaluating BEAR entries"
            )
            for asset in dbl_strong_up:
                data = buffer[asset]
                pm_yes = data["pm_yes"]
                if pm_yes is None:
                    self._last_signals[asset] = None
                    continue

                # PM t0 confirmation: for bear, pm0 should be <= bear_max
                t0_pm = self._triple_t0_pm.get(asset)
                if t0_pm is None:
                    self._last_signals[asset] = "triple-no-t0"
                    continue
                if t0_pm > self.triple_pm0_bear_max:
                    logger.debug(
                        f"[{asset}] TRIPLE: pm0={t0_pm:.3f} > {self.triple_pm0_bear_max} "
                        f"(PM0 not confirming bear)"
                    )
                    self._last_signals[asset] = "triple-pm0-fail"
                    continue

                # Current pm confirmation
                if pm_yes <= self.triple_bear_thresh:
                    await self._enter_triple_trade(asset, "bear", data)
                else:
                    self._last_signals[asset] = "bear-no-confirm"
        else:
            for asset in dbl_strong_up:
                self._last_signals[asset] = "triple-no-xasset"

        # Bull direction (from double strong DOWN)
        if len(dbl_strong_down) >= self.triple_xasset_min:
            logger.info(
                f"[TRIPLE] {len(dbl_strong_down)}/{len(buffer)} assets double-strong DOWN "
                f"(>= {self.triple_xasset_min}) → evaluating BULL entries"
            )
            for asset in dbl_strong_down:
                data = buffer[asset]
                pm_yes = data["pm_yes"]
                if pm_yes is None:
                    self._last_signals[asset] = None
                    continue

                # PM t0 confirmation: for bull, pm0 should be >= bull_min
                t0_pm = self._triple_t0_pm.get(asset)
                if t0_pm is None:
                    self._last_signals[asset] = "triple-no-t0"
                    continue
                if t0_pm < self.triple_pm0_bull_min:
                    logger.debug(
                        f"[{asset}] TRIPLE: pm0={t0_pm:.3f} < {self.triple_pm0_bull_min} "
                        f"(PM0 not confirming bull)"
                    )
                    self._last_signals[asset] = "triple-pm0-fail"
                    continue

                # Current pm confirmation
                if pm_yes >= self.triple_bull_thresh:
                    await self._enter_triple_trade(asset, "bull", data)
                else:
                    self._last_signals[asset] = "bull-no-confirm"
        else:
            for asset in dbl_strong_down:
                self._last_signals[asset] = "triple-no-xasset"

        # Mark assets with no double-strong prev
        all_qualifying = set(dbl_strong_up) | set(dbl_strong_down)
        for asset in buffer:
            if asset not in all_qualifying:
                if self._last_signals.get(asset) is None:
                    self._last_signals[asset] = None

    async def _enter_triple_trade(self, asset: str, direction: str, data: dict):
        """Enter a single triple_filter trade."""
        try:
            if asset in self.open_trades:
                return

            if self.state.pause_windows_remaining > 0:
                self._last_signals[asset] = None
                return

            sample = data["sample"]
            pm_yes = data["pm_yes"]

            self._last_signals[asset] = f"{direction}-triple"

            if direction == "bull":
                entry_price = pm_yes
            else:
                entry_price = 1.0 - pm_yes

            bet_size = min(self.base_bet, self.state.current_bankroll * self.max_bet_pct)

            # Signal metadata
            prev2 = self._prev2_window_pm.get(asset, 0)
            prev1 = self._prev_window_pm.get(asset, 0)
            t0_pm = self._triple_t0_pm.get(asset)
            _spot_vel = sample.spot_price_change_from_open
            _pm_mom = abs(pm_yes - t0_pm) if t0_pm is not None else abs(pm_yes - 0.50)

            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
                entry_mode="triple_filter",
                prev_pm=prev1,
                prev2_pm=prev2,
                spot_velocity=_spot_vel,
                pm_momentum=_pm_mom,
            )

            await self.trading_db.insert_trade(trade)
            self.open_trades[asset] = trade
            await self.trading_db.save_state(self.state)

            logger.info(
                f"[{asset}] TRIPLE {direction.upper()}: "
                f"prev2={prev2:.3f} prev1={prev1:.3f} pm0={t0_pm:.3f} "
                f"entry={entry_price:.3f} (pm={pm_yes:.3f}), "
                f"bet=${bet_size:.2f}, window={sample.window_id}"
            )

        except Exception as e:
            logger.error(f"[{asset}] Error in triple trade entry: {e}", exc_info=True)

    async def on_sample_at_triple_exit(self, asset: str, sample, state):
        """Handle triple_filter exit — same early-exit P&L as contrarian."""
        await self.on_sample_at_contrarian_exit(asset, sample, state)

    async def on_window_complete(self, asset: str, window: Window):
        """Handle window completion - resolve open trades and track previous window.

        This is called when a 15-minute window completes with a resolved outcome.
        For two_stage/single modes: resolve any open position for this asset.
        For contrarian mode: positions are already closed at t=12.5, so we just
        track the previous window state for next window's contrarian signal.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            window: Completed window with outcome
        """
        try:
            # For contrarian-family modes, track previous window's pm_yes at t=12.5
            # This is needed for the next window's entry decision
            if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
                pm_t12_5 = getattr(window, 'pm_yes_t12_5', None)

                if pm_t12_5 is None:
                    # Try to get from the asset database samples
                    if asset in self.asset_databases:
                        try:
                            samples = await self.asset_databases[asset].get_samples_for_window(
                                window.window_id, asset
                            )
                            for s in samples:
                                if hasattr(s, 't_minutes') and s.t_minutes == 12.5:
                                    if s.pm_yes_price is not None:
                                        pm_t12_5 = s.pm_yes_price
                                    break
                        except Exception as e:
                            logger.debug(f"[{asset}] Could not get t12.5 from samples: {e}")

                if pm_t12_5 is not None:
                    # Shift: current prev → prev2, then store new prev
                    if asset in self._prev_window_pm:
                        self._prev2_window_pm[asset] = self._prev_window_pm[asset]
                    self._prev_window_pm[asset] = pm_t12_5
                    logger.debug(
                        f"[{asset}] Stored prev window pm@t12.5={pm_t12_5:.3f}"
                        f" (prev2={self._prev2_window_pm.get(asset, 'N/A')})"
                    )

            # Resolve any open position for this asset (non-contrarian modes)
            # Contrarian-family trades are already closed at exit time
            if asset in self.open_trades:
                if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
                    # Trade wasn't closed at exit time (missed exit?)
                    # Fall back to binary resolution
                    logger.warning(
                        f"[{asset}] {self.entry_mode} trade still open at window complete! "
                        f"Resolving at binary outcome as fallback."
                    )
                await self._resolve_trade(asset, window)

            # Clear any stale pending signals for this asset (window is done)
            self._pending_signals.pop(asset, None)

            # Decrement pause counter if we're in a pause period
            # Only decrement once per window (use first asset alphabetically as trigger)
            # This prevents decrementing 4x for 4 assets in same window
            if self.state.pause_windows_remaining > 0 and asset == "BTC":
                self.state.pause_windows_remaining -= 1
                if self.state.pause_windows_remaining > 0:
                    logger.info(
                        f"Pause period: {self.state.pause_windows_remaining} windows remaining"
                    )
                else:
                    logger.info("Pause period ended - resuming trading")

            # Save state
            self.state.last_window_id = window.window_id
            self.state.timestamp = datetime.now(timezone.utc)
            await self.trading_db.save_state(self.state)

        except Exception as e:
            logger.error(f"[{asset}] Error in trading callback: {e}", exc_info=True)

    async def _resolve_trade(self, asset: str, window: Window):
        """Resolve an open trade with window outcome.

        P&L calculation matches backtest_suite.py:
        - Bull (BUY YES): Win if outcome='up', Lose if outcome='down'
        - Bear (BUY NO): Win if outcome='down', Lose if outcome='up'

        Payouts:
        - Bull win: profit = (1 - entry_price) * bet_size
        - Bull loss: loss = entry_price * bet_size
        - Bear win: profit = entry_price * bet_size (original YES price)
        - Bear loss: loss = (1 - entry_price) * bet_size

        Fees (matching actual Polymarket):
        - 0.1% (10 bps) taker fee on contract premium (all trades)
        - ~0.5% spread/slippage cost estimate (all trades)

        Args:
            asset: Asset symbol
            window: Completed window with outcome
        """
        trade = self.open_trades[asset]
        actual_outcome = window.outcome  # 'up' or 'down'

        if actual_outcome is None:
            logger.warning(f"[{asset}] Window has no outcome, cannot resolve trade")
            return

        # Determine if trade won
        if trade.direction == "bull":
            won = actual_outcome == "up"
        else:  # bear
            won = actual_outcome == "down"

        # Calculate P&L
        # Polymarket fee: 0.1% (10 bps) on total contract premium (all trades)
        # Spread: estimated market impact/slippage (all trades)

        # Contract premium = entry_price * bet_size (what we paid for contracts)
        contract_premium = trade.entry_price * trade.bet_size

        # Fee applies to contract premium on all trades (not just profits)
        fee = contract_premium * self.fee_rate

        # Spread cost applies to bet size (all trades)
        spread_fee = self.spread_cost * trade.bet_size

        if won:
            # Profit calculation
            if trade.direction == "bull":
                # Buy YES at entry_price, receive $1 if wins
                gross_profit = (1 - trade.entry_price) * trade.bet_size
            else:
                # Buy NO, entry_price is (1 - pm_yes), win means we get $1
                # profit = (1 - entry_price) * bet_size
                gross_profit = (1 - trade.entry_price) * trade.bet_size
        else:
            # Loss calculation
            if trade.direction == "bull":
                # Lose entry price (the YES price we paid)
                gross_profit = -trade.entry_price * trade.bet_size
            else:
                # Lose entry price (the NO price we paid, which is 1 - pm_yes)
                gross_profit = -trade.entry_price * trade.bet_size

        # Net P&L = gross - fee - spread
        net_pnl = gross_profit - fee - spread_fee

        # Update portfolio state
        self.state.current_bankroll += net_pnl
        self.state.total_pnl += net_pnl
        self.state.total_trades += 1

        if won:
            self.state.total_wins += 1
            self.state.current_win_streak += 1
            self.state.current_loss_streak = 0
            self.state.max_win_streak = max(
                self.state.max_win_streak,
                self.state.current_win_streak,
            )
        else:
            self.state.total_losses += 1
            self.state.current_loss_streak += 1
            self.state.current_win_streak = 0
            self.state.max_loss_streak = max(
                self.state.max_loss_streak,
                self.state.current_loss_streak,
            )
            # Trigger pause after loss - skip next N windows
            if self.pause_windows_after_loss > 0:
                self.state.pause_windows_remaining = self.pause_windows_after_loss
                logger.info(
                    f"[{asset}] LOSS DETECTED: Pausing for {self.pause_windows_after_loss} windows"
                )

        # Update bet size for next trade based on new win streak
        if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
            self.state.current_bet_size = self.base_bet
        else:
            # SlowGrowthSizer uses win streak count, not previous bet
            self.state.current_bet_size = self.sizer.calculate_bet_for_streak(
                self.state.current_win_streak,
                self.state.current_bankroll,
            )

        # Update drawdown tracking
        self.state.peak_bankroll = max(
            self.state.peak_bankroll,
            self.state.current_bankroll,
        )
        drawdown = self.state.current_bankroll - self.state.peak_bankroll
        drawdown_pct = (
            (drawdown / self.state.peak_bankroll * 100)
            if self.state.peak_bankroll > 0
            else 0
        )

        # Track max drawdown
        if drawdown < self.state.max_drawdown:
            self.state.max_drawdown = drawdown
        if drawdown_pct < self.state.max_drawdown_pct:
            self.state.max_drawdown_pct = drawdown_pct

        # Update trade record
        trade.exit_time = window.window_end_utc
        trade.exit_price = window.spot_close
        trade.outcome = "win" if won else "loss"
        trade.gross_pnl = gross_profit
        trade.fee_paid = fee
        trade.spread_cost = spread_fee
        trade.net_pnl = net_pnl
        trade.bankroll_after = self.state.current_bankroll
        trade.drawdown = drawdown
        trade.drawdown_pct = drawdown_pct
        trade.resolved_at = datetime.now(timezone.utc)

        # Save to database
        await self.trading_db.update_trade(trade)

        # Remove from open trades
        del self.open_trades[asset]

        # Log result
        result_str = "WIN" if won else "LOSS"
        pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
        logger.info(
            f"[{asset}] Trade resolved: {result_str} | "
            f"P&L: {pnl_str} | "
            f"Bankroll: ${self.state.current_bankroll:.2f} | "
            f"Win Rate: {self.state.win_rate:.1%}"
        )

    async def _try_resolve_pending_trade(self, trade: SimulatedTrade) -> bool:
        """Try to resolve a pending trade if its window already completed.

        This is called during startup recovery to handle trades that were
        open when the app crashed/stopped, but whose windows have since resolved.

        Args:
            trade: The pending trade to check

        Returns:
            True if the trade was resolved, False if window still pending
        """
        asset = trade.asset

        # Check if we have access to this asset's database
        if asset not in self.asset_databases:
            logger.warning(
                f"[{asset}] Cannot check window resolution - no database connection"
            )
            return False

        db = self.asset_databases[asset]

        # Look up the window in the asset database
        window = await db.get_window(trade.window_id, asset)

        if window is None:
            logger.debug(f"[{asset}] Window {trade.window_id} not found in database")
            return False

        if window.outcome is None:
            logger.debug(f"[{asset}] Window {trade.window_id} has no outcome yet")
            return False

        # Window has resolved! Process the trade
        logger.info(
            f"[{asset}] RECOVERY: Found resolved window for pending trade: "
            f"{trade.direction} -> outcome={window.outcome}"
        )

        # Temporarily add to open_trades so _resolve_trade can find it
        self.open_trades[asset] = trade

        # Use the normal resolution logic (this will remove it from open_trades)
        await self._resolve_trade(asset, window)

        # Save state after recovery resolution
        await self.trading_db.save_state(self.state)

        return True

    # =========================================================================
    # Accessors for Dashboard
    # =========================================================================

    def get_state(self) -> Optional[TradingState]:
        """Get current trading state."""
        return self.state

    def get_open_trades(self) -> Dict[str, SimulatedTrade]:
        """Get all open trades."""
        return self.open_trades

    async def get_recent_trades(self, limit: int = 10) -> List[SimulatedTrade]:
        """Get recent resolved trades."""
        return await self.trading_db.get_recent_trades(limit)

    async def get_asset_stats(self, asset: str) -> dict:
        """Get statistics for a specific asset."""
        return await self.trading_db.get_asset_stats(asset)

    async def get_today_stats(self) -> dict:
        """Get today's trading statistics."""
        return await self.trading_db.get_today_stats()

    def get_last_signal(self, asset: str) -> Optional[str]:
        """Get the last signal for an asset."""
        return self._last_signals.get(asset)

    def get_config(self) -> dict:
        """Get trader configuration."""
        config = {
            "entry_mode": self.entry_mode,
            "initial_bankroll": self.initial_bankroll,
            "base_bet": self.base_bet,
            "fee_rate": self.fee_rate,
            "spread_cost": self.spread_cost,
            "bull_threshold": self.bull_threshold,
            "bear_threshold": self.bear_threshold,
            "signal_threshold_bull": self.signal_threshold_bull,
            "signal_threshold_bear": self.signal_threshold_bear,
            "confirm_threshold_bull": self.confirm_threshold_bull,
            "confirm_threshold_bear": self.confirm_threshold_bear,
            "pause_windows_after_loss": self.pause_windows_after_loss,
            "min_trajectory": self.min_trajectory,
            "sizer": self.sizer.get_config(),
        }
        if self.entry_mode in ("contrarian", "contrarian_consensus"):
            config.update({
                "contrarian_prev_thresh": self.contrarian_prev_thresh,
                "contrarian_bull_thresh": self.contrarian_bull_thresh,
                "contrarian_bear_thresh": self.contrarian_bear_thresh,
                "contrarian_entry_time": self.contrarian_entry_time,
                "contrarian_exit_time": self.contrarian_exit_time,
            })
        if self.entry_mode == "contrarian_consensus":
            config.update({
                "consensus_min_agree": self.consensus_min_agree,
                "consensus_entry_time": self.consensus_entry_time,
                "consensus_exit_time": self.consensus_exit_time,
            })
        if self.entry_mode == "accel_dbl":
            config.update({
                "accel_neutral_band": self.accel_neutral_band,
                "accel_prev_thresh": self.accel_prev_thresh,
                "accel_bull_thresh": self.accel_bull_thresh,
                "accel_bear_thresh": self.accel_bear_thresh,
                "accel_entry_time": self.accel_entry_time,
                "accel_exit_time": self.accel_exit_time,
            })
        if self.entry_mode == "combo_dbl":
            config.update({
                "combo_prev_thresh": self.combo_prev_thresh,
                "combo_bull_thresh": self.combo_bull_thresh,
                "combo_bear_thresh": self.combo_bear_thresh,
                "combo_entry_time": self.combo_entry_time,
                "combo_exit_time": self.combo_exit_time,
                "combo_stop_time": self.combo_stop_time,
                "combo_stop_delta": self.combo_stop_delta,
                "combo_xasset_min": self.combo_xasset_min,
            })
        if self.entry_mode == "triple_filter":
            config.update({
                "triple_prev_thresh": self.triple_prev_thresh,
                "triple_bull_thresh": self.triple_bull_thresh,
                "triple_bear_thresh": self.triple_bear_thresh,
                "triple_entry_time": self.triple_entry_time,
                "triple_exit_time": self.triple_exit_time,
                "triple_xasset_min": self.triple_xasset_min,
                "triple_pm0_bull_min": self.triple_pm0_bull_min,
                "triple_pm0_bear_max": self.triple_pm0_bear_max,
            })
        return config

    def get_pending_signals(self) -> Dict[str, dict]:
        """Get pending signals awaiting confirmation."""
        return self._pending_signals

    # =========================================================================
    # Metrics Calculations
    # =========================================================================

    async def calculate_metrics(self) -> dict:
        """Calculate comprehensive trading metrics."""
        if self.state is None or self.state.total_trades == 0:
            return {}

        trades = await self.trading_db.get_all_trades(resolved_only=True)

        if not trades:
            return {}

        # Basic metrics
        pnls = [t.net_pnl for t in trades if t.net_pnl is not None]
        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]

        if not pnls:
            return {}

        import numpy as np

        pnl_array = np.array(pnls)
        win_pnls = [t.net_pnl for t in wins if t.net_pnl is not None]
        loss_pnls = [t.net_pnl for t in losses if t.net_pnl is not None]

        # Profit factor
        gross_wins = sum(p for p in win_pnls if p > 0) if win_pnls else 0
        gross_losses = abs(sum(p for p in loss_pnls if p < 0)) if loss_pnls else 0
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Average win/loss
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0

        # Expectancy
        win_rate = self.state.win_rate
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe ratio (annualized for 15-min windows: 96 per day * 252 trading days)
        if len(pnls) > 1 and np.std(pnl_array) > 0:
            sharpe = (np.mean(pnl_array) / np.std(pnl_array)) * np.sqrt(252 * 96)
        else:
            sharpe = 0

        # Sortino ratio (using downside deviation)
        downside = [p for p in pnls if p < 0]
        if downside and np.std(downside) > 0:
            sortino = (np.mean(pnl_array) / np.std(downside)) * np.sqrt(252 * 96)
        else:
            sortino = 0

        # Calmar ratio (annualized return / max drawdown)
        # Return as percentage of initial bankroll, annualized
        total_return_pct = (self.state.total_pnl / self.initial_bankroll) * 100
        # Rough annualization based on trades (assuming ~96 trades/day potential)
        if len(trades) > 0 and abs(self.state.max_drawdown_pct) > 0:
            calmar = total_return_pct / abs(self.state.max_drawdown_pct)
        else:
            calmar = 0

        # Recovery factor (total profit / max drawdown)
        # How many times over have we "earned back" our worst drop?
        if abs(self.state.max_drawdown) > 0:
            recovery_factor = self.state.total_pnl / abs(self.state.max_drawdown)
        else:
            recovery_factor = float("inf") if self.state.total_pnl > 0 else 0

        return {
            "total_trades": self.state.total_trades,
            "total_wins": self.state.total_wins,
            "total_losses": self.state.total_losses,
            "win_rate": self.state.win_rate,
            "total_pnl": self.state.total_pnl,
            "avg_pnl": float(np.mean(pnl_array)),
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "recovery_factor": recovery_factor,
            "max_drawdown": self.state.max_drawdown,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "current_win_streak": self.state.current_win_streak,
            "current_loss_streak": self.state.current_loss_streak,
            "max_win_streak": self.state.max_win_streak,
            "max_loss_streak": self.state.max_loss_streak,
        }
