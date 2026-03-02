"""Core simulated trading engine.

Supports both dry-run (simulated) and live trading modes. When live_trading=True
and an ExchangeClient with trading support is provided, trades are placed on the
exchange in parallel with the simulation tracking.
"""

import asyncio
import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List, TYPE_CHECKING

from ..db.database import Database
from ..db.models import Window
from .models import SimulatedTrade, TradingState, PerAssetStats
from .database import TradingDatabase
from .bet_sizing import SlowGrowthSizer

from ..clients.exchange import OrderResult

if TYPE_CHECKING:
    from ..clients.exchange import ExchangeClient

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
        # Exchange client (for live order placement, optional)
        exchange: "Optional[ExchangeClient]" = None,
        # Live trading mode
        live_trading: bool = False,
        # Exchange name (for display)
        exchange_name: str = "polymarket",
        # Fee model
        fee_model: str = "flat",
        # Bet scaling: increase base_bet every bet_scale_threshold% gain
        bet_scale_threshold: float = 0.0,  # 0 = disabled, e.g. 1.0 = every 100% gain
        bet_scale_increase: float = 0.0,   # e.g. 0.20 = +20% per threshold step
        # Regime filter: skip trades when volatility is in these regimes
        # e.g. ["high", "extreme"] to only trade in low/normal vol
        skip_regimes: Optional[list] = None,
        # Day-of-week filter: skip trades on these days (0=Mon, 5=Sat, 6=Sun)
        skip_days: Optional[list] = None,
        # Recovery sizing: step up bet after per-asset losses
        recovery_sizing: str = "none",      # "none", "linear", "mart_1.5x"
        recovery_step: float = 1.0,         # linear: add this multiple of base_bet per loss (e.g. 1.0 = +1× base)
        recovery_max_multiplier: int = 5,   # cap at base_bet * this
        # Triple filter
        triple_prev_thresh: float = DEFAULT_TRIPLE_PREV_THRESH,
        triple_bull_thresh: float = DEFAULT_TRIPLE_BULL_THRESH,
        triple_bear_thresh: float = DEFAULT_TRIPLE_BEAR_THRESH,
        triple_entry_time: str = DEFAULT_TRIPLE_ENTRY_TIME,
        triple_exit_time: str = DEFAULT_TRIPLE_EXIT_TIME,
        triple_xasset_min: int = DEFAULT_TRIPLE_XASSET_MIN,
        triple_pm0_bull_min: float = DEFAULT_TRIPLE_PM0_BULL_MIN,
        triple_pm0_bear_max: float = DEFAULT_TRIPLE_PM0_BEAR_MAX,
        # Adaptive direction filter
        adaptive_direction_n: int = 0,
        # Momentum confirmation
        momentum_min_threshold: float = 0.0,
        # Confidence-scaled sizing
        confidence_scaling_max: float = 1.0,
        confidence_scaling_ref: float = 0.20,
        # Daily loss limit
        daily_loss_limit: float = 0.0,
        # Asymmetric direction sizing
        bear_size_mult: float = 1.0,
        bull_size_mult: float = 1.0,
        # Anti-martingale
        anti_mart_mult: float = 0.0,
        anti_mart_max: float = 5.0,
        # Hold to binary resolution
        hold_to_resolution: bool = False,
        # Sweet spot filter
        sweet_spot_band: float = 0.0,
        # Prior momentum filter: require trend still building at entry
        prior_mom_filter: bool = False,
        prior_mom_min: float = 0.03,
        # Tiered exit: hold to resolution if n_agree >= threshold, else early exit
        tiered_exit: bool = False,
        tiered_resolution_threshold: int = 3,
        # Consecutive loss circuit breaker (0 = disabled)
        max_consec_losses: int = 0,
        # UTC hour whitelist (empty = all hours)
        allowed_hours: Optional[list] = None,
        # Auto-redeem settled positions after each window (live trading only)
        redeem_on_window_complete: bool = True,
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

        # Live trading
        self.exchange = exchange
        self.live_trading = live_trading

        # Configuration
        self.initial_bankroll = initial_bankroll
        self.base_bet = base_bet
        self.fee_rate = fee_rate
        self.spread_cost = spread_cost
        self.fee_model = fee_model
        self.exchange_name = exchange_name
        self.bet_scale_threshold = bet_scale_threshold
        self.bet_scale_increase = bet_scale_increase
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

        # Regime filter
        self.skip_regimes = set(skip_regimes) if skip_regimes else set()
        self.skip_days = set(skip_days) if skip_days else set()

        # Recovery sizing (per-asset step-up after losses)
        self.recovery_sizing = recovery_sizing
        self.recovery_step = recovery_step
        self.recovery_max_multiplier = recovery_max_multiplier

        # Adaptive direction filter
        self.adaptive_direction_n = adaptive_direction_n

        # Momentum confirmation
        self.momentum_min_threshold = momentum_min_threshold

        # Confidence-scaled sizing
        self.confidence_scaling_max = confidence_scaling_max
        self.confidence_scaling_ref = confidence_scaling_ref

        # Daily loss limit
        self.daily_loss_limit = daily_loss_limit

        # Asymmetric direction sizing
        self.bear_size_mult = bear_size_mult
        self.bull_size_mult = bull_size_mult

        # Anti-martingale
        self.anti_mart_mult = anti_mart_mult
        self.anti_mart_max = anti_mart_max

        # Hold to binary resolution
        self.hold_to_resolution = hold_to_resolution

        # Sweet spot filter
        self.sweet_spot_band = sweet_spot_band

        # Prior momentum filter
        self.prior_mom_filter = prior_mom_filter
        self.prior_mom_min = prior_mom_min

        # Tiered exit
        self.tiered_exit = tiered_exit
        self.tiered_resolution_threshold = tiered_resolution_threshold

        # Consecutive loss circuit breaker
        self.max_consec_losses = max_consec_losses

        # UTC hour whitelist
        self.allowed_hours = set(allowed_hours) if allowed_hours else set()

        # Auto-redeem settled positions after each window
        self.redeem_on_window_complete = redeem_on_window_complete
        self._last_redeem_time: Optional[datetime] = None
        self._last_loser_cleanup_time: Optional[datetime] = None
        self._redeem_creds_warned_at: Optional[datetime] = None
        # Funder/proxy wallet address for position queries
        self._funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")

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

        # Track previous window's volatility regime per asset (for regime filter)
        self._prev_window_regime: Dict[str, str] = {}

        # Recovery sizing: per-asset consecutive loss count
        self._asset_consecutive_losses: Dict[str, int] = {}

        # Adaptive direction: rolling history of (direction, won) tuples
        # Used to compute trailing bull/bear win rates
        self._adaptive_history: List[tuple] = []  # [(direction, won), ...]

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

        # Consensus: store pm_yes at t0 per asset for momentum/sweet-spot checks
        # Maps asset -> pm_yes at t0 of current window
        self._consensus_t0_pm: Dict[str, float] = {}

        # Daily P&L tracking for daily loss limit
        # Maps date string "YYYY-MM-DD" -> cumulative net P&L for that day
        self._daily_pnl: Dict[str, float] = {}

        # Tiered exit: per-asset binary resolution flag
        # True = hold to binary resolution; False = early exit at t12.5
        self._trade_use_resolution: Dict[str, bool] = {}

        # Group loss circuit breaker: track group-level P&L for max_consec_losses
        self._group_consec_losses: int = 0
        self._trade_group_key: Dict[str, str] = {}      # asset -> time_key
        self._group_entered_assets: Dict[str, set] = {}  # time_key -> pending assets
        self._group_pnl_acc: Dict[str, float] = {}       # time_key -> accumulated P&L

        # Session-start wallet balance (set once on first live balance snapshot)
        # Used as the baseline for bet scaling so config initial_bankroll doesn't
        # misalign when the actual wallet has a different starting balance.
        self._session_start_bankroll: Optional[float] = None

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
                # Fixed sizing for contrarian-family modes (with bet scaling)
                self.state.current_bet_size = self._get_scaled_bet()
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

        # Live trading: verify exchange connectivity, snapshot balance, cancel stale orders
        if self.live_trading:
            if not self.exchange:
                logger.critical(
                    "LIVE TRADING ENABLED but no exchange client provided! "
                    "Orders will NOT be placed."
                )
            elif not self.exchange.supports_trading:
                logger.critical(
                    "LIVE TRADING ENABLED but exchange.supports_trading=False! "
                    "Check POLYMARKET_PRIVATE_KEY and polymarket CLI installation. "
                    "Orders will NOT be placed."
                )
            else:
                logger.info("=" * 50)
                logger.info("LIVE TRADING VERIFICATION")
                try:
                    balances = await self.exchange.fetch_balance()
                    if balances:
                        for b in balances:
                            logger.info(
                                f"  Wallet: {b.currency} balance=${b.total:.2f} "
                                f"available=${b.available:.2f}"
                            )
                    else:
                        logger.warning("  No balance info returned — verify API credentials")
                    await self._cancel_stale_orders()
                    logger.info("  Live trading verification PASSED")
                except Exception as e:
                    logger.critical(
                        f"  Live trading verification FAILED: {e}\n"
                        f"  Orders may fail at runtime. Check API credentials.",
                        exc_info=True,
                    )

                # Check redemption readiness
                if self.redeem_on_window_complete:
                    await self._check_redeem_readiness()

                logger.info("=" * 50)
                await self._snapshot_wallet_balance(label="session_start")

    def _get_scaled_bet(self) -> float:
        """Get the current base bet, scaled up based on bankroll growth.

        If bet_scale_threshold > 0, for every threshold% the bankroll has grown
        above initial_bankroll, increase base_bet by bet_scale_increase%.

        Example: $1000 start, $50 bet, threshold=1.0 (100%), increase=0.20 (20%)
          - bankroll $1500 → 0 steps → $50
          - bankroll $2000 → 1 step  → $60
          - bankroll $3000 → 2 steps → $72
        """
        if self.bet_scale_threshold <= 0 or self.bet_scale_increase <= 0:
            return self.base_bet

        bankroll = self.state.current_bankroll if self.state else self.initial_bankroll
        baseline = self._session_start_bankroll if self._session_start_bankroll is not None else self.initial_bankroll
        gain_pct = (bankroll - baseline) / baseline
        if gain_pct <= 0:
            return self.base_bet

        steps = int(gain_pct / self.bet_scale_threshold)
        scaled = self.base_bet * (1 + self.bet_scale_increase) ** steps
        return round(scaled, 2)

    def _get_recovery_bet(self, asset: str) -> float:
        """Get bet size adjusted for per-asset recovery sizing.

        When recovery_sizing is enabled, steps up the bet after consecutive
        losses for the specific asset. Resets to base on a win.

        Linear: base + step * losses  (e.g. $25, $50, $75, $100, $125)
        Mart 1.5x: base * 1.5^losses  (e.g. $25, $37.5, $56.25, ...)

        Falls back to _get_scaled_bet() when recovery_sizing == "none".
        """
        if self.recovery_sizing == "none":
            return self._get_scaled_bet()

        losses = self._asset_consecutive_losses.get(asset, 0)
        base = self._get_scaled_bet()  # use bankroll-scaled base
        max_bet = base * self.recovery_max_multiplier

        if self.recovery_sizing == "linear":
            # recovery_step is a multiplier of base (e.g. 1.0 = add 1× base per loss)
            # e.g. base=$2, losses=1 → $2 + $2×1.0 = $4; losses=2 → $6 (capped at max)
            bet = base + base * self.recovery_step * losses
        elif self.recovery_sizing == "mart_1.5x":
            bet = base * (1.5 ** losses)
        else:
            bet = base

        bet = min(bet, max_bet)
        return round(bet, 2)

    def _record_daily_pnl(self, net_pnl: float):
        """Track cumulative daily P&L for daily loss limit."""
        if self.daily_loss_limit > 0:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._daily_pnl[today] = self._daily_pnl.get(today, 0.0) + net_pnl

    def _adaptive_direction_allowed(self, direction: str) -> bool:
        """Check if the given direction is allowed by the adaptive filter.

        Returns True if:
          - adaptive_direction_n == 0 (filter disabled)
          - not enough history yet (< adaptive_direction_n trades)
          - the proposed direction's trailing WR >= the other direction's WR
        """
        n = self.adaptive_direction_n
        if n <= 0:
            return True
        if len(self._adaptive_history) < n:
            return True

        recent = self._adaptive_history[-n:]
        bull_trades = [won for d, won in recent if d == "bull"]
        bear_trades = [won for d, won in recent if d == "bear"]

        bull_wr = sum(bull_trades) / len(bull_trades) if bull_trades else 0.5
        bear_wr = sum(bear_trades) / len(bear_trades) if bear_trades else 0.5

        if direction == "bull":
            return bull_wr >= bear_wr
        else:
            return bear_wr > bull_wr

    def _adaptive_record_outcome(self, direction: str, won: bool):
        """Record a trade outcome for the adaptive direction filter."""
        if self.adaptive_direction_n > 0:
            self._adaptive_history.append((direction, won))
            # Keep at most 2x the window to avoid unbounded growth
            max_len = self.adaptive_direction_n * 3
            if len(self._adaptive_history) > max_len:
                self._adaptive_history = self._adaptive_history[-max_len:]

    def _adaptive_status(self) -> str:
        """Return a short status string for the adaptive filter (for dashboard)."""
        n = self.adaptive_direction_n
        if n <= 0:
            return ""
        if len(self._adaptive_history) < n:
            return f"adapt{n}: warming ({len(self._adaptive_history)}/{n})"
        recent = self._adaptive_history[-n:]
        bull_trades = [won for d, won in recent if d == "bull"]
        bear_trades = [won for d, won in recent if d == "bear"]
        bull_wr = sum(bull_trades) / len(bull_trades) if bull_trades else 0.5
        bear_wr = sum(bear_trades) / len(bear_trades) if bear_trades else 0.5
        favored = "BULL" if bull_wr >= bear_wr else "BEAR"
        return f"adapt{n}: {favored} (B:{bull_wr:.0%} R:{bear_wr:.0%})"

    @staticmethod
    def _polymarket_crypto_fee(dollar_amount: float, price: float) -> float:
        """Polymarket 15-min crypto fee: C × 0.25 × (p × (1-p))^2.

        C = dollar amount of the order (NOT number of contracts).
        Max effective rate: 1.56% at p=0.50.
        Rounded to 4 dp, min 0.0001 USDC.
        """
        raw = dollar_amount * 0.25 * (price * (1.0 - price)) ** 2
        return max(round(raw, 4), 0.0001) if raw > 0 else 0.0

    def _calculate_fees(
        self, entry_contract: float, exit_contract: float, n_contracts: float, bet_size: float
    ) -> float:
        """Calculate total fees for a trade based on the configured fee model.

        Args:
            entry_contract: Contract price at entry (0-1)
            exit_contract: Contract price at exit (0-1)
            n_contracts: Number of contracts traded
            bet_size: Dollar bet size

        Returns:
            Total fees in dollars.
        """
        if self.fee_model == "probability_weighted":
            # Kalshi: ceil(0.07 * contracts * price * (1-price)) per side, in cents
            entry_fee = math.ceil(0.07 * n_contracts * entry_contract * (1 - entry_contract) * 100) / 100
            exit_fee = math.ceil(0.07 * n_contracts * exit_contract * (1 - exit_contract) * 100) / 100
            return entry_fee + exit_fee
        elif self.fee_model == "polymarket_crypto":
            # Polymarket 15-min crypto: fee = C × 0.25 × (p × (1-p))^2 per side
            # C = dollar amount of the order, max effective rate ~1.56% at p=0.50
            entry_dollars = n_contracts * entry_contract  # = bet_size
            exit_dollars = n_contracts * exit_contract
            entry_fee = self._polymarket_crypto_fee(entry_dollars, entry_contract)
            exit_fee = self._polymarket_crypto_fee(exit_dollars, exit_contract)
            return entry_fee + exit_fee
        else:
            # Flat model: fee_rate on contract premium + spread on trade value
            entry_fee = entry_contract * n_contracts * self.fee_rate
            exit_fee = exit_contract * n_contracts * self.fee_rate
            entry_spread = self.spread_cost * bet_size
            exit_spread = self.spread_cost * (n_contracts * exit_contract)
            return entry_fee + exit_fee + entry_spread + exit_spread

    async def _place_live_order(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        bet_size: float,
        trade: SimulatedTrade,
        side: str = "buy",
    ) -> "Optional[OrderResult]":
        """Place a live order on the exchange and enrich the trade with fill data.

        On successful fill, sets the live_* fields directly on the SimulatedTrade
        and persists via update_trade(). Returns the OrderResult so callers can
        use actual fill prices for P&L.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP).
            direction: "bull" or "bear".
            entry_price: Expected entry contract price (used as limit price).
            bet_size: Dollar amount of the trade.
            trade: The SimulatedTrade to enrich with live fill data.
            side: "buy" or "sell" (default: "buy" for opening positions).

        Returns:
            OrderResult if order was placed (and possibly filled), None otherwise.
        """
        if not self.live_trading or not self.exchange:
            logger.debug(
                f"[{asset}] Skipping live order: live_trading={self.live_trading}, "
                f"exchange={'set' if self.exchange else 'None'}"
            )
            return None

        if not self.exchange.supports_trading:
            logger.warning(
                f"[{asset}] Skipping live order: exchange.supports_trading=False "
                f"(polymarket CLI may have failed to initialize)"
            )
            return None

        try:
            market = self.exchange.get_cached_market(asset)
            if not market:
                logger.warning(f"[{asset}] No cached market for live order, skipping")
                return None

            # Determine outcome token: bull = YES, bear = NO
            outcome = "yes" if direction == "bull" else "no"

            # Number of contracts = bet_size / entry_price
            n_contracts = bet_size / entry_price if entry_price > 0.01 else 0
            if n_contracts <= 0:
                logger.warning(
                    f"[{asset}] Zero contracts — bet_size={bet_size}, "
                    f"entry_price={entry_price}, skipping live order"
                )
                return None

            logger.info(
                f"[{asset}] LIVE ORDER ATTEMPT: {side} {n_contracts:.1f} "
                f"{outcome.upper()} @ {entry_price:.4f} "
                f"(token={market.yes_token_id[:12] if outcome == 'yes' else market.no_token_id[:12]}...)"
            )

            result = await self.exchange.place_order(
                market=market,
                side=side,
                outcome=outcome,
                amount=n_contracts,
                price=entry_price,
                order_type="limit",
            )

            # Adapter may bump size to exchange minimum
            if result.amount > n_contracts:
                logger.info(
                    f"[{asset}] Order size bumped from {n_contracts:.1f} to "
                    f"{result.amount:.1f} contracts (exchange minimum). "
                    f"Actual cost: ${result.amount * entry_price:.2f}"
                )
                n_contracts = result.amount
                trade.bet_size = n_contracts * entry_price  # update sim trade to match reality

            logger.info(
                f"[{asset}] LIVE ORDER placed: {result.order_id} "
                f"{side} {n_contracts:.1f} {outcome.upper()} @ {entry_price:.3f} "
                f"status={result.status}"
            )

            # Poll for fill — cancel if not filled (we don't want stale entry orders)
            result = await self._poll_order_fill(
                result, max_wait=15.0, interval=1.0, cancel_if_unfilled=True
            )

            # Handle results
            if result.status in ("cancelled", "timeout", "expired", "unverified"):
                logger.warning(
                    f"[{asset}] Entry order not filled, status={result.status}. "
                    f"No live position opened."
                )
                return None

            # Determine actual fill data
            fill_price = result.price
            fill_contracts = result.filled if result.filled > 0 else None
            fill_fee = result.fee

            # Log partial fill without mutating bet_size — live_entry_contracts
            # already records what actually filled and is used in both exit paths.
            if result.status == "partial" and fill_contracts and fill_price:
                logger.info(
                    f"[{asset}] PARTIAL FILL: {fill_contracts:.1f}/{n_contracts:.1f} "
                    f"contracts @ {fill_price:.4f} (original bet ${trade.bet_size:.2f} preserved)"
                )

            # Enrich the trade with live fill data
            trade.live_order_id = result.order_id
            trade.live_entry_fill_price = fill_price
            trade.live_entry_contracts = fill_contracts
            trade.live_entry_fee = fill_fee
            trade.live_status = "open" if result.status in ("filled", "partial") else "entry_placed"
            trade.exchange = self.exchange_name

            # Persist live data on the trade
            await self.trading_db.update_trade(trade)

            logger.info(
                f"[{asset}] LIVE FILL: {fill_contracts or '?'} contracts "
                f"@ {fill_price or '?'} (requested {entry_price:.4f}) "
                f"fee={fill_fee} status={result.status}"
            )

            return result

        except Exception as e:
            logger.error(
                f"[{asset}] Live order placement failed: {e}", exc_info=True
            )
            return None  # Don't raise — simulation continues regardless

    async def _close_live_position(
        self,
        asset: str,
        direction: str,
        exit_price: float,
        n_contracts: float,
        trade: SimulatedTrade,
    ) -> "Optional[OrderResult]":
        """Sell/close a live position on the exchange using FAK sell loop.

        Uses repeated Fill-And-Kill orders to drain the position, handling
        partial fills and low liquidity gracefully. Falls back to binary
        resolution if the position can't be fully exited.

        Args:
            asset: Asset symbol.
            direction: "bull" or "bear" (the original trade direction).
            exit_price: Expected exit contract price (for P&L estimation).
            n_contracts: Number of contracts to sell (from sim tracking).
            trade: The SimulatedTrade with live entry data.

        Returns:
            OrderResult if exit was executed, None otherwise.
        """
        if not self.live_trading or not self.exchange:
            logger.debug(
                f"[{asset}] Skipping live exit: live_trading={self.live_trading}, "
                f"exchange={'set' if self.exchange else 'None'}"
            )
            return None

        if not self.exchange.supports_trading:
            logger.warning(
                f"[{asset}] Skipping live exit: exchange.supports_trading=False"
            )
            return None

        # Verify we actually have a live position to close
        if not trade.has_live_data:
            logger.warning(
                f"[{asset}] No live position found — skipping exit order. "
                f"Entry may not have filled."
            )
            return None

        try:
            market = self.exchange.get_cached_market(asset)
            if not market:
                logger.warning(f"[{asset}] No cached market for live exit, skipping")
                return None

            # Determine which token we're selling
            outcome = "yes" if direction == "bull" else "no"
            token_id = (
                market.yes_token_id if outcome == "yes" else market.no_token_id
            )

            # Use FAK sell loop if adapter supports it (PolymarketAdapter)
            if hasattr(self.exchange, 'sell_position_fak'):
                logger.info(
                    f"[{asset}] LIVE EXIT via FAK sell loop "
                    f"(token={token_id[:16]}...)"
                )
                total_usdc = await self.exchange.sell_position_fak(token_id)

                if total_usdc > 0:
                    # Estimate fill price from USDC received
                    actual_contracts = trade.live_entry_contracts or n_contracts
                    avg_exit_price = total_usdc / actual_contracts if actual_contracts > 0 else exit_price
                    trade.live_exit_fill_price = avg_exit_price
                    trade.live_exit_contracts = actual_contracts
                    trade.live_exit_fee = 0.0  # FAK fees are embedded in fill price
                    trade.live_status = "closed"

                    logger.info(
                        f"[{asset}] LIVE EXIT FILL: "
                        f"${total_usdc:.4f} USDC received "
                        f"(~{actual_contracts:.1f} contracts @ ~{avg_exit_price:.4f})"
                    )

                    # Build a synthetic OrderResult for callers
                    result = OrderResult(
                        order_id="fak_exit",
                        market_id=market.condition_id,
                        outcome_id=token_id,
                        side="sell",
                        order_type="market",
                        amount=actual_contracts,
                        price=avg_exit_price,
                        status="filled",
                        filled=actual_contracts,
                        remaining=0.0,
                    )
                else:
                    # FAK sell got no fills — position stays open for binary resolution
                    trade.live_status = "open"
                    logger.warning(
                        f"[{asset}] FAK EXIT got no fills. "
                        f"Position will settle at binary resolution."
                    )
                    result = None

                # Persist updated live data
                await self.trading_db.update_trade(trade)
                return result

            # Fallback: standard limit sell + poll (for non-Polymarket adapters)
            actual_contracts = trade.live_entry_contracts or n_contracts
            result = await self.exchange.place_order(
                market=market,
                side="sell",
                outcome=outcome,
                amount=actual_contracts,
                price=exit_price,
                order_type="limit",
            )

            logger.info(
                f"[{asset}] LIVE EXIT placed: {result.order_id} "
                f"sell {actual_contracts:.1f} {outcome.upper()} @ {exit_price:.3f} "
                f"status={result.status}"
            )

            result = await self._poll_order_fill(
                result, max_wait=15.0, interval=1.0, cancel_if_unfilled=True
            )

            trade.live_exit_order_id = result.order_id

            if result.status in ("filled", "partial"):
                trade.live_exit_fill_price = result.price
                trade.live_exit_contracts = result.filled if result.filled > 0 else None
                trade.live_exit_fee = result.fee
                trade.live_status = "closed"

                logger.info(
                    f"[{asset}] LIVE EXIT FILL: {result.filled:.1f} contracts "
                    f"@ {result.price} (requested {exit_price:.4f}) "
                    f"fee={result.fee}"
                )
            elif result.status in ("cancelled", "timeout", "unverified"):
                trade.live_status = "open"
                logger.warning(
                    f"[{asset}] EXIT ORDER NOT FILLED (status={result.status}). "
                    f"Position will settle at binary resolution."
                )
            else:
                trade.live_status = "exit_placed"

            await self.trading_db.update_trade(trade)
            return result

        except Exception as e:
            logger.error(f"[{asset}] Live exit order failed: {e}", exc_info=True)
            return None

    async def _poll_order_fill(
        self,
        order: "OrderResult",
        max_wait: float = 15.0,
        interval: float = 1.0,
        cancel_if_unfilled: bool = False,
    ) -> "OrderResult":
        """Poll exchange for order fill status using fetch_order().

        Uses fetch_order() to get actual fill price, filled amount, and fee
        from the exchange. This is critical for accurate P&L.

        If the order is not filled within max_wait and cancel_if_unfilled=True,
        cancels the order and returns the cancelled state.

        Handles partial fills: if some contracts filled but not all, marks as
        'partial' status with actual filled amount.

        Args:
            order: The OrderResult from place_order.
            max_wait: Maximum seconds to wait for fill (default: 15s).
            interval: Seconds between polls (default: 1s).
            cancel_if_unfilled: If True, cancel order after max_wait (default: False).

        Returns:
            Updated OrderResult with actual fill data from exchange.
        """
        if order.status == "filled":
            return order  # Already filled

        if not self.exchange or not self.exchange.supports_trading:
            return order

        elapsed = 0.0
        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval

            try:
                # Use fetch_order to get real fill data
                updated = await self.exchange.fetch_order(order.order_id)
                if updated.status:
                    updated.status = updated.status.lower()

                if updated.status in ("filled", "matched", "closed"):
                    # Fully filled — copy actual fill data
                    order.status = "filled"
                    order.filled = updated.filled if updated.filled > 0 else order.amount
                    order.remaining = updated.remaining
                    order.price = updated.price  # actual fill price
                    order.fee = updated.fee  # actual fee
                    logger.info(
                        f"Order {order.order_id} FILLED after {elapsed:.1f}s: "
                        f"{order.filled:.1f} contracts @ {order.price} "
                        f"fee={order.fee}"
                    )
                    return order

                elif updated.filled > 0 and updated.remaining > 0:
                    # Partial fill — update with what we have so far
                    order.filled = updated.filled
                    order.remaining = updated.remaining
                    order.price = updated.price
                    order.fee = updated.fee
                    logger.debug(
                        f"Order {order.order_id} partial fill: "
                        f"{updated.filled:.1f}/{order.amount:.1f} after {elapsed:.1f}s"
                    )
                    # Continue polling — might fill fully

                elif updated.status in ("cancelled", "expired"):
                    order.status = updated.status
                    order.filled = updated.filled
                    order.price = updated.price
                    order.fee = updated.fee
                    logger.warning(
                        f"Order {order.order_id} {updated.status} "
                        f"(filled={updated.filled})"
                    )
                    return order

            except Exception as e:
                logger.warning(f"fetch_order poll failed ({elapsed:.1f}s): {e}")
                # Fall back to open orders check
                try:
                    open_orders = await self.exchange.fetch_open_orders()
                    still_open = any(
                        o.order_id == order.order_id for o in open_orders
                    )
                    if not still_open:
                        # Disappeared from open orders — try fetch_order one more time
                        try:
                            final = await self.exchange.fetch_order(order.order_id)
                            order.status = "filled"
                            order.filled = final.filled if final.filled > 0 else order.amount
                            order.price = final.price
                            order.fee = final.fee
                            logger.info(
                                f"Order {order.order_id} filled (confirmed via fetch_order) "
                                f"after {elapsed:.1f}s (filled={order.filled})"
                            )
                        except Exception:
                            # Cannot confirm fill — do NOT assume it filled
                            order.status = "unverified"
                            logger.critical(
                                f"Order {order.order_id} disappeared from open orders but "
                                f"fetch_order also failed. Status set to 'unverified'. "
                                f"MANUAL CHECK REQUIRED — do not trust this fill."
                            )
                        return order
                except Exception as e2:
                    logger.warning(f"fetch_open_orders fallback also failed: {e2}")

        # Timed out — handle partial fills and optional cancellation
        if order.status != "filled":
            if order.filled > 0:
                # Partial fill: keep what we got
                order.status = "partial"
                logger.warning(
                    f"Order {order.order_id} PARTIAL fill after {max_wait}s: "
                    f"{order.filled:.1f}/{order.amount:.1f} contracts"
                )
            elif cancel_if_unfilled:
                # No fill — cancel it
                try:
                    cancel_result = await self.exchange.cancel_order(order.order_id)
                    order.status = "cancelled"
                    order.filled = cancel_result.filled
                    order.price = cancel_result.price
                    order.fee = cancel_result.fee
                    logger.warning(
                        f"Order {order.order_id} CANCELLED after {max_wait}s timeout "
                        f"(filled={order.filled})"
                    )
                except Exception as e:
                    logger.error(f"Failed to cancel timed-out order {order.order_id}: {e}")
                    order.status = "timeout"
            else:
                order.status = "timeout"
                logger.warning(
                    f"Order {order.order_id} not filled after {max_wait}s "
                    f"(status={order.status}, NOT cancelled)"
                )

        return order

    async def _fetch_wallet_balance(self) -> Optional[float]:
        """Fetch actual wallet balance from the exchange.

        This is the source of truth for bankroll when live trading.
        Returns the available USDC balance, or None if unavailable.
        """
        if not self.exchange or not self.exchange.supports_trading:
            return None

        try:
            balances = await self.exchange.fetch_balance()
            for bal in balances:
                if bal.currency.upper() in ("USDC", "USD"):
                    logger.debug(
                        f"Wallet: total=${bal.total:.2f} available=${bal.available:.2f} "
                        f"locked=${bal.locked:.2f}"
                    )
                    return bal.total  # total = available + locked in orders
            logger.debug("No USDC/USD balance found in wallet")
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch wallet balance: {e}")
            return None

    async def _cancel_stale_orders(self, asset: Optional[str] = None):
        """Cancel all open orders on the exchange, or for a specific asset.

        Called at window completion and before placing new entry orders to
        prevent stale limit orders from sitting on the book.

        Args:
            asset: If provided, only cancel orders for this asset's market.
                   If None, cancel ALL open orders.
        """
        if not self.exchange or not self.exchange.supports_trading:
            return

        try:
            market = None
            if asset:
                market = self.exchange.get_cached_market(asset)

            open_orders = await self.exchange.fetch_open_orders(market=market)

            if not open_orders:
                return

            for order in open_orders:
                try:
                    await self.exchange.cancel_order(order.order_id)
                    logger.info(
                        f"Cancelled stale order: {order.order_id} "
                        f"({order.side} {order.amount:.1f} @ {order.price})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to cancel stale order {order.order_id}: {e}")

            if open_orders:
                logger.info(f"Cleaned up {len(open_orders)} stale order(s)")

        except Exception as e:
            logger.warning(f"Failed to check for stale orders: {e}")

    async def _snapshot_wallet_balance(self, label: str = ""):
        """Snapshot the current wallet balance and log it.

        Called at session start and periodically for P&L tracking.
        The wallet balance is the ultimate source of truth — it captures
        all fees, slippage, partial fills, and settlement.

        Args:
            label: Optional label for the log message (e.g. "session_start").
        """
        if not self.exchange or not self.exchange.supports_trading:
            return

        try:
            balance = await self._fetch_wallet_balance()
            if balance is not None:
                prefix = f"[{label}] " if label else ""
                logger.info(
                    f"{prefix}Wallet balance snapshot: ${balance:.2f}"
                )

                # Capture session-start balance once (baseline for bet scaling)
                if self._session_start_bankroll is None:
                    self._session_start_bankroll = balance
                    logger.info(f"Session-start bankroll anchored at ${balance:.2f}")

                    # On a fresh live session with no prior trades, anchor
                    # state.initial_bankroll to the real wallet balance rather than
                    # the config value. Without this, return % shows -90% immediately
                    # if the wallet has less than the config's initial_bankroll.
                    if self.state and self.state.total_trades == 0 and balance != self.state.initial_bankroll:
                        logger.info(
                            f"Live session: adjusting initial_bankroll "
                            f"${self.state.initial_bankroll:.2f} (config) → ${balance:.2f} (wallet)"
                        )
                        self.state.initial_bankroll = balance
                        self.state.peak_bankroll = balance
                        await self.trading_db.save_state(self.state)

                # Use wallet as source of truth for bankroll
                if self.state:
                    self.state.current_bankroll = balance
            else:
                logger.debug("Wallet balance snapshot: unavailable (keeping last known)")
        except Exception as e:
            logger.warning(f"Wallet balance snapshot failed (keeping last known): {e}")

    async def _check_redeem_readiness(self):
        """Check redemption prerequisites at startup and log clear status."""
        issues = []

        try:
            from ..clients.polymarket_relayer import (
                RELAYER_ENABLED,
                cookies_valid,
                refresh_cookies,
                get_circuit_status,
            )
        except ImportError:
            logger.warning(
                "  Redeem: DISABLED — missing dependencies "
                "(polymarket_relayer/polymarket_claims not installed)"
            )
            return

        if not RELAYER_ENABLED:
            issues.append(
                "POLYMARKET_PRIVATE_KEY not set — relayer cannot sign requests"
            )

        # Attempt cookie refresh (will try Magic auth if configured)
        if RELAYER_ENABLED and not cookies_valid():
            refresh_cookies()

        if RELAYER_ENABLED and not cookies_valid():
            issues.append(
                "No Polymarket session cookies — gasless relayer cannot submit.\n"
                "           Fix: set POLYMARKET_AUTH_TYPE=magic in .env, OR\n"
                "           inject cookies via set_session_cookies(), OR\n"
                "           redeem manually at polymarket.com/portfolio"
            )

        wallet = self._funder_address
        if not wallet and self.exchange and hasattr(self.exchange, 'get_funder_address'):
            wallet = self.exchange.get_funder_address()
        if not wallet:
            issues.append(
                "No funder address — cannot query positions for redemption.\n"
                "           Fix: set POLYMARKET_FUNDER_ADDRESS in .env"
            )

        circuit = get_circuit_status()
        if circuit.get("open"):
            issues.append(
                f"Relayer circuit breaker OPEN — resets in {circuit.get('secondsRemaining', '?')}s"
            )

        if issues:
            logger.warning(
                "  Redeem: DEGRADED — auto-redemption will be skipped:\n"
                + "\n".join(f"    - {i}" for i in issues)
            )
            self._redeem_creds_warned_at = datetime.now(timezone.utc)
        else:
            logger.info("  Redeem: OK — cookies valid, relayer ready")

    async def try_redeem_settled(self):
        """Check for and redeem settled positions after window completes.

        Two sweep modes:
        1. **Winners** (every 15 min): redeem positions with value (redeem_losers=False)
        2. **Losers** (every hour): clean up zero-value settled positions (redeem_losers=True)

        Uses the Polymarket gasless relayer. Called from the window-complete
        callback when redeem_on_window_complete=True.
        """
        if not self.live_trading or not self.redeem_on_window_complete:
            return

        now = datetime.now(timezone.utc)

        # Determine which sweep to run based on timers
        winner_due = (
            self._last_redeem_time is None
            or (now - self._last_redeem_time).total_seconds() >= 900  # 15 min
        )
        loser_due = (
            self._last_loser_cleanup_time is None
            or (now - self._last_loser_cleanup_time).total_seconds() >= 3600  # 1 hour
        )

        if not winner_due and not loser_due:
            return

        # Resolve wallet address for position queries
        wallet = self._funder_address
        if not wallet and self.exchange and hasattr(self.exchange, 'get_funder_address'):
            wallet = self.exchange.get_funder_address()
        if not wallet:
            logger.debug("No funder address for redeem check")
            return

        try:
            from ..clients.polymarket_claims import (
                fetch_positions,
                sweep_and_redeem_claimables,
            )
            from ..clients.polymarket_relayer import (
                RELAYER_ENABLED,
                cookies_valid,
                get_circuit_status,
            )
        except ImportError as e:
            logger.debug(f"Redeem dependencies not available: {e}")
            return

        if not RELAYER_ENABLED:
            logger.debug("Relayer not enabled, skipping redeem")
            return

        circuit = get_circuit_status()
        if circuit.get("open"):
            logger.debug(
                f"Relayer circuit open — resets in {circuit.get('secondsRemaining', '?')}s"
            )
            return

        # Try to refresh cookies (e.g. Magic auth) before giving up
        from ..clients.polymarket_relayer import refresh_cookies
        if not cookies_valid():
            refresh_cookies()
        if not cookies_valid():
            # Warn once at first failure, then re-warn every 6 hours
            should_warn = (
                self._redeem_creds_warned_at is None
                or (now - self._redeem_creds_warned_at).total_seconds() >= 21600
            )
            if should_warn:
                logger.warning(
                    "Redeem skipped: no Polymarket session cookies.\n"
                    "  Fix: set POLYMARKET_AUTH_TYPE=magic in .env, OR\n"
                    "  inject cookies via set_session_cookies(), OR\n"
                    "  redeem manually at polymarket.com/portfolio"
                )
                self._redeem_creds_warned_at = now
            return

        # --- Winner sweep (every 15 min) ---
        if winner_due:
            try:
                positions = await fetch_positions(wallet)
                redeemable = [
                    p for p in positions
                    if p.get("redeemable") is True
                    and float(p.get("currentValue", 0) or 0) > 0
                ]

                if redeemable:
                    total_value = sum(
                        float(p.get("currentValue", 0) or 0) for p in redeemable
                    )
                    logger.info(
                        f"Found {len(redeemable)} redeemable winner(s) "
                        f"(~${total_value:.2f} USDC)"
                    )
                    await sweep_and_redeem_claimables(wallet, redeem_losers=False)
                    await asyncio.sleep(3)
                    await self._snapshot_wallet_balance(label="post_redeem")
                    logger.info("Winner redeem sweep completed")

                self._last_redeem_time = now

            except Exception as e:
                logger.error(f"Winner redeem sweep failed: {e}", exc_info=True)
                self._last_redeem_time = now

        # --- Loser cleanup (every hour) ---
        if loser_due:
            try:
                positions = await fetch_positions(wallet)
                losers = [
                    p for p in positions
                    if p.get("redeemable") is True
                    and float(p.get("currentValue", 0) or 0) == 0
                    and float(p.get("size", 0) or 0) > 0
                ]

                if losers:
                    logger.info(
                        f"Hourly cleanup: {len(losers)} zero-value loser(s) to clear"
                    )
                    await sweep_and_redeem_claimables(wallet, redeem_losers=True)
                    logger.info("Loser cleanup sweep completed")

                self._last_loser_cleanup_time = now

            except Exception as e:
                logger.error(f"Loser cleanup sweep failed: {e}", exc_info=True)
                self._last_loser_cleanup_time = now

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
            bet_size = min(self._get_scaled_bet(), self.state.current_bankroll * self.max_bet_pct)

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

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

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
            bet_size = min(self._get_scaled_bet(), self.state.current_bankroll * self.max_bet_pct)

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

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

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

            # Regime filter: skip if previous window was high/extreme vol
            if self.skip_regimes and self._prev_window_regime.get(asset) in self.skip_regimes:
                logger.debug(
                    f"[{asset}] Skipping — prev regime "
                    f"'{self._prev_window_regime[asset]}' in skip_regimes"
                )
                self._last_signals[asset] = "skip-regime"
                return

            # Day-of-week filter
            if self.skip_days:
                try:
                    from datetime import datetime as _dt
                    dt = _dt.fromisoformat(str(sample.sample_time_utc).replace('Z', '+00:00'))
                    if dt.weekday() in self.skip_days:
                        logger.debug(f"[{asset}] Skipping — {dt.strftime('%A')} in skip_days")
                        self._last_signals[asset] = "skip-day"
                        return
                except Exception:
                    pass

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

            # Bet size (recovery-aware for contrarian modes)
            bet_size = min(self._get_recovery_bet(asset), self.state.current_bankroll * self.max_bet_pct)

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

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

        except Exception as e:
            logger.error(f"[{asset}] Error in contrarian entry: {e}", exc_info=True)

    async def on_sample_at_contrarian_exit(self, asset: str, sample, state):
        """Handle sample at contrarian exit time (default t=12.5).

        Sell the position at current PM price. P&L is the price difference
        minus double fees (entry + exit).

        When live trading is enabled, places the exit order first and uses
        actual fill data from the exchange for P&L. Falls back to simulated
        estimates if live fill data is not available.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            sample: The sample collected at exit time
            state: The asset's current sampler state
        """
        try:
            if asset not in self.open_trades:
                return

            trade = self.open_trades[asset]

            # Tiered exit: if this trade is flagged for binary resolution, skip early exit
            if self._trade_use_resolution.get(asset, False):
                logger.info(
                    f"[{asset}] Tiered exit: consensus was strong, "
                    f"holding to binary resolution (skipping t12.5 exit)"
                )
                return

            pm_yes = sample.pm_yes_price

            if pm_yes is None:
                logger.warning(f"[{asset}] No pm_yes at exit time, cannot close position")
                return

            # Calculate simulated exit contract price
            if trade.direction == "bull":
                exit_contract = pm_yes
                entry_contract = trade.entry_price  # was pm_yes at entry
            else:
                exit_contract = 1.0 - pm_yes
                entry_contract = trade.entry_price  # was (1 - pm_yes) at entry

            if entry_contract <= 0.001:
                logger.warning(f"[{asset}] Entry contract price too low, skipping exit")
                return

            # Use actual fill count for live trades (handles partial fills correctly);
            # fall back to estimated count for simulation.
            n_contracts = (
                trade.live_entry_contracts
                if (self.live_trading and trade.live_entry_contracts)
                else trade.bet_size / entry_contract
            )

            # Place live exit order FIRST so we get actual fill data
            exit_result = await self._close_live_position(
                asset, trade.direction, exit_contract, n_contracts, trade=trade
            )

            # If the live exit order didn't fill, the position stays open on the exchange
            # and will settle at binary resolution. Don't resolve the trade here —
            # leave it in open_trades so on_window_complete → _resolve_trade handles it.
            if (
                self.live_trading
                and trade.has_live_data
                and trade.live_status == "open"
                and trade.live_exit_fill_price is None
            ):
                logger.warning(
                    f"[{asset}] Live exit order did not fill — deferring to binary resolution. "
                    f"Position stays open on exchange."
                )
                return

            use_live = (
                self.live_trading
                and trade.has_live_data
                and trade.live_exit_fill_price is not None
            )

            if use_live:
                # Use actual exchange fill data for P&L
                actual_entry = trade.live_entry_fill_price
                actual_exit = trade.live_exit_fill_price
                actual_contracts = trade.live_entry_contracts or n_contracts
                actual_entry_fee = trade.live_entry_fee or 0.0
                actual_exit_fee = trade.live_exit_fee or 0.0

                gross_pnl = actual_contracts * (actual_exit - actual_entry)
                total_fees = actual_entry_fee + actual_exit_fee
                net_pnl = gross_pnl - total_fees

                logger.info(
                    f"[{asset}] Using LIVE fill data: entry={actual_entry:.4f} "
                    f"exit={actual_exit:.4f} fees=${total_fees:.4f}"
                )
            else:
                # Fall back to simulated estimates
                gross_pnl = n_contracts * (exit_contract - entry_contract)
                total_fees = self._calculate_fees(
                    entry_contract, exit_contract, n_contracts, trade.bet_size
                )
                net_pnl = gross_pnl - total_fees

            # Update portfolio state
            # In live mode, fetch actual wallet balance as source of truth
            if self.live_trading and self.exchange and self.exchange.supports_trading:
                wallet_balance = await self._fetch_wallet_balance()
                if wallet_balance is not None:
                    self.state.current_bankroll = wallet_balance
                    # Wallet-derived P&L — the only source of truth when live
                    self.state.total_pnl = wallet_balance - self.state.initial_bankroll
                    logger.info(
                        f"[{asset}] Wallet balance (source of truth): ${wallet_balance:.2f}  "
                        f"P&L=${self.state.total_pnl:+.2f}"
                    )
                else:
                    self.state.current_bankroll += net_pnl
                    self.state.total_pnl += net_pnl
            else:
                self.state.current_bankroll += net_pnl
                self.state.total_pnl += net_pnl

            # Track daily P&L for daily loss limit
            self._record_daily_pnl(net_pnl)

            self.state.total_trades += 1

            won = net_pnl > 0

            # Record outcome for adaptive direction filter
            self._adaptive_record_outcome(trade.direction, won)

            if won:
                self.state.total_wins += 1
                self.state.current_win_streak += 1
                self.state.current_loss_streak = 0
                self.state.max_win_streak = max(
                    self.state.max_win_streak,
                    self.state.current_win_streak,
                )
                # Recovery sizing: reset per-asset loss streak on win
                if self.recovery_sizing != "none":
                    prev_losses = self._asset_consecutive_losses.get(asset, 0)
                    self._asset_consecutive_losses[asset] = 0
                    if prev_losses > 0:
                        logger.info(f"[{asset}] Recovery: WIN resets loss streak (was {prev_losses})")
            else:
                self.state.total_losses += 1
                self.state.current_loss_streak += 1
                self.state.current_win_streak = 0
                self.state.max_loss_streak = max(
                    self.state.max_loss_streak,
                    self.state.current_loss_streak,
                )
                # Recovery sizing: increment per-asset loss streak
                if self.recovery_sizing != "none":
                    self._asset_consecutive_losses[asset] = self._asset_consecutive_losses.get(asset, 0) + 1
                    next_bet = self._get_recovery_bet(asset)
                    logger.info(
                        f"[{asset}] Recovery: loss streak → {self._asset_consecutive_losses[asset]}, "
                        f"next bet ${next_bet:.2f}"
                    )
                # Trigger pause after loss
                if self.pause_windows_after_loss > 0:
                    self.state.pause_windows_remaining = self.pause_windows_after_loss
                    logger.info(
                        f"[{asset}] LOSS DETECTED: Pausing for {self.pause_windows_after_loss} windows"
                    )

            # Update bet size (fixed for contrarian-family, dynamic for others)
            if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
                self.state.current_bet_size = self._get_scaled_bet()
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
            trade.fee_paid = total_fees
            trade.spread_cost = 0.0
            trade.net_pnl = net_pnl
            trade.bankroll_after = self.state.current_bankroll
            trade.drawdown = drawdown
            trade.drawdown_pct = drawdown_pct
            trade.resolved_at = datetime.now(timezone.utc)

            # Save to database
            await self.trading_db.update_trade(trade)

            # Remove from open trades
            del self.open_trades[asset]

            # Update group loss counter for circuit breaker
            self._update_group_loss_counter(asset, net_pnl)

            # Log result
            result_str = "WIN" if won else "LOSS"
            pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
            source = "LIVE" if use_live else "SIM"
            logger.info(
                f"[{asset}] CONTRARIAN EXIT ({source}): {result_str} | "
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

    async def on_sample_at_consensus_t0(self, asset: str, sample, state):
        """Record pm_yes at t0 for momentum/sweet-spot checks."""
        pm_yes = sample.pm_yes_price
        if pm_yes is not None:
            self._consensus_t0_pm[asset] = pm_yes
        else:
            self._consensus_t0_pm.pop(asset, None)

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

        # --- Pre-filters: regime and day-of-week ---

        # Day-of-week filter (using time_key format "YYYYMMDD_HHMM")
        if self.skip_days:
            try:
                from datetime import datetime
                dt = datetime.strptime(time_key[:8], "%Y%m%d")
                if dt.weekday() in self.skip_days:
                    logger.info(
                        f"[CONSENSUS] Skipping — day {dt.strftime('%A')} is in skip_days"
                    )
                    for asset in buffer:
                        self._last_signals[asset] = "skip-day"
                    return
            except (ValueError, IndexError):
                pass  # Can't parse date, proceed anyway

        # Hour-of-day filter (UTC)
        if self.allowed_hours:
            try:
                hour = int(time_key[9:11])  # "YYYYMMDD_HHMM" → positions 9-10 = "HH"
                if hour not in self.allowed_hours:
                    logger.info(
                        f"[CONSENSUS] Skipping — UTC hour {hour:02d} not in allowed_hours"
                    )
                    for asset in buffer:
                        self._last_signals[asset] = "skip-hour"
                    return
            except (ValueError, IndexError):
                pass  # Can't parse hour, proceed anyway

        # Circuit breaker: skip window if too many consecutive group losses
        if self.max_consec_losses > 0 and self._group_consec_losses >= self.max_consec_losses:
            logger.info(
                f"[CONSENSUS] CIRCUIT BREAKER: {self._group_consec_losses} consecutive "
                f"group losses → skipping and resetting counter"
            )
            self._group_consec_losses = 0
            for asset in buffer:
                self._last_signals[asset] = "skip-circuit-break"
            return

        # Regime filter: skip if majority of assets had high/extreme previous window
        if self.skip_regimes:
            skip_count = sum(
                1 for asset in buffer
                if self._prev_window_regime.get(asset) in self.skip_regimes
            )
            if skip_count >= len(buffer) / 2:
                regime_summary = {
                    a: self._prev_window_regime.get(a, "?") for a in buffer
                }
                logger.info(
                    f"[CONSENSUS] Skipping — {skip_count}/{len(buffer)} assets had "
                    f"skip regime in prev window: {regime_summary}"
                )
                for asset in buffer:
                    self._last_signals[asset] = "skip-regime"
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

        # Prior momentum filter: require at least 1 asset with trend still building
        # For bear: prev_pm moved UP last window (prev - prev2 > threshold)
        # For bull: prev_pm moved DOWN last window (prev2 - prev > threshold)
        # Uses prev_window_pm and prev2_window_pm — both t0-valid (no look-ahead)
        if self.prior_mom_filter:
            n_confirm_prior = 0
            for asset in buffer:
                p1 = self._prev_window_pm.get(asset)
                p2 = self._prev2_window_pm.get(asset)
                if p1 is None or p2 is None:
                    continue
                delta = p1 - p2
                if contrarian_dir == "bear" and delta > self.prior_mom_min:
                    n_confirm_prior += 1
                elif contrarian_dir == "bull" and delta < -self.prior_mom_min:
                    n_confirm_prior += 1
            if n_confirm_prior < 1:
                logger.info(
                    f"[CONSENSUS] Prior momentum filter FAIL: 0/{len(buffer)} assets "
                    f"had |Δprev| > {self.prior_mom_min} for {contrarian_dir}"
                )
                for asset in buffer:
                    self._last_signals[asset] = f"{contrarian_dir}-priormom-fail"
                return
            logger.info(
                f"[CONSENSUS] Prior momentum filter PASS: {n_confirm_prior}/{len(buffer)} "
                f"assets had prev window building {contrarian_dir}"
            )

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
            f"confirm {contrarian_dir.upper()} reversal"
        )

        # Phase 2.5: Sweet spot filter — require t0 pm_yes near 0.50
        asset_momentum = {}  # asset -> signed momentum (for confidence scaling)
        if self.sweet_spot_band > 0:
            sweet_pass = []
            for asset in confirming_assets:
                t0_pm = self._consensus_t0_pm.get(asset)
                if t0_pm is not None and abs(t0_pm - 0.50) <= self.sweet_spot_band:
                    sweet_pass.append(asset)
                else:
                    self._last_signals[asset] = f"{contrarian_dir}-sweet-fail"
                    logger.debug(
                        f"[{asset}] Sweet spot filter: t0={t0_pm} "
                        f"outside band ±{self.sweet_spot_band}"
                    )
            if len(sweet_pass) < len(confirming_assets):
                logger.info(
                    f"[CONSENSUS] Sweet spot filter: {len(sweet_pass)}/{len(confirming_assets)} "
                    f"assets pass (t0 within ±{self.sweet_spot_band} of 0.50)"
                )
            confirming_assets = sweet_pass

        # Phase 3: Momentum confirmation — require price movement 0→entry confirms direction
        if self.momentum_min_threshold > 0:
            mom_pass = []
            for asset in confirming_assets:
                t0_pm = self._consensus_t0_pm.get(asset)
                current_pm = buffer[asset]["pm_yes"]
                if t0_pm is None or current_pm is None:
                    self._last_signals[asset] = f"{contrarian_dir}-mom-nodata"
                    continue
                momentum = current_pm - t0_pm  # signed: positive = price rising
                asset_momentum[asset] = momentum
                if contrarian_dir == "bull" and momentum >= self.momentum_min_threshold:
                    mom_pass.append(asset)
                elif contrarian_dir == "bear" and momentum <= -self.momentum_min_threshold:
                    mom_pass.append(asset)
                else:
                    self._last_signals[asset] = f"{contrarian_dir}-mom-fail"
                    logger.debug(
                        f"[{asset}] Momentum filter: {contrarian_dir} needs "
                        f"{'>=+' if contrarian_dir == 'bull' else '<=-'}"
                        f"{self.momentum_min_threshold}, got {momentum:+.4f}"
                    )
            logger.info(
                f"[CONSENSUS] Momentum filter (>={self.momentum_min_threshold}): "
                f"{len(mom_pass)}/{len(confirming_assets)} assets pass"
            )
            confirming_assets = mom_pass
        else:
            # No momentum filter — compute momentum anyway for confidence scaling
            for asset in confirming_assets:
                t0_pm = self._consensus_t0_pm.get(asset)
                current_pm = buffer[asset]["pm_yes"]
                if t0_pm is not None and current_pm is not None:
                    asset_momentum[asset] = current_pm - t0_pm

        if not confirming_assets:
            logger.info(f"[CONSENSUS] No assets remain after filters")
            return

        # Daily loss limit check
        if self.daily_loss_limit > 0:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_pnl = self._daily_pnl.get(today, 0.0)
            if daily_pnl <= -self.daily_loss_limit:
                logger.info(
                    f"[CONSENSUS] DAILY LIMIT: P&L today ${daily_pnl:.2f} "
                    f"exceeds -${self.daily_loss_limit:.2f} limit. Skipping."
                )
                for asset in buffer:
                    self._last_signals[asset] = f"{contrarian_dir}-daily-limit"
                return

        # Adaptive direction filter: skip if this direction isn't favored
        if not self._adaptive_direction_allowed(contrarian_dir):
            logger.info(
                f"[CONSENSUS] ADAPTIVE SKIP: {contrarian_dir.upper()} not favored. "
                f"{self._adaptive_status()}"
            )
            for asset in buffer:
                self._last_signals[asset] = f"{contrarian_dir}-adapt-skip"
            return

        logger.info(f"[CONSENSUS] → ENTERING TRADES ({len(confirming_assets)} assets)")

        # Tiered exit: use binary resolution if consensus is strong enough
        use_resolution = (
            self.tiered_exit and len(confirming_assets) >= self.tiered_resolution_threshold
        ) or self.hold_to_resolution
        if self.tiered_exit:
            logger.info(
                f"[CONSENSUS] Tiered exit: n_agree={len(confirming_assets)}, "
                f"threshold={self.tiered_resolution_threshold} → "
                f"{'RESOLUTION' if use_resolution else 'EARLY EXIT'}"
            )

        # Enter trades for all confirming assets
        for asset in confirming_assets:
            await self._enter_consensus_trade(
                asset, contrarian_dir, buffer[asset],
                momentum=asset_momentum.get(asset),
                time_key=time_key,
                use_resolution=use_resolution,
            )

        # Mark non-confirming assets
        for asset in buffer:
            if asset not in confirming_assets and asset not in self._last_signals:
                self._last_signals[asset] = f"{contrarian_dir}-no-confirm"

    async def _enter_consensus_trade(
        self, asset: str, direction: str, data: dict,
        momentum: float = None,
        time_key: str = None,
        use_resolution: bool = False,
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

            # --- Bet sizing with all modifiers ---
            base = self._get_recovery_bet(asset)

            # 1. Confidence scaling (based on momentum strength)
            confidence_scale = 1.0
            if self.confidence_scaling_max > 1.0 and momentum is not None:
                abs_mom = abs(momentum)
                max_extra = self.confidence_scaling_max - 1.0
                confidence_scale = 1.0 + min(abs_mom / self.confidence_scaling_ref, max_extra)

            # 2. Asymmetric direction sizing
            dir_mult = self.bear_size_mult if direction == "bear" else self.bull_size_mult

            # 3. Anti-martingale (scale up after consecutive wins)
            anti_mart_scale = 1.0
            if self.anti_mart_mult > 0 and self.state.current_win_streak > 0:
                anti_mart_scale = min(
                    self.anti_mart_mult ** self.state.current_win_streak,
                    self.anti_mart_max,
                )

            bet_size = base * confidence_scale * dir_mult * anti_mart_scale
            bet_size = min(bet_size, self.state.current_bankroll * self.max_bet_pct)

            # Compute signal metadata
            _spot_vel = sample.spot_price_change_from_open
            _pm_mom = abs(momentum) if momentum is not None else abs(pm_yes - 0.50)

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

            # Per-trade resolution mode (tiered exit)
            self._trade_use_resolution[asset] = use_resolution

            # Group loss tracking (circuit breaker)
            if time_key and self.max_consec_losses > 0:
                self._trade_group_key[asset] = time_key
                self._group_entered_assets.setdefault(time_key, set()).add(asset)
                self._group_pnl_acc.setdefault(time_key, 0.0)

            # Save state
            await self.trading_db.save_state(self.state)

            # Build sizing detail string
            sizing_parts = [f"bet=${bet_size:.2f}"]
            if confidence_scale > 1.001:
                sizing_parts.append(f"conf={confidence_scale:.2f}x")
            if dir_mult != 1.0:
                sizing_parts.append(f"dir={dir_mult:.2f}x")
            if anti_mart_scale > 1.001:
                sizing_parts.append(f"antimart={anti_mart_scale:.2f}x")
            if momentum is not None:
                sizing_parts.append(f"mom={momentum:+.4f}")
            sizing_str = " ".join(sizing_parts)

            logger.info(
                f"[{asset}] OPENED CONSENSUS {direction.upper()} position: "
                f"entry_contract={entry_price:.3f} (pm_yes={pm_yes:.3f}), "
                f"{sizing_str}, window={sample.window_id}"
            )

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

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

            bet_size = min(self._get_recovery_bet(asset), self.state.current_bankroll * self.max_bet_pct)

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

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

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

            bet_size = min(self._get_recovery_bet(asset), self.state.current_bankroll * self.max_bet_pct)

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

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

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

            bet_size = min(self._get_recovery_bet(asset), self.state.current_bankroll * self.max_bet_pct)

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

            # Place live order if enabled; abort sim trade if exchange rejects
            live_result = await self._place_live_order(
                asset, direction, entry_price, bet_size, trade=trade,
            )
            if live_result is None and self.live_trading:
                # Exchange order failed — don't keep phantom sim position
                logger.warning(
                    f"[{asset}] Live order failed — removing sim trade "
                    f"(no position on exchange)"
                )
                self.open_trades.pop(asset, None)
                trade.outcome = "cancelled"
                trade.live_status = "entry_failed"
                await self.trading_db.update_trade(trade)
                return

        except Exception as e:
            logger.error(f"[{asset}] Error in triple trade entry: {e}", exc_info=True)

    async def on_sample_at_triple_exit(self, asset: str, sample, state):
        """Handle triple_filter exit — same early-exit P&L as contrarian."""
        await self.on_sample_at_contrarian_exit(asset, sample, state)

    def _update_group_loss_counter(self, asset: str, net_pnl: float) -> None:
        """Update group-level consecutive loss counter for the circuit breaker.

        Called whenever a consensus trade closes (early exit or binary resolution).
        Cleans up per-trade state and, once the last trade in a group resolves,
        updates _group_consec_losses used by max_consec_losses in _evaluate_consensus.

        Args:
            asset: Asset that just closed its trade
            net_pnl: Net P&L of the closed trade
        """
        # Always clean up per-trade resolution flag
        self._trade_use_resolution.pop(asset, None)

        # Only track group losses if circuit breaker is enabled
        group_key = self._trade_group_key.pop(asset, None)
        if group_key is None or self.max_consec_losses <= 0:
            return

        # Accumulate P&L for this group
        self._group_pnl_acc[group_key] = self._group_pnl_acc.get(group_key, 0.0) + net_pnl

        # Remove this asset from the group's pending set
        group_assets = self._group_entered_assets.get(group_key)
        if group_assets is not None:
            group_assets.discard(asset)
            if not group_assets:
                # All trades from this group have resolved — update streak
                group_pnl = self._group_pnl_acc.pop(group_key, 0.0)
                self._group_entered_assets.pop(group_key, None)
                if group_pnl > 0:
                    self._group_consec_losses = 0
                else:
                    self._group_consec_losses += 1
                logger.info(
                    f"[GROUP {group_key}] All trades closed: "
                    f"P&L=${group_pnl:.2f}, "
                    f"consecutive losses={self._group_consec_losses}"
                )

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
            # Cancel any stale open orders from this window before moving to next
            await self._cancel_stale_orders(asset)

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

                # Store previous window's volatility regime
                regime = getattr(window, 'volatility_regime', None)
                if regime:
                    self._prev_window_regime[asset] = regime

            # Resolve any open position for this asset (non-contrarian modes)
            # Contrarian-family trades are already closed at exit time, UNLESS
            # hold_to_resolution=True or this trade was flagged for tiered exit.
            if asset in self.open_trades:
                if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
                    use_res = (
                        self.hold_to_resolution
                        or self._trade_use_resolution.get(asset, False)
                    )
                    if use_res:
                        mode = (
                            "tiered_exit"
                            if self._trade_use_resolution.get(asset, False)
                            else "hold_to_resolution"
                        )
                        logger.info(
                            f"[{asset}] {mode}: resolving at binary outcome"
                        )
                    else:
                        # Trade wasn't closed at exit time (missed exit?)
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

        For binary resolution (held to expiry), the payout is 1.0 (win) or 0.0 (loss).
        When live trading, uses actual entry fill price and fees from exchange.
        In live mode, also fetches the actual wallet balance as source of truth.

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

        # Check for live fill data on the trade itself
        use_live = self.live_trading and trade.has_live_data

        if use_live:
            # Use actual entry fill price and fee from exchange
            actual_entry = trade.live_entry_fill_price
            actual_contracts = trade.live_entry_contracts or (
                trade.bet_size / actual_entry if actual_entry > 0.001 else 0
            )
            entry_fee = trade.live_entry_fee or 0.0

            # Binary settlement: payout is 1.0 (win) or 0.0 (loss)
            payout = 1.0 if won else 0.0
            gross_profit = actual_contracts * (payout - actual_entry)
            # No exit fee on binary resolution (exchange settles automatically)
            total_fees = entry_fee
            net_pnl = gross_profit - total_fees

            # Update live status on the trade
            trade.live_status = "settled"

            logger.info(
                f"[{asset}] Using LIVE fill data for resolution: "
                f"entry={actual_entry:.4f} contracts={actual_contracts:.1f} "
                f"fee=${entry_fee:.4f}"
            )
        else:
            # Simulated P&L
            n_contracts = trade.bet_size / trade.entry_price if trade.entry_price > 0.001 else 0

            if won:
                gross_profit = (1 - trade.entry_price) * trade.bet_size
            else:
                gross_profit = -trade.entry_price * trade.bet_size

            exit_contract = 1.0 if won else 0.0
            total_fees = self._calculate_fees(
                trade.entry_price, exit_contract, n_contracts, trade.bet_size
            )
            net_pnl = gross_profit - total_fees

        # In live mode, fetch actual wallet balance as source of truth
        if self.live_trading and self.exchange and self.exchange.supports_trading:
            wallet_balance = await self._fetch_wallet_balance()
            if wallet_balance is not None:
                self.state.current_bankroll = wallet_balance
                # Wallet-derived P&L — the only source of truth when live
                self.state.total_pnl = wallet_balance - self.state.initial_bankroll
                logger.info(
                    f"[{asset}] Wallet balance (source of truth): ${wallet_balance:.2f}  "
                    f"P&L=${self.state.total_pnl:+.2f}"
                )
            else:
                self.state.current_bankroll += net_pnl
                self.state.total_pnl += net_pnl
        else:
            self.state.current_bankroll += net_pnl
            self.state.total_pnl += net_pnl

        # Track daily P&L for daily loss limit
        self._record_daily_pnl(net_pnl)

        self.state.total_trades += 1

        if won:
            self.state.total_wins += 1
            self.state.current_win_streak += 1
            self.state.current_loss_streak = 0
            self.state.max_win_streak = max(
                self.state.max_win_streak,
                self.state.current_win_streak,
            )
            # Recovery sizing: reset per-asset loss streak on win
            if self.recovery_sizing != "none":
                prev_losses = self._asset_consecutive_losses.get(asset, 0)
                self._asset_consecutive_losses[asset] = 0
                if prev_losses > 0:
                    logger.info(f"[{asset}] Recovery: WIN resets loss streak (was {prev_losses})")
        else:
            self.state.total_losses += 1
            self.state.current_loss_streak += 1
            self.state.current_win_streak = 0
            self.state.max_loss_streak = max(
                self.state.max_loss_streak,
                self.state.current_loss_streak,
            )
            # Recovery sizing: increment per-asset loss streak
            if self.recovery_sizing != "none":
                self._asset_consecutive_losses[asset] = self._asset_consecutive_losses.get(asset, 0) + 1
                next_bet = self._get_recovery_bet(asset)
                logger.info(
                    f"[{asset}] Recovery: loss streak → {self._asset_consecutive_losses[asset]}, "
                    f"next bet ${next_bet:.2f}"
                )
            if self.pause_windows_after_loss > 0:
                self.state.pause_windows_remaining = self.pause_windows_after_loss
                logger.info(
                    f"[{asset}] LOSS DETECTED: Pausing for {self.pause_windows_after_loss} windows"
                )

        # Update bet size for next trade based on new win streak
        if self.entry_mode in ("contrarian", "contrarian_consensus", "accel_dbl", "combo_dbl", "triple_filter"):
            self.state.current_bet_size = self._get_scaled_bet()
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
        trade.exit_time = window.window_end_utc
        trade.exit_price = window.spot_close
        trade.outcome = "win" if won else "loss"
        trade.gross_pnl = gross_profit
        trade.fee_paid = total_fees
        trade.spread_cost = 0.0
        trade.net_pnl = net_pnl
        trade.bankroll_after = self.state.current_bankroll
        trade.drawdown = drawdown
        trade.drawdown_pct = drawdown_pct
        trade.resolved_at = datetime.now(timezone.utc)

        # Save to database
        await self.trading_db.update_trade(trade)

        # Remove from open trades
        del self.open_trades[asset]

        # Update group loss counter for circuit breaker
        self._update_group_loss_counter(asset, net_pnl)

        # Log result
        result_str = "WIN" if won else "LOSS"
        pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
        source = "LIVE" if use_live else "SIM"
        logger.info(
            f"[{asset}] Trade resolved ({source}): {result_str} | "
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
            "scaled_bet": self._get_scaled_bet(),
            "bet_scale_threshold": self.bet_scale_threshold,
            "bet_scale_increase": self.bet_scale_increase,
            "fee_rate": self.fee_rate,
            "spread_cost": self.spread_cost,
            "fee_model": self.fee_model,
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
            if self.momentum_min_threshold > 0:
                config["momentum_min_threshold"] = self.momentum_min_threshold
            if self.confidence_scaling_max > 1.0:
                config["confidence_scaling_max"] = self.confidence_scaling_max
                config["confidence_scaling_ref"] = self.confidence_scaling_ref
            if self.daily_loss_limit > 0:
                config["daily_loss_limit"] = self.daily_loss_limit
            if self.bear_size_mult != 1.0 or self.bull_size_mult != 1.0:
                config["bear_size_mult"] = self.bear_size_mult
                config["bull_size_mult"] = self.bull_size_mult
            if self.anti_mart_mult > 0:
                config["anti_mart_mult"] = self.anti_mart_mult
                config["anti_mart_max"] = self.anti_mart_max
            if self.hold_to_resolution:
                config["hold_to_resolution"] = True
            if self.sweet_spot_band > 0:
                config["sweet_spot_band"] = self.sweet_spot_band
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
        """Calculate comprehensive trading metrics.

        All metrics are computed purely from trade history (matching
        chart.py methodology) rather than cached sim_state values,
        which may be stale across sessions/resets.
        """
        if self.state is None or self.state.total_trades == 0:
            return {}

        trades = await self.trading_db.get_all_trades(resolved_only=True)

        if not trades:
            return {}

        # --- Compute everything from trade records (like chart.py) ---
        pnls = [t.net_pnl for t in trades if t.net_pnl is not None]
        wins = [t for t in trades if t.outcome == "win"]
        losses = [t for t in trades if t.outcome == "loss"]

        if not pnls:
            return {}

        import numpy as np

        pnl_array = np.array(pnls)
        total_pnl = float(np.sum(pnl_array))

        # Profit factor (matches chart.py: sum positive win pnls / abs sum negative loss pnls)
        gross_wins = sum(t.net_pnl for t in wins if t.net_pnl and t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in losses if t.net_pnl and t.net_pnl < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Average win/loss
        win_pnls = [t.net_pnl for t in wins if t.net_pnl is not None]
        loss_pnls = [t.net_pnl for t in losses if t.net_pnl is not None]
        avg_win = float(np.mean(win_pnls)) if win_pnls else 0
        avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0

        # Win rate from actual trades
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = len(trades)
        win_rate = n_wins / n_total if n_total > 0 else 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # --- Rebuild equity curve from trades (matches chart.py exactly) ---
        equity = [self.initial_bankroll]
        for t in trades:
            if t.bankroll_after is not None:
                equity.append(t.bankroll_after)

        # Drawdown from equity curve (matches chart.py)
        peak = self.initial_bankroll
        drawdowns = []
        for val in equity:
            peak = max(peak, val)
            dd_pct = ((val - peak) / peak) * 100 if peak > 0 else 0
            drawdowns.append(dd_pct)

        max_dd_pct = min(drawdowns) if drawdowns else 0

        # Max drawdown in dollars (matches chart.py recovery factor calc)
        max_dd_dollars = min(
            equity[i] - max(equity[:i + 1]) for i in range(len(equity))
        )

        # Calmar ratio (matches chart.py: return_pct / |max_dd_pct|)
        total_return_pct = (total_pnl / self.initial_bankroll) * 100
        calmar = total_return_pct / abs(max_dd_pct) if max_dd_pct < 0 else 0

        # Recovery factor (matches chart.py: total_pnl / |max_dd_dollars|)
        recovery_factor = total_pnl / abs(max_dd_dollars) if max_dd_dollars < 0 else float("inf")

        # Sharpe ratio (per-trade, not annualized — avoids inflated values
        # since the bot doesn't trade every 15-min window)
        if len(pnls) > 1 and np.std(pnl_array) > 0:
            sharpe = float(np.mean(pnl_array) / np.std(pnl_array))
        else:
            sharpe = 0

        # Sortino ratio (per-trade, using downside deviation)
        downside = np.array([p for p in pnls if p < 0])
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(np.mean(pnl_array) / np.std(downside))
        else:
            sortino = 0

        return {
            "total_trades": n_total,
            "total_wins": n_wins,
            "total_losses": n_losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": float(np.mean(pnl_array)),
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "recovery_factor": recovery_factor,
            "max_drawdown": max_dd_dollars,
            "max_drawdown_pct": max_dd_pct,
            "current_win_streak": self.state.current_win_streak,
            "current_loss_streak": self.state.current_loss_streak,
            "max_win_streak": self.state.max_win_streak,
            "max_loss_streak": self.state.max_loss_streak,
        }
