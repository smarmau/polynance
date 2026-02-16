"""Trading configuration management."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = Path("config/config.json")


@dataclass
class TradingConfig:
    """Trading configuration settings."""

    # Exchange: "polymarket" or "kalshi"
    exchange: str = "polymarket"

    # Fee model: "flat" (polymarket) or "probability_weighted" (kalshi)
    fee_model: str = "flat"

    # Entry mode: "two_stage", "single", "contrarian", "contrarian_consensus",
    #              "accel_dbl", "combo_dbl", or "triple_filter"
    # - two_stage: signal at t=7.5, confirm at t=10, hold to resolution
    # - single: enter at t=7.5, hold to resolution
    # - contrarian: after strong prev window, enter at t=0, sell at t=12.5
    # - contrarian_consensus: contrarian + require N-of-4 assets to agree
    # - accel_dbl: double contrarian + t0 near neutral (acceleration filter)
    # - combo_dbl: double contrarian + stop-loss at t7.5 + cross-asset filter
    # - triple_filter: double contrarian + cross-asset consensus + PM t0 confirmation
    entry_mode: str = "two_stage"

    # Two-stage thresholds (entry_mode="two_stage")
    signal_threshold_bull: float = 0.70  # t=7.5: initial signal if pm_yes >= this
    signal_threshold_bear: float = 0.30  # t=7.5: initial signal if pm_yes <= this
    confirm_threshold_bull: float = 0.85  # t=10: confirm and enter if pm_yes >= this
    confirm_threshold_bear: float = 0.15  # t=10: confirm and enter if pm_yes <= this

    # Single-mode thresholds (entry_mode="single", also used as legacy fallback)
    bull_threshold: float = 0.80  # BUY YES if pm_yes >= this
    bear_threshold: float = 0.20  # BUY NO if pm_yes <= this

    # Contrarian thresholds (entry_mode="contrarian" or "contrarian_consensus")
    # Previous window strength: pm@t12.5 must be >= this (or <= 1-this) to trigger
    contrarian_prev_thresh: float = 0.75
    # Current window confirmation: entry pm must confirm reversal direction
    contrarian_bull_thresh: float = 0.60  # enter bull if pm_yes >= this (after prev strong DOWN)
    contrarian_bear_thresh: float = 0.40  # enter bear if pm_yes <= this (after prev strong UP)
    # Entry/exit timing for contrarian
    contrarian_entry_time: str = "t0"    # when to enter: t0, t2.5, t5
    contrarian_exit_time: str = "t12.5"  # when to exit: t10, t12.5

    # Consensus settings (entry_mode="contrarian_consensus")
    # Requires N-of-4 assets to agree on both previous window strength AND current reversal
    consensus_min_agree: int = 3          # minimum assets agreeing (out of 4)
    consensus_entry_time: str = "t5"      # when to check consensus & enter (t0, t2.5, t5)
    consensus_exit_time: str = "t12.5"    # when to exit consensus trades

    # ACCEL_DBL settings (entry_mode="accel_dbl")
    # Double contrarian + t0 near neutral acceleration filter
    # Requires TWO consecutive strong prev windows, then t0 must be near 0.50
    accel_neutral_band: float = 0.15      # t0 must be within this of 0.50
    accel_prev_thresh: float = 0.75       # prev window strength threshold
    accel_bull_thresh: float = 0.55       # entry bull threshold at t5
    accel_bear_thresh: float = 0.45       # entry bear threshold at t5
    accel_entry_time: str = "t5"          # when to enter
    accel_exit_time: str = "t12.5"        # when to exit

    # COMBO_DBL settings (entry_mode="combo_dbl")
    # Double contrarian + stop-loss at t7.5 + cross-asset agreement
    combo_prev_thresh: float = 0.75       # prev window strength threshold
    combo_bull_thresh: float = 0.55       # entry bull threshold at t5
    combo_bear_thresh: float = 0.45       # entry bear threshold at t5
    combo_entry_time: str = "t5"          # when to enter
    combo_exit_time: str = "t12.5"        # normal exit time
    combo_stop_time: str = "t7.5"         # when to check stop-loss
    combo_stop_delta: float = 0.10        # exit early if position moves against by this
    combo_xasset_min: int = 2             # min OTHER assets also double-strong prev

    # TRIPLE FILTER settings (entry_mode="triple_filter")
    # Double contrarian + cross-asset consensus (N dbl-strong) + PM t0 confirmation
    triple_prev_thresh: float = 0.70       # prev window strength threshold
    triple_bull_thresh: float = 0.55       # entry bull threshold
    triple_bear_thresh: float = 0.45       # entry bear threshold
    triple_entry_time: str = "t5"          # when to enter
    triple_exit_time: str = "t12.5"        # when to exit
    triple_xasset_min: int = 3             # min assets with double-strong prev
    triple_pm0_bull_min: float = 0.50      # pm t0 must be >= this for bull
    triple_pm0_bear_max: float = 0.50      # pm t0 must be <= this for bear

    # Portfolio settings
    initial_bankroll: float = 1000.0
    base_bet: float = 25.0

    # Fee structure
    fee_rate: float = 0.02  # 2% on profits
    spread_cost: float = 0.006  # 0.6% on all trades

    # Bet sizing (Slow Growth)
    growth_per_win: float = 0.10  # 10% growth per consecutive win
    max_bet_multiplier: float = 2.0  # Cap at 2x base bet
    max_bet_pct: float = 0.05  # 5% of bankroll

    # Bet scaling: step-increase base_bet as bankroll grows
    bet_scale_threshold: float = 0.0  # 0=disabled, 1.0=every 100% gain
    bet_scale_increase: float = 0.0   # 0.20 = +20% per threshold step

    # Live trading (CAUTION: places real orders with real money)
    live_trading: bool = False  # False = dry-run only (default, safe)

    # Wallet signature type for py-clob-client:
    #   0 = EOA / external wallet (MetaMask, hardware wallet) — default
    #   1 = Polymarket proxy / email wallet (the web app wallet)
    #   2 = Browser wallet proxy / Gnosis Safe
    signature_type: int = 1

    # Signal quality filter
    min_trajectory: float = 0.20  # Min PM price movement from t=0 to entry

    # Risk management
    pause_windows_after_loss: int = 2  # Skip N windows after any loss

    # Recovery sizing: step up bet size after consecutive per-asset losses
    # "none" = flat base_bet, "linear" = base + step*losses, "mart_1.5x" = base * 1.5^losses
    recovery_sizing: str = "none"
    recovery_step: float = 25.0           # linear: add this per loss (e.g. $25)
    recovery_max_multiplier: int = 5      # cap at base_bet * this (e.g. 5 → $125 max)

    # Regime filter: skip entry when previous window's volatility was in these regimes
    # Valid values: "low", "normal", "high", "extreme"
    # e.g. ["high", "extreme"] to only trade after low/normal vol windows
    skip_regimes: list = field(default_factory=lambda: ["high", "extreme"])

    # Day-of-week filter: skip entry on these days (0=Mon..6=Sun)
    # e.g. [5] to skip Saturdays (data shows Saturday is consistently unprofitable)
    skip_days: list = field(default_factory=lambda: [5])

    # Assets to trade
    assets: list = field(default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"])

    # Data directory
    data_dir: str = "data"

    # Dashboard settings
    show_dashboard: bool = True
    dashboard_refresh_rate: float = 2.0

    # Analysis settings
    run_analysis: bool = False  # Disable by default for trading mode

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_trading_config(self) -> dict:
        """Convert to trading_config dict for Application."""
        return {
            "exchange": self.exchange,
            "fee_model": self.fee_model,
            "entry_mode": self.entry_mode,
            "signal_threshold_bull": self.signal_threshold_bull,
            "signal_threshold_bear": self.signal_threshold_bear,
            "confirm_threshold_bull": self.confirm_threshold_bull,
            "confirm_threshold_bear": self.confirm_threshold_bear,
            "initial_bankroll": self.initial_bankroll,
            "base_bet": self.base_bet,
            "fee_rate": self.fee_rate,
            "spread_cost": self.spread_cost,
            "bull_threshold": self.bull_threshold,
            "bear_threshold": self.bear_threshold,
            "max_bet_pct": self.max_bet_pct,
            "bet_scale_threshold": self.bet_scale_threshold,
            "bet_scale_increase": self.bet_scale_increase,
            "pause_windows_after_loss": self.pause_windows_after_loss,
            "growth_per_win": self.growth_per_win,
            "max_bet_multiplier": self.max_bet_multiplier,
            "min_trajectory": self.min_trajectory,
            "contrarian_prev_thresh": self.contrarian_prev_thresh,
            "contrarian_bull_thresh": self.contrarian_bull_thresh,
            "contrarian_bear_thresh": self.contrarian_bear_thresh,
            "contrarian_entry_time": self.contrarian_entry_time,
            "contrarian_exit_time": self.contrarian_exit_time,
            "consensus_min_agree": self.consensus_min_agree,
            "consensus_entry_time": self.consensus_entry_time,
            "consensus_exit_time": self.consensus_exit_time,
            "accel_neutral_band": self.accel_neutral_band,
            "accel_prev_thresh": self.accel_prev_thresh,
            "accel_bull_thresh": self.accel_bull_thresh,
            "accel_bear_thresh": self.accel_bear_thresh,
            "accel_entry_time": self.accel_entry_time,
            "accel_exit_time": self.accel_exit_time,
            "combo_prev_thresh": self.combo_prev_thresh,
            "combo_bull_thresh": self.combo_bull_thresh,
            "combo_bear_thresh": self.combo_bear_thresh,
            "combo_entry_time": self.combo_entry_time,
            "combo_exit_time": self.combo_exit_time,
            "combo_stop_time": self.combo_stop_time,
            "combo_stop_delta": self.combo_stop_delta,
            "combo_xasset_min": self.combo_xasset_min,
            "triple_prev_thresh": self.triple_prev_thresh,
            "triple_bull_thresh": self.triple_bull_thresh,
            "triple_bear_thresh": self.triple_bear_thresh,
            "triple_entry_time": self.triple_entry_time,
            "triple_exit_time": self.triple_exit_time,
            "triple_xasset_min": self.triple_xasset_min,
            "triple_pm0_bull_min": self.triple_pm0_bull_min,
            "triple_pm0_bear_max": self.triple_pm0_bear_max,
            "live_trading": self.live_trading,
            "signature_type": self.signature_type,
            "skip_regimes": self.skip_regimes,
            "skip_days": self.skip_days,
            "recovery_sizing": self.recovery_sizing,
            "recovery_step": self.recovery_step,
            "recovery_max_multiplier": self.recovery_max_multiplier,
        }

    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file."""
        path = path or DEFAULT_CONFIG_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TradingConfig":
        """Load configuration from JSON file.

        If file doesn't exist, returns default config and creates the file.
        """
        path = path or DEFAULT_CONFIG_PATH
        path = Path(path)

        if not path.exists():
            logger.info(f"Config file not found at {path}, using defaults")
            config = cls()
            config.save(path)
            return config

        try:
            with open(path) as f:
                data = json.load(f)

            # Handle assets field specially (may be stored as list)
            if "assets" in data and isinstance(data["assets"], list):
                pass  # Keep as list

            config = cls(**data)
            logger.info(f"Configuration loaded from {path}")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            logger.info("Using default configuration")
            return cls()

        except TypeError as e:
            logger.error(f"Invalid config values: {e}")
            logger.info("Using default configuration")
            return cls()

    @classmethod
    def create_default(cls, path: Optional[Path] = None) -> "TradingConfig":
        """Create and save a default configuration file."""
        config = cls()
        config.save(path)
        return config


def get_default_config_template() -> str:
    """Get a formatted JSON template with comments for documentation."""
    return """{
  // Entry Mode: "two_stage" (signal at t=7.5, confirm at t=10) or "single" (enter at t=7.5)
  "entry_mode": "two_stage",

  // Two-Stage Thresholds
  "signal_threshold_bull": 0.70,   // t=7.5: initial bull signal if pm_yes >= this
  "signal_threshold_bear": 0.30,   // t=7.5: initial bear signal if pm_yes <= this
  "confirm_threshold_bull": 0.85,  // t=10: confirm bull entry if pm_yes >= this
  "confirm_threshold_bear": 0.15,  // t=10: confirm bear entry if pm_yes <= this

  // Single-Mode Thresholds (legacy / entry_mode="single")
  "bull_threshold": 0.80,    // BUY YES if pm_yes >= this value
  "bear_threshold": 0.20,    // BUY NO if pm_yes <= this value

  // Portfolio Settings
  "initial_bankroll": 1000.0,  // Starting bankroll in USD
  "base_bet": 25.0,            // Base bet size in USD

  // Fee Structure (matches Polymarket)
  "fee_rate": 0.02,      // 2% fee on profits only
  "spread_cost": 0.006,  // 0.6% spread cost on all trades

  // Bet Sizing
  "max_bet_pct": 0.05,         // Maximum bet as % of bankroll (5%)

  // Signal Quality
  "min_trajectory": 0.20,  // Min PM price movement from t=0 to entry (filters weak signals)

  // Risk Management
  "pause_windows_after_loss": 2,  // Skip N windows after any loss (avoids clustering)

  // Assets to Trade
  "assets": ["BTC", "ETH", "SOL", "XRP"],

  // Data Directory
  "data_dir": "data",

  // Dashboard Settings
  "show_dashboard": true,
  "dashboard_refresh_rate": 2.0,

  // Analysis (run statistical analysis after each window)
  "run_analysis": false
}"""
