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

    # Entry mode: "two_stage", "single", "contrarian", or "contrarian_consensus"
    # - two_stage: signal at t=7.5, confirm at t=10, hold to resolution
    # - single: enter at t=7.5, hold to resolution
    # - contrarian: after strong prev window, enter at t=0, sell at t=12.5
    # - contrarian_consensus: contrarian + require N-of-4 assets to agree
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

    # Signal quality filter
    min_trajectory: float = 0.20  # Min PM price movement from t=0 to entry

    # Risk management
    pause_windows_after_loss: int = 2  # Skip N windows after any loss

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
