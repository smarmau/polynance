"""Trading configuration management."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = Path("config/trading.json")


@dataclass
class TradingConfig:
    """Trading configuration settings."""

    # Strategy thresholds
    bull_threshold: float = 0.80  # BUY YES if pm_yes >= this
    bear_threshold: float = 0.20  # BUY NO if pm_yes <= this

    # Portfolio settings
    initial_bankroll: float = 1000.0
    base_bet: float = 25.0

    # Fee structure
    fee_rate: float = 0.02  # 2% on profits
    spread_cost: float = 0.006  # 0.6% on all trades

    # Bet sizing (Anti-Martingale)
    win_multiplier: float = 2.0
    loss_multiplier: float = 0.5
    max_bet_pct: float = 0.05  # 5% of bankroll
    min_bet_pct: float = 0.10  # 10% of base bet (floor)

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
            "initial_bankroll": self.initial_bankroll,
            "base_bet": self.base_bet,
            "fee_rate": self.fee_rate,
            "spread_cost": self.spread_cost,
            "bull_threshold": self.bull_threshold,
            "bear_threshold": self.bear_threshold,
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
  // Strategy Thresholds
  "bull_threshold": 0.80,    // BUY YES if pm_yes >= this value
  "bear_threshold": 0.20,    // BUY NO if pm_yes <= this value

  // Portfolio Settings
  "initial_bankroll": 1000.0,  // Starting bankroll in USD
  "base_bet": 25.0,            // Base bet size in USD

  // Fee Structure (matches Polymarket)
  "fee_rate": 0.02,      // 2% fee on profits only
  "spread_cost": 0.006,  // 0.6% spread cost on all trades

  // Bet Sizing (Anti-Martingale)
  "win_multiplier": 2.0,   // Multiply bet by this after a win
  "loss_multiplier": 0.5,  // Multiply bet by this after a loss
  "max_bet_pct": 0.05,     // Maximum bet as % of bankroll (5%)
  "min_bet_pct": 0.10,     // Minimum bet as % of base bet (10% = $2.50)

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
