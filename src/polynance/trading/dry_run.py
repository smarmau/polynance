#!/usr/bin/env python3
"""Simulated trading bot entry point (dry run mode).

This module provides the CLI entry point for running the polynance
simulated trading bot. It collects live data from Polymarket and
executes simulated trades based on the configured strategy.

Configuration is loaded from config/trading.json by default.

Usage:
    python -m polynance.trading.dry_run
    python -m polynance.trading.dry_run --config path/to/config.json
    polynance-trade

Strategy (default):
    At the 7.5-minute mark of each 15-minute window:
    - If pm_yes_price >= 0.80: BUY YES (betting price goes UP)
    - If pm_yes_price <= 0.20: BUY NO (betting price goes DOWN)
    - Otherwise: NO TRADE

Bet Sizing (Anti-Martingale):
    - Base bet: $25
    - After win: multiply by 2x
    - After loss: multiply by 0.5x
    - Cap: 5% of current bankroll
    - Floor: $2.50 (10% of base)

Fees:
    - 2% fee on profits only
    - 0.6% spread cost on all trades
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from ..main import Application
from .config import TradingConfig, DEFAULT_CONFIG_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polynance Simulated Trading Bot (Dry Run Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
    Settings are loaded from config/trading.json by default.
    Edit that file to change strategy parameters, bankroll, etc.

    To create a fresh config file with defaults:
        python -m polynance.trading.dry_run --init-config

Examples:
    # Run with config file (default: config/trading.json)
    python -m polynance.trading.dry_run

    # Use a different config file
    python -m polynance.trading.dry_run --config my_config.json

    # Reset trading state and start fresh
    python -m polynance.trading.dry_run --reset

Strategy:
    At t=7.5 min: If pm_yes >= bull_threshold → BUY YES (bull)
    At t=7.5 min: If pm_yes <= bear_threshold → BUY NO (bear)
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config JSON file (default: {DEFAULT_CONFIG_PATH})",
    )

    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create a default config file and exit",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration and exit",
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Override config: disable the trading dashboard",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset trading state (start fresh)",
    )

    return parser.parse_args()


async def async_main(args, config: TradingConfig):
    """Async main function."""
    logger.info("=" * 60)
    logger.info("POLYNANCE SIMULATED TRADING BOT (DRY RUN)")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Assets: {config.assets}")
    logger.info(f"Initial Bankroll: ${config.initial_bankroll:,.2f}")
    logger.info(f"Base Bet: ${config.base_bet:.2f}")
    logger.info(f"Bull Threshold: >= {config.bull_threshold}")
    logger.info(f"Bear Threshold: <= {config.bear_threshold}")
    logger.info(f"Fee Rate: {config.fee_rate*100:.1f}%")
    logger.info(f"Spread Cost: {config.spread_cost*100:.1f}%")
    logger.info("=" * 60)

    # Determine dashboard setting (CLI override takes precedence)
    show_dashboard = config.show_dashboard and not args.no_dashboard

    # Create application with trading enabled
    app = Application(
        assets=config.assets,
        data_dir=Path(config.data_dir),
        run_analysis=config.run_analysis,
        show_dashboard=False,  # We'll use trading dashboard instead
        enable_trading=True,
        trading_config=config.to_trading_config(),
    )

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await app.initialize()

        # Reset state if requested
        if args.reset and app.trading_db:
            logger.warning("Resetting trading state...")
            await app.trading_db.reset_state()
            # Reinitialize trader with fresh state
            await app.trader.initialize()

        # Replace dashboard with trading dashboard if enabled
        if show_dashboard:
            from .dashboard import TradingDashboard

            app.dashboard = TradingDashboard(
                trader=app.trader,
                sampler=app.sampler,
                refresh_rate=config.dashboard_refresh_rate,
            )

        # Run app until shutdown signal
        app_task = asyncio.create_task(app.run())

        # Wait for either app to finish or shutdown signal
        done, pending = await asyncio.wait(
            [app_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        # Log final state
        if app.trader and app.trader.state:
            state = app.trader.state
            logger.info("=" * 60)
            logger.info("FINAL TRADING STATE")
            logger.info("=" * 60)
            logger.info(f"Bankroll: ${state.current_bankroll:,.2f}")
            logger.info(f"Total P&L: ${state.total_pnl:+,.2f}")
            logger.info(f"Return: {state.return_pct:+.2f}%")
            logger.info(f"Total Trades: {state.total_trades}")
            logger.info(f"Win Rate: {state.win_rate:.1%}")
            logger.info(f"Max Drawdown: {state.max_drawdown_pct:.1f}%")
            logger.info("=" * 60)

        await app.shutdown()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle --init-config
    if args.init_config:
        config = TradingConfig.create_default(args.config)
        print(f"Created default config at: {args.config}")
        print("\nConfiguration:")
        import json
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    # Load configuration
    config = TradingConfig.load(args.config)

    # Handle --show-config
    if args.show_config:
        print(f"Configuration from: {args.config}")
        print()
        import json
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    try:
        asyncio.run(async_main(args, config))
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
