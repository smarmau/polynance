#!/usr/bin/env python3
"""Simulated trading bot entry point (dry run mode).

This module provides the CLI entry point for running the polynance
simulated trading bot. It collects live data from Polymarket and
executes simulated trades based on the configured strategy.

Configuration is loaded from config/config.json by default.

Usage:
    python -m polynance.trading.dry_run
    python -m polynance.trading.dry_run --config path/to/config.json
    polynance-trade

Strategy (default - two-stage):
    Stage 1 (t=7.5 min): Check for initial signal
    - If pm_yes >= 0.70 → pending BULL signal
    - If pm_yes <= 0.30 → pending BEAR signal

    Stage 2 (t=10 min): Confirm signal still strong
    - If pm_yes >= 0.85 → CONFIRM BULL, enter trade at t=10 price
    - If pm_yes <= 0.15 → CONFIRM BEAR, enter trade at t=10 price
    - Otherwise → signal "faded", NO TRADE (filters ~13% of false signals)

    This filters out "faders" - signals that appear strong at t=7.5
    but reverse by t=10, which have ~61% WR vs 95% for confirmed signals.

Bet Sizing:
    - Fixed $50 per trade (capped at 5% of bankroll)

Fees:
    - 0.1% taker fee on contract premium
    - 0.5% spread/slippage estimate
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
    Settings are loaded from config/config.json by default.
    Edit that file to change strategy parameters, bankroll, etc.

    To create a fresh config file with defaults:
        python -m polynance.trading.dry_run --init-config

Examples:
    # Run with config file (default: config/config.json)
    python -m polynance.trading.dry_run

    # Use a different config file
    python -m polynance.trading.dry_run --config my_config.json

    # Reset trading state and start fresh
    python -m polynance.trading.dry_run --reset

Strategy (two-stage default):
    At t=7.5: Signal if pm_yes >= signal_threshold_bull or <= signal_threshold_bear
    At t=10:  Confirm if pm_yes >= confirm_threshold_bull or <= confirm_threshold_bear
    Filters out "fader" signals that reverse between t=7.5 and t=10
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
    logger.info(f"Entry Mode: {config.entry_mode}")
    logger.info(f"Assets: {config.assets}")
    logger.info(f"Initial Bankroll: ${config.initial_bankroll:,.2f}")
    logger.info(f"Base Bet: ${config.base_bet:.2f}")
    if config.bet_scale_threshold > 0:
        logger.info(f"Bet Scale: +{config.bet_scale_increase*100:.0f}% per {config.bet_scale_threshold*100:.0f}% gain")
    if config.live_trading:
        logger.info("*** LIVE TRADING ENABLED — REAL ORDERS WILL BE PLACED ***")
    if config.entry_mode == "triple_filter":
        logger.info(f"Prev Window Thresh: >= {config.triple_prev_thresh} (double required)")
        logger.info(f"Cross-Asset: >= {config.triple_xasset_min} assets double-strong")
        logger.info(f"PM0 Confirm: Bull >= {config.triple_pm0_bull_min}, Bear <= {config.triple_pm0_bear_max}")
        logger.info(f"Entry: {config.triple_entry_time}, Exit: {config.triple_exit_time}")
        logger.info(f"Bull Confirm: >= {config.triple_bull_thresh}, Bear Confirm: <= {config.triple_bear_thresh}")
    elif config.entry_mode == "accel_dbl":
        logger.info(f"Prev Window Thresh: >= {config.accel_prev_thresh} (double required)")
        logger.info(f"Neutral Band: t0 within {config.accel_neutral_band} of 0.50")
        logger.info(f"Entry: {config.accel_entry_time}, Exit: {config.accel_exit_time}")
        logger.info(f"Bull Confirm: >= {config.accel_bull_thresh}, Bear Confirm: <= {config.accel_bear_thresh}")
    elif config.entry_mode == "combo_dbl":
        logger.info(f"Prev Window Thresh: >= {config.combo_prev_thresh} (double required)")
        logger.info(f"Cross-Asset: >= {config.combo_xasset_min} other assets double-strong")
        logger.info(f"Stop-Loss: delta >= {config.combo_stop_delta} at {config.combo_stop_time}")
        logger.info(f"Entry: {config.combo_entry_time}, Exit: {config.combo_exit_time}")
        logger.info(f"Bull Confirm: >= {config.combo_bull_thresh}, Bear Confirm: <= {config.combo_bear_thresh}")
    elif config.entry_mode == "contrarian_consensus":
        logger.info(f"Prev Window Thresh: >= {config.contrarian_prev_thresh}")
        logger.info(f"Consensus: {config.consensus_min_agree}/4 assets must agree")
        logger.info(f"Entry: {config.consensus_entry_time}, Exit: {config.consensus_exit_time}")
        logger.info(f"Bull Confirm: >= {config.contrarian_bull_thresh}, Bear Confirm: <= {config.contrarian_bear_thresh}")
    elif config.entry_mode == "contrarian":
        logger.info(f"Prev Window Thresh: >= {config.contrarian_prev_thresh}")
        logger.info(f"Entry: {config.contrarian_entry_time}, Exit: {config.contrarian_exit_time}")
        logger.info(f"Bull Confirm: >= {config.contrarian_bull_thresh}, Bear Confirm: <= {config.contrarian_bear_thresh}")
    elif config.entry_mode == "two_stage":
        logger.info(f"Signal (t=7.5): Bull >= {config.signal_threshold_bull}, Bear <= {config.signal_threshold_bear}")
        logger.info(f"Confirm (t=10): Bull >= {config.confirm_threshold_bull}, Bear <= {config.confirm_threshold_bear}")
    else:
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
