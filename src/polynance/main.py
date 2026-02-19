"""Main entry point for polynance."""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from .clients.exchange import ExchangeClient, create_exchange
from .clients.binance import BinanceClient
from .db.database import Database
from .db.models import Window
from .sampler import Sampler
from .dashboard.terminal import TerminalDashboard
from .analysis.analyzer import Analyzer
from .analysis.hourly_analyzer import HourlyAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default assets to track
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP"]

# Data directory
DATA_DIR = Path("data")


class Application:
    """Main application coordinator."""

    def __init__(
        self,
        assets: List[str],
        data_dir: Path,
        run_analysis: bool = True,
        show_dashboard: bool = True,
        enable_trading: bool = False,
        trading_config: Optional[dict] = None,
    ):
        self.assets = assets
        self.data_dir = data_dir
        self.run_analysis = run_analysis
        self.show_dashboard = show_dashboard
        self.enable_trading = enable_trading
        self.trading_config = trading_config or {}

        # Components
        self.databases: Dict[str, Database] = {}
        self.exchange: ExchangeClient = None
        self.binance: BinanceClient = None
        self.sampler: Sampler = None
        self.dashboard: TerminalDashboard = None
        self.analyzers: Dict[str, Analyzer] = {}
        self.hourly_analyzer: HourlyAnalyzer = None

        # Trading components (optional)
        self.trader = None
        self.trading_db = None

        # Running flag
        self._running = False
        self._shutting_down = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing polynance for assets: {self.assets}")

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize databases (one per asset)
        for asset in self.assets:
            db_path = self.data_dir / f"{asset.lower()}.db"
            db = Database(db_path)
            await db.connect()
            self.databases[asset] = db
            logger.info(f"Initialized database for {asset}: {db_path}")

        # Initialize API clients
        exchange_name = self.trading_config.get("exchange", "polymarket")
        live_trading = self.trading_config.get("live_trading", False)
        exchange_kwargs = {}
        sig_type = self.trading_config.get("signature_type")
        if sig_type is not None:
            exchange_kwargs["signature_type"] = sig_type
        self.exchange = create_exchange(exchange_name, live_trading=live_trading, **exchange_kwargs)
        await self.exchange.connect()

        self.binance = BinanceClient()
        await self.binance.__aenter__()

        # Initialize analyzers
        reports_dir = self.data_dir / "reports"
        for asset in self.assets:
            self.analyzers[asset] = Analyzer(self.databases[asset], reports_dir)

        # Initialize sampler with callbacks for analysis/trading
        # We need a special database that routes to the right one per asset
        # Enable callback if either analysis or trading is enabled
        needs_callback = self.run_analysis or self.enable_trading
        self.sampler = Sampler(
            db=DatabaseRouter(self.databases),
            exchange=self.exchange,
            binance=self.binance,
            assets=self.assets,
            on_window_complete=self._on_window_complete if needs_callback else None,
            on_sample_collected=self._on_sample_collected if self.enable_trading else None,
        )

        await self.sampler.initialize()

        # Initialize dashboard
        if self.show_dashboard:
            self.dashboard = TerminalDashboard(
                sampler=self.sampler,
                databases=self.databases,
            )

        # Initialize hourly analyzer
        reports_dir = self.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.hourly_analyzer = HourlyAnalyzer(
            databases=self.databases,
            assets=self.assets,
            reports_dir=reports_dir,
        )

        # Initialize trading engine if enabled
        if self.enable_trading:
            await self._initialize_trading()

        logger.info("Initialization complete")
        if not self.show_dashboard:
            logger.info("Dashboard disabled - verbose logging mode active")

    async def _initialize_trading(self):
        """Initialize the trading engine."""
        from .trading.database import TradingDatabase
        from .trading.trader import SimulatedTrader

        logger.info("Initializing trading engine...")

        # Use trading.db, falling back to legacy sim_trading.db if it exists
        trading_db_path = self.data_dir / "trading.db"
        legacy_db_path = self.data_dir / "sim_trading.db"
        if not trading_db_path.exists() and legacy_db_path.exists():
            logger.info(f"Migrating legacy database: {legacy_db_path} → {trading_db_path}")
            legacy_db_path.rename(trading_db_path)
        self.trading_db = TradingDatabase(trading_db_path)
        await self.trading_db.connect()

        # Check if live trading is enabled
        live_trading = self.trading_config.get("live_trading", False)

        # Create trader with config
        self.trader = SimulatedTrader(
            trading_db=self.trading_db,
            asset_databases=self.databases,
            exchange=self.exchange,
            live_trading=live_trading,
            initial_bankroll=self.trading_config.get("initial_bankroll", 1000.0),
            base_bet=self.trading_config.get("base_bet", 25.0),
            fee_rate=self.trading_config.get("fee_rate", 0.02),
            spread_cost=self.trading_config.get("spread_cost", 0.006),
            fee_model=self.trading_config.get("fee_model", "flat"),
            exchange_name=self.trading_config.get("exchange", "polymarket"),
            bull_threshold=self.trading_config.get("bull_threshold", 0.80),
            bear_threshold=self.trading_config.get("bear_threshold", 0.20),
            max_bet_pct=self.trading_config.get("max_bet_pct", 0.05),
            bet_scale_threshold=self.trading_config.get("bet_scale_threshold", 0.0),
            bet_scale_increase=self.trading_config.get("bet_scale_increase", 0.0),
            pause_windows_after_loss=self.trading_config.get("pause_windows_after_loss", 2),
            growth_per_win=self.trading_config.get("growth_per_win", 0.10),
            max_bet_multiplier=self.trading_config.get("max_bet_multiplier", 2.0),
            min_trajectory=self.trading_config.get("min_trajectory", 0.20),
            entry_mode=self.trading_config.get("entry_mode", "two_stage"),
            signal_threshold_bull=self.trading_config.get("signal_threshold_bull", 0.70),
            signal_threshold_bear=self.trading_config.get("signal_threshold_bear", 0.30),
            confirm_threshold_bull=self.trading_config.get("confirm_threshold_bull", 0.85),
            confirm_threshold_bear=self.trading_config.get("confirm_threshold_bear", 0.15),
            contrarian_prev_thresh=self.trading_config.get("contrarian_prev_thresh", 0.75),
            contrarian_bull_thresh=self.trading_config.get("contrarian_bull_thresh", 0.60),
            contrarian_bear_thresh=self.trading_config.get("contrarian_bear_thresh", 0.40),
            contrarian_entry_time=self.trading_config.get("contrarian_entry_time", "t0"),
            contrarian_exit_time=self.trading_config.get("contrarian_exit_time", "t12.5"),
            consensus_min_agree=self.trading_config.get("consensus_min_agree", 3),
            consensus_entry_time=self.trading_config.get("consensus_entry_time", "t5"),
            consensus_exit_time=self.trading_config.get("consensus_exit_time", "t12.5"),
            accel_neutral_band=self.trading_config.get("accel_neutral_band", 0.15),
            accel_prev_thresh=self.trading_config.get("accel_prev_thresh", 0.75),
            accel_bull_thresh=self.trading_config.get("accel_bull_thresh", 0.55),
            accel_bear_thresh=self.trading_config.get("accel_bear_thresh", 0.45),
            accel_entry_time=self.trading_config.get("accel_entry_time", "t5"),
            accel_exit_time=self.trading_config.get("accel_exit_time", "t12.5"),
            combo_prev_thresh=self.trading_config.get("combo_prev_thresh", 0.75),
            combo_bull_thresh=self.trading_config.get("combo_bull_thresh", 0.55),
            combo_bear_thresh=self.trading_config.get("combo_bear_thresh", 0.45),
            combo_entry_time=self.trading_config.get("combo_entry_time", "t5"),
            combo_exit_time=self.trading_config.get("combo_exit_time", "t12.5"),
            combo_stop_time=self.trading_config.get("combo_stop_time", "t7.5"),
            combo_stop_delta=self.trading_config.get("combo_stop_delta", 0.10),
            combo_xasset_min=self.trading_config.get("combo_xasset_min", 2),
            triple_prev_thresh=self.trading_config.get("triple_prev_thresh", 0.70),
            triple_bull_thresh=self.trading_config.get("triple_bull_thresh", 0.55),
            triple_bear_thresh=self.trading_config.get("triple_bear_thresh", 0.45),
            triple_entry_time=self.trading_config.get("triple_entry_time", "t5"),
            triple_exit_time=self.trading_config.get("triple_exit_time", "t12.5"),
            triple_xasset_min=self.trading_config.get("triple_xasset_min", 3),
            triple_pm0_bull_min=self.trading_config.get("triple_pm0_bull_min", 0.50),
            triple_pm0_bear_max=self.trading_config.get("triple_pm0_bear_max", 0.50),
            skip_regimes=self.trading_config.get("skip_regimes"),
            skip_days=self.trading_config.get("skip_days"),
            recovery_sizing=self.trading_config.get("recovery_sizing", "none"),
            recovery_step=self.trading_config.get("recovery_step", 25.0),
            recovery_max_multiplier=self.trading_config.get("recovery_max_multiplier", 5),
            adaptive_direction_n=self.trading_config.get("adaptive_direction_n", 0),
        )
        await self.trader.initialize()

        entry_mode = self.trading_config.get("entry_mode", "two_stage")
        if entry_mode == "triple_filter":
            logger.info(
                f"Trading engine initialized (TRIPLE FILTER): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Prev thresh={self.trading_config.get('triple_prev_thresh', 0.70)}, "
                f"XAsset min={self.trading_config.get('triple_xasset_min', 3)}, "
                f"PM0 bull>={self.trading_config.get('triple_pm0_bull_min', 0.50)} "
                f"bear<={self.trading_config.get('triple_pm0_bear_max', 0.50)}, "
                f"Entry={self.trading_config.get('triple_entry_time', 't5')} "
                f"Exit={self.trading_config.get('triple_exit_time', 't12.5')}, "
                f"Bull>={self.trading_config.get('triple_bull_thresh', 0.55)} "
                f"Bear<={self.trading_config.get('triple_bear_thresh', 0.45)}"
            )
        elif entry_mode == "accel_dbl":
            logger.info(
                f"Trading engine initialized (ACCEL_DBL): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Prev thresh={self.trading_config.get('accel_prev_thresh', 0.75)}, "
                f"Neutral band={self.trading_config.get('accel_neutral_band', 0.15)}, "
                f"Entry={self.trading_config.get('accel_entry_time', 't5')} "
                f"Exit={self.trading_config.get('accel_exit_time', 't12.5')}, "
                f"Bull>={self.trading_config.get('accel_bull_thresh', 0.55)} "
                f"Bear<={self.trading_config.get('accel_bear_thresh', 0.45)}"
            )
        elif entry_mode == "combo_dbl":
            logger.info(
                f"Trading engine initialized (COMBO_DBL): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Prev thresh={self.trading_config.get('combo_prev_thresh', 0.75)}, "
                f"Entry={self.trading_config.get('combo_entry_time', 't5')} "
                f"Exit={self.trading_config.get('combo_exit_time', 't12.5')}, "
                f"Stop={self.trading_config.get('combo_stop_time', 't7.5')} "
                f"delta={self.trading_config.get('combo_stop_delta', 0.10)}, "
                f"XAsset min={self.trading_config.get('combo_xasset_min', 2)}, "
                f"Bull>={self.trading_config.get('combo_bull_thresh', 0.55)} "
                f"Bear<={self.trading_config.get('combo_bear_thresh', 0.45)}"
            )
        elif entry_mode == "contrarian_consensus":
            logger.info(
                f"Trading engine initialized (CONTRARIAN+CONSENSUS): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Prev thresh={self.trading_config.get('contrarian_prev_thresh', 0.75)}, "
                f"Min agree={self.trading_config.get('consensus_min_agree', 3)}/4, "
                f"Entry={self.trading_config.get('consensus_entry_time', 't5')} "
                f"Exit={self.trading_config.get('consensus_exit_time', 't12.5')}, "
                f"Bull>={self.trading_config.get('contrarian_bull_thresh', 0.60)} "
                f"Bear<={self.trading_config.get('contrarian_bear_thresh', 0.40)}"
            )
        elif entry_mode == "contrarian":
            logger.info(
                f"Trading engine initialized (CONTRARIAN): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Prev thresh={self.trading_config.get('contrarian_prev_thresh', 0.75)}, "
                f"Entry={self.trading_config.get('contrarian_entry_time', 't0')} "
                f"Exit={self.trading_config.get('contrarian_exit_time', 't12.5')}, "
                f"Bull>={self.trading_config.get('contrarian_bull_thresh', 0.60)} "
                f"Bear<={self.trading_config.get('contrarian_bear_thresh', 0.40)}"
            )
        elif entry_mode == "two_stage":
            logger.info(
                f"Trading engine initialized (TWO-STAGE): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Signal: Bull>={self.trading_config.get('signal_threshold_bull', 0.70)} Bear<={self.trading_config.get('signal_threshold_bear', 0.30)}, "
                f"Confirm: Bull>={self.trading_config.get('confirm_threshold_bull', 0.85)} Bear<={self.trading_config.get('confirm_threshold_bear', 0.15)}"
            )
        else:
            logger.info(
                f"Trading engine initialized (SINGLE): "
                f"Bankroll=${self.trader.state.current_bankroll:.2f}, "
                f"Bull>={self.trading_config.get('bull_threshold', 0.80)}, "
                f"Bear<={self.trading_config.get('bear_threshold', 0.20)}"
            )

    async def _on_sample_collected(self, asset: str, sample, state):
        """Called when a sample is collected - check for trade signals.

        Two-stage mode: signal at t=7.5, confirm at t=10.
        Single mode: entry at t=7.5.
        Contrarian mode: entry at configured time (default t=0), exit at configured time (default t=12.5).
        """
        if not self.trader:
            return

        try:
            entry_mode = self.trading_config.get("entry_mode", "two_stage")

            if entry_mode == "triple_filter":
                # TRIPLE: t0 for PM0 check, entry at configured time, exit at configured time
                entry_t = self.trader._time_to_minutes.get(self.trader.triple_entry_time, 5.0)
                exit_t = self.trader._time_to_minutes.get(self.trader.triple_exit_time, 12.5)

                if sample.t_minutes == 0.0:
                    await self.trader.on_sample_at_triple_t0(asset, sample, state)
                elif sample.t_minutes == entry_t:
                    await self.trader.on_sample_at_triple_entry(asset, sample, state)
                elif sample.t_minutes == exit_t:
                    await self.trader.on_sample_at_triple_exit(asset, sample, state)
            elif entry_mode == "accel_dbl":
                # ACCEL_DBL: t0 for acceleration check, entry at configured time, exit at configured time
                entry_t = self.trader._time_to_minutes.get(self.trader.accel_entry_time, 5.0)
                exit_t = self.trader._time_to_minutes.get(self.trader.accel_exit_time, 12.5)

                if sample.t_minutes == 0.0:
                    await self.trader.on_sample_at_accel_t0(asset, sample, state)
                elif sample.t_minutes == entry_t:
                    await self.trader.on_sample_at_accel_entry(asset, sample, state)
                elif sample.t_minutes == exit_t:
                    await self.trader.on_sample_at_accel_exit(asset, sample, state)
            elif entry_mode == "combo_dbl":
                # COMBO_DBL: entry at configured time, stop-loss check, exit at configured time
                entry_t = self.trader._time_to_minutes.get(self.trader.combo_entry_time, 5.0)
                stop_t = self.trader._time_to_minutes.get(self.trader.combo_stop_time, 7.5)
                exit_t = self.trader._time_to_minutes.get(self.trader.combo_exit_time, 12.5)

                if sample.t_minutes == entry_t:
                    await self.trader.on_sample_at_combo_entry(asset, sample, state)
                elif sample.t_minutes == stop_t:
                    await self.trader.on_sample_at_combo_stop(asset, sample, state)
                elif sample.t_minutes == exit_t:
                    await self.trader.on_sample_at_combo_exit(asset, sample, state)
            elif entry_mode == "contrarian_consensus":
                # Consensus: entry and exit at consensus-specific times
                entry_t = self.trader._time_to_minutes.get(
                    self.trader.consensus_entry_time, 5.0
                )
                exit_t = self.trader._time_to_minutes.get(
                    self.trader.consensus_exit_time, 12.5
                )

                if sample.t_minutes == entry_t:
                    await self.trader.on_sample_at_consensus_entry(asset, sample, state)
                elif sample.t_minutes == exit_t:
                    await self.trader.on_sample_at_consensus_exit(asset, sample, state)
            elif entry_mode == "contrarian":
                # Contrarian: entry and exit at configured sample points
                entry_t = self.trader._time_to_minutes.get(
                    self.trader.contrarian_entry_time, 0.0
                )
                exit_t = self.trader._time_to_minutes.get(
                    self.trader.contrarian_exit_time, 12.5
                )

                if sample.t_minutes == entry_t:
                    await self.trader.on_sample_at_contrarian_entry(asset, sample, state)
                elif sample.t_minutes == exit_t:
                    await self.trader.on_sample_at_contrarian_exit(asset, sample, state)
            else:
                # Two-stage / single modes
                if sample.t_minutes == 7.5:
                    await self.trader.on_sample_at_signal(asset, sample, state)
                elif sample.t_minutes == 10.0:
                    await self.trader.on_sample_at_confirm(asset, sample, state)
        except Exception as e:
            logger.error(f"[{asset}] Trading signal/entry error: {e}", exc_info=True)

    async def _on_window_complete(self, asset: str, window: Window):
        """Called when a window completes - run analysis and trading."""
        # Run analysis if enabled
        if self.run_analysis:
            logger.info(f"[{asset}] Running analysis after window completion...")

            analyzer = self.analyzers.get(asset)
            if analyzer:
                try:
                    results = await analyzer.run_full_analysis(asset, min_windows=5)

                    if results.get("status") != "insufficient_data":
                        # Log summary
                        summary = analyzer.format_summary(results)
                        logger.info(summary)

                        # Update dashboard
                        if self.dashboard:
                            self.dashboard.update_analysis(asset, results)
                except Exception as e:
                    logger.error(f"[{asset}] Analysis error: {e}")

        # Run trading if enabled
        if self.trader:
            try:
                await self.trader.on_window_complete(asset, window)
            except Exception as e:
                logger.error(f"[{asset}] Trading error: {e}", exc_info=True)

    async def run(self):
        """Run the main application loop."""
        self._running = True

        # Start sampler task
        sampler_task = asyncio.create_task(self.sampler.run())
        self._tasks.append(sampler_task)

        # Start dashboard task if enabled
        if self.dashboard:
            dashboard_task = asyncio.create_task(self.dashboard.run())
            self._tasks.append(dashboard_task)

        # Start hourly analyzer task
        if self.hourly_analyzer:
            analyzer_task = asyncio.create_task(self.hourly_analyzer.run())
            self._tasks.append(analyzer_task)

        logger.info("Application running. Press Ctrl+C to stop.")

        try:
            # Wait for tasks
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Application cancelled")

    async def shutdown(self):
        """Shutdown all components."""
        # Prevent double shutdown
        if self._shutting_down:
            return
        self._shutting_down = True

        logger.info("Shutting down...")

        self._running = False

        # Stop sampler first (this stops the sampling loop)
        if self.sampler:
            self.sampler.stop()

        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop()

        # Stop hourly analyzer
        if self.hourly_analyzer:
            self.hourly_analyzer.stop()

        # Cancel tasks with a timeout
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete in time")

        # Close API clients
        try:
            if self.exchange:
                await asyncio.wait_for(
                    self.exchange.close(),
                    timeout=2.0
                )
        except asyncio.TimeoutError:
            logger.warning("Exchange client close timed out")
        except Exception as e:
            logger.warning(f"Error closing exchange client: {e}")

        try:
            if self.binance:
                await asyncio.wait_for(
                    self.binance.__aexit__(None, None, None),
                    timeout=2.0
                )
        except asyncio.TimeoutError:
            logger.warning("Binance client close timed out")
        except Exception as e:
            logger.warning(f"Error closing Binance client: {e}")

        # Close databases
        for asset, db in self.databases.items():
            try:
                await asyncio.wait_for(db.close(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(f"Database close timed out for {asset}")
            except Exception as e:
                logger.warning(f"Error closing database for {asset}: {e}")

        # Close trading databases
        if self.trading_db:
            try:
                await asyncio.wait_for(self.trading_db.close(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Trading database close timed out")
            except Exception as e:
                logger.warning(f"Error closing trading database: {e}")

        logger.info("Shutdown complete")


class DatabaseRouter:
    """Routes database calls to the correct per-asset database."""

    def __init__(self, databases: Dict[str, Database]):
        self.databases = databases

    async def insert_sample(self, sample):
        db = self.databases.get(sample.asset)
        if db:
            return await db.insert_sample(sample)

    async def insert_window(self, window):
        db = self.databases.get(window.asset)
        if db:
            return await db.insert_window(window)

    async def get_samples_for_window(self, window_id: str, asset: str):
        db = self.databases.get(asset)
        if db:
            return await db.get_samples_for_window(window_id, asset)
        return []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polynance - Polymarket 15-min crypto prediction logger and analyzer"
    )

    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=DEFAULT_ASSETS,
        help=f"Assets to track (default: {DEFAULT_ASSETS})",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory for data storage (default: {DATA_DIR})",
    )

    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the terminal dashboard",
    )

    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable automatic analysis after each window",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def async_main(args):
    """Async main function."""
    app = Application(
        assets=args.assets,
        data_dir=args.data_dir,
        run_analysis=not args.no_analysis,
        show_dashboard=not args.no_dashboard,
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
        logger.error(f"Application error: {e}")
    finally:
        await app.shutdown()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
