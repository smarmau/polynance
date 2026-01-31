"""Core simulated trading engine."""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List

from ..db.database import Database
from ..db.models import Window
from .models import SimulatedTrade, TradingState, PerAssetStats
from .database import TradingDatabase
from .bet_sizing import AntiMartingaleSizer

logger = logging.getLogger(__name__)


class SimulatedTrader:
    """Simulated trading engine for dry-run mode.

    Evaluates trading signals from completed windows and tracks
    simulated positions, P&L, and portfolio performance.

    Strategy (from user spec):
    - At t=7.5 min: If pm_yes >= 0.80 → BUY YES (bull)
    - At t=7.5 min: If pm_yes <= 0.20 → BUY NO (bear)
    - Otherwise: NO TRADE

    Payout mechanics (matches backtest_suite.py):
    - Buy YES at price P: Win → profit = (1-P), Lose → lose P
    - Buy NO at price (1-P): Win → profit = P, Lose → lose (1-P)
    - 2% fee on profits only
    - 0.6% spread cost on all trades
    """

    # Default configuration
    DEFAULT_INITIAL_BANKROLL = 1000.0
    DEFAULT_BASE_BET = 25.0
    DEFAULT_FEE_RATE = 0.02  # 2% on profits
    DEFAULT_SPREAD_COST = 0.006  # 0.6%
    DEFAULT_BULL_THRESHOLD = 0.80
    DEFAULT_BEAR_THRESHOLD = 0.20

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
    ):
        """Initialize the simulated trader.

        Args:
            trading_db: Database for trading state and trades
            asset_databases: Optional dict of per-asset databases
            initial_bankroll: Starting bankroll (default: $1000)
            base_bet: Base bet size (default: $25)
            fee_rate: Fee rate on profits (default: 0.02 = 2%)
            spread_cost: Spread cost on all trades (default: 0.006 = 0.6%)
            bull_threshold: YES price threshold for bull signal (default: 0.80)
            bear_threshold: YES price threshold for bear signal (default: 0.20)
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

        # Bet sizing
        self.sizer = AntiMartingaleSizer(base_bet=base_bet)

        # State (loaded from DB or initialized)
        self.state: Optional[TradingState] = None

        # Track open positions per asset
        self.open_trades: Dict[str, SimulatedTrade] = {}

        # Per-asset statistics cache
        self._asset_stats: Dict[str, PerAssetStats] = {}

        # Last signals for dashboard display
        self._last_signals: Dict[str, Optional[str]] = {}

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
            logger.info(
                f"Recovered trading state: "
                f"Bankroll=${self.state.current_bankroll:.2f}, "
                f"Trades={self.state.total_trades}, "
                f"Win Rate={self.state.win_rate:.1%}, "
                f"P&L=${self.state.total_pnl:.2f}"
            )

        # Load any pending trades (for restart recovery)
        pending = await self.trading_db.get_pending_trades()
        for trade in pending:
            self.open_trades[trade.asset] = trade
            logger.info(
                f"Recovered pending trade: {trade.asset} {trade.direction} "
                f"at ${trade.entry_price:.3f}"
            )

        if pending:
            logger.info(f"Recovered {len(pending)} pending trade(s)")

    async def on_sample_at_entry(self, asset: str, sample, state):
        """Handle sample at t=7.5 - check for trade entry.

        This is called when a sample is collected at t=7.5 minutes.
        We check if there's a signal and open a position immediately.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            sample: The sample collected at t=7.5
            state: The asset's current sampler state
        """
        try:
            # Don't open if we already have a position for this asset
            if asset in self.open_trades:
                logger.debug(f"[{asset}] Already have open position, skipping entry check")
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

            # Store last signal for dashboard
            self._last_signals[asset] = direction

            # Calculate bet size
            bet_size = self.sizer.calculate_actual_bet(
                self.state.current_bet_size,
                self.state.current_bankroll,
            )

            # Create trade
            trade = SimulatedTrade(
                window_id=sample.window_id,
                asset=asset,
                direction=direction,
                entry_time=sample.sample_time_utc,
                entry_price=entry_price,
                bet_size=bet_size,
                outcome="pending",
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

    async def on_window_complete(self, asset: str, window: Window):
        """Handle window completion - resolve open trades.

        This is called when a 15-minute window completes with a resolved outcome.
        We resolve any open position for this asset.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
            window: Completed window with outcome
        """
        try:
            # Resolve any open position for this asset
            if asset in self.open_trades:
                await self._resolve_trade(asset, window)

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

        # Update bet size for next trade
        self.state.current_bet_size = self.sizer.calculate_next_bet(
            self.state.current_bet_size,
            self.state.current_bankroll,
            "win" if won else "loss",
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
        return {
            "initial_bankroll": self.initial_bankroll,
            "base_bet": self.base_bet,
            "fee_rate": self.fee_rate,
            "spread_cost": self.spread_cost,
            "bull_threshold": self.bull_threshold,
            "bear_threshold": self.bear_threshold,
            "sizer": self.sizer.get_config(),
        }

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
