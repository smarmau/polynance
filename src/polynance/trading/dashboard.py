"""Trading dashboard using Rich library."""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..sampler import Sampler, get_window_boundaries
from .trader import SimulatedTrader
from .models import SimulatedTrade, TradingState


class TradingDashboard:
    """Rich terminal dashboard for simulated trading."""

    def __init__(
        self,
        trader: SimulatedTrader,
        sampler: Sampler,
        refresh_rate: float = 2.0,
    ):
        """Initialize the trading dashboard.

        Args:
            trader: SimulatedTrader instance
            sampler: Sampler instance for live data
            refresh_rate: Dashboard refresh rate in seconds
        """
        self.trader = trader
        self.sampler = sampler
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False

        # Cached data
        self._recent_trades: List[SimulatedTrade] = []
        self._asset_stats: Dict[str, dict] = {}
        self._today_stats: dict = {}
        self._metrics: dict = {}
        self._asset_regimes: Dict[str, str] = {}  # latest volatility regime per asset

    async def run(self):
        """Run the dashboard in a live loop."""
        self._running = True

        with Live(
            self._generate_layout(),
            console=self.console,
            refresh_per_second=2,
            screen=True,
        ) as live:
            while self._running:
                try:
                    # Update cached data
                    await self._update_caches()

                    # Update display
                    live.update(self._generate_layout())

                    await asyncio.sleep(self.refresh_rate)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.console.print(f"[red]Dashboard error: {e}[/red]")
                    await asyncio.sleep(1)

    def stop(self):
        """Stop the dashboard."""
        self._running = False

    async def _update_caches(self):
        """Update cached data from trader."""
        try:
            self._recent_trades = await self.trader.get_recent_trades(10)
        except Exception:
            pass

        try:
            for asset in self.sampler.assets:
                self._asset_stats[asset] = await self.trader.get_asset_stats(asset)
        except Exception:
            pass

        try:
            self._today_stats = await self.trader.get_today_stats()
        except Exception:
            pass

        try:
            self._metrics = await self.trader.calculate_metrics()
        except Exception:
            pass

        # Fetch latest volatility regime per asset from most recent window
        try:
            for asset in self.sampler.assets:
                recent = await self.sampler.db.get_recent_windows(asset, limit=1)
                if recent and recent[0].volatility_regime:
                    self._asset_regimes[asset] = recent[0].volatility_regime
        except Exception:
            pass

    def _generate_layout(self) -> Layout:
        """Generate the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(self._make_header())

        # Main content - split into left and right
        layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )

        # Left side
        layout["left"].split_column(
            Layout(name="portfolio", size=9),
            Layout(name="open_positions", size=10),
            Layout(name="recent_trades"),
        )

        layout["left"]["portfolio"].update(self._make_portfolio_panel())
        layout["left"]["open_positions"].update(self._make_open_positions_panel())
        layout["left"]["recent_trades"].update(self._make_recent_trades_panel())

        # Right side
        layout["right"].split_column(
            Layout(name="metrics", size=14),
            Layout(name="per_asset", size=9),
            Layout(name="slot_status", size=9),
        )

        layout["right"]["metrics"].update(self._make_metrics_panel())
        layout["right"]["per_asset"].update(self._make_per_asset_panel())
        layout["right"]["slot_status"].update(self._make_slot_status_panel())

        # Footer
        layout["footer"].update(self._make_footer())

        return layout

    def _make_header(self) -> Panel:
        """Create the header panel with countdown timers."""
        now = datetime.now(timezone.utc)
        window_start, window_end = get_window_boundaries(now)

        # Calculate time into window (in seconds for precision)
        time_into_window_sec = (now - window_start).total_seconds()
        time_into_window_min = time_into_window_sec / 60

        is_contrarian = self.trader.entry_mode == "contrarian"
        is_consensus = self.trader.entry_mode == "contrarian_consensus"
        is_accel = self.trader.entry_mode == "accel_dbl"
        is_combo = self.trader.entry_mode == "combo_dbl"
        is_contrarian_family = is_contrarian or is_consensus or is_accel or is_combo
        is_two_stage = self.trader.entry_mode == "two_stage"

        # Resolution time is at t=15 minutes (900 seconds)
        resolution_time_sec = 15 * 60  # 900 seconds
        time_to_resolution = resolution_time_sec - time_into_window_sec

        if is_contrarian_family:
            # Contrarian-family: entry at configured time, exit at configured time
            if is_accel:
                entry_min = self.trader._time_to_minutes.get(self.trader.accel_entry_time, 5.0)
                exit_min = self.trader._time_to_minutes.get(self.trader.accel_exit_time, 12.5)
            elif is_combo:
                entry_min = self.trader._time_to_minutes.get(self.trader.combo_entry_time, 5.0)
                exit_min = self.trader._time_to_minutes.get(self.trader.combo_exit_time, 12.5)
            elif is_consensus:
                entry_min = self.trader._time_to_minutes.get(self.trader.consensus_entry_time, 5.0)
                exit_min = self.trader._time_to_minutes.get(self.trader.consensus_exit_time, 12.5)
            else:
                entry_min = self.trader._time_to_minutes.get(self.trader.contrarian_entry_time, 0.0)
                exit_min = self.trader._time_to_minutes.get(self.trader.contrarian_exit_time, 12.5)
            entry_sec = entry_min * 60
            exit_sec = exit_min * 60
            time_to_entry = entry_sec - time_into_window_sec
            time_to_exit = exit_sec - time_into_window_sec

            if time_into_window_min < entry_min:
                phase = "PRE-ENTRY"
                phase_style = "yellow"
                entry_m = int(time_to_entry // 60)
                entry_s = int(time_to_entry % 60)
                signal_str = f"{entry_m:02d}:{entry_s:02d}"
                signal_style = "yellow bold"
            elif time_into_window_min < entry_min + 0.5:
                phase = "ENTRY!"
                phase_style = "green bold blink"
                signal_str = "NOW"
                signal_style = "green bold"
            elif time_into_window_min < exit_min:
                phase = "HOLDING"
                phase_style = "cyan"
                exit_m = int(time_to_exit // 60)
                exit_s = int(time_to_exit % 60)
                signal_str = f"EXIT in {exit_m:02d}:{exit_s:02d}"
                signal_style = "cyan bold"
            elif time_into_window_min < exit_min + 0.5:
                phase = "EXIT!"
                phase_style = "magenta bold blink"
                signal_str = "CLOSING"
                signal_style = "magenta bold"
            else:
                phase = "DONE"
                phase_style = "dim"
                signal_str = "--:--"
                signal_style = "dim"
        else:
            # Two-stage / single mode timers
            signal_time_sec = 7.5 * 60
            time_to_signal = signal_time_sec - time_into_window_sec
            confirm_time_sec = 10.0 * 60
            time_to_confirm = confirm_time_sec - time_into_window_sec

            if time_into_window_min < 7.5:
                phase = "PRE-SIGNAL"
                phase_style = "yellow"
                signal_min = int(time_to_signal // 60)
                signal_sec = int(time_to_signal % 60)
                signal_str = f"{signal_min:02d}:{signal_sec:02d}"
                signal_style = "yellow bold"
            elif time_into_window_min < 8.0:
                phase = "SIGNAL!"
                phase_style = "green bold blink"
                signal_str = "NOW"
                signal_style = "green bold"
            elif is_two_stage and time_into_window_min < 10.0:
                phase = "CONFIRMING"
                phase_style = "yellow"
                confirm_min = int(time_to_confirm // 60)
                confirm_sec = int(time_to_confirm % 60)
                signal_str = f"CONFIRM in {confirm_min:02d}:{confirm_sec:02d}"
                signal_style = "yellow bold"
            elif is_two_stage and time_into_window_min < 10.5:
                phase = "CONFIRM!"
                phase_style = "green bold blink"
                signal_str = "CONFIRMING"
                signal_style = "green bold"
            else:
                phase = "WAITING"
                phase_style = "dim"
                signal_str = "--:--"
                signal_style = "dim"

        # Resolution countdown (always counting down to window end)
        if time_to_resolution > 0:
            res_min = int(time_to_resolution // 60)
            res_sec = int(time_to_resolution % 60)
            resolution_str = f"{res_min:02d}:{res_sec:02d}"
            resolution_style = "cyan"
        else:
            resolution_str = "NOW"
            resolution_style = "cyan bold"

        # Build header content
        title = Text()
        title.append("POLYNANCE TRADING BOT ", style="bold cyan")
        if is_accel:
            title.append("(Accel+DblContrarian Mode)", style="magenta")
        elif is_combo:
            title.append("(Combo+DblContrarian Mode)", style="magenta")
        elif is_consensus:
            title.append("(Contrarian+Consensus Mode)", style="magenta")
        elif is_contrarian:
            title.append("(Contrarian Mode)", style="magenta")
        else:
            title.append("(Dry Run Mode)", style="dim")

        # Line 2: Time info
        line2 = Text()
        line2.append(f"\n  UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}  ", style="white")
        line2.append(f"| Window: {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}  ", style="green")
        line2.append(f"| t={time_into_window_min:.1f}m  ", style="dim")
        line2.append(f"| ", style="dim")
        line2.append(f"{phase}", style=phase_style)

        # Line 3: Countdown timers
        line3 = Text()
        line3.append(f"\n  ", style="dim")
        if is_contrarian_family:
            line3.append(f"ENTRY in: ", style="yellow")
        else:
            line3.append(f"SIGNAL in: ", style="yellow")
        line3.append(f"{signal_str}", style=signal_style)
        line3.append(f"    |    ", style="dim")
        line3.append(f"RESOLUTION in: ", style="cyan")
        line3.append(f"{resolution_str}", style=resolution_style)

        # Show open positions count
        open_count = len(self.trader.get_open_trades())
        if open_count > 0:
            line3.append(f"    |    ", style="dim")
            line3.append(f"OPEN: {open_count}", style="green bold")

        content = Text()
        content.append_text(title)
        content.append_text(line2)
        content.append_text(line3)

        return Panel(content, style="bold")

    def _make_portfolio_panel(self) -> Panel:
        """Create the portfolio status panel."""
        state = self.trader.get_state()

        if state is None:
            return Panel("No trading state", title="Portfolio", border_style="blue")

        # Create grid layout
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # Row 1: Bankroll info
        pnl_color = "green" if state.total_pnl >= 0 else "red"
        return_color = "green" if state.return_pct >= 0 else "red"

        grid.add_row(
            Text.assemble(("Bankroll: ", "dim"), (f"${state.current_bankroll:,.2f}", "bold white")),
            Text.assemble(("Starting: ", "dim"), (f"${state.initial_bankroll:,.2f}", "white")),
            Text.assemble(("Total P&L: ", "dim"), (f"${state.total_pnl:+,.2f}", pnl_color)),
            Text.assemble(("Return: ", "dim"), (f"{state.return_pct:+.2f}%", return_color)),
        )

        # Row 2: Bet sizing and drawdown
        dd_color = "red" if state.max_drawdown_pct < -10 else "yellow" if state.max_drawdown_pct < -5 else "green"

        grid.add_row(
            Text.assemble(("Current Bet: ", "dim"), (f"${state.current_bet_size:.2f}", "cyan")),
            Text.assemble(("Base Bet: ", "dim"), (f"${state.base_bet_size:.2f}", "dim")),
            Text.assemble(("Peak: ", "dim"), (f"${state.peak_bankroll:,.2f}", "white")),
            Text.assemble(("Max DD: ", "dim"), (f"{state.max_drawdown_pct:.1f}%", dd_color)),
        )

        # Row 3: Today's stats
        today = self._today_stats
        today_pnl_color = "green" if today.get("pnl", 0) >= 0 else "red"

        grid.add_row(
            Text.assemble(("Today Trades: ", "dim"), (f"{today.get('trades', 0)}", "white")),
            Text.assemble(("Today Wins: ", "dim"), (f"{today.get('wins', 0)}", "green")),
            Text.assemble(("Today P&L: ", "dim"), (f"${today.get('pnl', 0):+,.2f}", today_pnl_color)),
            Text.assemble(("", "dim"), ("", "dim")),
        )

        return Panel(grid, title="Portfolio Status", border_style="blue")

    def _make_metrics_panel(self) -> Panel:
        """Create the performance metrics panel."""
        state = self.trader.get_state()
        metrics = self._metrics

        if state is None:
            return Panel("No metrics", title="Performance Metrics", border_style="magenta")

        lines = []

        # Win rate
        wr = state.win_rate * 100
        wr_color = "green" if wr >= 55 else "yellow" if wr >= 50 else "red"
        lines.append(
            f"[dim]Win Rate:[/dim]      [{wr_color}]{wr:.1f}%[/{wr_color}] "
            f"[dim]({state.total_wins}W / {state.total_losses}L)[/dim]"
        )

        # Total trades
        lines.append(f"[dim]Total Trades:[/dim]  [white]{state.total_trades}[/white]")

        # Profit factor
        pf = metrics.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf < 100 else "Inf"
        pf_color = "green" if pf > 1.5 else "yellow" if pf > 1.0 else "red"
        lines.append(f"[dim]Profit Factor:[/dim] [{pf_color}]{pf_str}[/{pf_color}]")

        # Expectancy
        exp = metrics.get("expectancy", 0)
        exp_color = "green" if exp > 0 else "red"
        lines.append(f"[dim]Expectancy:[/dim]   [{exp_color}]${exp:.2f}[/{exp_color}]")

        # Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_color = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
        lines.append(f"[dim]Sharpe:[/dim]       [{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]")

        # Sortino ratio
        sortino = metrics.get("sortino_ratio", 0)
        sortino_color = "green" if sortino > 1 else "yellow" if sortino > 0 else "red"
        lines.append(f"[dim]Sortino:[/dim]      [{sortino_color}]{sortino:.2f}[/{sortino_color}]")

        # Calmar ratio
        calmar = metrics.get("calmar_ratio", 0)
        calmar_color = "green" if calmar > 1 else "yellow" if calmar > 0 else "red"
        lines.append(f"[dim]Calmar:[/dim]       [{calmar_color}]{calmar:.2f}[/{calmar_color}]")

        # Recovery factor
        rf = metrics.get("recovery_factor", 0)
        rf_str = f"{rf:.2f}" if rf < 100 else "Inf"
        rf_color = "green" if rf > 2 else "yellow" if rf > 1 else "red"
        lines.append(f"[dim]Recovery F:[/dim]   [{rf_color}]{rf_str}[/{rf_color}]")

        # Average win/loss
        avg_win = metrics.get("avg_win", 0)
        avg_loss = metrics.get("avg_loss", 0)
        lines.append(f"[dim]Avg Win:[/dim]      [green]${avg_win:.2f}[/green]")
        lines.append(f"[dim]Avg Loss:[/dim]     [red]${avg_loss:.2f}[/red]")

        # Streaks
        lines.append(f"[dim]Win Streak:[/dim]   [green]{state.current_win_streak}[/green] [dim](max {state.max_win_streak})[/dim]")
        lines.append(f"[dim]Loss Streak:[/dim]  [red]{state.current_loss_streak}[/red] [dim](max {state.max_loss_streak})[/dim]")

        content = "\n".join(lines)
        return Panel(content, title="Performance Metrics", border_style="magenta")

    def _make_open_positions_panel(self) -> Panel:
        """Create the open positions panel."""
        open_trades = self.trader.get_open_trades()

        if not open_trades:
            return Panel(
                "[dim]No open positions[/dim]",
                title="Open Positions",
                border_style="yellow",
            )

        table = Table(show_header=True, header_style="bold yellow", expand=True)
        table.add_column("Asset", width=6)
        table.add_column("Dir", width=5)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Bet", justify="right", width=8)
        table.add_column("Entry Time", width=12)
        table.add_column("Status", width=8)

        for asset, trade in open_trades.items():
            dir_color = "green" if trade.direction == "bull" else "red"
            dir_str = f"[{dir_color}]{trade.direction.upper()}[/{dir_color}]"

            entry_time = trade.entry_time.strftime("%H:%M") if trade.entry_time else "--"

            table.add_row(
                f"[bold]{asset}[/bold]",
                dir_str,
                f"{trade.entry_price:.3f}",
                f"${trade.bet_size:.2f}",
                entry_time,
                "[yellow]PENDING[/yellow]",
            )

        return Panel(table, title="Open Positions", border_style="yellow")

    def _make_recent_trades_panel(self) -> Panel:
        """Create the recent trades panel."""
        if not self._recent_trades:
            return Panel(
                "[dim]No recent trades[/dim]",
                title="Recent Trades (Last 10)",
                border_style="cyan",
            )

        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Time", width=6)
        table.add_column("Asset", width=5)
        table.add_column("Dir", width=5)
        table.add_column("Entry", justify="right", width=6)
        table.add_column("Bet", justify="right", width=7)
        table.add_column("P&L", justify="right", width=8)
        table.add_column("Result", width=6)
        table.add_column("Balance", justify="right", width=10)

        for trade in self._recent_trades:
            time_str = trade.entry_time.strftime("%H:%M") if trade.entry_time else "--"

            dir_color = "green" if trade.direction == "bull" else "red"
            dir_str = f"[{dir_color}]{trade.direction[:4].upper()}[/{dir_color}]"

            pnl = trade.net_pnl or 0
            pnl_color = "green" if pnl >= 0 else "red"
            pnl_str = f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]"

            result_color = "green" if trade.outcome == "win" else "red"
            result_str = f"[{result_color}]{trade.outcome.upper()}[/{result_color}]"

            balance = trade.bankroll_after or 0

            table.add_row(
                time_str,
                trade.asset,
                dir_str,
                f"{trade.entry_price:.2f}",
                f"${trade.bet_size:.0f}",
                pnl_str,
                result_str,
                f"${balance:,.0f}",
            )

        return Panel(table, title="Recent Trades (Last 10)", border_style="cyan")

    def _make_per_asset_panel(self) -> Panel:
        """Create the per-asset summary panel."""
        table = Table(show_header=True, header_style="bold green", expand=True)
        table.add_column("Asset", width=5)
        table.add_column("Trd", justify="right", width=4)
        table.add_column("Win%", justify="right", width=6)
        table.add_column("P&L", justify="right", width=8)
        table.add_column("Sig", width=5)
        table.add_column("Vol", width=4)

        for asset in self.sampler.assets:
            stats = self._asset_stats.get(asset, {})
            trades = stats.get("trades", 0)
            win_rate = stats.get("win_rate", 0) * 100
            total_pnl = stats.get("total_pnl", 0)
            last_signal = self.trader.get_last_signal(asset)
            regime = self._asset_regimes.get(asset)

            wr_color = "green" if win_rate >= 55 else "yellow" if win_rate >= 50 else "red" if trades > 0 else "dim"
            pnl_color = "green" if total_pnl > 0 else "red" if total_pnl < 0 else "dim"

            signal_str = ""
            if last_signal in ("bull", "bull-contrarian", "bull-consensus", "bull-accel", "bull-combo"):
                signal_str = "[green]BULL[/green]"
            elif last_signal in ("bear", "bear-contrarian", "bear-consensus", "bear-accel", "bear-combo"):
                signal_str = "[red]BEAR[/red]"
            elif last_signal and "no-consensus" in last_signal:
                signal_str = "[yellow]N-CS[/yellow]"
            elif last_signal and "no-xasset" in last_signal:
                signal_str = "[yellow]N-XA[/yellow]"
            elif last_signal and "no-confirm" in last_signal:
                signal_str = "[yellow]N-CF[/yellow]"
            elif last_signal and "no-t0" in last_signal:
                signal_str = "[yellow]N-T0[/yellow]"
            elif last_signal and "filtered" in last_signal:
                signal_str = "[yellow]FILT[/yellow]"
            elif last_signal and "pending" in last_signal:
                signal_str = "[yellow]PEND[/yellow]"
            elif last_signal and "faded" in last_signal:
                signal_str = "[yellow]FADE[/yellow]"
            else:
                signal_str = "[dim]-[/dim]"

            # Regime display
            if regime == "extreme":
                regime_str = "[red bold]EXT[/red bold]"
            elif regime == "high":
                regime_str = "[yellow]HI[/yellow]"
            elif regime == "normal":
                regime_str = "[green]NRM[/green]"
            elif regime == "low":
                regime_str = "[dim]LOW[/dim]"
            else:
                regime_str = "[dim]-[/dim]"

            table.add_row(
                f"[bold]{asset}[/bold]",
                str(trades),
                f"[{wr_color}]{win_rate:.0f}%[/{wr_color}]" if trades > 0 else "[dim]-[/dim]",
                f"[{pnl_color}]${total_pnl:+,.0f}[/{pnl_color}]" if trades > 0 else "[dim]-[/dim]",
                signal_str,
                regime_str,
            )

        return Panel(table, title="Per-Asset Summary", border_style="green")

    def _make_slot_status_panel(self) -> Panel:
        """Create the current window slot status panel.

        Shows different columns depending on entry mode:
        - Contrarian/Consensus: Prev PM, @entry, Now, Status
        - Two-stage: @t7.5, @t10, Now, Status
        - Single: @t7.5, Now, Status
        """
        is_contrarian = self.trader.entry_mode == "contrarian"
        is_consensus = self.trader.entry_mode == "contrarian_consensus"
        is_accel = self.trader.entry_mode == "accel_dbl"
        is_combo = self.trader.entry_mode == "combo_dbl"
        is_contrarian_family = is_contrarian or is_consensus or is_accel or is_combo
        is_two_stage = self.trader.entry_mode == "two_stage"

        # Determine entry time label for contrarian family
        if is_accel:
            entry_t_label = self.trader.accel_entry_time
        elif is_combo:
            entry_t_label = self.trader.combo_entry_time
        elif is_consensus:
            entry_t_label = self.trader.consensus_entry_time
        elif is_contrarian:
            entry_t_label = self.trader.contrarian_entry_time
        else:
            entry_t_label = "t7.5"

        # Get entry_t_minutes for extracting sample price
        entry_t_min = self.trader._time_to_minutes.get(entry_t_label, 0.0) if is_contrarian_family else 7.5

        table = Table(show_header=True, header_style="bold white", expand=True, box=None)
        table.add_column("Asset", width=5)

        if is_contrarian_family:
            if is_accel or is_combo:
                table.add_column("Pv2", justify="right", width=5)
            table.add_column("Prev", justify="right", width=5)
            if is_accel:
                table.add_column("@t0", justify="right", width=5)
            table.add_column(f"@{entry_t_label}", justify="right", width=5)
        else:
            table.add_column("@t7.5", justify="right", width=6)
            if is_two_stage:
                table.add_column("@t10", justify="right", width=6)

        table.add_column("Now", justify="right", width=5)
        table.add_column("Status", width=8)

        pending_signals = self.trader.get_pending_signals()
        open_trades = self.trader.get_open_trades()

        for asset in self.sampler.assets:
            state = self.sampler.get_current_state(asset)

            # Get prices at key timepoints
            price_at_entry = None
            price_at_7_5 = None
            price_at_10 = None
            current_price = None

            if state and state.samples:
                for sample in state.samples:
                    if is_contrarian_family and sample.t_minutes == entry_t_min:
                        price_at_entry = sample.pm_yes_price
                    if sample.t_minutes == 7.5:
                        price_at_7_5 = sample.pm_yes_price
                    elif sample.t_minutes == 10.0:
                        price_at_10 = sample.pm_yes_price
                # Get latest price
                current_price = state.samples[-1].pm_yes_price

            if is_contrarian_family:
                # Determine threshold for this mode
                if is_accel:
                    thresh = self.trader.accel_prev_thresh
                elif is_combo:
                    thresh = self.trader.combo_prev_thresh
                else:
                    thresh = self.trader.contrarian_prev_thresh

                # Previous2 window PM@t12.5 (for double contrarian)
                prev2_str = None
                if is_accel or is_combo:
                    prev2_pm = self.trader._prev2_window_pm.get(asset)
                    if prev2_pm is not None:
                        if prev2_pm >= thresh:
                            prev2_str = f"[green]{prev2_pm:.2f}[/green]"
                        elif prev2_pm <= (1.0 - thresh):
                            prev2_str = f"[red]{prev2_pm:.2f}[/red]"
                        else:
                            prev2_str = f"[dim]{prev2_pm:.2f}[/dim]"
                    else:
                        prev2_str = "[dim]--[/dim]"

                # Previous window PM@t12.5
                prev_pm = self.trader._prev_window_pm.get(asset)
                if prev_pm is not None:
                    if prev_pm >= thresh:
                        prev_str = f"[green]{prev_pm:.2f}[/green]"
                    elif prev_pm <= (1.0 - thresh):
                        prev_str = f"[red]{prev_pm:.2f}[/red]"
                    else:
                        prev_str = f"[dim]{prev_pm:.2f}[/dim]"
                else:
                    prev_str = "[dim]--[/dim]"

                # t0 price for accel mode
                t0_str = None
                if is_accel:
                    t0_pm = self.trader._accel_t0_pm.get(asset)
                    if t0_pm is not None:
                        band = self.trader.accel_neutral_band
                        if abs(t0_pm - 0.50) <= band:
                            t0_str = f"[green]{t0_pm:.2f}[/green]"
                        else:
                            t0_str = f"[yellow]{t0_pm:.2f}[/yellow]"
                    else:
                        t0_str = "[dim]--[/dim]"

                # Entry price at configured time
                if price_at_entry is not None:
                    e_color = "green" if price_at_entry >= 0.6 else "red" if price_at_entry <= 0.4 else "white"
                    entry_str = f"[{e_color}]{price_at_entry:.2f}[/{e_color}]"
                else:
                    entry_str = "[dim]--[/dim]"
            else:
                # Format t=7.5 price
                if price_at_7_5 is not None:
                    p_color = "green" if price_at_7_5 >= 0.7 else "red" if price_at_7_5 <= 0.3 else "white"
                    t75_str = f"[{p_color}]{price_at_7_5:.2f}[/{p_color}]"
                else:
                    t75_str = "[dim]--[/dim]"

                # Format t=10 price
                if price_at_10 is not None:
                    c_color = "green" if price_at_10 >= 0.85 else "red" if price_at_10 <= 0.15 else "white"
                    t10_str = f"[{c_color}]{price_at_10:.2f}[/{c_color}]"
                else:
                    t10_str = "[dim]--[/dim]"

            # Format current price
            if current_price is not None:
                c_color = "green" if current_price >= 0.6 else "red" if current_price <= 0.4 else "white"
                now_str = f"[{c_color}]{current_price:.2f}[/{c_color}]"
            else:
                now_str = "[dim]--[/dim]"

            # Determine status
            has_open_trade = asset in open_trades
            has_pending = asset in pending_signals
            last_signal = self.trader.get_last_signal(asset)

            if has_open_trade:
                status_str = "[green bold]OPEN[/green bold]"
            elif has_pending:
                direction = pending_signals[asset]["direction"]
                dir_color = "green" if direction == "bull" else "red"
                status_str = f"[{dir_color}]PEND[/{dir_color}]"
            elif last_signal and "no-consensus" in str(last_signal):
                status_str = "[yellow]NO-CS[/yellow]"
            elif last_signal and "no-confirm" in str(last_signal):
                status_str = "[yellow]NO-CF[/yellow]"
            elif last_signal and "faded" in str(last_signal):
                status_str = "[yellow]FADED[/yellow]"
            elif last_signal and "filtered" in str(last_signal):
                status_str = "[yellow]FILT[/yellow]"
            elif last_signal and "no-xasset" in str(last_signal):
                status_str = "[yellow]NO-XA[/yellow]"
            elif last_signal and "no-t0" in str(last_signal):
                status_str = "[yellow]NO-T0[/yellow]"
            elif last_signal and ("contrarian" in str(last_signal) or "consensus" in str(last_signal)
                                  or "accel" in str(last_signal) or "combo" in str(last_signal)):
                direction = last_signal.split("-")[0]
                dir_color = "green" if direction == "bull" else "red"
                status_str = f"[{dir_color}]DONE[/{dir_color}]"
            else:
                status_str = "[dim]SKIP[/dim]"

            if is_contrarian_family:
                row = [asset]
                if is_accel or is_combo:
                    row.append(prev2_str)
                row.append(prev_str)
                if is_accel:
                    row.append(t0_str)
                row.extend([entry_str, now_str, status_str])
                table.add_row(*row)
            elif is_two_stage:
                table.add_row(asset, t75_str, t10_str, now_str, status_str)
            else:
                table.add_row(asset, t75_str, now_str, status_str)

        if is_accel:
            title = "Window Status (Accel+DblContrarian)"
        elif is_combo:
            title = "Window Status (Combo+DblContrarian)"
        elif is_consensus:
            title = "Window Status (Consensus)"
        elif is_contrarian:
            title = "Window Status (Contrarian)"
        else:
            title = "Window Status"
        return Panel(table, title=title, border_style="white")

    def _make_footer(self) -> Panel:
        """Create the footer panel."""
        now = datetime.now(timezone.utc)
        state = self.trader.get_state()

        content = Text()

        if self.trader.entry_mode == "accel_dbl":
            content.append("Strategy: ", style="dim")
            content.append("Accel+DblContrarian ", style="magenta bold")
            content.append(
                f"Prev>={self.trader.accel_prev_thresh} "
                f"Band={self.trader.accel_neutral_band} "
                f"Bull>={self.trader.accel_bull_thresh} "
                f"Bear<={self.trader.accel_bear_thresh} "
                f"({self.trader.accel_entry_time}→{self.trader.accel_exit_time})",
                style="magenta",
            )
        elif self.trader.entry_mode == "combo_dbl":
            content.append("Strategy: ", style="dim")
            content.append("Combo+DblContrarian ", style="magenta bold")
            content.append(
                f"Prev>={self.trader.combo_prev_thresh} "
                f"XAsset>={self.trader.combo_xasset_min} "
                f"Stop={self.trader.combo_stop_delta}@{self.trader.combo_stop_time} "
                f"Bull>={self.trader.combo_bull_thresh} "
                f"Bear<={self.trader.combo_bear_thresh} "
                f"({self.trader.combo_entry_time}→{self.trader.combo_exit_time})",
                style="magenta",
            )
        elif self.trader.entry_mode == "contrarian_consensus":
            content.append("Strategy: ", style="dim")
            content.append("Contrarian+Consensus ", style="magenta bold")
            content.append(
                f"Prev>={self.trader.contrarian_prev_thresh} "
                f"{self.trader.consensus_min_agree}/4agree "
                f"Bull>={self.trader.contrarian_bull_thresh} "
                f"Bear<={self.trader.contrarian_bear_thresh} "
                f"({self.trader.consensus_entry_time}→{self.trader.consensus_exit_time})",
                style="magenta",
            )
        elif self.trader.entry_mode == "contrarian":
            content.append("Strategy: ", style="dim")
            content.append("Contrarian ", style="magenta bold")
            content.append(
                f"Prev>={self.trader.contrarian_prev_thresh} "
                f"Bull>={self.trader.contrarian_bull_thresh} "
                f"Bear<={self.trader.contrarian_bear_thresh} "
                f"({self.trader.contrarian_entry_time}→{self.trader.contrarian_exit_time})",
                style="magenta",
            )
        elif self.trader.entry_mode == "two_stage":
            content.append("Strategy: ", style="dim")
            content.append("Two-Stage ", style="cyan bold")
            content.append(
                f"Signal t7.5: >={self.trader.signal_threshold_bull}/<={self.trader.signal_threshold_bear}",
                style="cyan",
            )
            content.append(" → ", style="dim")
            content.append(
                f"Confirm t10: >={self.trader.confirm_threshold_bull}/<={self.trader.confirm_threshold_bear}",
                style="cyan",
            )
        else:
            content.append("Strategy: ", style="dim")
            content.append(
                f"BULL >= {self.trader.bull_threshold} | BEAR <= {self.trader.bear_threshold}",
                style="cyan",
            )

        content.append("  |  ", style="dim")
        content.append("Fees: ", style="dim")
        content.append(f"{self.trader.fee_rate*100:.1f}% + {self.trader.spread_cost*100:.1f}% spread", style="cyan")
        content.append("  |  ", style="dim")
        content.append("Sizing: ", style="dim")
        content.append(f"Fixed ${self.trader.base_bet:.0f}", style="cyan")
        content.append("  |  ", style="dim")
        content.append("Ctrl+C", style="bold")
        content.append(" to exit", style="dim")

        return Panel(content, style="dim")
