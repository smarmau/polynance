"""Terminal dashboard for arbitrage tracking.

Displays real-time signals and lock opportunity status.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .database import ArbitrageDatabase
from .signals import SignalState


class ArbitrageDashboard:
    """Real-time dashboard for arbitrage signal monitoring."""

    def __init__(
        self,
        db: ArbitrageDatabase,
        assets: List[str],
        refresh_rate: float = 0.5,
    ):
        self.db = db
        self.assets = assets
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Current signal states (updated by sampler callback)
        self.signal_states: Dict[str, SignalState] = {}

        # Stats cache
        self._stats_cache: Dict = {}
        self._last_stats_update: Optional[datetime] = None

        # Dashboard state
        self._running = False
        self._start_time: Optional[datetime] = None

    def update_signal(self, state: SignalState):
        """Update signal state (called by sampler)."""
        self.signal_states[state.asset] = state

    async def run(self):
        """Run the dashboard."""
        self._running = True
        self._start_time = datetime.now(timezone.utc)

        with Live(self._generate_layout(), console=self.console, refresh_per_second=2) as live:
            while self._running:
                try:
                    await self._update_stats_cache()
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

    async def _update_stats_cache(self):
        """Update cached stats periodically."""
        now = datetime.now(timezone.utc)
        if self._last_stats_update and (now - self._last_stats_update).seconds < 5:
            return

        try:
            self._stats_cache = await self.db.get_stats()
            self._last_stats_update = now
        except Exception:
            pass

    def _generate_layout(self) -> Layout:
        """Generate the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="signals", ratio=2),
            Layout(name="stats", ratio=1),
        )

        layout["header"].update(self._make_header())
        layout["signals"].update(self._make_signals_panel())
        layout["stats"].update(self._make_stats_panel())
        layout["footer"].update(self._make_footer())

        return layout

    def _make_header(self) -> Panel:
        """Create header panel."""
        title = Text()
        title.append("ARBITRAGE TRACKER", style="bold magenta")
        title.append(" - High-Frequency Signal Monitor", style="dim")
        return Panel(title, style="magenta")

    def _make_signals_panel(self) -> Panel:
        """Create the main signals panel."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)

        table.add_column("Asset", width=6)
        table.add_column("YES", width=6, justify="right")
        table.add_column("Spread", width=7, justify="right")
        table.add_column("RHR", width=7, justify="right")
        table.add_column("OBI", width=7, justify="right")
        table.add_column("Flips", width=6, justify="center")
        table.add_column("Pattern", width=10)
        table.add_column("Lock", width=8, justify="center")
        table.add_column("Guards", width=12)

        for asset in self.assets:
            state = self.signal_states.get(asset)

            if not state:
                table.add_row(
                    asset, "-", "-", "-", "-", "-", "-", "-", "[dim]No data[/dim]"
                )
                continue

            # YES price
            yes_str = f"{state.yes_price:.3f}"
            if state.yes_price >= 0.65:
                yes_style = "green"
            elif state.yes_price <= 0.35:
                yes_style = "red"
            else:
                yes_style = "yellow"

            # Spread
            spread_str = f"{state.spread*100:.2f}%"
            spread_style = "green" if state.spread <= 0.01 else ("yellow" if state.spread <= 0.02 else "red")

            # RHR
            rhr_str = f"{state.rhr*100:.1f}%"
            rhr_style = "red" if state.rhr_guard_triggered else "green"

            # OBI
            obi_str = f"{state.obi:+.2f}"
            obi_style = "red" if state.obi_guard_triggered else "green"

            # Flips
            flip_str = str(state.flip_count)
            flip_style = "red" if state.flip_doom else "green"

            # Pattern
            pattern_styles = {
                'reversal': 'green',
                'trending': 'yellow',
                'choppy': 'red',
                'flat': 'dim',
            }
            pattern_style = pattern_styles.get(state.pattern, 'white')

            # Lock status
            if state.lock_achievable:
                lock_str = f"[green]✓ {state.estimated_lock_profit*100:.1f}%[/green]"
            else:
                lock_str = "[dim]--[/dim]"

            # Guards
            guards = []
            if state.rhr_guard_triggered:
                guards.append("[red]RHR[/red]")
            if state.obi_guard_triggered:
                guards.append("[red]OBI[/red]")
            if state.flip_doom:
                guards.append("[red]FLIP[/red]")
            if state.ptb_doom:
                guards.append("[red]PTB[/red]")
            guards_str = " ".join(guards) if guards else "[green]CLEAR[/green]"

            table.add_row(
                asset,
                f"[{yes_style}]{yes_str}[/{yes_style}]",
                f"[{spread_style}]{spread_str}[/{spread_style}]",
                f"[{rhr_style}]{rhr_str}[/{rhr_style}]",
                f"[{obi_style}]{obi_str}[/{obi_style}]",
                f"[{flip_style}]{flip_str}[/{flip_style}]",
                f"[{pattern_style}]{state.pattern}[/{pattern_style}]",
                lock_str,
                guards_str,
            )

        return Panel(table, title="Live Signals", border_style="cyan")

    def _make_stats_panel(self) -> Panel:
        """Create stats panel."""
        lines = []

        stats = self._stats_cache
        if stats:
            lines.append("[bold]Session Stats[/bold]")
            lines.append(f"  Ticks: {stats.get('tick_count', 0):,}")
            lines.append(f"  Windows: {stats.get('window_count', 0)}")
            lines.append(f"  Lockable: {stats.get('lockable_pct', 0):.1f}%")
            lines.append("")

        # Current window summary
        lines.append("[bold]Current Windows[/bold]")
        for asset in self.assets:
            state = self.signal_states.get(asset)
            if state:
                range_pct = (state.yes_high - state.yes_low) * 100
                t_min = state.t_seconds / 60
                lines.append(f"  {asset}: {t_min:.1f}m, range {range_pct:.1f}%")

        # Guard summary
        lines.append("")
        lines.append("[bold]Guard Status[/bold]")
        clear_count = 0
        blocked_count = 0
        for asset in self.assets:
            state = self.signal_states.get(asset)
            if state:
                if state.rhr_guard_triggered or state.obi_guard_triggered or state.flip_doom:
                    blocked_count += 1
                else:
                    clear_count += 1

        lines.append(f"  [green]Clear: {clear_count}[/green]")
        lines.append(f"  [red]Blocked: {blocked_count}[/red]")

        content = "\n".join(lines)
        return Panel(content, title="Stats", border_style="blue")

    def _make_footer(self) -> Panel:
        """Create footer panel."""
        now = datetime.now(timezone.utc)

        # Uptime
        uptime_str = "--:--:--"
        if self._start_time:
            uptime = now - self._start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Build footer
        content = Text()
        content.append("Uptime: ", style="dim")
        content.append(uptime_str, style="cyan")
        content.append("  |  ", style="dim")
        content.append("Sampling: ", style="dim")
        content.append("30s", style="cyan")
        content.append("  |  ", style="dim")
        content.append("Time: ", style="dim")
        content.append(now.strftime("%H:%M:%S UTC"), style="cyan")
        content.append("  |  ", style="dim")
        content.append("Ctrl+C to exit", style="dim")

        return Panel(content, style="dim")


async def run_dashboard_standalone(data_dir: str = "data"):
    """Run dashboard in standalone mode (reads from DB only)."""
    from pathlib import Path

    db_path = Path(data_dir) / "arbitrage.db"

    async with ArbitrageDatabase(db_path) as db:
        dashboard = ArbitrageDashboard(
            db=db,
            assets=['BTC', 'ETH', 'SOL', 'XRP'],
        )
        await dashboard.run()


if __name__ == "__main__":
    asyncio.run(run_dashboard_standalone())
