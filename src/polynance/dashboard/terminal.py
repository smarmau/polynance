"""Terminal dashboard using Rich library."""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn

from ..db.database import Database
from ..db.models import Window
from ..sampler import Sampler, AssetState, get_window_boundaries


class TerminalDashboard:
    """Rich terminal dashboard for live monitoring."""

    def __init__(self, sampler: Sampler, databases: Dict[str, Database], refresh_rate: float = 1.0):
        self.sampler = sampler
        self.databases = databases
        self.refresh_rate = refresh_rate
        self.console = Console()
        self._running = False

        # Cache for recent windows per asset
        self._recent_windows: Dict[str, List[Window]] = {}

        # Cache for total window counts per asset
        self._window_counts: Dict[str, int] = {}

        # Latest analysis results per asset
        self._analysis_results: Dict[str, dict] = {}

        # Status tracking
        self._start_time: Optional[datetime] = None
        self._last_sample_time: Dict[str, datetime] = {}
        self._total_samples: Dict[str, int] = {}
        self._api_status: Dict[str, str] = {"polymarket": "unknown", "binance": "unknown"}

    async def run(self):
        """Run the dashboard in a live loop."""
        self._running = True
        self._start_time = datetime.now(timezone.utc)

        with Live(self._generate_layout(), console=self.console, refresh_per_second=2) as live:
            while self._running:
                try:
                    # Update recent windows cache periodically
                    await self._update_caches()

                    # Update status info
                    self._update_status()

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

    def _update_status(self):
        """Update status tracking info."""
        for asset in self.sampler.assets:
            state = self.sampler.get_current_state(asset)
            if state and state.samples:
                self._last_sample_time[asset] = state.samples[-1].sample_time_utc
                self._total_samples[asset] = len(state.samples)

                # Update API status based on recent samples
                latest = state.samples[-1]
                if latest.pm_yes_price is not None:
                    self._api_status["polymarket"] = "ok"
                if latest.spot_price is not None:
                    self._api_status["binance"] = "ok"

    async def _update_caches(self):
        """Update cached data."""
        for asset in self.sampler.assets:
            # Get the database for this asset
            db = self.databases.get(asset)
            if not db:
                continue

            try:
                windows = await db.get_recent_windows(asset, limit=10)
                self._recent_windows[asset] = windows
            except Exception:
                pass

            try:
                count = await db.get_window_count(asset)
                self._window_counts[asset] = count
            except Exception:
                pass

            try:
                analysis = await db.get_latest_analysis(asset)
                if analysis:
                    self._analysis_results[asset] = analysis
            except Exception:
                pass

    def _generate_layout(self) -> Layout:
        """Generate the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(self._make_header())

        # Main content
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        # Left side: current windows
        layout["left"].update(self._make_current_windows_panel())

        # Right side: recent results and stats
        layout["right"].split_column(
            Layout(name="recent", ratio=2),
            Layout(name="stats", ratio=1),
        )
        layout["right"]["recent"].update(self._make_recent_windows_panel())
        layout["right"]["stats"].update(self._make_stats_panel())

        # Footer
        layout["footer"].update(self._make_footer())

        return layout

    def _make_header(self) -> Panel:
        """Create the header panel."""
        now = datetime.now(timezone.utc)
        window_start, window_end = get_window_boundaries(now)

        time_remaining = (window_end - now).total_seconds()
        minutes = int(time_remaining // 60)
        seconds = int(time_remaining % 60)

        title = Text()
        title.append("POLYNANCE ", style="bold cyan")
        title.append("- Polymarket 15-Min Crypto Tracker", style="dim")

        status = Text()
        status.append(f"  UTC: {now.strftime('%H:%M:%S')}  ", style="dim")
        status.append(f"Window: {window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}  ", style="green")
        status.append(f"Remaining: {minutes:02d}:{seconds:02d}", style="yellow")

        content = Text()
        content.append_text(title)
        content.append("\n")
        content.append_text(status)

        return Panel(content, style="bold")

    def _make_current_windows_panel(self) -> Panel:
        """Create the current windows panel."""
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Asset", width=6)
        table.add_column("YES Price", justify="right", width=10)
        table.add_column("Spread", justify="right", width=8)
        table.add_column("Spot", justify="right", width=12)
        table.add_column("Change", justify="right", width=10)
        table.add_column("Samples", justify="center", width=8)
        table.add_column("Progress", width=20)

        now = datetime.now(timezone.utc)
        window_start, window_end = get_window_boundaries(now)

        for asset in self.sampler.assets:
            state = self.sampler.get_current_state(asset)

            if state is None or state.market is None:
                table.add_row(
                    asset,
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]N/A[/dim]",
                    "[dim]0[/dim]",
                    "[dim]No market[/dim]",
                )
                continue

            # Get latest sample
            latest = state.samples[-1] if state.samples else None

            # YES price with color
            yes_price = latest.pm_yes_price if latest else None
            if yes_price is not None:
                if yes_price > 0.6:
                    yes_str = f"[green]{yes_price:.3f}[/green]"
                elif yes_price < 0.4:
                    yes_str = f"[red]{yes_price:.3f}[/red]"
                else:
                    yes_str = f"{yes_price:.3f}"
            else:
                yes_str = "[dim]--[/dim]"

            # Spread
            spread = latest.pm_spread if latest else None
            spread_str = f"{spread:.3f}" if spread else "[dim]--[/dim]"

            # Spot price
            spot = latest.spot_price if latest else None
            spot_str = f"${spot:,.2f}" if spot else "[dim]--[/dim]"

            # Change from open
            change = latest.spot_price_change_from_open if latest else None
            if change is not None:
                if change > 0:
                    change_str = f"[green]+{change:.3f}%[/green]"
                elif change < 0:
                    change_str = f"[red]{change:.3f}%[/red]"
                else:
                    change_str = "0.000%"
            else:
                change_str = "[dim]--[/dim]"

            # Samples count
            sample_count = len(state.samples)

            # Progress bar
            elapsed = (now - window_start).total_seconds()
            progress_pct = min(elapsed / 900 * 100, 100)  # 900 seconds = 15 min
            progress_bar = self._make_progress_bar(progress_pct)

            table.add_row(
                f"[bold]{asset}[/bold]",
                yes_str,
                spread_str,
                spot_str,
                change_str,
                str(sample_count),
                progress_bar,
            )

        return Panel(table, title="Current Windows", border_style="green")

    def _make_progress_bar(self, pct: float) -> str:
        """Create a simple ASCII progress bar."""
        filled = int(pct / 5)  # 20 chars total
        empty = 20 - filled
        return f"[{'█' * filled}{'░' * empty}] {pct:.0f}%"

    def _make_recent_windows_panel(self) -> Panel:
        """Create the recent windows panel."""
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Time", width=6)
        table.add_column("Asset", width=5)
        table.add_column("YES@t5", justify="right", width=7)
        table.add_column("Result", justify="center", width=6)
        table.add_column("Move", justify="right", width=8)
        table.add_column("", width=3)

        # Collect and sort recent windows from all assets
        all_windows = []
        for asset, windows in self._recent_windows.items():
            for w in windows[:5]:  # Last 5 per asset
                all_windows.append(w)

        # Sort by time (most recent first)
        all_windows.sort(key=lambda w: w.window_start_utc, reverse=True)

        for w in all_windows[:10]:  # Show top 10
            time_str = w.window_start_utc.strftime("%H:%M")

            # YES@t5
            yes_t5 = w.pm_yes_t5
            if yes_t5 is not None:
                yes_str = f"{yes_t5:.2f}"
            else:
                yes_str = "[dim]--[/dim]"

            # Outcome
            if w.outcome == "up":
                outcome_str = "[green]UP[/green]"
            elif w.outcome == "down":
                outcome_str = "[red]DOWN[/red]"
            else:
                outcome_str = "[dim]--[/dim]"

            # Move
            if w.spot_change_bps is not None:
                if w.spot_change_bps > 0:
                    move_str = f"[green]+{w.spot_change_bps:.0f}bps[/green]"
                else:
                    move_str = f"[red]{w.spot_change_bps:.0f}bps[/red]"
            else:
                move_str = "[dim]--[/dim]"

            # Prediction check
            check = ""
            if yes_t5 is not None and w.outcome:
                predicted_up = yes_t5 > 0.55
                predicted_down = yes_t5 < 0.45
                actual_up = w.outcome == "up"

                if (predicted_up and actual_up) or (predicted_down and not actual_up):
                    check = "[green]✓[/green]"
                elif predicted_up or predicted_down:
                    check = "[red]✗[/red]"

            table.add_row(time_str, w.asset, yes_str, outcome_str, move_str, check)

        return Panel(table, title="Recent Windows", border_style="blue")

    def _make_stats_panel(self) -> Panel:
        """Create the stats panel showing data collection progress."""
        lines = []

        # Minimum sample size for meaningful analysis
        MIN_SAMPLE_SIZE = 100

        for asset in self.sampler.assets:
            windows = self._recent_windows.get(asset, [])

            # Get actual total window count from cached DB query
            total = self._window_counts.get(asset, 0)

            if total == 0:
                lines.append(f"[bold]{asset}[/bold]: No data collected yet")
                continue

            # Show collection progress
            if total < MIN_SAMPLE_SIZE:
                remaining = MIN_SAMPLE_SIZE - total
                pct = (total / MIN_SAMPLE_SIZE) * 100

                # Progress bar for collection
                filled = int(pct / 5)  # 20 chars
                empty = 20 - filled
                bar = f"[{'█' * filled}{'░' * empty}]"

                lines.append(
                    f"[bold]{asset}[/bold]: {total} windows collected  {bar} {pct:.0f}%"
                )
                lines.append(f"  [dim]Need {remaining} more for statistical analysis[/dim]")
            else:
                # Show basic directional stats (descriptive only, from recent windows)
                recent_total = len(windows)
                up_count = sum(1 for w in windows if w.outcome == "up")
                up_rate = up_count / recent_total * 100 if recent_total > 0 else 0

                # Count strong predictions (YES > 0.6 or < 0.4)
                strong_predictions = sum(
                    1 for w in windows
                    if w.pm_yes_t5 is not None and (w.pm_yes_t5 > 0.6 or w.pm_yes_t5 < 0.4)
                )

                lines.append(
                    f"[bold]{asset}[/bold]: {total} windows collected"
                )
                lines.append(f"  [dim]Recent: {up_rate:.0f}% up, {strong_predictions}/{recent_total} strong signals[/dim]")
                lines.append(f"  [green]✓ Ready for statistical analysis[/green]")

        content = "\n".join(lines) if lines else "No data yet"
        return Panel(content, title="Data Collection Progress", border_style="yellow")

    def _make_footer(self) -> Panel:
        """Create the footer panel with status information."""
        now = datetime.now(timezone.utc)

        # Calculate uptime
        uptime_str = "--:--:--"
        if self._start_time:
            uptime = now - self._start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Total samples across all assets
        total_samples = sum(self._total_samples.values())

        # Total windows across all assets (from actual DB counts)
        total_windows = sum(self._window_counts.values())

        # Last sample time (most recent across all assets)
        last_sample_str = "--:--:--"
        if self._last_sample_time:
            most_recent = max(self._last_sample_time.values())
            last_sample_str = most_recent.strftime("%H:%M:%S")
            # Show how long ago
            ago = (now - most_recent).total_seconds()
            if ago < 60:
                last_sample_str += f" ({int(ago)}s ago)"
            else:
                last_sample_str += f" ({int(ago/60)}m ago)"

        # API status
        pm_status = self._api_status.get("polymarket", "unknown")
        bn_status = self._api_status.get("binance", "unknown")

        pm_style = "green" if pm_status == "ok" else ("red" if pm_status == "error" else "yellow")
        bn_style = "green" if bn_status == "ok" else ("red" if bn_status == "error" else "yellow")

        content = Text()
        content.append("Uptime: ", style="dim")
        content.append(uptime_str, style="cyan")
        content.append("  |  ", style="dim")
        content.append("Samples: ", style="dim")
        content.append(str(total_samples), style="cyan")
        content.append("  |  ", style="dim")
        content.append("Windows: ", style="dim")
        content.append(str(total_windows), style="cyan")
        content.append("  |  ", style="dim")
        content.append("Last: ", style="dim")
        content.append(last_sample_str, style="cyan")
        content.append("  |  ", style="dim")
        content.append("Polymarket: ", style="dim")
        content.append(pm_status.upper(), style=pm_style)
        content.append("  ", style="dim")
        content.append("Binance: ", style="dim")
        content.append(bn_status.upper(), style=bn_style)
        content.append("  |  ", style="dim")
        content.append("Ctrl+C", style="bold")
        content.append(" to exit", style="dim")

        return Panel(content, style="dim")

    def update_analysis(self, asset: str, results: dict):
        """Update analysis results for display."""
        self._analysis_results[asset] = results
