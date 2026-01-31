"""Hourly statistical analysis runner."""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

from ..db.database import Database
from .statistical_tests import analyze_market_predictiveness, format_analysis_report

logger = logging.getLogger(__name__)


class HourlyAnalyzer:
    """Runs statistical analysis on the hour."""

    def __init__(
        self,
        databases: Dict[str, Database],
        assets: List[str],
        reports_dir: Path,
    ):
        self.databases = databases
        self.assets = assets
        self.reports_dir = reports_dir
        self._running = False
        self._last_analysis_time: Dict[str, datetime] = {}

    async def run(self):
        """Run the hourly analysis loop."""
        self._running = True
        logger.info("Starting hourly analysis loop...")

        # Run first analysis after 5 minutes (to let some data accumulate)
        await asyncio.sleep(300)

        while self._running:
            try:
                await self._run_analysis()

                # Wait until next hour
                now = datetime.now(timezone.utc)
                minutes_until_next_hour = 60 - now.minute
                seconds = minutes_until_next_hour * 60 - now.second

                logger.info(f"Next analysis in {minutes_until_next_hour} minutes")
                await asyncio.sleep(seconds)

            except asyncio.CancelledError:
                logger.info("Hourly analysis loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in hourly analysis: {e}", exc_info=True)
                await asyncio.sleep(300)  # Retry in 5 minutes

    def stop(self):
        """Stop the analysis loop."""
        self._running = False

    async def _run_analysis(self):
        """Run analysis for all assets."""
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d %H:%M UTC")

        logger.info("=" * 70)
        logger.info(f"Running hourly analysis at {timestamp}")
        logger.info("=" * 70)

        for asset in self.assets:
            try:
                await self._analyze_asset(asset, now)
            except Exception as e:
                logger.error(f"Error analyzing {asset}: {e}", exc_info=True)

    async def _analyze_asset(self, asset: str, now: datetime):
        """Run analysis for a single asset."""
        db = self.databases.get(asset)
        if not db:
            logger.warning(f"No database for {asset}")
            return

        # Get all windows for this asset
        windows = await db.get_all_windows(asset)

        if not windows:
            logger.info(f"[{asset}] No windows to analyze yet")
            return

        logger.info(f"[{asset}] Analyzing {len(windows)} windows...")

        # Run statistical analysis
        summary = analyze_market_predictiveness(windows, asset, threshold=0.5)

        # Format and log report
        report = format_analysis_report(summary)
        logger.info(f"\n{report}")

        # Save report to file
        self._save_report(asset, now, report, summary)

        # Update last analysis time
        self._last_analysis_time[asset] = now

    def _save_report(self, asset: str, timestamp: datetime, report: str, summary):
        """Save analysis report to file."""
        try:
            # Create reports directory
            self.reports_dir.mkdir(parents=True, exist_ok=True)

            # Hourly report
            hourly_file = self.reports_dir / f"{asset.lower()}_hourly_{timestamp.strftime('%Y%m%d_%H%M')}.txt"
            hourly_file.write_text(report)

            # Also update latest report
            latest_file = self.reports_dir / f"{asset.lower()}_latest.txt"
            latest_file.write_text(report)

            # Save CSV summary for easy analysis
            csv_file = self.reports_dir / f"{asset.lower()}_analysis_log.csv"

            # Create header if file doesn't exist
            if not csv_file.exists():
                csv_file.write_text(
                    "timestamp,total_windows,up_rate,predictions_made,accuracy,"
                    "p_value,significant,auc_score,optimal_threshold\n"
                )

            # Append row
            p_value = summary.binomial_test.p_value if summary.binomial_test else None
            significant = summary.binomial_test.significant if summary.binomial_test else False

            row = (
                f"{timestamp.isoformat()},"
                f"{summary.total_windows},"
                f"{summary.up_rate:.4f},"
                f"{summary.predictions_made},"
                f"{summary.accuracy:.4f},"
                f"{p_value if p_value is not None else ''},"
                f"{1 if significant else 0},"
                f"{summary.auc_score if summary.auc_score is not None else ''},"
                f"{summary.optimal_threshold if summary.optimal_threshold is not None else ''}\n"
            )

            with csv_file.open('a') as f:
                f.write(row)

            logger.info(f"[{asset}] Saved reports to {self.reports_dir}")

        except Exception as e:
            logger.error(f"Error saving report for {asset}: {e}")
