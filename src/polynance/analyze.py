"""Standalone analysis script for polynance data."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List

from .db.database import Database
from .analysis.analyzer import Analyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP"]


async def analyze_asset(asset: str, data_dir: Path, generate_report: bool = True):
    """Run analysis for a single asset."""
    db_path = data_dir / f"{asset.lower()}.db"

    if not db_path.exists():
        logger.warning(f"[{asset}] Database not found: {db_path}")
        return None

    async with Database(db_path) as db:
        # Get stats first
        stats = await db.get_stats(asset)
        logger.info(f"[{asset}] Database stats: {stats}")

        if stats["total_windows"] < 5:
            logger.warning(f"[{asset}] Not enough data ({stats['total_windows']} windows)")
            return None

        # Run analysis
        analyzer = Analyzer(db, data_dir / "reports")
        results = await analyzer.run_full_analysis(asset, min_windows=5)

        # Print summary
        print(analyzer.format_summary(results))

        # Generate markdown report
        if generate_report and results.get("status") != "insufficient_data":
            report_path = await analyzer.generate_report(asset)
            logger.info(f"[{asset}] Report saved to: {report_path}")

        return results


async def analyze_all(assets: List[str], data_dir: Path, generate_report: bool = True):
    """Run analysis for all assets."""
    results = {}

    for asset in assets:
        try:
            result = await analyze_asset(asset, data_dir, generate_report)
            if result:
                results[asset] = result
        except Exception as e:
            logger.error(f"[{asset}] Analysis failed: {e}")

    return results


async def show_stats(assets: List[str], data_dir: Path):
    """Show basic stats for all assets."""
    print("\n" + "=" * 60)
    print("POLYNANCE DATABASE STATS")
    print("=" * 60)

    for asset in assets:
        db_path = data_dir / f"{asset.lower()}.db"

        if not db_path.exists():
            print(f"\n{asset}: No database found")
            continue

        async with Database(db_path) as db:
            stats = await db.get_stats(asset)

            print(f"\n{asset}:")
            print(f"  Windows: {stats['total_windows']}")
            print(f"  Samples: {stats['total_samples']}")
            print(f"  Up/Down: {stats['up_count']}/{stats['down_count']} ({stats['up_rate']*100:.1f}% up)")
            if stats['avg_move_bps']:
                print(f"  Avg Move: {stats['avg_move_bps']:.1f} bps")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze polynance data"
    )

    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=DEFAULT_ASSETS,
        help=f"Assets to analyze (default: {DEFAULT_ASSETS})",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory for data storage (default: {DATA_DIR})",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show basic stats, no full analysis",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating markdown reports",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def async_main(args):
    """Async main function."""
    if args.stats_only:
        await show_stats(args.assets, args.data_dir)
    else:
        await analyze_all(args.assets, args.data_dir, not args.no_report)


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
