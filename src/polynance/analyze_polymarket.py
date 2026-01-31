#!/usr/bin/env python3
"""CLI for Polymarket trading viability analysis."""

import asyncio
import argparse
from pathlib import Path

from .analysis.polymarket_trading import run_polymarket_analysis


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze Polymarket trading viability"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing database files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for report (optional, prints to stdout if not specified)",
    )

    args = parser.parse_args()

    report = await run_polymarket_analysis(args.data_dir)

    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    asyncio.run(main())
