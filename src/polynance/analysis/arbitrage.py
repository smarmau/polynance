"""Arbitrage opportunity analysis for Polymarket 15-min markets.

Analyzes conditions that enable market-neutral "lock" strategies where
profit is guaranteed regardless of outcome by holding both YES and NO.

The key insight: if you can buy YES at price P_yes and NO at price P_no
where P_yes + P_no < 1.0, you have an arbitrage (guaranteed profit).

In practice, with spreads this is rare. The "Incremental Pair" strategy
instead builds positions over time as prices move, achieving a lock state
where the weighted average costs guarantee profit either way.

This module analyzes:
1. Price volatility within windows (needed for position building)
2. Spread dynamics (cost of crossing)
3. "Lockability" conditions - when can a lock be achieved?
4. Optimal entry timing
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..db.database import Database

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageWindow:
    """Analysis of a single window for arbitrage potential."""
    window_id: str
    asset: str
    window_start: datetime

    # Price movement
    yes_open: float
    yes_close: float
    yes_high: float
    yes_low: float
    yes_range: float  # high - low

    # Spread stats
    avg_spread: float
    max_spread: float
    min_spread: float

    # Velocity (price change per minute)
    max_velocity: float
    avg_velocity: float

    # Lock potential metrics
    distance_from_fair: float  # abs(yes_price - 0.5) at t=5
    price_range_vs_spread: float  # range / avg_spread (higher = more opportunity)

    # Did a lock opportunity exist?
    lock_achievable: bool
    lock_profit_potential: float  # Estimated max profit if locked


@dataclass
class ArbitrageAnalysisResult:
    """Complete arbitrage analysis result."""
    analysis_time: str
    total_windows: int
    assets: List[str]

    # Summary stats
    avg_yes_range: float
    avg_spread: float
    pct_lockable: float
    avg_lock_profit: float

    # By asset
    by_asset: Dict[str, Dict]

    # By time of day
    by_hour: Dict[int, Dict]

    # Conditions analysis
    best_conditions: Dict
    worst_conditions: Dict

    # Recommendations
    recommendations: List[str]


class ArbitrageAnalyzer:
    """Analyzer for arbitrage/lock opportunities in Polymarket."""

    # Minimum price range (as multiple of spread) to consider lockable
    MIN_RANGE_SPREAD_RATIO = 3.0

    # Estimated cost to achieve lock (crossing spread multiple times)
    LOCK_COST_ESTIMATE = 0.03  # 3% of position

    def __init__(self, databases: Dict[str, Database]):
        self.databases = databases

    async def run_analysis(self) -> ArbitrageAnalysisResult:
        """Run complete arbitrage opportunity analysis."""

        all_samples = []
        assets = []

        for asset, db in self.databases.items():
            samples = await self._get_all_samples(db, asset)
            all_samples.extend(samples)
            assets.append(asset)

        if not all_samples:
            return self._empty_result(assets)

        df = pd.DataFrame(all_samples)

        # Analyze each window
        windows = self._analyze_windows(df)

        if not windows:
            return self._empty_result(assets)

        # Aggregate analysis
        windows_df = pd.DataFrame([self._window_to_dict(w) for w in windows])

        # Overall stats
        avg_yes_range = windows_df['yes_range'].mean()
        avg_spread = windows_df['avg_spread'].mean()
        pct_lockable = windows_df['lock_achievable'].mean() * 100
        avg_lock_profit = windows_df[windows_df['lock_achievable']]['lock_profit_potential'].mean()

        # By asset
        by_asset = {}
        for asset in assets:
            asset_df = windows_df[windows_df['asset'] == asset]
            if len(asset_df) > 0:
                by_asset[asset] = {
                    'n_windows': len(asset_df),
                    'avg_range': float(asset_df['yes_range'].mean()),
                    'avg_spread': float(asset_df['avg_spread'].mean()),
                    'pct_lockable': float(asset_df['lock_achievable'].mean() * 100),
                    'avg_lock_profit': float(asset_df[asset_df['lock_achievable']]['lock_profit_potential'].mean()) if asset_df['lock_achievable'].any() else 0,
                }

        # By hour (UTC)
        windows_df['hour'] = pd.to_datetime(windows_df['window_start']).dt.hour
        by_hour = {}
        for hour in range(24):
            hour_df = windows_df[windows_df['hour'] == hour]
            if len(hour_df) >= 5:
                by_hour[hour] = {
                    'n_windows': len(hour_df),
                    'avg_range': float(hour_df['yes_range'].mean()),
                    'pct_lockable': float(hour_df['lock_achievable'].mean() * 100),
                }

        # Best/worst conditions
        lockable = windows_df[windows_df['lock_achievable']]
        not_lockable = windows_df[~windows_df['lock_achievable']]

        best_conditions = {
            'avg_range': float(lockable['yes_range'].mean()) if len(lockable) > 0 else 0,
            'avg_spread': float(lockable['avg_spread'].mean()) if len(lockable) > 0 else 0,
            'avg_distance_from_fair': float(lockable['distance_from_fair'].mean()) if len(lockable) > 0 else 0,
        }

        worst_conditions = {
            'avg_range': float(not_lockable['yes_range'].mean()) if len(not_lockable) > 0 else 0,
            'avg_spread': float(not_lockable['avg_spread'].mean()) if len(not_lockable) > 0 else 0,
            'avg_distance_from_fair': float(not_lockable['distance_from_fair'].mean()) if len(not_lockable) > 0 else 0,
        }

        # Recommendations
        recommendations = self._generate_recommendations(
            avg_yes_range, avg_spread, pct_lockable, by_asset, by_hour
        )

        return ArbitrageAnalysisResult(
            analysis_time=datetime.now(timezone.utc).isoformat(),
            total_windows=len(windows),
            assets=assets,
            avg_yes_range=float(avg_yes_range),
            avg_spread=float(avg_spread),
            pct_lockable=float(pct_lockable),
            avg_lock_profit=float(avg_lock_profit) if not np.isnan(avg_lock_profit) else 0,
            by_asset=by_asset,
            by_hour=by_hour,
            best_conditions=best_conditions,
            worst_conditions=worst_conditions,
            recommendations=recommendations,
        )

    async def _get_all_samples(self, db: Database, asset: str) -> List[Dict]:
        """Get all samples for analysis."""
        sql = """
        SELECT
            window_id, asset, window_start_utc, sample_time_utc, t_minutes,
            pm_yes_price, pm_no_price, pm_yes_bid, pm_yes_ask, pm_spread,
            spot_price
        FROM samples
        WHERE asset = ?
        ORDER BY window_start_utc, t_minutes
        """
        cursor = await db.conn.execute(sql, (asset,))
        rows = await cursor.fetchall()

        return [
            {
                'window_id': row[0],
                'asset': row[1],
                'window_start': row[2],
                'sample_time': row[3],
                't_minutes': row[4],
                'yes_price': row[5],
                'no_price': row[6],
                'yes_bid': row[7],
                'yes_ask': row[8],
                'spread': row[9],
                'spot_price': row[10],
            }
            for row in rows
        ]

    def _analyze_windows(self, df: pd.DataFrame) -> List[ArbitrageWindow]:
        """Analyze each window for arbitrage potential."""
        windows = []

        for window_id in df['window_id'].unique():
            window_df = df[df['window_id'] == window_id].sort_values('t_minutes')

            if len(window_df) < 3:
                continue

            asset = window_df['asset'].iloc[0]
            window_start = window_df['window_start'].iloc[0]

            # Price movement
            yes_prices = window_df['yes_price'].dropna()
            if len(yes_prices) < 2:
                continue

            yes_open = yes_prices.iloc[0]
            yes_close = yes_prices.iloc[-1]
            yes_high = yes_prices.max()
            yes_low = yes_prices.min()
            yes_range = yes_high - yes_low

            # Spread stats
            spreads = window_df['spread'].dropna()
            avg_spread = spreads.mean() if len(spreads) > 0 else 0.01
            max_spread = spreads.max() if len(spreads) > 0 else 0.01
            min_spread = spreads.min() if len(spreads) > 0 else 0.01

            # Velocity (price change between samples)
            price_changes = yes_prices.diff().abs()
            time_diffs = window_df['t_minutes'].diff()
            velocities = (price_changes / time_diffs).dropna()
            max_velocity = velocities.max() if len(velocities) > 0 else 0
            avg_velocity = velocities.mean() if len(velocities) > 0 else 0

            # Distance from fair value at t=5
            t5_row = window_df[window_df['t_minutes'] == 5.0]
            if len(t5_row) > 0:
                distance_from_fair = abs(t5_row['yes_price'].iloc[0] - 0.5)
            else:
                distance_from_fair = abs(yes_prices.mean() - 0.5)

            # Price range vs spread ratio
            price_range_vs_spread = yes_range / avg_spread if avg_spread > 0 else 0

            # Lock achievability estimate
            # A lock is achievable if price moves enough to build offsetting positions
            # Rule of thumb: need range > 3x spread to overcome crossing costs
            lock_achievable = price_range_vs_spread >= self.MIN_RANGE_SPREAD_RATIO

            # Estimate lock profit potential
            # If we can capture half the range minus costs
            if lock_achievable:
                captured_range = yes_range * 0.5  # Assume we capture half
                costs = self.LOCK_COST_ESTIMATE + avg_spread  # Spread + est trading costs
                lock_profit_potential = max(0, captured_range - costs)
            else:
                lock_profit_potential = 0

            windows.append(ArbitrageWindow(
                window_id=window_id,
                asset=asset,
                window_start=window_start,
                yes_open=float(yes_open),
                yes_close=float(yes_close),
                yes_high=float(yes_high),
                yes_low=float(yes_low),
                yes_range=float(yes_range),
                avg_spread=float(avg_spread),
                max_spread=float(max_spread),
                min_spread=float(min_spread),
                max_velocity=float(max_velocity),
                avg_velocity=float(avg_velocity),
                distance_from_fair=float(distance_from_fair),
                price_range_vs_spread=float(price_range_vs_spread),
                lock_achievable=lock_achievable,
                lock_profit_potential=float(lock_profit_potential),
            ))

        return windows

    def _window_to_dict(self, w: ArbitrageWindow) -> Dict:
        """Convert window to dict for DataFrame."""
        return {
            'window_id': w.window_id,
            'asset': w.asset,
            'window_start': w.window_start,
            'yes_open': w.yes_open,
            'yes_close': w.yes_close,
            'yes_high': w.yes_high,
            'yes_low': w.yes_low,
            'yes_range': w.yes_range,
            'avg_spread': w.avg_spread,
            'max_spread': w.max_spread,
            'min_spread': w.min_spread,
            'max_velocity': w.max_velocity,
            'avg_velocity': w.avg_velocity,
            'distance_from_fair': w.distance_from_fair,
            'price_range_vs_spread': w.price_range_vs_spread,
            'lock_achievable': w.lock_achievable,
            'lock_profit_potential': w.lock_profit_potential,
        }

    def _generate_recommendations(
        self,
        avg_range: float,
        avg_spread: float,
        pct_lockable: float,
        by_asset: Dict,
        by_hour: Dict,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        # Overall viability
        if pct_lockable > 50:
            recs.append(f"Lock strategy viable: {pct_lockable:.0f}% of windows are lockable")
        elif pct_lockable > 25:
            recs.append(f"Lock strategy marginal: {pct_lockable:.0f}% lockable - be selective")
        else:
            recs.append(f"Lock strategy difficult: only {pct_lockable:.0f}% lockable")

        # Best asset
        if by_asset:
            best_asset = max(by_asset.items(), key=lambda x: x[1].get('pct_lockable', 0))
            worst_asset = min(by_asset.items(), key=lambda x: x[1].get('pct_lockable', 0))
            recs.append(f"Best asset: {best_asset[0]} ({best_asset[1]['pct_lockable']:.0f}% lockable)")
            if best_asset[1]['pct_lockable'] > worst_asset[1]['pct_lockable'] + 10:
                recs.append(f"Avoid: {worst_asset[0]} ({worst_asset[1]['pct_lockable']:.0f}% lockable)")

        # Best hours
        if by_hour:
            sorted_hours = sorted(by_hour.items(), key=lambda x: x[1].get('pct_lockable', 0), reverse=True)
            if len(sorted_hours) >= 3:
                best_hours = [h[0] for h in sorted_hours[:3]]
                recs.append(f"Best hours (UTC): {best_hours}")

        # Spread vs range
        range_spread_ratio = avg_range / avg_spread if avg_spread > 0 else 0
        if range_spread_ratio < 2:
            recs.append("Warning: low range/spread ratio - spreads eating profits")
        elif range_spread_ratio > 4:
            recs.append("Good range/spread ratio - favorable for locking")

        return recs

    def _empty_result(self, assets: List[str]) -> ArbitrageAnalysisResult:
        """Return empty result when no data."""
        return ArbitrageAnalysisResult(
            analysis_time=datetime.now(timezone.utc).isoformat(),
            total_windows=0,
            assets=assets,
            avg_yes_range=0,
            avg_spread=0,
            pct_lockable=0,
            avg_lock_profit=0,
            by_asset={},
            by_hour={},
            best_conditions={},
            worst_conditions={},
            recommendations=["Insufficient data for analysis"],
        )

    def format_report(self, result: ArbitrageAnalysisResult) -> str:
        """Format analysis as readable report."""
        lines = []

        lines.append("=" * 70)
        lines.append("ARBITRAGE / LOCK OPPORTUNITY ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Analysis Time: {result.analysis_time}")
        lines.append(f"Total Windows: {result.total_windows}")
        lines.append(f"Assets: {', '.join(result.assets)}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"  Avg YES price range: {result.avg_yes_range*100:.2f}%")
        lines.append(f"  Avg spread: {result.avg_spread*100:.2f}%")
        lines.append(f"  Range/Spread ratio: {result.avg_yes_range/result.avg_spread:.1f}x" if result.avg_spread > 0 else "  Range/Spread ratio: N/A")
        lines.append(f"  Windows lockable: {result.pct_lockable:.1f}%")
        lines.append(f"  Avg lock profit: {result.avg_lock_profit*100:.2f}% (when achievable)")
        lines.append("")

        lines.append("-" * 70)
        lines.append("BY ASSET")
        lines.append("-" * 70)
        for asset, stats in result.by_asset.items():
            lines.append(f"  {asset}:")
            lines.append(f"    Windows: {stats['n_windows']}, Range: {stats['avg_range']*100:.2f}%, Lockable: {stats['pct_lockable']:.0f}%")
        lines.append("")

        if result.by_hour:
            lines.append("-" * 70)
            lines.append("BY HOUR (UTC) - Top 5")
            lines.append("-" * 70)
            sorted_hours = sorted(result.by_hour.items(), key=lambda x: x[1]['pct_lockable'], reverse=True)[:5]
            for hour, stats in sorted_hours:
                lines.append(f"  {hour:02d}:00 - {stats['n_windows']} windows, {stats['pct_lockable']:.0f}% lockable")
            lines.append("")

        lines.append("-" * 70)
        lines.append("LOCKABLE vs NON-LOCKABLE CONDITIONS")
        lines.append("-" * 70)
        lines.append(f"  Lockable windows:")
        lines.append(f"    Avg range: {result.best_conditions.get('avg_range', 0)*100:.2f}%")
        lines.append(f"    Avg spread: {result.best_conditions.get('avg_spread', 0)*100:.2f}%")
        lines.append(f"    Avg distance from 0.50: {result.best_conditions.get('avg_distance_from_fair', 0)*100:.1f}%")
        lines.append(f"  Non-lockable windows:")
        lines.append(f"    Avg range: {result.worst_conditions.get('avg_range', 0)*100:.2f}%")
        lines.append(f"    Avg spread: {result.worst_conditions.get('avg_spread', 0)*100:.2f}%")
        lines.append("")

        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for rec in result.recommendations:
            lines.append(f"  • {rec}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


async def run_arbitrage_analysis(data_dir: Path = Path("data")) -> str:
    """Run arbitrage analysis and return formatted report."""
    from ..db.database import Database

    assets = ['BTC', 'ETH', 'SOL', 'XRP']
    databases = {}

    for asset in assets:
        db_path = data_dir / f"{asset.lower()}.db"
        if db_path.exists():
            db = Database(db_path)
            await db.connect()
            databases[asset] = db

    try:
        analyzer = ArbitrageAnalyzer(databases)
        result = await analyzer.run_analysis()
        return analyzer.format_report(result)
    finally:
        for db in databases.values():
            await db.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        report = await run_arbitrage_analysis()
        print(report)

    asyncio.run(main())
