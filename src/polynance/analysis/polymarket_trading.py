"""Polymarket trading viability analysis.

Analyzes whether trading Polymarket 15-min crypto markets directly is profitable.
Key questions:
1. Is there alpha? (win rate vs odds)
2. Can we execute? (spreads, liquidity)
3. What's the expected return after fees?

Polymarket mechanics:
- Buy YES at price P: win (1-P) if YES, lose P if NO
- Buy NO at price (1-P): win P if NO, lose (1-P) if YES
- ~2% fee on profits only
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
from ..db.models import Window

logger = logging.getLogger(__name__)


@dataclass
class TradeScenario:
    """Results for a specific trading scenario."""
    name: str
    description: str
    n_trades: int
    win_rate: float
    avg_entry_price: float
    avg_spread: float
    gross_ev_pct: float
    net_ev_pct: float  # after fees
    net_ev_after_spread_pct: float  # after fees and spread
    confidence_interval_95: Tuple[float, float]


@dataclass
class PolymarketAnalysisResult:
    """Complete Polymarket trading analysis."""
    analysis_time: str
    total_windows: int
    assets_analyzed: List[str]

    # Data quality
    spread_stats: Dict

    # Trading scenarios
    scenarios: List[TradeScenario]

    # Timing analysis
    best_entry_time: str
    timing_details: List[Dict]

    # Recommendations
    is_viable: bool
    min_recommended_bankroll: float
    expected_monthly_return_pct: float
    key_risks: List[str]


class PolymarketTradingAnalyzer:
    """Analyzer for Polymarket direct trading viability."""

    POLYMARKET_FEE = 0.02  # 2% on profits
    MIN_SAMPLE_SIZE = 100  # Minimum windows for reliable analysis
    CONFIDENCE_LEVEL = 0.95

    def __init__(self, databases: Dict[str, Database]):
        self.databases = databases

    async def run_analysis(self) -> PolymarketAnalysisResult:
        """Run complete Polymarket trading viability analysis."""

        # Load all data
        all_windows = []
        all_samples = []
        assets = []

        for asset, db in self.databases.items():
            windows = await db.get_all_windows(asset, resolved_only=True)
            all_windows.extend(windows)
            assets.append(asset)

            # Get samples for spread analysis
            samples = await self._get_samples(db, asset)
            all_samples.extend(samples)

        if len(all_windows) < self.MIN_SAMPLE_SIZE:
            logger.warning(f"Insufficient data: {len(all_windows)} windows (need {self.MIN_SAMPLE_SIZE})")
            return self._insufficient_data_result(len(all_windows), assets)

        # Convert to DataFrames
        windows_df = self._windows_to_df(all_windows)
        samples_df = pd.DataFrame(all_samples) if all_samples else pd.DataFrame()

        # Run analyses
        spread_stats = self._analyze_spreads(samples_df)
        scenarios = self._analyze_trading_scenarios(windows_df, spread_stats)
        timing = self._analyze_timing(windows_df)

        # Generate recommendations
        viable, bankroll, monthly_return, risks = self._generate_recommendations(
            scenarios, spread_stats, len(all_windows)
        )

        return PolymarketAnalysisResult(
            analysis_time=datetime.now(timezone.utc).isoformat(),
            total_windows=len(all_windows),
            assets_analyzed=assets,
            spread_stats=spread_stats,
            scenarios=scenarios,
            best_entry_time=timing['best_time'],
            timing_details=timing['by_time'],
            is_viable=viable,
            min_recommended_bankroll=bankroll,
            expected_monthly_return_pct=monthly_return,
            key_risks=risks,
        )

    async def _get_samples(self, db: Database, asset: str) -> List[Dict]:
        """Get sample data for spread analysis."""
        # Query samples with spread data
        sql = """
        SELECT asset, t_minutes, pm_yes_price, pm_yes_bid, pm_yes_ask, pm_spread
        FROM samples
        WHERE asset = ? AND pm_spread IS NOT NULL
        """
        cursor = await db.conn.execute(sql, (asset,))
        rows = await cursor.fetchall()

        return [
            {
                'asset': row[0],
                't_minutes': row[1],
                'pm_yes_price': row[2],
                'pm_yes_bid': row[3],
                'pm_yes_ask': row[4],
                'pm_spread': row[5],
            }
            for row in rows
        ]

    def _windows_to_df(self, windows: List[Window]) -> pd.DataFrame:
        """Convert windows to DataFrame."""
        data = []
        for w in windows:
            data.append({
                'asset': w.asset,
                'window_id': w.window_id,
                'outcome': w.outcome,
                'outcome_binary': w.outcome_binary,
                'spot_change_bps': w.spot_change_bps,
                'spot_range_bps': w.spot_range_bps,
                'pm_yes_t0': w.pm_yes_t0,
                'pm_yes_t2_5': w.pm_yes_t2_5,
                'pm_yes_t5': w.pm_yes_t5,
                'pm_yes_t7_5': w.pm_yes_t7_5,
                'pm_yes_t10': w.pm_yes_t10,
                'pm_yes_t12_5': w.pm_yes_t12_5,
                'pm_spread_t0': w.pm_spread_t0,
                'pm_spread_t5': w.pm_spread_t5,
            })
        return pd.DataFrame(data)

    def _analyze_spreads(self, samples_df: pd.DataFrame) -> Dict:
        """Analyze spread/liquidity data."""
        if samples_df.empty or 'pm_spread' not in samples_df.columns:
            return {
                'mean_spread': None,
                'median_spread': None,
                'p75_spread': None,
                'p95_spread': None,
                'data_points': 0,
            }

        spreads = samples_df['pm_spread'].dropna()

        return {
            'mean_spread': float(spreads.mean()),
            'median_spread': float(spreads.median()),
            'p75_spread': float(spreads.quantile(0.75)),
            'p95_spread': float(spreads.quantile(0.95)),
            'max_spread': float(spreads.max()),
            'data_points': len(spreads),
            'spread_as_pct': float(spreads.mean() * 100),  # Convert to percentage
        }

    def _analyze_trading_scenarios(
        self, df: pd.DataFrame, spread_stats: Dict
    ) -> List[TradeScenario]:
        """Analyze different trading scenarios."""
        scenarios = []

        avg_spread = spread_stats.get('mean_spread', 0.01) or 0.01

        # Scenario definitions: (name, description, filter_func, is_bullish)
        scenario_defs = [
            (
                "strong_bull_t5",
                "Buy YES when pm_yes_t5 >= 0.70",
                lambda d: d['pm_yes_t5'] >= 0.70,
                True,
            ),
            (
                "bull_t5",
                "Buy YES when pm_yes_t5 >= 0.60",
                lambda d: d['pm_yes_t5'] >= 0.60,
                True,
            ),
            (
                "strong_bear_t5",
                "Buy NO when pm_yes_t5 < 0.30",
                lambda d: d['pm_yes_t5'] < 0.30,
                False,
            ),
            (
                "bear_t5",
                "Buy NO when pm_yes_t5 < 0.40",
                lambda d: d['pm_yes_t5'] < 0.40,
                False,
            ),
            (
                "extreme_bull_t5",
                "Buy YES when pm_yes_t5 >= 0.80",
                lambda d: d['pm_yes_t5'] >= 0.80,
                True,
            ),
            (
                "extreme_bear_t5",
                "Buy NO when pm_yes_t5 < 0.20",
                lambda d: d['pm_yes_t5'] < 0.20,
                False,
            ),
            # Later timing scenarios
            (
                "strong_bull_t7_5",
                "Buy YES when pm_yes_t7_5 >= 0.70",
                lambda d: d['pm_yes_t7_5'] >= 0.70,
                True,
            ),
            (
                "strong_bear_t7_5",
                "Buy NO when pm_yes_t7_5 < 0.30",
                lambda d: d['pm_yes_t7_5'] < 0.30,
                False,
            ),
        ]

        for name, desc, filter_func, is_bullish in scenario_defs:
            try:
                scenario = self._compute_scenario(
                    df, name, desc, filter_func, is_bullish, avg_spread
                )
                if scenario:
                    scenarios.append(scenario)
            except Exception as e:
                logger.warning(f"Error computing scenario {name}: {e}")

        # Sort by net EV
        scenarios.sort(key=lambda s: s.net_ev_after_spread_pct, reverse=True)

        return scenarios

    def _compute_scenario(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        filter_func,
        is_bullish: bool,
        avg_spread: float,
    ) -> Optional[TradeScenario]:
        """Compute metrics for a trading scenario."""

        subset = df[filter_func(df)].copy()

        if len(subset) < 5:
            return None

        # Determine win condition
        if is_bullish:
            # Buying YES - win if outcome is UP
            wins = subset['outcome'] == 'up'
            price_col = 'pm_yes_t5' if 't5' in name else 'pm_yes_t7_5'
        else:
            # Buying NO - win if outcome is DOWN
            wins = subset['outcome'] == 'down'
            price_col = 'pm_yes_t5' if 't5' in name else 'pm_yes_t7_5'

        n_trades = len(subset)
        win_rate = wins.mean()

        # Average entry price
        if is_bullish:
            avg_entry = subset[price_col].mean()  # Buying YES at this price
            profit_if_win = 1 - avg_entry
            loss_if_lose = avg_entry
        else:
            avg_yes = subset[price_col].mean()
            avg_entry = 1 - avg_yes  # Buying NO at (1 - yes_price)
            profit_if_win = avg_yes  # If NO wins, we get (1 - entry) = yes_price
            loss_if_lose = 1 - avg_yes

        # Gross EV (before fees)
        gross_ev = win_rate * profit_if_win - (1 - win_rate) * loss_if_lose

        # Net EV (after 2% fee on profits)
        fee_drag = self.POLYMARKET_FEE * win_rate * profit_if_win
        net_ev = gross_ev - fee_drag

        # Net EV after spread (half spread on entry)
        spread_cost = avg_spread / 2  # Pay half spread to cross
        net_ev_after_spread = net_ev - spread_cost

        # Confidence interval using Wilson score
        ci_low, ci_high = self._wilson_ci(int(wins.sum()), n_trades)

        # Convert CI to EV range
        ev_low = ci_low * profit_if_win - (1 - ci_low) * loss_if_lose - fee_drag - spread_cost
        ev_high = ci_high * profit_if_win - (1 - ci_high) * loss_if_lose - fee_drag - spread_cost

        return TradeScenario(
            name=name,
            description=description,
            n_trades=n_trades,
            win_rate=float(win_rate),
            avg_entry_price=float(avg_entry),
            avg_spread=float(avg_spread),
            gross_ev_pct=float(gross_ev * 100),
            net_ev_pct=float(net_ev * 100),
            net_ev_after_spread_pct=float(net_ev_after_spread * 100),
            confidence_interval_95=(float(ev_low * 100), float(ev_high * 100)),
        )

    def _wilson_ci(self, wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Wilson score confidence interval for proportions."""
        if n == 0:
            return (0.0, 1.0)

        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = wins / n

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def _analyze_timing(self, df: pd.DataFrame) -> Dict:
        """Analyze optimal entry timing."""
        time_cols = [
            ('pm_yes_t0', 't=0', 15.0),
            ('pm_yes_t2_5', 't=2.5', 12.5),
            ('pm_yes_t5', 't=5', 10.0),
            ('pm_yes_t7_5', 't=7.5', 7.5),
            ('pm_yes_t10', 't=10', 5.0),
            ('pm_yes_t12_5', 't=12.5', 2.5),
        ]

        results = []

        for col, label, time_remaining in time_cols:
            if col not in df.columns:
                continue

            valid = df[[col, 'outcome_binary', 'outcome']].dropna()

            if len(valid) < 10:
                continue

            # Correlation with outcome
            corr, pval = stats.pearsonr(valid[col], valid['outcome_binary'])

            # Strong signal accuracy (>=0.70 or <0.30)
            strong_bull = valid[valid[col] >= 0.70]
            strong_bear = valid[valid[col] < 0.30]

            bull_acc = (strong_bull['outcome'] == 'up').mean() if len(strong_bull) >= 5 else None
            bear_acc = (strong_bear['outcome'] == 'down').mean() if len(strong_bear) >= 5 else None

            results.append({
                'time': label,
                'column': col,
                'time_remaining_min': time_remaining,
                'correlation': float(corr),
                'p_value': float(pval),
                'strong_bull_n': len(strong_bull),
                'strong_bull_accuracy': float(bull_acc) if bull_acc else None,
                'strong_bear_n': len(strong_bear),
                'strong_bear_accuracy': float(bear_acc) if bear_acc else None,
            })

        # Find best time (balance of accuracy and time remaining)
        # Score = accuracy * sqrt(time_remaining) - want high accuracy with enough time
        best = None
        best_score = -1

        for r in results:
            bull_acc = r.get('strong_bull_accuracy')
            bear_acc = r.get('strong_bear_accuracy')

            if bull_acc and bear_acc:
                avg_acc = (bull_acc + bear_acc) / 2
                time_factor = np.sqrt(r['time_remaining_min'])
                score = avg_acc * time_factor

                if score > best_score:
                    best_score = score
                    best = r['time']

        return {
            'by_time': results,
            'best_time': best or 't=5',
        }

    def _generate_recommendations(
        self,
        scenarios: List[TradeScenario],
        spread_stats: Dict,
        n_windows: int,
    ) -> Tuple[bool, float, float, List[str]]:
        """Generate trading recommendations."""

        risks = []

        # Check data sufficiency
        if n_windows < self.MIN_SAMPLE_SIZE:
            risks.append(f"Insufficient data ({n_windows} windows, need {self.MIN_SAMPLE_SIZE}+)")
        elif n_windows < 500:
            risks.append(f"Limited data ({n_windows} windows) - results may not be stable")

        # Check spreads
        mean_spread = spread_stats.get('mean_spread', 0.01)
        if mean_spread and mean_spread > 0.03:
            risks.append(f"High spreads ({mean_spread*100:.1f}%) reduce profitability")

        # Find best scenario
        profitable_scenarios = [s for s in scenarios if s.net_ev_after_spread_pct > 0]

        if not profitable_scenarios:
            risks.append("No scenarios show positive expected value after costs")
            return False, 0, 0, risks

        best = profitable_scenarios[0]

        # Check confidence interval
        ci_low, ci_high = best.confidence_interval_95
        if ci_low < 0:
            risks.append(f"95% CI includes negative EV ({ci_low:.1f}% to {ci_high:.1f}%)")

        # Check sample size for best scenario
        if best.n_trades < 30:
            risks.append(f"Best scenario has few trades ({best.n_trades}) - need more data")

        # Estimate monthly return
        # Assume ~4 windows per hour * 24 hours * 30 days = ~2880 potential windows
        # But only some will trigger our signal
        signal_frequency = best.n_trades / n_windows
        monthly_windows = 2880 * 4  # 4 assets
        monthly_trades = monthly_windows * signal_frequency
        monthly_return = monthly_trades * (best.net_ev_after_spread_pct / 100)

        # Bankroll recommendation (Kelly-ish, conservative)
        # With win_rate and payoff, recommend enough to survive drawdowns
        min_bankroll = 1000  # Minimum practical amount

        # Is it viable?
        viable = (
            len(profitable_scenarios) > 0 and
            best.net_ev_after_spread_pct > 1.0 and  # At least 1% EV per trade
            best.n_trades >= 20 and
            ci_low > -2.0  # CI doesn't go too negative
        )

        return viable, min_bankroll, monthly_return, risks

    def _insufficient_data_result(
        self, n_windows: int, assets: List[str]
    ) -> PolymarketAnalysisResult:
        """Return result when insufficient data."""
        return PolymarketAnalysisResult(
            analysis_time=datetime.now(timezone.utc).isoformat(),
            total_windows=n_windows,
            assets_analyzed=assets,
            spread_stats={},
            scenarios=[],
            best_entry_time='unknown',
            timing_details=[],
            is_viable=False,
            min_recommended_bankroll=0,
            expected_monthly_return_pct=0,
            key_risks=[f"Insufficient data: {n_windows} windows (need {self.MIN_SAMPLE_SIZE}+)"],
        )

    def format_report(self, result: PolymarketAnalysisResult) -> str:
        """Format analysis result as readable report."""
        lines = []

        lines.append("=" * 70)
        lines.append("POLYMARKET TRADING VIABILITY ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Analysis Time: {result.analysis_time}")
        lines.append(f"Total Windows: {result.total_windows}")
        lines.append(f"Assets: {', '.join(result.assets_analyzed)}")
        lines.append("")

        # Spread analysis
        lines.append("-" * 70)
        lines.append("SPREAD / LIQUIDITY ANALYSIS")
        lines.append("-" * 70)
        if result.spread_stats:
            ss = result.spread_stats
            lines.append(f"  Mean spread:   {ss.get('mean_spread', 0)*100:.2f}%")
            lines.append(f"  Median spread: {ss.get('median_spread', 0)*100:.2f}%")
            lines.append(f"  95th pct:      {ss.get('p95_spread', 0)*100:.2f}%")
            lines.append(f"  Data points:   {ss.get('data_points', 0)}")
        else:
            lines.append("  No spread data available")
        lines.append("")

        # Trading scenarios
        lines.append("-" * 70)
        lines.append("TRADING SCENARIOS (sorted by Net EV)")
        lines.append("-" * 70)
        lines.append(f"{'Scenario':<22} {'N':>6} {'Win%':>7} {'Gross':>8} {'Net':>8} {'95% CI':>16}")
        lines.append("-" * 70)

        for s in result.scenarios:
            ci_str = f"({s.confidence_interval_95[0]:+.1f}, {s.confidence_interval_95[1]:+.1f})"
            lines.append(
                f"{s.name:<22} {s.n_trades:>6} {s.win_rate*100:>6.1f}% "
                f"{s.gross_ev_pct:>+7.2f}% {s.net_ev_after_spread_pct:>+7.2f}% {ci_str:>16}"
            )
        lines.append("")

        # Timing
        lines.append("-" * 70)
        lines.append("TIMING ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"Best entry time: {result.best_entry_time}")
        lines.append("")
        lines.append(f"{'Time':<8} {'Corr':>8} {'Bull N':>8} {'Bull Acc':>10} {'Bear N':>8} {'Bear Acc':>10}")
        lines.append("-" * 60)

        for t in result.timing_details:
            bull_acc = f"{t['strong_bull_accuracy']*100:.1f}%" if t.get('strong_bull_accuracy') else "N/A"
            bear_acc = f"{t['strong_bear_accuracy']*100:.1f}%" if t.get('strong_bear_accuracy') else "N/A"
            lines.append(
                f"{t['time']:<8} {t['correlation']:>8.3f} {t['strong_bull_n']:>8} {bull_acc:>10} "
                f"{t['strong_bear_n']:>8} {bear_acc:>10}"
            )
        lines.append("")

        # Recommendations
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)

        if result.is_viable:
            lines.append("  STATUS: ✓ POTENTIALLY VIABLE")
        else:
            lines.append("  STATUS: ✗ NOT YET VIABLE (need more data or edge not confirmed)")

        lines.append(f"  Expected monthly return: {result.expected_monthly_return_pct:.1f}%")
        lines.append(f"  Minimum bankroll: ${result.min_recommended_bankroll:.0f}")
        lines.append("")

        lines.append("  RISKS:")
        for risk in result.key_risks:
            lines.append(f"    - {risk}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


async def run_polymarket_analysis(data_dir: Path = Path("data")) -> str:
    """Run Polymarket trading analysis and return formatted report."""
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
        analyzer = PolymarketTradingAnalyzer(databases)
        result = await analyzer.run_analysis()
        return analyzer.format_report(result)
    finally:
        for db in databases.values():
            await db.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        report = await run_polymarket_analysis()
        print(report)

    asyncio.run(main())
