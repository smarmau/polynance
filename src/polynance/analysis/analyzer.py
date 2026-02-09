"""Analysis module for polynance - correlations, signals, and calibration."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from ..db.database import Database
from ..db.models import Window

logger = logging.getLogger(__name__)


@dataclass
class SignalBucket:
    """Analysis results for a signal bucket."""

    bucket_name: str
    lower: float
    upper: float
    n: int
    win_rate: float
    avg_move_on_win_bps: float
    avg_move_on_loss_bps: float
    median_move_on_win_bps: float
    std_move_bps: float
    expected_value_bps: float


@dataclass
class CalibrationPoint:
    """Calibration data for a predicted probability bin."""

    predicted_prob_lower: float
    predicted_prob_upper: float
    predicted_prob_mid: float
    actual_frequency: float
    n: int
    calibration_error: float  # predicted - actual


class Analyzer:
    """Analysis engine for Polymarket prediction data."""

    def __init__(self, db: Database, reports_dir: Optional[Path] = None):
        self.db = db
        self.reports_dir = reports_dir or Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def run_full_analysis(self, asset: str, min_windows: int = 10) -> dict:
        """Run full analysis suite for an asset.

        Args:
            asset: Asset symbol
            min_windows: Minimum windows required for analysis

        Returns:
            Dict with all analysis results
        """
        windows = await self.db.get_all_windows(asset, resolved_only=True)

        if len(windows) < min_windows:
            logger.warning(f"[{asset}] Not enough data for analysis ({len(windows)} windows)")
            return {"status": "insufficient_data", "window_count": len(windows)}

        # Convert to DataFrame for analysis
        df = self._windows_to_dataframe(windows)

        results = {
            "asset": asset,
            "analysis_time": datetime.now(timezone.utc).isoformat(),
            "window_count": len(windows),
        }

        # Run each analysis
        try:
            results["correlations"] = self._compute_correlations(df)
        except Exception as e:
            logger.error(f"Error computing correlations: {e}")
            results["correlations"] = {}

        try:
            results["signal_buckets"] = self._compute_signal_buckets(df)
        except Exception as e:
            logger.error(f"Error computing signal buckets: {e}")
            results["signal_buckets"] = []

        try:
            results["calibration"] = self._compute_calibration(df)
        except Exception as e:
            logger.error(f"Error computing calibration: {e}")
            results["calibration"] = {}

        try:
            results["time_decay"] = self._compute_time_decay(df)
        except Exception as e:
            logger.error(f"Error computing time decay: {e}")
            results["time_decay"] = {}

        try:
            results["magnitude_analysis"] = self._compute_magnitude_analysis(df)
        except Exception as e:
            logger.error(f"Error computing magnitude analysis: {e}")
            results["magnitude_analysis"] = {}

        # Store in database
        await self._store_analysis_results(asset, results)

        return results

    def _windows_to_dataframe(self, windows: List[Window]) -> pd.DataFrame:
        """Convert windows to a pandas DataFrame."""
        data = []
        for w in windows:
            data.append({
                "window_id": w.window_id,
                "window_start": w.window_start_utc,
                "outcome": w.outcome,
                "outcome_binary": w.outcome_binary,
                "spot_open": w.spot_open,
                "spot_close": w.spot_close,
                "spot_change_pct": w.spot_change_pct,
                "spot_change_bps": w.spot_change_bps,
                "spot_high": w.spot_high,
                "spot_low": w.spot_low,
                "spot_range_bps": w.spot_range_bps,
                "pm_yes_t0": w.pm_yes_t0,
                "pm_yes_t2_5": w.pm_yes_t2_5,
                "pm_yes_t5": w.pm_yes_t5,
                "pm_yes_t7_5": w.pm_yes_t7_5,
                "pm_yes_t10": w.pm_yes_t10,
                "pm_yes_t12_5": w.pm_yes_t12_5,
                "pm_spread_t0": w.pm_spread_t0,
                "pm_spread_t5": w.pm_spread_t5,
                "pm_momentum_0_5": w.pm_price_momentum_0_to_5,
                "pm_momentum_5_10": w.pm_price_momentum_5_to_10,
            })

        df = pd.DataFrame(data)

        # Add derived columns
        df["signal_strength_t5"] = (df["pm_yes_t5"] - 0.5).abs()
        df["signal_strength_t10"] = (df["pm_yes_t10"] - 0.5).abs()
        df["abs_move_bps"] = df["spot_change_bps"].abs()

        return df

    def _compute_correlations(self, df: pd.DataFrame) -> dict:
        """Compute correlation matrix for key variables."""
        # Columns for correlation
        cols = [
            "pm_yes_t0", "pm_yes_t5", "pm_yes_t7_5", "pm_yes_t10",
            "pm_momentum_0_5", "pm_momentum_5_10",
            "pm_spread_t0", "pm_spread_t5",
            "signal_strength_t5",
            "outcome_binary", "spot_change_bps", "abs_move_bps"
        ]

        # Filter to existing columns with data
        available_cols = [c for c in cols if c in df.columns and df[c].notna().sum() >= 5]

        if len(available_cols) < 2:
            return {}

        # Compute correlation matrix
        corr_matrix = df[available_cols].corr()

        # Extract key correlations
        result = {
            "matrix": corr_matrix.to_dict(),
        }

        # Key correlations we care about
        key_pairs = [
            ("pm_yes_t5", "outcome_binary", "yes_t5_vs_outcome"),
            ("pm_yes_t10", "outcome_binary", "yes_t10_vs_outcome"),
            ("pm_momentum_0_5", "outcome_binary", "momentum_0_5_vs_outcome"),
            ("pm_momentum_5_10", "outcome_binary", "momentum_5_10_vs_outcome"),
            ("signal_strength_t5", "abs_move_bps", "signal_strength_vs_magnitude"),
        ]

        for col1, col2, name in key_pairs:
            if col1 in available_cols and col2 in available_cols:
                valid = df[[col1, col2]].dropna()
                if len(valid) >= 5:
                    corr, pval = stats.pearsonr(valid[col1], valid[col2])
                    result[name] = {
                        "correlation": float(corr),
                        "p_value": float(pval),
                        "n": len(valid),
                        "significant": pval < 0.05,
                    }

        return result

    def _compute_signal_buckets(self, df: pd.DataFrame) -> List[dict]:
        """Compute signal bucket analysis."""
        buckets = []

        # Bullish buckets (yes price > 0.5)
        bullish_ranges = [
            (0.50, 0.55, "yes_0.50-0.55"),
            (0.55, 0.60, "yes_0.55-0.60"),
            (0.60, 0.65, "yes_0.60-0.65"),
            (0.65, 0.70, "yes_0.65-0.70"),
            (0.70, 1.00, "yes_0.70+"),
        ]

        # Bearish buckets (yes price < 0.5)
        bearish_ranges = [
            (0.45, 0.50, "yes_0.45-0.50"),
            (0.40, 0.45, "yes_0.40-0.45"),
            (0.35, 0.40, "yes_0.35-0.40"),
            (0.30, 0.35, "yes_0.30-0.35"),
            (0.00, 0.30, "yes_<0.30"),
        ]

        # Analyze at t=5 (primary signal)
        if "pm_yes_t5" not in df.columns:
            return buckets

        for lower, upper, name in bullish_ranges + bearish_ranges:
            mask = (df["pm_yes_t5"] >= lower) & (df["pm_yes_t5"] < upper)
            subset = df[mask]

            if len(subset) < 3:
                continue

            # Determine expected direction based on bucket
            expected_up = lower >= 0.5  # Bullish if yes price >= 0.5

            # Win rate (market predicts correctly)
            if expected_up:
                wins = subset[subset["outcome"] == "up"]
                losses = subset[subset["outcome"] == "down"]
            else:
                wins = subset[subset["outcome"] == "down"]
                losses = subset[subset["outcome"] == "up"]

            n = len(subset)
            win_rate = len(wins) / n if n > 0 else 0

            # Move magnitudes (in bps, absolute values)
            avg_win = wins["abs_move_bps"].mean() if len(wins) > 0 else 0
            avg_loss = losses["abs_move_bps"].mean() if len(losses) > 0 else 0
            median_win = wins["abs_move_bps"].median() if len(wins) > 0 else 0
            std_move = subset["abs_move_bps"].std() if n > 1 else 0

            # Expected value (in bps)
            ev = win_rate * avg_win - (1 - win_rate) * avg_loss

            bucket_result = {
                "name": name,
                "lower": lower,
                "upper": upper,
                "n": n,
                "win_rate": float(win_rate),
                "avg_move_on_win_bps": float(avg_win),
                "avg_move_on_loss_bps": float(avg_loss),
                "median_move_on_win_bps": float(median_win),
                "std_move_bps": float(std_move),
                "expected_value_bps": float(ev),
            }
            buckets.append(bucket_result)

        return buckets

    def _compute_calibration(self, df: pd.DataFrame) -> dict:
        """Compute calibration analysis - is Polymarket well-calibrated?"""
        if "pm_yes_t5" not in df.columns or "outcome_binary" not in df.columns:
            return {}

        # Bin by predicted probability
        bins = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]

        df["prob_bin"] = pd.cut(df["pm_yes_t5"], bins=bins, labels=labels, include_lowest=True)

        calibration_points = []
        total_calibration_error = 0
        total_n = 0

        for i, label in enumerate(labels):
            subset = df[df["prob_bin"] == label]
            n = len(subset)

            if n < 2:
                continue

            predicted_mid = (bins[i] + bins[i+1]) / 2
            actual_freq = subset["outcome_binary"].mean()
            error = predicted_mid - actual_freq

            calibration_points.append({
                "bin": label,
                "lower": bins[i],
                "upper": bins[i+1],
                "predicted_prob": float(predicted_mid),
                "actual_frequency": float(actual_freq),
                "n": int(n),
                "calibration_error": float(error),
            })

            total_calibration_error += abs(error) * n
            total_n += n

        mean_abs_error = total_calibration_error / total_n if total_n > 0 else None

        return {
            "points": calibration_points,
            "mean_absolute_error": float(mean_abs_error) if mean_abs_error else None,
            "interpretation": self._interpret_calibration(mean_abs_error),
        }

    def _interpret_calibration(self, mae: Optional[float]) -> str:
        """Interpret the calibration error."""
        if mae is None:
            return "Insufficient data"
        if mae < 0.05:
            return "Excellent calibration"
        if mae < 0.10:
            return "Good calibration"
        if mae < 0.15:
            return "Fair calibration"
        return "Poor calibration - possible systematic bias"

    def _compute_time_decay(self, df: pd.DataFrame) -> dict:
        """Analyze how signal quality changes through the window."""
        time_points = ["pm_yes_t0", "pm_yes_t2_5", "pm_yes_t5", "pm_yes_t7_5", "pm_yes_t10", "pm_yes_t12_5"]
        time_labels = ["t=0", "t=2.5", "t=5", "t=7.5", "t=10", "t=12.5"]

        results = []

        for col, label in zip(time_points, time_labels):
            if col not in df.columns:
                continue

            valid = df[[col, "outcome_binary"]].dropna()
            if len(valid) < 5:
                continue

            # Correlation with outcome
            corr, pval = stats.pearsonr(valid[col], valid["outcome_binary"])

            # Predictive accuracy (yes > 0.5 predicts up correctly)
            correct = ((valid[col] > 0.5) & (valid["outcome_binary"] == 1)) | \
                      ((valid[col] < 0.5) & (valid["outcome_binary"] == 0))
            accuracy = correct.sum() / len(valid)

            results.append({
                "time": label,
                "column": col,
                "correlation": float(corr),
                "p_value": float(pval),
                "accuracy": float(accuracy),
                "n": len(valid),
            })

        # Find optimal time point
        if results:
            best = max(results, key=lambda x: x["correlation"])
            worst = min(results, key=lambda x: x["correlation"])
        else:
            best = worst = None

        return {
            "by_time": results,
            "best_time": best["time"] if best else None,
            "worst_time": worst["time"] if worst else None,
            "interpretation": self._interpret_time_decay(results),
        }

    def _interpret_time_decay(self, results: list) -> str:
        """Interpret time decay analysis."""
        if not results:
            return "Insufficient data"

        # Check if later times have better correlation
        early = [r for r in results if r["time"] in ["t=0", "t=2.5", "t=5"]]
        late = [r for r in results if r["time"] in ["t=7.5", "t=10", "t=12.5"]]

        if not early or not late:
            return "Insufficient time points for analysis"

        early_avg = np.mean([r["correlation"] for r in early])
        late_avg = np.mean([r["correlation"] for r in late])

        if late_avg > early_avg + 0.1:
            return "Signal improves later in window - consider waiting"
        if early_avg > late_avg + 0.1:
            return "Signal degrades later in window - act early"
        return "Signal quality relatively stable through window"

    def _compute_magnitude_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze relationship between signal strength and move magnitude."""
        if "signal_strength_t5" not in df.columns or "abs_move_bps" not in df.columns:
            return {}

        valid = df[["signal_strength_t5", "abs_move_bps", "pm_yes_t5", "spot_change_bps"]].dropna()

        if len(valid) < 5:
            return {}

        # Correlation between signal strength and magnitude
        corr, pval = stats.pearsonr(valid["signal_strength_t5"], valid["abs_move_bps"])

        # Quartile analysis
        quartiles = []
        for q in [0.25, 0.5, 0.75, 1.0]:
            q_low = valid["signal_strength_t5"].quantile(q - 0.25)
            q_high = valid["signal_strength_t5"].quantile(q)
            subset = valid[(valid["signal_strength_t5"] >= q_low) &
                          (valid["signal_strength_t5"] <= q_high)]

            if len(subset) >= 2:
                quartiles.append({
                    "quartile": int(q * 4),
                    "signal_range": f"{q_low:.3f}-{q_high:.3f}",
                    "avg_move_bps": float(subset["abs_move_bps"].mean()),
                    "n": len(subset),
                })

        return {
            "correlation": float(corr),
            "p_value": float(pval),
            "significant": pval < 0.05,
            "by_quartile": quartiles,
            "interpretation": self._interpret_magnitude(corr, pval),
        }

    def _interpret_magnitude(self, corr: float, pval: float) -> str:
        """Interpret magnitude correlation."""
        if pval > 0.05:
            return "No significant relationship between signal strength and move size"
        if corr > 0.3:
            return "Strong positive relationship - higher conviction predicts larger moves"
        if corr > 0.15:
            return "Moderate positive relationship - some predictive value for move size"
        if corr > 0:
            return "Weak positive relationship"
        return "No clear relationship - signal strength does not predict magnitude"

    async def _store_analysis_results(self, asset: str, results: dict):
        """Store analysis results (deprecated — analysis_results table removed)."""
        logger.debug(f"Skipping analysis storage for {asset} (table dropped)")

    def format_summary(self, results: dict) -> str:
        """Format analysis results as a human-readable summary."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"ANALYSIS SUMMARY - {results.get('asset', 'Unknown')}")
        lines.append(f"Time: {results.get('analysis_time', 'N/A')}")
        lines.append(f"Windows analyzed: {results.get('window_count', 0)}")
        lines.append(f"{'='*60}\n")

        # Correlations
        if "correlations" in results and results["correlations"]:
            lines.append("KEY CORRELATIONS:")
            corr = results["correlations"]
            for key in ["yes_t5_vs_outcome", "yes_t10_vs_outcome", "signal_strength_vs_magnitude"]:
                if key in corr:
                    data = corr[key]
                    sig = "*" if data.get("significant") else ""
                    lines.append(f"  {key}: {data['correlation']:.3f} (p={data['p_value']:.3f}){sig}")
            lines.append("")

        # Signal buckets
        if "signal_buckets" in results and results["signal_buckets"]:
            lines.append("SIGNAL BUCKETS (at t=5):")
            lines.append(f"  {'Bucket':<15} {'N':>5} {'Win%':>7} {'Avg Win':>10} {'EV (bps)':>10}")
            lines.append(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*10} {'-'*10}")

            for bucket in results["signal_buckets"]:
                lines.append(
                    f"  {bucket['name']:<15} {bucket['n']:>5} "
                    f"{bucket['win_rate']*100:>6.1f}% {bucket['avg_move_on_win_bps']:>9.1f} "
                    f"{bucket['expected_value_bps']:>+9.1f}"
                )
            lines.append("")

        # Calibration
        if "calibration" in results and results["calibration"]:
            cal = results["calibration"]
            lines.append("CALIBRATION:")
            lines.append(f"  Mean Absolute Error: {cal.get('mean_absolute_error', 'N/A'):.3f}")
            lines.append(f"  Interpretation: {cal.get('interpretation', 'N/A')}")
            lines.append("")

        # Time decay
        if "time_decay" in results and results["time_decay"]:
            td = results["time_decay"]
            lines.append("TIME DECAY:")
            lines.append(f"  Best time: {td.get('best_time', 'N/A')}")
            lines.append(f"  Interpretation: {td.get('interpretation', 'N/A')}")
            lines.append("")

        # Magnitude
        if "magnitude_analysis" in results and results["magnitude_analysis"]:
            mag = results["magnitude_analysis"]
            lines.append("MAGNITUDE PREDICTION:")
            lines.append(f"  Correlation: {mag.get('correlation', 'N/A'):.3f}")
            lines.append(f"  Interpretation: {mag.get('interpretation', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    async def generate_report(self, asset: str, output_path: Optional[Path] = None) -> Path:
        """Generate a full markdown report for an asset."""
        results = await self.run_full_analysis(asset)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.reports_dir / f"{asset}_analysis_{timestamp}.md"

        report = self._generate_markdown_report(asset, results)

        output_path.write_text(report)
        logger.info(f"Report written to {output_path}")

        return output_path

    def _generate_markdown_report(self, asset: str, results: dict) -> str:
        """Generate markdown report content."""
        lines = []

        lines.append(f"# Polynance Analysis Report - {asset}")
        lines.append(f"\nGenerated: {results.get('analysis_time', 'N/A')}")
        lines.append(f"\nWindows Analyzed: {results.get('window_count', 0)}")

        # Correlations
        lines.append("\n## Correlation Analysis\n")
        if "correlations" in results and results["correlations"]:
            lines.append("| Metric | Correlation | P-Value | Significant |")
            lines.append("|--------|-------------|---------|-------------|")

            corr = results["correlations"]
            for key in ["yes_t5_vs_outcome", "yes_t10_vs_outcome",
                       "momentum_0_5_vs_outcome", "signal_strength_vs_magnitude"]:
                if key in corr:
                    data = corr[key]
                    sig = "Yes" if data.get("significant") else "No"
                    lines.append(f"| {key} | {data['correlation']:.3f} | {data['p_value']:.4f} | {sig} |")
        else:
            lines.append("*Insufficient data for correlation analysis*")

        # Signal Buckets
        lines.append("\n## Signal Bucket Analysis\n")
        if "signal_buckets" in results and results["signal_buckets"]:
            lines.append("| Bucket | N | Win Rate | Avg Win (bps) | Avg Loss (bps) | EV (bps) |")
            lines.append("|--------|---|----------|---------------|----------------|----------|")

            for bucket in results["signal_buckets"]:
                lines.append(
                    f"| {bucket['name']} | {bucket['n']} | {bucket['win_rate']*100:.1f}% | "
                    f"{bucket['avg_move_on_win_bps']:.1f} | {bucket['avg_move_on_loss_bps']:.1f} | "
                    f"{bucket['expected_value_bps']:+.1f} |"
                )
        else:
            lines.append("*Insufficient data for bucket analysis*")

        # Calibration
        lines.append("\n## Calibration Analysis\n")
        if "calibration" in results and results["calibration"]:
            cal = results["calibration"]
            lines.append(f"**Mean Absolute Error:** {cal.get('mean_absolute_error', 'N/A'):.4f}")
            lines.append(f"\n**Interpretation:** {cal.get('interpretation', 'N/A')}")

            if "points" in cal and cal["points"]:
                lines.append("\n| Predicted Prob | Actual Freq | N | Error |")
                lines.append("|----------------|-------------|---|-------|")
                for point in cal["points"]:
                    lines.append(
                        f"| {point['predicted_prob']:.2f} | {point['actual_frequency']:.3f} | "
                        f"{point['n']} | {point['calibration_error']:+.3f} |"
                    )
        else:
            lines.append("*Insufficient data for calibration analysis*")

        # Time Decay
        lines.append("\n## Time Decay Analysis\n")
        if "time_decay" in results and results["time_decay"]:
            td = results["time_decay"]
            lines.append(f"**Best Time Point:** {td.get('best_time', 'N/A')}")
            lines.append(f"\n**Interpretation:** {td.get('interpretation', 'N/A')}")

            if "by_time" in td and td["by_time"]:
                lines.append("\n| Time | Correlation | Accuracy | N |")
                lines.append("|------|-------------|----------|---|")
                for point in td["by_time"]:
                    lines.append(
                        f"| {point['time']} | {point['correlation']:.3f} | "
                        f"{point['accuracy']*100:.1f}% | {point['n']} |"
                    )
        else:
            lines.append("*Insufficient data for time decay analysis*")

        # Magnitude
        lines.append("\n## Magnitude Prediction\n")
        if "magnitude_analysis" in results and results["magnitude_analysis"]:
            mag = results["magnitude_analysis"]
            lines.append(f"**Signal-Magnitude Correlation:** {mag.get('correlation', 'N/A'):.3f}")
            lines.append(f"\n**Interpretation:** {mag.get('interpretation', 'N/A')}")
        else:
            lines.append("*Insufficient data for magnitude analysis*")

        return "\n".join(lines)
