"""Data models for polynance database."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Sample:
    """Sample data point within a 15-minute window."""

    window_id: str
    window_start_utc: datetime
    sample_time_utc: datetime
    t_minutes: float  # 0, 2.5, 5, 7.5, 10, 12.5, 15
    asset: str  # BTC, ETH, SOL, XRP

    # Polymarket data
    pm_yes_price: float
    pm_no_price: float
    pm_yes_bid: float
    pm_yes_ask: float
    pm_spread: float
    pm_midpoint: float

    # Spot price data
    spot_price: float
    spot_price_change_from_open: Optional[float] = None

    # Market metadata
    pm_market_id: Optional[str] = None
    pm_condition_id: Optional[str] = None

    # Primary key (set after insert)
    id: Optional[int] = None


@dataclass
class Window:
    """Summary of a completed 15-minute window."""

    window_id: str
    asset: str  # BTC, ETH, SOL, XRP
    window_start_utc: datetime
    window_end_utc: datetime

    # Outcome
    outcome: Optional[str] = None  # 'up' or 'down'
    outcome_binary: Optional[int] = None  # 1 = up, 0 = down

    # Spot price movement
    spot_open: Optional[float] = None
    spot_close: Optional[float] = None
    spot_change_pct: Optional[float] = None
    spot_change_bps: Optional[float] = None
    spot_high: Optional[float] = None
    spot_low: Optional[float] = None
    spot_range_bps: Optional[float] = None

    # Polymarket prices at key times
    pm_yes_t0: Optional[float] = None
    pm_yes_t2_5: Optional[float] = None
    pm_yes_t5: Optional[float] = None
    pm_yes_t7_5: Optional[float] = None
    pm_yes_t10: Optional[float] = None
    pm_yes_t12_5: Optional[float] = None
    pm_spread_t0: Optional[float] = None
    pm_spread_t5: Optional[float] = None

    # Derived signals
    pm_price_momentum_0_to_5: Optional[float] = None
    pm_price_momentum_5_to_10: Optional[float] = None

    # Resolution
    resolved_at_utc: Optional[datetime] = None
    resolution_source: Optional[str] = None


@dataclass
class AnalysisResult:
    """Results from a correlation/signal analysis."""

    analysis_time: datetime
    asset: str
    window_count: int

    # Correlation coefficients
    corr_yes_t5_vs_outcome: Optional[float] = None
    corr_yes_t10_vs_outcome: Optional[float] = None
    corr_momentum_vs_outcome: Optional[float] = None
    corr_signal_strength_vs_magnitude: Optional[float] = None

    # Signal performance
    accuracy_yes_gt_55: Optional[float] = None  # Win rate when yes > 0.55
    accuracy_yes_gt_60: Optional[float] = None  # Win rate when yes > 0.60
    accuracy_yes_lt_45: Optional[float] = None  # Win rate when yes < 0.45
    accuracy_yes_lt_40: Optional[float] = None  # Win rate when yes < 0.40

    # Expected values
    ev_yes_gt_55_bps: Optional[float] = None
    ev_yes_gt_60_bps: Optional[float] = None

    # Calibration
    calibration_error: Optional[float] = None  # Mean absolute calibration error

    # Raw data for detailed analysis
    raw_data: Optional[dict] = field(default=None)
