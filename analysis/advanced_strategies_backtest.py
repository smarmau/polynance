#!/usr/bin/env python3
"""
Advanced Strategies Backtest: Cross-Asset Consensus, Contrarian, & Combinations

New strategies discovered through data mining:

1. CROSS-ASSET CONSENSUS: Only trade when 3+ or 4/4 assets agree on direction at t5
   - When all 4 agree bull (pm_yes >= 0.65): 82-83% up rate per asset
   - When all 4 agree bear (pm_yes <= 0.35): 84-87% down rate per asset

2. CONTRARIAN / CROSS-WINDOW: After strong previous resolution, bet reversal
   - After strong up (prev pm@t12.5 >= 0.85), next window tends down (37-41% up)
   - After strong down (prev pm@t12.5 <= 0.15), next window tends up

3. PM VELOCITY: Speed of PM movement from t0→t2.5 as quality filter
   - Fast moves (15%+ in 2.5 min) show high directional accuracy

4. OVERSHOOT FROM NEUTRAL: PM starts neutral (0.40-0.60) → extreme by t5
   - Bull overshoot: 80-86% up   Bear overshoot: 83-87% down

5. COMBINED: Cross-asset + best momentum (t0→t12.5 or t5→t12.5)

All use early exit (sell position), not binary resolution.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import json

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "reports"
ASSETS = ["btc", "eth", "sol", "xrp"]

INITIAL_BANKROLL = 1000.0
BASE_BET = 50.0
FEE_RATE = 0.001
SPREAD_COST = 0.005


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data() -> pd.DataFrame:
    """Load and combine data from all asset databases."""
    all_data = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        query = """
            SELECT
                window_id, window_start_utc, outcome,
                pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5,
                pm_spread_t0, pm_spread_t5
            FROM windows
            WHERE outcome IS NOT NULL
              AND pm_yes_t0 IS NOT NULL
              AND pm_yes_t2_5 IS NOT NULL
              AND pm_yes_t5 IS NOT NULL
              AND pm_yes_t7_5 IS NOT NULL
              AND pm_yes_t10 IS NOT NULL
              AND pm_yes_t12_5 IS NOT NULL
            ORDER BY window_start_utc
        """
        df = pd.read_sql_query(query, conn)
        df['asset'] = asset.upper()
        df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
        all_data.append(df)
        conn.close()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('window_start_utc').reset_index(drop=True)
    return combined


def build_cross_asset_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a view where each row is a unique window_start_utc with all 4 asset PM values.
    This enables cross-asset consensus checks.
    """
    # Round timestamps to nearest 15 min (they should already be aligned)
    df['window_ts'] = df['window_start_utc'].dt.floor('15min')

    # Pivot to get per-asset columns
    pivoted = df.pivot_table(
        index='window_ts',
        columns='asset',
        values=['pm_yes_t0', 'pm_yes_t2_5', 'pm_yes_t5', 'pm_yes_t7_5', 'pm_yes_t10', 'pm_yes_t12_5', 'outcome'],
        aggfunc='first'
    )

    # Flatten column names
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()

    return pivoted


def build_window_sequence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous window's PM values for contrarian/cross-window strategies.
    Each asset gets its own lagged values.
    """
    result = df.copy()

    for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
        asset_mask = result['asset'] == asset
        asset_df = result[asset_mask].sort_values('window_start_utc').copy()

        # Previous window values
        asset_df[f'prev_pm_t12_5'] = asset_df['pm_yes_t12_5'].shift(1)
        asset_df[f'prev_pm_t5'] = asset_df['pm_yes_t5'].shift(1)
        asset_df[f'prev_outcome'] = asset_df['outcome'].shift(1)
        asset_df[f'prev_pm_t0'] = asset_df['pm_yes_t0'].shift(1)

        result.loc[asset_mask, 'prev_pm_t12_5'] = asset_df['prev_pm_t12_5'].values
        result.loc[asset_mask, 'prev_pm_t5'] = asset_df['prev_pm_t5'].values
        result.loc[asset_mask, 'prev_outcome'] = asset_df['prev_outcome'].values
        result.loc[asset_mask, 'prev_pm_t0'] = asset_df['prev_pm_t0'].values

    return result


# =============================================================================
# P&L CALCULATION
# =============================================================================

def pnl_early_exit(direction: str, entry_pm_yes: float, exit_pm_yes: float, bet_size: float) -> float:
    """P&L for selling position before resolution. Double fees."""
    if direction == 'bull':
        entry_contract = entry_pm_yes
        exit_contract = exit_pm_yes
    else:
        entry_contract = 1.0 - entry_pm_yes
        exit_contract = 1.0 - exit_pm_yes

    if entry_contract <= 0.001:
        return 0.0

    n_contracts = bet_size / entry_contract
    gross = n_contracts * (exit_contract - entry_contract)

    entry_fee = entry_contract * n_contracts * FEE_RATE
    exit_fee = exit_contract * n_contracts * FEE_RATE
    entry_spread = SPREAD_COST * bet_size
    exit_spread = SPREAD_COST * (n_contracts * exit_contract)

    return gross - entry_fee - exit_fee - entry_spread - exit_spread


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(name: str, pnls: List[float], extra_info: Dict = None) -> Dict:
    """Calculate comprehensive trading metrics from a list of P&Ls."""
    if not pnls:
        return {"name": name, "n_trades": 0}

    pnls = np.array(pnls)
    n = len(pnls)
    n_wins = int(np.sum(pnls > 0))
    n_losses = int(np.sum(pnls <= 0))

    win_pnls = pnls[pnls > 0]
    loss_pnls = pnls[pnls <= 0]

    total_pnl = float(np.sum(pnls))
    win_rate = n_wins / n
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if n > 1 else 0.0

    avg_win = float(np.mean(win_pnls)) if len(win_pnls) > 0 else 0.0
    avg_loss = float(np.mean(loss_pnls)) if len(loss_pnls) > 0 else 0.0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    gross_wins = float(np.sum(win_pnls)) if len(win_pnls) > 0 else 0.0
    gross_losses = abs(float(np.sum(loss_pnls))) if len(loss_pnls) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Equity curve
    equity = np.concatenate([[INITIAL_BANKROLL], INITIAL_BANKROLL + np.cumsum(pnls)])
    peak_eq = np.maximum.accumulate(equity)
    dd = equity - peak_eq
    dd_pct = (dd / peak_eq) * 100
    max_dd = float(np.min(dd))
    max_dd_pct = float(np.min(dd_pct))

    # Risk metrics
    periods_per_year = 96 * 365
    sharpe = (avg_pnl / std_pnl) * np.sqrt(periods_per_year) if std_pnl > 0 else 0.0

    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = (avg_pnl / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    total_return_pct = (equity[-1] - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    calmar = total_return_pct / abs(max_dd_pct) if max_dd_pct < 0 else 0.0
    recovery_factor = total_pnl / abs(max_dd) if max_dd < 0 else (float('inf') if total_pnl > 0 else 0.0)

    # Streaks
    max_cw, max_cl, cur_w, cur_l = 0, 0, 0, 0
    for p in pnls:
        if p > 0:
            cur_w += 1; cur_l = 0; max_cw = max(max_cw, cur_w)
        else:
            cur_l += 1; cur_w = 0; max_cl = max(max_cl, cur_l)

    result = {
        'name': name,
        'n_trades': n,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_bankroll': float(equity[-1]),
        'total_return_pct': total_return_pct,
        'avg_pnl': avg_pnl,
        'std_pnl': std_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'max_dd': max_dd,
        'max_dd_pct': max_dd_pct,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'recovery_factor': recovery_factor,
        'max_consec_wins': max_cw,
        'max_consec_losses': max_cl,
    }

    if extra_info:
        result.update(extra_info)

    return result


# =============================================================================
# STRATEGY 1: CROSS-ASSET CONSENSUS
# =============================================================================

def run_cross_asset_consensus(df: pd.DataFrame,
                               min_agree: int = 4,
                               bull_thresh: float = 0.65,
                               bear_thresh: float = 0.35,
                               signal_time: str = 't5',
                               entry_time: str = 't5',
                               exit_time: str = 't12.5') -> Dict:
    """
    Only enter trades when min_agree assets agree on direction at signal_time.
    Enter each agreeing asset at entry_time, exit at exit_time.
    """
    signal_col = f'pm_yes_{signal_time.replace(".", "_")}'
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    # Group by window timestamp
    df['window_ts'] = df['window_start_utc'].dt.floor('15min')
    grouped = df.groupby('window_ts')

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}
    n_consensus_windows = 0
    direction_counts = {'bull': 0, 'bear': 0}

    for ts, group in grouped:
        if len(group) < 4:
            continue

        # Check consensus
        signals = {}
        for _, row in group.iterrows():
            asset = row['asset']
            pm_signal = row[signal_col]
            if pd.isna(pm_signal):
                continue
            signals[asset] = pm_signal

        if len(signals) < 4:
            continue

        # Count agreements
        n_bull = sum(1 for v in signals.values() if v >= bull_thresh)
        n_bear = sum(1 for v in signals.values() if v <= bear_thresh)

        direction = None
        if n_bull >= min_agree:
            direction = 'bull'
        elif n_bear >= min_agree:
            direction = 'bear'
        else:
            continue

        n_consensus_windows += 1
        direction_counts[direction] += 1

        # Trade ALL agreeing assets
        for _, row in group.iterrows():
            asset = row['asset']
            pm_signal = row[signal_col]
            entry_pm = row[entry_col]
            exit_pm = row[exit_col]

            if pd.isna(entry_pm) or pd.isna(exit_pm) or pd.isna(pm_signal):
                continue

            # Only trade if this asset agrees
            if direction == 'bull' and pm_signal < bull_thresh:
                continue
            if direction == 'bear' and pm_signal > bear_thresh:
                continue

            pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
            all_pnls.append(pnl)
            per_asset_pnls[asset].append(pnl)

    extra = {
        'consensus_windows': n_consensus_windows,
        'direction_counts': direction_counts,
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    return calculate_metrics(
        f"CONSENSUS_{min_agree}of4_{signal_time}_{entry_time}→{exit_time}_{bull_thresh:.2f}",
        all_pnls, extra
    )


# =============================================================================
# STRATEGY 2: CONTRARIAN / CROSS-WINDOW MEAN REVERSION
# =============================================================================

def run_contrarian(df: pd.DataFrame,
                   prev_strong_thresh: float = 0.85,
                   entry_time: str = 't5',
                   exit_time: str = 't12.5',
                   bull_thresh: float = 0.60,
                   bear_thresh: float = 0.40,
                   require_contrarian_signal: bool = True) -> Dict:
    """
    After a strong previous window (pm@t12.5 >= 0.85 or <= 0.15),
    bet on reversal in the next window.

    If require_contrarian_signal=True, also check that current window's PM
    at entry_time confirms the reversal direction.
    """
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    # Need previous window data
    df_seq = build_window_sequence(df)

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}
    signal_counts = {'contrarian_bear': 0, 'contrarian_bull': 0}

    for _, row in df_seq.iterrows():
        prev_t12_5 = row.get('prev_pm_t12_5')
        if pd.isna(prev_t12_5) if not isinstance(prev_t12_5, (int, float)) else False:
            continue
        if prev_t12_5 is None or (isinstance(prev_t12_5, float) and np.isnan(prev_t12_5)):
            continue

        entry_pm = row[entry_col]
        exit_pm = row[exit_col]
        if pd.isna(entry_pm) or pd.isna(exit_pm):
            continue

        direction = None

        # After strong UP previous window → expect DOWN (contrarian bear)
        if prev_t12_5 >= prev_strong_thresh:
            if require_contrarian_signal:
                # Current window should show bearish signal
                if entry_pm <= bear_thresh:
                    direction = 'bear'
            else:
                direction = 'bear'

        # After strong DOWN previous window → expect UP (contrarian bull)
        elif prev_t12_5 <= (1.0 - prev_strong_thresh):
            if require_contrarian_signal:
                # Current window should show bullish signal
                if entry_pm >= bull_thresh:
                    direction = 'bull'
            else:
                direction = 'bull'

        if direction is None:
            continue

        signal_counts[f'contrarian_{direction}'] += 1

        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        all_pnls.append(pnl)
        per_asset_pnls[row['asset']].append(pnl)

    extra = {
        'signal_counts': signal_counts,
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    sig_label = "confirmed" if require_contrarian_signal else "blind"
    return calculate_metrics(
        f"CONTRARIAN_{sig_label}_{prev_strong_thresh:.2f}_{entry_time}→{exit_time}",
        all_pnls, extra
    )


# =============================================================================
# STRATEGY 3: PM VELOCITY FILTER
# =============================================================================

def run_velocity_filtered(df: pd.DataFrame,
                          min_velocity: float = 0.15,
                          entry_time: str = 't2.5',
                          exit_time: str = 't12.5',
                          bull_thresh: float = 0.58,
                          bear_thresh: float = 0.42) -> Dict:
    """
    Use PM velocity (speed of PM movement from t0→t2.5) as quality filter.
    Fast initial moves indicate strong directional conviction.

    Enter at entry_time if PM at entry_time passes threshold AND velocity is high.
    """
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}

    for _, row in df.iterrows():
        pm_t0 = row['pm_yes_t0']
        pm_t2_5 = row['pm_yes_t2_5']
        entry_pm = row[entry_col]
        exit_pm = row[exit_col]

        if pd.isna(pm_t0) or pd.isna(pm_t2_5) or pd.isna(entry_pm) or pd.isna(exit_pm):
            continue

        velocity = pm_t2_5 - pm_t0  # Positive = bullish move, negative = bearish

        direction = None
        if velocity >= min_velocity and entry_pm >= bull_thresh:
            direction = 'bull'
        elif velocity <= -min_velocity and entry_pm <= bear_thresh:
            direction = 'bear'

        if direction is None:
            continue

        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        all_pnls.append(pnl)
        per_asset_pnls[row['asset']].append(pnl)

    extra = {
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    return calculate_metrics(
        f"VELOCITY_{min_velocity:.2f}_{entry_time}→{exit_time}_{bull_thresh:.2f}",
        all_pnls, extra
    )


# =============================================================================
# STRATEGY 4: OVERSHOOT FROM NEUTRAL
# =============================================================================

def run_overshoot_neutral(df: pd.DataFrame,
                          neutral_range: Tuple[float, float] = (0.40, 0.60),
                          bull_thresh: float = 0.65,
                          bear_thresh: float = 0.35,
                          entry_time: str = 't5',
                          exit_time: str = 't12.5') -> Dict:
    """
    When PM starts neutral at t0 but reaches extreme by entry_time,
    this is highly predictive of direction.

    PM starting neutral means the market has no bias, so a strong move
    to extreme by t5 reflects genuine new information.
    """
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}

    for _, row in df.iterrows():
        pm_t0 = row['pm_yes_t0']
        entry_pm = row[entry_col]
        exit_pm = row[exit_col]

        if pd.isna(pm_t0) or pd.isna(entry_pm) or pd.isna(exit_pm):
            continue

        # PM must start neutral
        if pm_t0 < neutral_range[0] or pm_t0 > neutral_range[1]:
            continue

        direction = None
        if entry_pm >= bull_thresh:
            direction = 'bull'
        elif entry_pm <= bear_thresh:
            direction = 'bear'

        if direction is None:
            continue

        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        all_pnls.append(pnl)
        per_asset_pnls[row['asset']].append(pnl)

    extra = {
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    return calculate_metrics(
        f"OVERSHOOT_{neutral_range}_{entry_time}→{exit_time}_{bull_thresh:.2f}",
        all_pnls, extra
    )


# =============================================================================
# STRATEGY 5: COMBINED — Cross-Asset Consensus + Momentum
# =============================================================================

def run_consensus_momentum(df: pd.DataFrame,
                            min_agree: int = 3,
                            consensus_thresh_bull: float = 0.60,
                            consensus_thresh_bear: float = 0.40,
                            entry_bull: float = 0.58,
                            entry_bear: float = 0.42,
                            entry_time: str = 't5',
                            exit_time: str = 't12.5',
                            min_trajectory: float = 0.0) -> Dict:
    """
    Combine cross-asset consensus with momentum entry.
    1. Check consensus at t5 — at least min_agree assets must agree
    2. Enter assets that pass the entry threshold
    3. Optional trajectory filter from t0
    """
    signal_col = f'pm_yes_{entry_time.replace(".", "_")}'  # consensus check at same time
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    df['window_ts'] = df['window_start_utc'].dt.floor('15min')
    grouped = df.groupby('window_ts')

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}
    n_consensus = 0

    for ts, group in grouped:
        if len(group) < min_agree:
            continue

        signals = {}
        for _, row in group.iterrows():
            pm = row[signal_col]
            if pd.notna(pm):
                signals[row['asset']] = pm

        if len(signals) < min_agree:
            continue

        n_bull = sum(1 for v in signals.values() if v >= consensus_thresh_bull)
        n_bear = sum(1 for v in signals.values() if v <= consensus_thresh_bear)

        direction = None
        if n_bull >= min_agree:
            direction = 'bull'
        elif n_bear >= min_agree:
            direction = 'bear'
        else:
            continue

        n_consensus += 1

        for _, row in group.iterrows():
            entry_pm = row[entry_col]
            exit_pm = row[exit_col]
            pm_t0 = row['pm_yes_t0']

            if pd.isna(entry_pm) or pd.isna(exit_pm) or pd.isna(pm_t0):
                continue

            # Asset-level entry threshold
            if direction == 'bull' and entry_pm < entry_bull:
                continue
            if direction == 'bear' and entry_pm > entry_bear:
                continue

            # Optional trajectory filter
            if min_trajectory > 0:
                if direction == 'bull':
                    traj = entry_pm - pm_t0
                else:
                    traj = pm_t0 - entry_pm
                if traj < min_trajectory:
                    continue

            pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
            all_pnls.append(pnl)
            per_asset_pnls[row['asset']].append(pnl)

    extra = {
        'consensus_windows': n_consensus,
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    traj_str = f"_traj{min_trajectory:.2f}" if min_trajectory > 0 else ""
    return calculate_metrics(
        f"COMBO_{min_agree}of4_{entry_time}→{exit_time}_{entry_bull:.2f}{traj_str}",
        all_pnls, extra
    )


# =============================================================================
# STRATEGY 6: CONTRARIAN + CONSENSUS
# =============================================================================

def run_contrarian_consensus(df: pd.DataFrame,
                              prev_strong_thresh: float = 0.80,
                              min_agree: int = 3,
                              bull_thresh: float = 0.60,
                              bear_thresh: float = 0.40,
                              entry_time: str = 't5',
                              exit_time: str = 't12.5') -> Dict:
    """
    After strong previous window, check if current window shows consensus
    in the contrarian direction. Only trade if both signals agree.
    """
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    # Build sequence data
    df_seq = build_window_sequence(df)
    df_seq['window_ts'] = df_seq['window_start_utc'].dt.floor('15min')

    grouped = df_seq.groupby('window_ts')

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}
    n_signals = 0

    for ts, group in grouped:
        if len(group) < min_agree:
            continue

        # Check if any asset had a strong previous window
        # We check if the majority of assets had strong previous windows
        prev_strong_up = 0
        prev_strong_down = 0
        for _, row in group.iterrows():
            prev = row.get('prev_pm_t12_5')
            if isinstance(prev, float) and not np.isnan(prev):
                if prev >= prev_strong_thresh:
                    prev_strong_up += 1
                elif prev <= (1.0 - prev_strong_thresh):
                    prev_strong_down += 1

        contrarian_dir = None
        if prev_strong_up >= min_agree:
            contrarian_dir = 'bear'  # After strong up, expect reversal
        elif prev_strong_down >= min_agree:
            contrarian_dir = 'bull'  # After strong down, expect reversal
        else:
            continue

        # Now check current consensus in contrarian direction
        signals = {}
        for _, row in group.iterrows():
            pm = row[entry_col]
            if pd.notna(pm):
                signals[row['asset']] = pm

        if contrarian_dir == 'bear':
            n_confirming = sum(1 for v in signals.values() if v <= bear_thresh)
        else:
            n_confirming = sum(1 for v in signals.values() if v >= bull_thresh)

        if n_confirming < min_agree:
            continue

        n_signals += 1

        # Trade confirming assets
        for _, row in group.iterrows():
            entry_pm = row[entry_col]
            exit_pm = row[exit_col]

            if pd.isna(entry_pm) or pd.isna(exit_pm):
                continue

            if contrarian_dir == 'bull' and entry_pm < bull_thresh:
                continue
            if contrarian_dir == 'bear' and entry_pm > bear_thresh:
                continue

            pnl = pnl_early_exit(contrarian_dir, entry_pm, exit_pm, BASE_BET)
            all_pnls.append(pnl)
            per_asset_pnls[row['asset']].append(pnl)

    extra = {
        'n_signals': n_signals,
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    return calculate_metrics(
        f"CONTRA_CONSENSUS_{min_agree}of4_{prev_strong_thresh:.2f}_{entry_time}→{exit_time}",
        all_pnls, extra
    )


# =============================================================================
# REFERENCE: Best known momentum (from strategy_b_tuning)
# =============================================================================

def run_best_momentum(df: pd.DataFrame,
                      entry_time: str = 't0',
                      exit_time: str = 't12.5',
                      bull_thresh: float = 0.58,
                      bear_thresh: float = 0.42) -> Dict:
    """Reference: best momentum strategy from previous tuning."""
    entry_col = f'pm_yes_{entry_time.replace(".", "_")}'
    exit_col = f'pm_yes_{exit_time.replace(".", "_")}'

    all_pnls = []
    per_asset_pnls = {a: [] for a in ['BTC', 'ETH', 'SOL', 'XRP']}

    for _, row in df.iterrows():
        entry_pm = row[entry_col]
        exit_pm = row[exit_col]

        if pd.isna(entry_pm) or pd.isna(exit_pm):
            continue

        direction = None
        if entry_pm >= bull_thresh:
            direction = 'bull'
        elif entry_pm <= bear_thresh:
            direction = 'bear'
        else:
            continue

        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        all_pnls.append(pnl)
        per_asset_pnls[row['asset']].append(pnl)

    extra = {
        'per_asset_trades': {a: len(v) for a, v in per_asset_pnls.items()},
        'per_asset_pnl': {a: sum(v) for a, v in per_asset_pnls.items()},
        'per_asset_wr': {a: (sum(1 for p in v if p > 0) / len(v) if v else 0) for a, v in per_asset_pnls.items()},
    }

    return calculate_metrics(
        f"MOMENTUM_{entry_time}→{exit_time}_{bull_thresh:.2f}",
        all_pnls, extra
    )


# =============================================================================
# DISPLAY
# =============================================================================

def print_results_table(results: List[Dict], title: str):
    """Print formatted results table."""
    print()
    print("=" * 140)
    print(title)
    print("=" * 140)

    # Filter out empty results
    results = [r for r in results if r.get('n_trades', 0) > 0]

    if not results:
        print("  No results with trades.")
        return

    header = (f"  {'Strategy':<55} {'Trades':>7} {'Win%':>7} {'TotalPnL':>10} "
              f"{'AvgPnL':>8} {'AvgWin':>8} {'AvgLoss':>8} {'R:R':>6} {'PF':>6} "
              f"{'MaxDD%':>8} {'Sharpe':>7} {'Sortino':>7} {'Calmar':>7}")
    print(header)
    print(f"  {'─' * 135}")

    for r in results:
        def sv(key, default=0):
            v = r.get(key, default)
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                return default
            return v

        pf = min(sv('profit_factor'), 99.9)
        line = (f"  {r['name']:<55} {r['n_trades']:>7} {r['win_rate']:>6.1%} "
                f"${r['total_pnl']:>8,.0f} ${r['avg_pnl']:>7.3f} "
                f"${r['avg_win']:>7.3f} ${r['avg_loss']:>7.3f} "
                f"{sv('rr_ratio'):>6.2f} {pf:>6.2f} "
                f"{sv('max_dd_pct'):>7.1f}% {sv('sharpe'):>7.1f} "
                f"{sv('sortino'):>7.1f} {sv('calmar'):>7.2f}")
        print(line)


def print_detailed(result: Dict):
    """Print detailed view of a single strategy."""
    r = result
    if r.get('n_trades', 0) == 0:
        print(f"  {r['name']}: No trades")
        return

    def sv(key, default=0):
        v = r.get(key, default)
        if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
            return default
        return v

    print(f"\n{'─'*80}")
    print(f"  {r['name']}")
    print(f"{'─'*80}")
    print(f"  Trades: {r['n_trades']:,}    Win Rate: {r['win_rate']:.1%}")
    print(f"  Total P&L: ${r['total_pnl']:,.2f}    Return: {r['total_return_pct']:.1f}%")
    print(f"  Avg Win: ${r['avg_win']:.3f}    Avg Loss: ${r['avg_loss']:.3f}    R:R: {sv('rr_ratio'):.2f}")
    print(f"  Profit Factor: {sv('profit_factor'):.2f}    Expectancy: ${r['expectancy']:.3f}")
    print(f"  Sharpe: {sv('sharpe'):.1f}    Sortino: {sv('sortino'):.1f}    Calmar: {sv('calmar'):.2f}")
    print(f"  Max DD: ${r['max_dd']:,.2f} ({sv('max_dd_pct'):.1f}%)")
    print(f"  Recovery Factor: {sv('recovery_factor'):.2f}")
    print(f"  Max Consec Wins: {r['max_consec_wins']}    Max Consec Losses: {r['max_consec_losses']}")

    if 'consensus_windows' in r:
        print(f"  Consensus Windows: {r['consensus_windows']}")
    if 'direction_counts' in r:
        print(f"  Direction: {r['direction_counts']}")
    if 'signal_counts' in r:
        print(f"  Signals: {r['signal_counts']}")
    if 'n_signals' in r:
        print(f"  Combined Signals: {r['n_signals']}")

    # Per-asset
    pa_trades = r.get('per_asset_trades', {})
    pa_pnl = r.get('per_asset_pnl', {})
    pa_wr = r.get('per_asset_wr', {})
    if pa_trades:
        print(f"  Per-Asset:")
        for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
            t = pa_trades.get(asset, 0)
            if t > 0:
                p = pa_pnl.get(asset, 0)
                w = pa_wr.get(asset, 0)
                print(f"    {asset}: {t} trades, {w:.1%} WR, ${p:.2f} P&L, ${p/t:.3f}/trade")


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} windows across 4 assets")
    print(f"Date range: {df['window_start_utc'].min()} to {df['window_start_utc'].max()}")
    print()

    all_results = []

    # =========================================================================
    # REFERENCE: Best known momentum strategies
    # =========================================================================
    print("Running reference momentum strategies...")

    # Best from tuning: t0→t12.5 @ 0.58/0.42
    r = run_best_momentum(df, 't0', 't12.5', 0.58, 0.42)
    all_results.append(r)

    # t0→t12.5 @ 0.55/0.45 (wider, more trades)
    r = run_best_momentum(df, 't0', 't12.5', 0.55, 0.45)
    all_results.append(r)

    # t5→t12.5 @ 0.60/0.40
    r = run_best_momentum(df, 't5', 't12.5', 0.60, 0.40)
    all_results.append(r)

    print_results_table(all_results, "REFERENCE: Best Known Momentum Strategies")

    # =========================================================================
    # 1. CROSS-ASSET CONSENSUS
    # =========================================================================
    print("\n\nRunning cross-asset consensus strategies...")
    consensus_results = []

    # Vary: min_agree (3 or 4), thresholds, signal/entry/exit times
    for min_agree in [3, 4]:
        for bull_th, bear_th in [(0.60, 0.40), (0.65, 0.35), (0.55, 0.45)]:
            for entry_t, exit_t in [('t5', 't12.5'), ('t5', 't10'), ('t0', 't12.5')]:
                r = run_cross_asset_consensus(
                    df, min_agree=min_agree,
                    bull_thresh=bull_th, bear_thresh=bear_th,
                    signal_time=entry_t, entry_time=entry_t, exit_time=exit_t
                )
                if r.get('n_trades', 0) >= 20:
                    consensus_results.append(r)

    # Sort by total P&L
    consensus_results.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(consensus_results, "1. CROSS-ASSET CONSENSUS STRATEGIES")
    all_results.extend(consensus_results)

    # =========================================================================
    # 2. CONTRARIAN / CROSS-WINDOW
    # =========================================================================
    print("\n\nRunning contrarian strategies...")
    contrarian_results = []

    for prev_thresh in [0.80, 0.85, 0.75]:
        for entry_t, exit_t in [('t5', 't12.5'), ('t0', 't12.5'), ('t2.5', 't12.5')]:
            for bull_th, bear_th in [(0.60, 0.40), (0.55, 0.45), (0.50, 0.50)]:
                # With contrarian signal confirmation
                r = run_contrarian(df, prev_strong_thresh=prev_thresh,
                                   entry_time=entry_t, exit_time=exit_t,
                                   bull_thresh=bull_th, bear_thresh=bear_th,
                                   require_contrarian_signal=True)
                if r.get('n_trades', 0) >= 20:
                    contrarian_results.append(r)

                # Blind contrarian (no signal check, just direction)
                r = run_contrarian(df, prev_strong_thresh=prev_thresh,
                                   entry_time=entry_t, exit_time=exit_t,
                                   bull_thresh=bull_th, bear_thresh=bear_th,
                                   require_contrarian_signal=False)
                if r.get('n_trades', 0) >= 20:
                    contrarian_results.append(r)

    contrarian_results.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(contrarian_results[:20], "2. CONTRARIAN / CROSS-WINDOW STRATEGIES (Top 20)")
    all_results.extend(contrarian_results)

    # =========================================================================
    # 3. PM VELOCITY FILTER
    # =========================================================================
    print("\n\nRunning velocity-filtered strategies...")
    velocity_results = []

    for min_vel in [0.10, 0.15, 0.20, 0.08]:
        for entry_t, exit_t in [('t2.5', 't12.5'), ('t5', 't12.5'), ('t2.5', 't10')]:
            for bull_th, bear_th in [(0.55, 0.45), (0.58, 0.42), (0.60, 0.40)]:
                r = run_velocity_filtered(df, min_velocity=min_vel,
                                          entry_time=entry_t, exit_time=exit_t,
                                          bull_thresh=bull_th, bear_thresh=bear_th)
                if r.get('n_trades', 0) >= 20:
                    velocity_results.append(r)

    velocity_results.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(velocity_results[:20], "3. PM VELOCITY-FILTERED STRATEGIES (Top 20)")
    all_results.extend(velocity_results)

    # =========================================================================
    # 4. OVERSHOOT FROM NEUTRAL
    # =========================================================================
    print("\n\nRunning overshoot-from-neutral strategies...")
    overshoot_results = []

    for neutral in [(0.40, 0.60), (0.45, 0.55), (0.42, 0.58)]:
        for bull_th, bear_th in [(0.60, 0.40), (0.65, 0.35), (0.55, 0.45)]:
            for entry_t, exit_t in [('t5', 't12.5'), ('t2.5', 't12.5'), ('t5', 't10')]:
                r = run_overshoot_neutral(df, neutral_range=neutral,
                                          bull_thresh=bull_th, bear_thresh=bear_th,
                                          entry_time=entry_t, exit_time=exit_t)
                if r.get('n_trades', 0) >= 20:
                    overshoot_results.append(r)

    overshoot_results.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(overshoot_results[:20], "4. OVERSHOOT FROM NEUTRAL STRATEGIES (Top 20)")
    all_results.extend(overshoot_results)

    # =========================================================================
    # 5. COMBINED: CONSENSUS + MOMENTUM
    # =========================================================================
    print("\n\nRunning combined consensus+momentum strategies...")
    combo_results = []

    for min_agree in [3, 4]:
        for con_bull, con_bear in [(0.55, 0.45), (0.60, 0.40), (0.50, 0.50)]:
            for ent_bull, ent_bear in [(0.55, 0.45), (0.58, 0.42), (0.60, 0.40)]:
                for entry_t, exit_t in [('t5', 't12.5'), ('t0', 't12.5')]:
                    for traj in [0.0, 0.03, 0.05]:
                        r = run_consensus_momentum(
                            df, min_agree=min_agree,
                            consensus_thresh_bull=con_bull,
                            consensus_thresh_bear=con_bear,
                            entry_bull=ent_bull, entry_bear=ent_bear,
                            entry_time=entry_t, exit_time=exit_t,
                            min_trajectory=traj
                        )
                        if r.get('n_trades', 0) >= 20:
                            combo_results.append(r)

    combo_results.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(combo_results[:25], "5. COMBINED: CONSENSUS + MOMENTUM (Top 25)")
    all_results.extend(combo_results)

    # =========================================================================
    # 6. CONTRARIAN + CONSENSUS
    # =========================================================================
    print("\n\nRunning contrarian+consensus strategies...")
    contra_con_results = []

    for prev_thresh in [0.75, 0.80, 0.85]:
        for min_agree in [2, 3]:
            for bull_th, bear_th in [(0.55, 0.45), (0.60, 0.40)]:
                for entry_t, exit_t in [('t5', 't12.5'), ('t0', 't12.5')]:
                    r = run_contrarian_consensus(
                        df, prev_strong_thresh=prev_thresh,
                        min_agree=min_agree,
                        bull_thresh=bull_th, bear_thresh=bear_th,
                        entry_time=entry_t, exit_time=exit_t
                    )
                    if r.get('n_trades', 0) >= 10:
                        contra_con_results.append(r)

    contra_con_results.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(contra_con_results[:15], "6. CONTRARIAN + CONSENSUS (Top 15)")
    all_results.extend(contra_con_results)

    # =========================================================================
    # FINAL COMPARISON: TOP STRATEGIES ACROSS ALL CATEGORIES
    # =========================================================================
    print("\n")

    # Filter profitable strategies with enough trades
    profitable = [r for r in all_results if r.get('total_pnl', 0) > 0 and r.get('n_trades', 0) >= 30]

    # By total P&L
    by_pnl = sorted(profitable, key=lambda x: x.get('total_pnl', 0), reverse=True)
    print_results_table(by_pnl[:20], "FINAL RANKING: TOP 20 BY TOTAL P&L (all categories)")

    # By Sharpe
    by_sharpe = sorted(profitable, key=lambda x: x.get('sharpe', 0) if not np.isinf(x.get('sharpe', 0)) else 0, reverse=True)
    print_results_table(by_sharpe[:20], "FINAL RANKING: TOP 20 BY SHARPE RATIO (all categories)")

    # By composite: Sharpe * PF * sqrt(trades)
    for r in profitable:
        pf = r.get('profit_factor', 0)
        if isinstance(pf, float) and (np.isinf(pf) or np.isnan(pf)):
            pf = 0
        sh = r.get('sharpe', 0)
        if isinstance(sh, float) and (np.isinf(sh) or np.isnan(sh)):
            sh = 0
        r['composite'] = sh * min(pf, 10) * np.sqrt(r['n_trades']) if sh > 0 and pf > 0 else 0

    by_composite = sorted(profitable, key=lambda x: x.get('composite', 0), reverse=True)
    print_results_table(by_composite[:20], "FINAL RANKING: TOP 20 BY COMPOSITE SCORE (all categories)")

    # Detailed view of top 8
    print()
    print("=" * 140)
    print("DETAILED VIEW: TOP 8 OVERALL BY COMPOSITE SCORE")
    print("=" * 140)
    for r in by_composite[:8]:
        print_detailed(r)

    # =========================================================================
    # COMPARISON VS BEST MOMENTUM
    # =========================================================================
    print()
    print("=" * 140)
    print("HEAD-TO-HEAD: New Strategies vs Best Known Momentum")
    print("=" * 140)

    # Best momentum reference
    best_mom = [r for r in all_results if r['name'].startswith('MOMENTUM_')]
    best_new = [r for r in by_composite[:5] if not r['name'].startswith('MOMENTUM_')]

    comparison = best_mom[:2] + best_new[:5]
    print_results_table(comparison, "Head-to-Head Comparison")

    for r in comparison:
        print_detailed(r)

    # =========================================================================
    # SAVE
    # =========================================================================
    json_results = []
    for r in by_composite[:80]:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                jr[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                jr[k] = float(v)
            elif isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            else:
                jr[k] = v
        json_results.append(jr)

    output_path = OUTPUT_DIR / f"advanced_strategies_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nSaved top 80 results to: {output_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
