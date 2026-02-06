#!/usr/bin/env python3
"""
Strategy Comparison Backtest: A vs B vs C vs Baseline

Compares four approaches head-to-head:

BASELINE (current): Two-stage entry at t=7.5/t=10, hold to resolution (binary payout)
STRATEGY A: Early Profit-Taking - Same two-stage entry, but sell position before resolution
STRATEGY B: Momentum Scalping - Enter earlier at t=5 when PM is mid-range, exit at t=10/t=12.5
STRATEGY C: Hybrid - Two-stage confirmed entry, profit target + stop loss, fallback to resolution

All strategies use the same fee model:
  - 0.1% taker fee on contract premium (entry AND exit for non-resolution trades)
  - 0.5% spread/slippage (per leg - so 2x for round-trip early exits)
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "reports"
ASSETS = ["btc", "eth", "sol", "xrp"]

# Shared constants
INITIAL_BANKROLL = 1000.0
BASE_BET = 50.0
FEE_RATE = 0.001      # 0.1% on contract premium per leg
SPREAD_COST = 0.005   # 0.5% per leg


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
                window_id, window_start_utc, outcome, outcome_binary,
                spot_open, spot_close, spot_change_pct,
                pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5,
                pm_spread_t0, pm_spread_t5
            FROM windows
            WHERE outcome IS NOT NULL
              AND pm_yes_t0 IS NOT NULL
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


# =============================================================================
# P&L CALCULATIONS
# =============================================================================

def pnl_binary(direction: str, entry_price: float, outcome: str, bet_size: float) -> float:
    """
    P&L for holding to binary resolution (current strategy).
    Entry fee + spread on entry only.
    """
    if direction == 'bull':
        contract_price = entry_price
        won = (outcome == 'up')
    else:  # bear
        contract_price = 1.0 - entry_price  # NO price
        won = (outcome == 'down')

    fee = contract_price * FEE_RATE * bet_size
    spread = SPREAD_COST * bet_size

    if won:
        gross = (1.0 - contract_price) * bet_size
    else:
        gross = -contract_price * bet_size

    return gross - fee - spread


def pnl_early_exit(direction: str, entry_pm_yes: float, exit_pm_yes: float, bet_size: float) -> float:
    """
    P&L for selling position before resolution.
    Double fees: entry fee + exit fee, entry spread + exit spread.

    We buy contracts at entry, sell at exit. The P&L is the price difference
    times the number of contracts we hold.

    For bull (bought YES):
      - num_contracts = bet_size / entry_price
      - But simpler: P&L per dollar bet = (exit_price - entry_price) / entry_price
      - Actually: P&L = (exit_pm_yes - entry_pm_yes) * bet_size (for normalized positions)

    Polymarket contracts: you buy N contracts at price P.
      N = bet_size / P
      Value at exit = N * exit_price = bet_size * (exit_price / entry_price)
      P&L = bet_size * (exit_price / entry_price - 1)
      = bet_size * (exit_price - entry_price) / entry_price

    For bear (bought NO at 1-pm_yes):
      entry_no = 1 - entry_pm_yes
      exit_no = 1 - exit_pm_yes
      N = bet_size / entry_no
      P&L = bet_size * (exit_no - entry_no) / entry_no
      = bet_size * (entry_pm_yes - exit_pm_yes) / (1 - entry_pm_yes)
    """
    if direction == 'bull':
        entry_contract = entry_pm_yes
        exit_contract = exit_pm_yes
    else:
        entry_contract = 1.0 - entry_pm_yes
        exit_contract = 1.0 - exit_pm_yes

    # Number of contracts
    if entry_contract <= 0.001:
        return 0.0
    n_contracts = bet_size / entry_contract

    # Gross P&L
    gross = n_contracts * (exit_contract - entry_contract)

    # Fees: on entry premium AND exit premium
    entry_fee = entry_contract * n_contracts * FEE_RATE
    exit_fee = exit_contract * n_contracts * FEE_RATE

    # Spread: on both legs
    entry_spread = SPREAD_COST * bet_size
    exit_spread = SPREAD_COST * (n_contracts * exit_contract)  # based on exit value

    total_fees = entry_fee + exit_fee + entry_spread + exit_spread

    return gross - total_fees


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

@dataclass
class TradeResult:
    timestamp: datetime
    asset: str
    direction: str
    entry_time: str       # e.g., "t7.5", "t10", "t5"
    entry_pm_yes: float
    exit_time: str        # e.g., "t15_resolution", "t12.5_profit_target", "t10_stop"
    exit_pm_yes: float
    outcome: str          # "win", "loss"
    pnl: float
    bet_size: float
    bankroll_after: float = 0.0
    exit_reason: str = ""  # "resolution", "profit_target", "stop_loss", "time_exit"


def run_baseline(df: pd.DataFrame) -> List[TradeResult]:
    """
    BASELINE: Current two-stage strategy.
    Signal at t=7.5 (>=0.70 bull, <=0.30 bear)
    Confirm at t=10 (>=0.85 bull, <=0.15 bear)
    Hold to resolution at t=15.
    """
    trades = []
    bankroll = INITIAL_BANKROLL

    for _, row in df.iterrows():
        pm_7_5 = row['pm_yes_t7_5']
        pm_10 = row['pm_yes_t10']

        # Stage 1: signal
        if pm_7_5 >= 0.70:
            direction = 'bull'
        elif pm_7_5 <= 0.30:
            direction = 'bear'
        else:
            continue

        # Stage 2: confirm
        if direction == 'bull' and pm_10 < 0.85:
            continue
        if direction == 'bear' and pm_10 > 0.15:
            continue

        # Entry at t=10 price, hold to resolution
        pnl = pnl_binary(direction, pm_10, row['outcome'], BASE_BET)
        bankroll += pnl
        won = (direction == 'bull' and row['outcome'] == 'up') or \
              (direction == 'bear' and row['outcome'] == 'down')

        trades.append(TradeResult(
            timestamp=row['window_start_utc'],
            asset=row['asset'],
            direction=direction,
            entry_time="t10",
            entry_pm_yes=pm_10,
            exit_time="t15",
            exit_pm_yes=1.0 if won else 0.0,
            outcome="win" if won else "loss",
            pnl=pnl,
            bet_size=BASE_BET,
            bankroll_after=bankroll,
            exit_reason="resolution",
        ))

    return trades


def run_strategy_a(df: pd.DataFrame, profit_target: float = 0.03,
                   stop_loss: float = 0.03, exit_time: str = "t12.5") -> List[TradeResult]:
    """
    STRATEGY A: Early Profit-Taking.
    Same two-stage entry (signal t=7.5, confirm t=10).
    Instead of holding to resolution, sell position at t=12.5.

    Variants tested:
    - Pure time exit at t=12.5 (no targets/stops)
    - With profit target and stop loss checked at t=12.5
    """
    trades = []
    bankroll = INITIAL_BANKROLL

    for _, row in df.iterrows():
        pm_7_5 = row['pm_yes_t7_5']
        pm_10 = row['pm_yes_t10']

        # Stage 1: signal
        if pm_7_5 >= 0.70:
            direction = 'bull'
        elif pm_7_5 <= 0.30:
            direction = 'bear'
        else:
            continue

        # Stage 2: confirm
        if direction == 'bull' and pm_10 < 0.85:
            continue
        if direction == 'bear' and pm_10 > 0.15:
            continue

        # Entry at t=10 price
        entry_pm = pm_10
        exit_pm = row['pm_yes_t12_5']

        if pd.isna(exit_pm):
            continue

        # Check movement
        if direction == 'bull':
            move = exit_pm - entry_pm
        else:
            move = entry_pm - exit_pm  # For bear, pm_yes dropping is good

        # Always exit at t=12.5 (early exit, sell position)
        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        bankroll += pnl

        exit_reason = "time_exit"
        if move >= profit_target:
            exit_reason = "profit_target"
        elif move <= -stop_loss:
            exit_reason = "stop_loss"

        trades.append(TradeResult(
            timestamp=row['window_start_utc'],
            asset=row['asset'],
            direction=direction,
            entry_time="t10",
            entry_pm_yes=entry_pm,
            exit_time="t12.5",
            exit_pm_yes=exit_pm,
            outcome="win" if pnl > 0 else "loss",
            pnl=pnl,
            bet_size=BASE_BET,
            bankroll_after=bankroll,
            exit_reason=exit_reason,
        ))

    return trades


def run_strategy_b(df: pd.DataFrame, entry_bull: float = 0.60, entry_bear: float = 0.40,
                   exit_time: str = "t10") -> List[TradeResult]:
    """
    STRATEGY B: Momentum Scalping.
    Enter at t=5 when PM is in mid-range, ride the momentum to t=10 or t=12.5.
    No binary resolution dependency.

    We're trading the PM price trajectory, not the binary outcome.
    """
    trades = []
    bankroll = INITIAL_BANKROLL

    for _, row in df.iterrows():
        pm_5 = row['pm_yes_t5']
        pm_0 = row['pm_yes_t0']

        if pd.isna(pm_5) or pd.isna(pm_0):
            continue

        # Trajectory filter: PM must already be moving (momentum from t=0 to t=5)
        trajectory = pm_5 - pm_0

        direction = None
        if pm_5 >= entry_bull and trajectory > 0.05:
            direction = 'bull'
        elif pm_5 <= entry_bear and trajectory < -0.05:
            direction = 'bear'

        if direction is None:
            continue

        entry_pm = pm_5

        # Exit at t=10
        if exit_time == "t10":
            exit_pm = row['pm_yes_t10']
        elif exit_time == "t12.5":
            exit_pm = row['pm_yes_t12_5']
        else:
            exit_pm = row['pm_yes_t10']

        if pd.isna(exit_pm):
            continue

        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        bankroll += pnl

        trades.append(TradeResult(
            timestamp=row['window_start_utc'],
            asset=row['asset'],
            direction=direction,
            entry_time="t5",
            entry_pm_yes=entry_pm,
            exit_time=exit_time,
            exit_pm_yes=exit_pm,
            outcome="win" if pnl > 0 else "loss",
            pnl=pnl,
            bet_size=BASE_BET,
            bankroll_after=bankroll,
            exit_reason="time_exit",
        ))

    return trades


def run_strategy_b_t7_5(df: pd.DataFrame, entry_bull: float = 0.65, entry_bear: float = 0.35,
                         exit_time: str = "t12.5") -> List[TradeResult]:
    """
    STRATEGY B variant: Enter at t=7.5, exit at t=12.5.
    Slightly later entry for better signal, still avoids resolution.
    """
    trades = []
    bankroll = INITIAL_BANKROLL

    for _, row in df.iterrows():
        pm_7_5 = row['pm_yes_t7_5']
        pm_0 = row['pm_yes_t0']

        if pd.isna(pm_7_5) or pd.isna(pm_0):
            continue

        trajectory = pm_7_5 - pm_0

        direction = None
        if pm_7_5 >= entry_bull and trajectory > 0.05:
            direction = 'bull'
        elif pm_7_5 <= entry_bear and trajectory < -0.05:
            direction = 'bear'

        if direction is None:
            continue

        entry_pm = pm_7_5

        if exit_time == "t10":
            exit_pm = row['pm_yes_t10']
        else:
            exit_pm = row['pm_yes_t12_5']

        if pd.isna(exit_pm):
            continue

        pnl = pnl_early_exit(direction, entry_pm, exit_pm, BASE_BET)
        bankroll += pnl

        trades.append(TradeResult(
            timestamp=row['window_start_utc'],
            asset=row['asset'],
            direction=direction,
            entry_time="t7.5",
            entry_pm_yes=entry_pm,
            exit_time=exit_time,
            exit_pm_yes=exit_pm,
            outcome="win" if pnl > 0 else "loss",
            pnl=pnl,
            bet_size=BASE_BET,
            bankroll_after=bankroll,
            exit_reason="time_exit",
        ))

    return trades


def run_strategy_c(df: pd.DataFrame, profit_target_pm: float = 0.04,
                   stop_loss_pm: float = 0.04) -> List[TradeResult]:
    """
    STRATEGY C: Hybrid - Confirmed entry with targets.
    Same two-stage entry.
    Check at t=12.5:
      - If profit target hit (PM moved in our favor by X) → exit early
      - If stop loss hit (PM moved against by X) → exit early
      - Otherwise → hold to resolution (binary payout)

    This captures the best of both worlds: take early profits when available,
    cut losses early, and let the 93% resolution winrate work for borderline cases.
    """
    trades = []
    bankroll = INITIAL_BANKROLL

    for _, row in df.iterrows():
        pm_7_5 = row['pm_yes_t7_5']
        pm_10 = row['pm_yes_t10']

        # Stage 1: signal
        if pm_7_5 >= 0.70:
            direction = 'bull'
        elif pm_7_5 <= 0.30:
            direction = 'bear'
        else:
            continue

        # Stage 2: confirm
        if direction == 'bull' and pm_10 < 0.85:
            continue
        if direction == 'bear' and pm_10 > 0.15:
            continue

        entry_pm = pm_10
        pm_12_5 = row['pm_yes_t12_5']

        if pd.isna(pm_12_5):
            # No t=12.5 data, hold to resolution
            pnl = pnl_binary(direction, pm_10, row['outcome'], BASE_BET)
            bankroll += pnl
            won = (direction == 'bull' and row['outcome'] == 'up') or \
                  (direction == 'bear' and row['outcome'] == 'down')
            trades.append(TradeResult(
                timestamp=row['window_start_utc'],
                asset=row['asset'],
                direction=direction,
                entry_time="t10",
                entry_pm_yes=entry_pm,
                exit_time="t15",
                exit_pm_yes=1.0 if won else 0.0,
                outcome="win" if won else "loss",
                pnl=pnl,
                bet_size=BASE_BET,
                bankroll_after=bankroll,
                exit_reason="resolution_no_data",
            ))
            continue

        # Check PM move at t=12.5
        if direction == 'bull':
            move = pm_12_5 - entry_pm
        else:
            move = entry_pm - pm_12_5

        # Decision logic
        if move >= profit_target_pm:
            # Profit target hit - exit early
            pnl = pnl_early_exit(direction, entry_pm, pm_12_5, BASE_BET)
            bankroll += pnl
            trades.append(TradeResult(
                timestamp=row['window_start_utc'],
                asset=row['asset'],
                direction=direction,
                entry_time="t10",
                entry_pm_yes=entry_pm,
                exit_time="t12.5",
                exit_pm_yes=pm_12_5,
                outcome="win" if pnl > 0 else "loss",
                pnl=pnl,
                bet_size=BASE_BET,
                bankroll_after=bankroll,
                exit_reason="profit_target",
            ))
        elif move <= -stop_loss_pm:
            # Stop loss hit - exit early to cut loss
            pnl = pnl_early_exit(direction, entry_pm, pm_12_5, BASE_BET)
            bankroll += pnl
            trades.append(TradeResult(
                timestamp=row['window_start_utc'],
                asset=row['asset'],
                direction=direction,
                entry_time="t10",
                entry_pm_yes=entry_pm,
                exit_time="t12.5",
                exit_pm_yes=pm_12_5,
                outcome="win" if pnl > 0 else "loss",
                pnl=pnl,
                bet_size=BASE_BET,
                bankroll_after=bankroll,
                exit_reason="stop_loss",
            ))
        else:
            # Neutral zone - hold to resolution
            pnl = pnl_binary(direction, pm_10, row['outcome'], BASE_BET)
            bankroll += pnl
            won = (direction == 'bull' and row['outcome'] == 'up') or \
                  (direction == 'bear' and row['outcome'] == 'down')
            trades.append(TradeResult(
                timestamp=row['window_start_utc'],
                asset=row['asset'],
                direction=direction,
                entry_time="t10",
                entry_pm_yes=entry_pm,
                exit_time="t15",
                exit_pm_yes=1.0 if won else 0.0,
                outcome="win" if won else "loss",
                pnl=pnl,
                bet_size=BASE_BET,
                bankroll_after=bankroll,
                exit_reason="resolution",
            ))

    return trades


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_all_metrics(name: str, trades: List[TradeResult]) -> Dict:
    """Calculate comprehensive trading metrics for a strategy."""
    if not trades:
        return {"name": name, "n_trades": 0}

    n = len(trades)
    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    n_wins = len(wins)
    n_losses = len(losses)

    pnls = np.array([t.pnl for t in trades])
    win_pnls = np.array([t.pnl for t in wins]) if wins else np.array([])
    loss_pnls = np.array([t.pnl for t in losses]) if losses else np.array([])

    # Basic
    total_pnl = float(np.sum(pnls))
    win_rate = n_wins / n
    avg_pnl = float(np.mean(pnls))
    median_pnl = float(np.median(pnls))
    std_pnl = float(np.std(pnls)) if n > 1 else 0.0

    # Win/loss averages
    avg_win = float(np.mean(win_pnls)) if len(win_pnls) > 0 else 0.0
    avg_loss = float(np.mean(loss_pnls)) if len(loss_pnls) > 0 else 0.0

    # Risk-reward ratio
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Profit factor
    gross_wins = float(np.sum(win_pnls[win_pnls > 0])) if len(win_pnls) > 0 else 0.0
    gross_losses = abs(float(np.sum(loss_pnls[loss_pnls < 0]))) if len(loss_pnls) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Equity curve and drawdown
    equity = [INITIAL_BANKROLL]
    for t in trades:
        equity.append(equity[-1] + t.pnl)
    equity = np.array(equity)

    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    drawdown_pct = (drawdown / peak) * 100
    max_dd = float(np.min(drawdown))
    max_dd_pct = float(np.min(drawdown_pct))
    avg_dd_pct = float(np.mean(drawdown_pct[drawdown_pct < 0])) if np.any(drawdown_pct < 0) else 0.0

    # Consecutive streaks
    max_consec_wins = 0
    max_consec_losses = 0
    cur_w = 0
    cur_l = 0
    for t in trades:
        if t.outcome == "win":
            cur_w += 1
            cur_l = 0
            max_consec_wins = max(max_consec_wins, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_consec_losses = max(max_consec_losses, cur_l)

    # Sharpe (annualized for 15-min windows: 96/day * 365 days)
    periods_per_year = 96 * 365
    sharpe = (avg_pnl / std_pnl) * np.sqrt(periods_per_year) if std_pnl > 0 else 0.0

    # Sortino
    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = (avg_pnl / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    # Calmar
    total_return_pct = (equity[-1] - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    calmar = total_return_pct / abs(max_dd_pct) if max_dd_pct < 0 else 0.0

    # Recovery factor
    recovery_factor = total_pnl / abs(max_dd) if max_dd < 0 else float('inf') if total_pnl > 0 else 0.0

    # Daily analysis
    df_trades = pd.DataFrame([{
        'date': t.timestamp.date() if isinstance(t.timestamp, datetime) else pd.Timestamp(t.timestamp).date(),
        'pnl': t.pnl,
    } for t in trades])
    daily_pnl = df_trades.groupby('date')['pnl'].sum()
    n_days = len(daily_pnl)
    trades_per_day = n / n_days if n_days > 0 else 0
    best_day = float(daily_pnl.max()) if n_days > 0 else 0.0
    worst_day = float(daily_pnl.min()) if n_days > 0 else 0.0
    profitable_days_pct = float((daily_pnl > 0).sum() / n_days * 100) if n_days > 0 else 0.0

    # Per-asset breakdown
    per_asset = {}
    for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
        at = [t for t in trades if t.asset == asset]
        if at:
            aw = len([t for t in at if t.outcome == 'win'])
            ap = sum(t.pnl for t in at)
            per_asset[asset] = {
                'trades': len(at),
                'wins': aw,
                'win_rate': aw / len(at),
                'total_pnl': ap,
                'avg_pnl': ap / len(at),
            }

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {'count': 0, 'wins': 0, 'total_pnl': 0.0}
        exit_reasons[r]['count'] += 1
        if t.outcome == 'win':
            exit_reasons[r]['wins'] += 1
        exit_reasons[r]['total_pnl'] += t.pnl
    for r in exit_reasons:
        c = exit_reasons[r]['count']
        exit_reasons[r]['win_rate'] = exit_reasons[r]['wins'] / c if c > 0 else 0
        exit_reasons[r]['avg_pnl'] = exit_reasons[r]['total_pnl'] / c if c > 0 else 0

    # Price movement analysis
    entry_prices = [t.entry_pm_yes for t in trades]
    exit_prices = [t.exit_pm_yes for t in trades]

    return {
        'name': name,
        'n_trades': n,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'median_pnl': median_pnl,
        'std_pnl': std_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr_ratio': rr_ratio,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'total_return_pct': total_return_pct,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'avg_drawdown_pct': avg_dd_pct,
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'recovery_factor': recovery_factor,
        'trades_per_day': trades_per_day,
        'best_day': best_day,
        'worst_day': worst_day,
        'profitable_days_pct': profitable_days_pct,
        'per_asset': per_asset,
        'exit_reasons': exit_reasons,
        'equity_curve': equity.tolist(),
        'final_bankroll': float(equity[-1]),
        'avg_entry_price': float(np.mean(entry_prices)),
    }


# =============================================================================
# REPORT GENERATION
# =============================================================================

def print_comparison_report(all_results: Dict[str, Dict]):
    """Print side-by-side comparison of all strategies."""

    print()
    print("=" * 120)
    print("STRATEGY COMPARISON BACKTEST - COMPREHENSIVE RESULTS")
    print("=" * 120)
    print()

    # Data overview
    strats = list(all_results.keys())
    first = all_results[strats[0]]
    print(f"Base Bet: ${BASE_BET:.0f}  |  Fee: {FEE_RATE*100:.1f}%  |  Spread: {SPREAD_COST*100:.1f}%  |  Initial Bankroll: ${INITIAL_BANKROLL:,.0f}")
    print()

    # =========================================================================
    # MAIN COMPARISON TABLE
    # =========================================================================
    print("─" * 120)
    header = f"{'Metric':<28}"
    for name in strats:
        header += f"  {name:>18}"
    print(header)
    print("─" * 120)

    metrics_to_show = [
        ('Trades', 'n_trades', '{:>18,}'),
        ('Win Rate', 'win_rate', '{:>17.1%}'),
        ('', None, None),  # spacer
        ('Total P&L', 'total_pnl', '${:>16,.2f}'),
        ('Final Bankroll', 'final_bankroll', '${:>16,.2f}'),
        ('Total Return %', 'total_return_pct', '{:>17.2f}%'),
        ('', None, None),
        ('Avg P&L/Trade', 'avg_pnl', '${:>16.3f}'),
        ('Median P&L/Trade', 'median_pnl', '${:>16.3f}'),
        ('Std Dev P&L', 'std_pnl', '${:>16.3f}'),
        ('', None, None),
        ('Avg Win', 'avg_win', '${:>16.3f}'),
        ('Avg Loss', 'avg_loss', '${:>16.3f}'),
        ('Risk/Reward Ratio', 'rr_ratio', '{:>18.2f}'),
        ('Profit Factor', 'profit_factor', '{:>18.2f}'),
        ('Expectancy/Trade', 'expectancy', '${:>16.3f}'),
        ('', None, None),
        ('Max Drawdown $', 'max_drawdown', '${:>16,.2f}'),
        ('Max Drawdown %', 'max_drawdown_pct', '{:>17.2f}%'),
        ('Avg Drawdown %', 'avg_drawdown_pct', '{:>17.2f}%'),
        ('', None, None),
        ('Sharpe Ratio', 'sharpe', '{:>18.1f}'),
        ('Sortino Ratio', 'sortino', '{:>18.1f}'),
        ('Calmar Ratio', 'calmar', '{:>18.2f}'),
        ('Recovery Factor', 'recovery_factor', '{:>18.2f}'),
        ('', None, None),
        ('Max Consec Wins', 'max_consec_wins', '{:>18}'),
        ('Max Consec Losses', 'max_consec_losses', '{:>18}'),
        ('Trades/Day', 'trades_per_day', '{:>18.1f}'),
        ('Best Day', 'best_day', '${:>16,.2f}'),
        ('Worst Day', 'worst_day', '${:>16,.2f}'),
        ('% Profitable Days', 'profitable_days_pct', '{:>17.1f}%'),
    ]

    for label, key, fmt in metrics_to_show:
        if key is None:
            print()
            continue
        row = f"{label:<28}"
        for name in strats:
            val = all_results[name].get(key, 0)
            if isinstance(val, float) and (val == float('inf') or val == float('-inf')):
                row += f"  {'Inf':>18}"
            else:
                try:
                    row += f"  {fmt.format(val)}"
                except (ValueError, TypeError):
                    row += f"  {str(val):>18}"
        print(row)

    print("─" * 120)
    print()

    # =========================================================================
    # EXIT REASON BREAKDOWN
    # =========================================================================
    print("EXIT REASON BREAKDOWN")
    print("─" * 100)
    for name in strats:
        reasons = all_results[name].get('exit_reasons', {})
        if not reasons:
            continue
        print(f"\n  {name}:")
        print(f"    {'Reason':<25} {'Count':>8} {'Win%':>8} {'Total P&L':>12} {'Avg P&L':>10}")
        print(f"    {'─'*65}")
        for reason, data in sorted(reasons.items(), key=lambda x: -x[1]['count']):
            print(f"    {reason:<25} {data['count']:>8} {data['win_rate']:>7.1%} "
                  f"${data['total_pnl']:>10,.2f} ${data['avg_pnl']:>9.3f}")

    print()

    # =========================================================================
    # PER-ASSET BREAKDOWN
    # =========================================================================
    print("PER-ASSET BREAKDOWN")
    print("─" * 100)
    for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
        print(f"\n  {asset}:")
        header = f"    {'Strategy':<28} {'Trades':>8} {'Win%':>8} {'Total P&L':>12} {'Avg P&L':>10}"
        print(header)
        print(f"    {'─'*68}")
        for name in strats:
            pa = all_results[name].get('per_asset', {}).get(asset, {})
            if pa:
                print(f"    {name:<28} {pa['trades']:>8} {pa['win_rate']:>7.1%} "
                      f"${pa['total_pnl']:>10,.2f} ${pa['avg_pnl']:>9.3f}")
            else:
                print(f"    {name:<28} {'N/A':>8}")

    print()

    # =========================================================================
    # RISK-ADJUSTED RANKING
    # =========================================================================
    print("=" * 80)
    print("STRATEGY RANKING")
    print("=" * 80)

    ranking_criteria = [
        ('By Total P&L', 'total_pnl', True),
        ('By Win Rate', 'win_rate', True),
        ('By Risk/Reward', 'rr_ratio', True),
        ('By Sharpe Ratio', 'sharpe', True),
        ('By Profit Factor', 'profit_factor', True),
        ('By Max Drawdown %', 'max_drawdown_pct', False),  # less negative = better
        ('By Expectancy', 'expectancy', True),
        ('By Worst Day', 'worst_day', False),  # Less negative = better, but False inverts to max
    ]

    for label, key, higher_is_better in ranking_criteria:
        sorted_strats = sorted(strats, key=lambda s: all_results[s].get(key, 0),
                               reverse=higher_is_better)
        values = [f"{s}: {all_results[s].get(key, 0):.3f}" for s in sorted_strats]
        print(f"  {label:<22} → {' > '.join(values)}")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading data from all asset databases...")
    df = load_all_data()
    print(f"Loaded {len(df)} windows with full price data")
    print(f"Date range: {df['window_start_utc'].min()} to {df['window_start_utc'].max()}")
    print(f"Assets: {sorted(df['asset'].unique().tolist())}")

    up = (df['outcome'] == 'up').sum()
    down = (df['outcome'] == 'down').sum()
    print(f"Outcome balance: UP {up} ({up/len(df)*100:.1f}%) / DOWN {down} ({down/len(df)*100:.1f}%)")
    print()

    # =========================================================================
    # RUN ALL STRATEGIES
    # =========================================================================
    print("Running backtests...")
    all_trades = {}
    all_metrics = {}

    # BASELINE
    print("  [1/8] Baseline (current two-stage, hold to resolution)...")
    trades = run_baseline(df)
    all_trades['BASELINE'] = trades
    all_metrics['BASELINE'] = calculate_all_metrics('BASELINE', trades)

    # STRATEGY A: Early profit taking (sell at t=12.5)
    print("  [2/8] Strategy A: Early Profit Take (enter t=10, exit t=12.5)...")
    trades = run_strategy_a(df)
    all_trades['A_EXIT_T12.5'] = trades
    all_metrics['A_EXIT_T12.5'] = calculate_all_metrics('A_EXIT_T12.5', trades)

    # STRATEGY B: Momentum scalp (enter t=5, exit t=10)
    print("  [3/8] Strategy B: Momentum Scalp (enter t=5 @0.60/0.40, exit t=10)...")
    trades = run_strategy_b(df, entry_bull=0.60, entry_bear=0.40, exit_time="t10")
    all_trades['B_t5→t10'] = trades
    all_metrics['B_t5→t10'] = calculate_all_metrics('B_t5→t10', trades)

    # STRATEGY B variant: enter t=5, exit t=12.5 (longer hold)
    print("  [4/8] Strategy B: Momentum Scalp (enter t=5 @0.60/0.40, exit t=12.5)...")
    trades = run_strategy_b(df, entry_bull=0.60, entry_bear=0.40, exit_time="t12.5")
    all_trades['B_t5→t12.5'] = trades
    all_metrics['B_t5→t12.5'] = calculate_all_metrics('B_t5→t12.5', trades)

    # STRATEGY B variant: enter t=7.5, exit t=12.5
    print("  [5/8] Strategy B: Momentum Scalp (enter t=7.5 @0.65/0.35, exit t=12.5)...")
    trades = run_strategy_b_t7_5(df, entry_bull=0.65, entry_bear=0.35, exit_time="t12.5")
    all_trades['B_t7.5→t12.5'] = trades
    all_metrics['B_t7.5→t12.5'] = calculate_all_metrics('B_t7.5→t12.5', trades)

    # STRATEGY C: Hybrid (tight targets)
    print("  [6/8] Strategy C: Hybrid (target=0.03, stop=0.03)...")
    trades = run_strategy_c(df, profit_target_pm=0.03, stop_loss_pm=0.03)
    all_trades['C_0.03/0.03'] = trades
    all_metrics['C_0.03/0.03'] = calculate_all_metrics('C_0.03/0.03', trades)

    # STRATEGY C variant: wider targets
    print("  [7/8] Strategy C: Hybrid (target=0.05, stop=0.05)...")
    trades = run_strategy_c(df, profit_target_pm=0.05, stop_loss_pm=0.05)
    all_trades['C_0.05/0.05'] = trades
    all_metrics['C_0.05/0.05'] = calculate_all_metrics('C_0.05/0.05', trades)

    # STRATEGY C variant: asymmetric (wider profit, tighter stop)
    print("  [8/8] Strategy C: Hybrid (target=0.05, stop=0.03)...")
    trades = run_strategy_c(df, profit_target_pm=0.05, stop_loss_pm=0.03)
    all_trades['C_0.05/0.03'] = trades
    all_metrics['C_0.05/0.03'] = calculate_all_metrics('C_0.05/0.03', trades)

    print()

    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    print_comparison_report(all_metrics)

    # =========================================================================
    # PM PRICE MOVEMENT ANALYSIS (for context)
    # =========================================================================
    print("=" * 80)
    print("PM PRICE MOVEMENT ANALYSIS")
    print("=" * 80)
    print()
    print("How much does PM YES price move between sample points?")
    print("(This determines how much profit is available for early-exit strategies)")
    print()

    # t=10 to t=12.5 movement (relevant for Strategy A)
    confirmed_bull = df[df['pm_yes_t10'] >= 0.85]
    confirmed_bear = df[df['pm_yes_t10'] <= 0.15]

    if len(confirmed_bull) > 0:
        move_bull = confirmed_bull['pm_yes_t12_5'] - confirmed_bull['pm_yes_t10']
        print(f"BULL confirmed trades (pm_yes@t10 >= 0.85, n={len(confirmed_bull)}):")
        print(f"  PM move t10→t12.5:  mean={move_bull.mean():+.4f}  std={move_bull.std():.4f}  "
              f"min={move_bull.min():+.4f}  max={move_bull.max():+.4f}")
        print(f"  % that went higher:  {(move_bull > 0).sum()}/{len(move_bull)} ({(move_bull > 0).mean()*100:.1f}%)")
        print(f"  % that moved > 0.03: {(move_bull > 0.03).sum()}/{len(move_bull)} ({(move_bull > 0.03).mean()*100:.1f}%)")
        print(f"  % that dropped > 0.05: {(move_bull < -0.05).sum()}/{len(move_bull)} ({(move_bull < -0.05).mean()*100:.1f}%)")
        print()

    if len(confirmed_bear) > 0:
        move_bear = confirmed_bear['pm_yes_t12_5'] - confirmed_bear['pm_yes_t10']
        print(f"BEAR confirmed trades (pm_yes@t10 <= 0.15, n={len(confirmed_bear)}):")
        print(f"  PM move t10→t12.5:  mean={move_bear.mean():+.4f}  std={move_bear.std():.4f}  "
              f"min={move_bear.min():+.4f}  max={move_bear.max():+.4f}")
        print(f"  % that went lower:  {(move_bear < 0).sum()}/{len(move_bear)} ({(move_bear < 0).mean()*100:.1f}%)")
        print(f"  % that moved < -0.03: {(move_bear < -0.03).sum()}/{len(move_bear)} ({(move_bear < -0.03).mean()*100:.1f}%)")
        print(f"  % that rose > 0.05: {(move_bear > 0.05).sum()}/{len(move_bear)} ({(move_bear > 0.05).mean()*100:.1f}%)")
        print()

    # t=5 to t=10 movement (relevant for Strategy B)
    mom_bull = df[df['pm_yes_t5'] >= 0.60]
    mom_bear = df[df['pm_yes_t5'] <= 0.40]

    if len(mom_bull) > 0:
        move_b_bull = mom_bull['pm_yes_t10'] - mom_bull['pm_yes_t5']
        print(f"Momentum BULL entries (pm_yes@t5 >= 0.60, n={len(mom_bull)}):")
        print(f"  PM move t5→t10:  mean={move_b_bull.mean():+.4f}  std={move_b_bull.std():.4f}")
        print(f"  % that went higher: {(move_b_bull > 0).sum()}/{len(move_b_bull)} ({(move_b_bull > 0).mean()*100:.1f}%)")
        print()

    if len(mom_bear) > 0:
        move_b_bear = mom_bear['pm_yes_t10'] - mom_bear['pm_yes_t5']
        print(f"Momentum BEAR entries (pm_yes@t5 <= 0.40, n={len(mom_bear)}):")
        print(f"  PM move t5→t10:  mean={move_b_bear.mean():+.4f}  std={move_b_bear.std():.4f}")
        print(f"  % that went lower: {(move_b_bear < 0).sum()}/{len(move_b_bear)} ({(move_b_bear < 0).mean()*100:.1f}%)")
        print()

    # =========================================================================
    # SAVE JSON
    # =========================================================================
    print("Saving results...")

    # Remove equity curves for JSON (too large)
    json_results = {}
    for name, m in all_metrics.items():
        json_results[name] = {k: v for k, v in m.items() if k != 'equity_curve'}
        # Convert numpy types
        for k, v in json_results[name].items():
            if isinstance(v, (np.integer, np.int64)):
                json_results[name][k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                json_results[name][k] = float(v)

    output_path = OUTPUT_DIR / f"strategy_comparison_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved: {output_path}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    return all_metrics


if __name__ == "__main__":
    main()
