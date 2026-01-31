"""
Polynance Strategy Backtest & Statistical Validation

Run this script on new data to validate the trading strategy.
All conclusions are derived from the actual test results.

Usage:
    python backtest_analysis.py

Output:
    - Console output with all test results
    - Saves results to analysis/results_YYYYMMDD_HHMMSS.json
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom
from datetime import datetime
from pathlib import Path
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS = ['btc', 'eth', 'sol', 'xrp']

# Trading parameters
BET_SIZE = 25.0
POLYMARKET_FEE = 0.02
STARTING_BANKROLL = 1000.0

# Strategies to test (name, price_column, bull_threshold, bear_threshold)
STRATEGIES = [
    ('t5_0.60_0.40', 'pm_yes_t5', 0.60, 0.40),
    ('t7.5_0.60_0.40', 'pm_yes_t7_5', 0.60, 0.40),
    ('t7.5_0.75_0.25', 'pm_yes_t7_5', 0.75, 0.25),
    ('t7.5_0.80_0.20', 'pm_yes_t7_5', 0.80, 0.20),
    ('t10_0.60_0.40', 'pm_yes_t10', 0.60, 0.40),
]

# Statistical thresholds
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(STRATEGIES)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all window data from SQLite databases."""
    all_windows = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset}.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            df = pd.read_sql('SELECT * FROM windows WHERE outcome IS NOT NULL', conn)
            conn.close()
            all_windows.append(df)

    if not all_windows:
        raise FileNotFoundError(f"No database files found in {DATA_DIR}")

    df = pd.concat(all_windows, ignore_index=True)
    df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
    df = df.sort_values('window_start_utc').reset_index(drop=True)
    return df


# =============================================================================
# TRADE GENERATION
# =============================================================================

def generate_trades(df, price_col, bull_thresh, bear_thresh):
    """Generate trade list for a strategy."""
    valid = df[df[price_col].notna()].copy()
    trades = []

    for _, row in valid.iterrows():
        price = row[price_col]
        spread = row['pm_spread_t5'] if pd.notna(row.get('pm_spread_t5')) else 0.01
        outcome = row['outcome']

        if price >= bull_thresh:
            entry = price + spread / 2
            win = outcome == 'up'
            if win:
                pnl = (1 - entry) * BET_SIZE * (1 - POLYMARKET_FEE)
            else:
                pnl = -entry * BET_SIZE
            trades.append({
                'win': win,
                'pnl': pnl,
                'signal': 'bull',
                'asset': row['asset'],
                'timestamp': row['window_start_utc'],
                'price': price,
            })

        elif price < bear_thresh:
            entry = (1 - price) + spread / 2
            win = outcome == 'down'
            if win:
                pnl = (1 - entry) * BET_SIZE * (1 - POLYMARKET_FEE)
            else:
                pnl = -entry * BET_SIZE
            trades.append({
                'win': win,
                'pnl': pnl,
                'signal': 'bear',
                'asset': row['asset'],
                'timestamp': row['window_start_utc'],
                'price': price,
            })

    return pd.DataFrame(trades)


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def binomial_test(wins, n, null_prob=0.5):
    """
    One-sided binomial test.
    H0: true win rate = null_prob
    H1: true win rate > null_prob
    Returns p-value.
    """
    # P(X >= wins) under null hypothesis
    p_value = 1 - binom.cdf(wins - 1, n, null_prob)
    return p_value


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """
    Bootstrap confidence interval for mean.
    Returns (lower, upper, point_estimate).
    """
    data = np.array(data)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(sample.mean())

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    return lower, upper, data.mean()


def permutation_test(df, price_col, bull_thresh, bear_thresh, n_permutations=1000):
    """
    Permutation test for strategy edge.
    Shuffles outcomes and compares to actual win rate.
    Returns (actual_wr, mean_shuffled_wr, std_shuffled_wr, p_value).
    """
    valid = df[df[price_col].notna()].copy()
    bull_mask = valid[price_col] >= bull_thresh
    bear_mask = valid[price_col] < bear_thresh
    trade_mask = bull_mask | bear_mask

    trade_df = valid[trade_mask].copy()
    if len(trade_df) == 0:
        return None, None, None, None

    outcomes = (trade_df['outcome'] == 'up').values
    prices = trade_df[price_col].values
    is_bull = prices >= bull_thresh

    # Actual win rate
    actual_wins = (is_bull & outcomes) | (~is_bull & ~outcomes)
    actual_wr = actual_wins.mean()

    # Permutation distribution
    permuted_wrs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(outcomes)
        wins = (is_bull & shuffled) | (~is_bull & ~shuffled)
        permuted_wrs.append(wins.mean())

    permuted_wrs = np.array(permuted_wrs)
    p_value = (permuted_wrs >= actual_wr).sum() / n_permutations

    return actual_wr, permuted_wrs.mean(), permuted_wrs.std(), p_value


def runs_test(sequence):
    """
    Wald-Wolfowitz runs test for randomness.
    Returns (n_runs, expected_runs, z_stat, p_value).
    """
    sequence = np.array(sequence).astype(int)
    n = len(sequence)
    n1 = sequence.sum()
    n0 = n - n1

    if n0 == 0 or n1 == 0:
        return None, None, None, None

    # Count runs
    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i-1]:
            runs += 1

    # Expected runs and variance under null
    expected = (2 * n0 * n1) / n + 1
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))

    if var <= 0:
        return runs, expected, None, None

    z = (runs - expected) / np.sqrt(var)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return runs, expected, z, p_value


# =============================================================================
# BACKTEST WITH SIZING
# =============================================================================

def backtest_flat(trades_df, bet_size=BET_SIZE):
    """Run backtest with flat bet sizing."""
    if len(trades_df) == 0:
        return None

    trades = trades_df.copy()
    trades['cumulative_pnl'] = trades['pnl'].cumsum()
    trades['bankroll'] = STARTING_BANKROLL + trades['cumulative_pnl']

    peak = trades['bankroll'].cummax()
    drawdown = trades['bankroll'] - peak

    return {
        'n_trades': len(trades),
        'wins': trades['win'].sum(),
        'win_rate': trades['win'].mean(),
        'total_pnl': trades['pnl'].sum(),
        'max_drawdown': drawdown.min(),
        'max_drawdown_pct': (drawdown / peak).min() * 100,
        'sharpe': (trades['pnl'].mean() / trades['pnl'].std() * np.sqrt(252 * 96)) if trades['pnl'].std() > 0 else 0,
        'avg_pnl': trades['pnl'].mean(),
        'std_pnl': trades['pnl'].std(),
    }


def backtest_antimartingale(trades_df, base_bet=25, win_mult=2.0, loss_mult=0.5, max_pct=0.05):
    """Run backtest with anti-martingale sizing."""
    if len(trades_df) == 0:
        return None

    bankroll = STARTING_BANKROLL
    current_bet = base_bet
    results = []

    for _, row in trades_df.iterrows():
        bet_size = max(1, min(current_bet, bankroll * max_pct))

        # Scale PnL by bet size ratio
        pnl_ratio = bet_size / BET_SIZE
        pnl = row['pnl'] * pnl_ratio

        bankroll += pnl
        results.append({'pnl': pnl, 'bankroll': bankroll, 'bet_size': bet_size})

        if row['win']:
            current_bet = min(base_bet * win_mult, bankroll * max_pct)
        else:
            current_bet = base_bet * loss_mult

        if bankroll <= 0:
            break

    results_df = pd.DataFrame(results)
    peak = results_df['bankroll'].cummax()
    drawdown = results_df['bankroll'] - peak

    return {
        'n_trades': len(results_df),
        'total_pnl': results_df['pnl'].sum(),
        'final_bankroll': results_df['bankroll'].iloc[-1],
        'max_drawdown': drawdown.min(),
        'max_drawdown_pct': (drawdown / peak).min() * 100,
        'sharpe': (results_df['pnl'].mean() / results_df['pnl'].std() * np.sqrt(252 * 96)) if results_df['pnl'].std() > 0 else 0,
        'avg_bet': results_df['bet_size'].mean(),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis():
    """Run complete analysis and return results dictionary."""

    print("=" * 80)
    print("POLYNANCE STRATEGY ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    df = load_data()

    print(f"Total windows: {len(df)}")
    print(f"Date range: {df['window_start_utc'].min()} to {df['window_start_utc'].max()}")
    print(f"Assets: {df['asset'].unique().tolist()}")
    print()

    # Sample balance
    up_count = (df['outcome'] == 'up').sum()
    down_count = (df['outcome'] == 'down').sum()
    print(f"Outcome balance: UP {up_count} ({up_count/len(df)*100:.1f}%), DOWN {down_count} ({down_count/len(df)*100:.1f}%)")
    print()

    results = {
        'run_timestamp': datetime.now().isoformat(),
        'data': {
            'total_windows': len(df),
            'date_range': [str(df['window_start_utc'].min()), str(df['window_start_utc'].max())],
            'up_count': int(up_count),
            'down_count': int(down_count),
        },
        'strategies': {},
    }

    # ==========================================================================
    # STRATEGY ANALYSIS
    # ==========================================================================

    print("=" * 80)
    print("STRATEGY PERFORMANCE")
    print("=" * 80)
    print()

    for name, price_col, bull_thresh, bear_thresh in STRATEGIES:
        print(f"--- {name} ---")

        trades = generate_trades(df, price_col, bull_thresh, bear_thresh)

        if len(trades) == 0:
            print("  No trades generated\n")
            continue

        # Basic stats
        n = len(trades)
        wins = int(trades['win'].sum())
        win_rate = wins / n

        print(f"  Trades: {n}")
        print(f"  Wins: {wins} ({win_rate*100:.1f}%)")

        # Backtest results
        bt_flat = backtest_flat(trades)
        bt_am = backtest_antimartingale(trades)

        print(f"  Flat $25: PnL=${bt_flat['total_pnl']:.2f}, MaxDD={bt_flat['max_drawdown_pct']:.1f}%, Sharpe={bt_flat['sharpe']:.1f}")
        print(f"  Anti-Martingale: PnL=${bt_am['total_pnl']:.2f}, MaxDD={bt_am['max_drawdown_pct']:.1f}%, Sharpe={bt_am['sharpe']:.1f}")

        # Store results
        results['strategies'][name] = {
            'config': {
                'price_col': price_col,
                'bull_thresh': bull_thresh,
                'bear_thresh': bear_thresh,
            },
            'n_trades': n,
            'wins': wins,
            'win_rate': win_rate,
            'backtest_flat': bt_flat,
            'backtest_am': bt_am,
            'tests': {},
        }

        print()

    # ==========================================================================
    # STATISTICAL TESTS
    # ==========================================================================

    print("=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)
    print()

    for name, price_col, bull_thresh, bear_thresh in STRATEGIES:
        if name not in results['strategies']:
            continue

        print(f"--- {name} ---")

        trades = generate_trades(df, price_col, bull_thresh, bear_thresh)
        n = len(trades)
        wins = int(trades['win'].sum())
        win_rate = wins / n

        strat_results = results['strategies'][name]

        # 1. Binomial test
        p_binom = binomial_test(wins, n, 0.5)
        sig_binom = p_binom < ALPHA
        sig_bonf = p_binom < BONFERRONI_ALPHA

        print(f"  Binomial test: p={p_binom:.2e}, significant={sig_binom}, significant_bonferroni={sig_bonf}")

        strat_results['tests']['binomial'] = {
            'p_value': p_binom,
            'significant_at_0.05': sig_binom,
            'significant_bonferroni': sig_bonf,
        }

        # 2. Bootstrap CI
        ci_lower, ci_upper, ci_mean = bootstrap_ci(trades['win'].astype(int).values)

        print(f"  Bootstrap 95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")

        strat_results['tests']['bootstrap_ci'] = {
            'lower': ci_lower,
            'upper': ci_upper,
            'mean': ci_mean,
            'lower_above_50': ci_lower > 0.5,
        }

        # 3. Train/test split
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        train_trades = generate_trades(train_df, price_col, bull_thresh, bear_thresh)
        test_trades = generate_trades(test_df, price_col, bull_thresh, bear_thresh)

        if len(train_trades) > 0 and len(test_trades) > 0:
            train_wr = train_trades['win'].mean()
            test_wr = test_trades['win'].mean()
            wr_diff = test_wr - train_wr

            print(f"  Train/Test: train={train_wr*100:.1f}%, test={test_wr*100:.1f}%, diff={wr_diff*100:+.1f}%")

            strat_results['tests']['train_test'] = {
                'train_win_rate': train_wr,
                'test_win_rate': test_wr,
                'difference': wr_diff,
                'test_degrades': wr_diff < -0.05,
            }

        # 4. Permutation test
        actual_wr, mean_perm, std_perm, p_perm = permutation_test(df, price_col, bull_thresh, bear_thresh, n_permutations=1000)

        if p_perm is not None:
            z_score = (actual_wr - mean_perm) / std_perm if std_perm > 0 else 0
            print(f"  Permutation test: actual={actual_wr*100:.1f}%, shuffled={mean_perm*100:.1f}%±{std_perm*100:.1f}%, z={z_score:.1f}, p={p_perm:.4f}")

            strat_results['tests']['permutation'] = {
                'actual_win_rate': actual_wr,
                'shuffled_mean': mean_perm,
                'shuffled_std': std_perm,
                'z_score': z_score,
                'p_value': p_perm,
                'significant': p_perm < ALPHA,
            }

        # 5. K-fold cross-validation
        n_folds = 5
        fold_size = len(df) // n_folds
        fold_wrs = []

        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else len(df)
            fold_df = df.iloc[start:end]
            fold_trades = generate_trades(fold_df, price_col, bull_thresh, bear_thresh)
            if len(fold_trades) > 0:
                fold_wrs.append(fold_trades['win'].mean())

        if fold_wrs:
            fold_std = np.std(fold_wrs)
            print(f"  K-fold CV (5 folds): mean={np.mean(fold_wrs)*100:.1f}%, std={fold_std*100:.1f}%")

            strat_results['tests']['kfold'] = {
                'fold_win_rates': fold_wrs,
                'mean': np.mean(fold_wrs),
                'std': fold_std,
                'high_variance': fold_std > 0.05,
            }

        # 6. Runs test
        runs, expected, z_runs, p_runs = runs_test(trades['win'].values)

        if p_runs is not None:
            print(f"  Runs test: runs={runs}, expected={expected:.1f}, z={z_runs:.2f}, p={p_runs:.4f}")

            strat_results['tests']['runs'] = {
                'observed_runs': runs,
                'expected_runs': expected,
                'z_stat': z_runs,
                'p_value': p_runs,
                'clustering_detected': p_runs < ALPHA,
            }

        # 7. Per-asset analysis
        asset_results = {}
        for asset in ASSETS:
            asset_df = df[df['asset'] == asset.upper()]
            asset_trades = generate_trades(asset_df, price_col, bull_thresh, bear_thresh)
            if len(asset_trades) > 10:
                asset_wins = int(asset_trades['win'].sum())
                asset_n = len(asset_trades)
                asset_wr = asset_wins / asset_n
                asset_p = binomial_test(asset_wins, asset_n, 0.5)
                asset_results[asset.upper()] = {
                    'n': asset_n,
                    'wins': asset_wins,
                    'win_rate': asset_wr,
                    'p_value': asset_p,
                    'significant': asset_p < ALPHA,
                }

        print(f"  Per-asset: " + ", ".join([f"{k}={v['win_rate']*100:.1f}%" for k, v in asset_results.items()]))

        strat_results['tests']['per_asset'] = asset_results

        # 8. Expected value t-test
        pnls = trades['pnl'].values
        t_stat, p_ttest = stats.ttest_1samp(pnls, 0)
        p_ttest_onesided = p_ttest / 2 if t_stat > 0 else 1 - p_ttest / 2

        print(f"  EV t-test: mean=${pnls.mean():.2f}, t={t_stat:.2f}, p={p_ttest_onesided:.4f}")

        strat_results['tests']['ev_ttest'] = {
            'mean_pnl': pnls.mean(),
            'std_pnl': pnls.std(),
            't_stat': t_stat,
            'p_value_onesided': p_ttest_onesided,
            'significant': p_ttest_onesided < ALPHA and pnls.mean() > 0,
        }

        print()

    # ==========================================================================
    # CALIBRATION ANALYSIS
    # ==========================================================================

    print("=" * 80)
    print("MARKET CALIBRATION")
    print("=" * 80)
    print()

    # Use t7.5 prices
    valid = df[df['pm_yes_t7_5'].notna()].copy()
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    calibration = []
    print(f"{'Bin':<12} {'N':>8} {'Predicted':>12} {'Actual':>12} {'Error':>10}")
    print("-" * 55)

    for i in range(len(bins) - 1):
        mask = (valid['pm_yes_t7_5'] >= bins[i]) & (valid['pm_yes_t7_5'] < bins[i + 1])
        subset = valid[mask]

        if len(subset) > 20:
            predicted = (bins[i] + bins[i + 1]) / 2
            actual = (subset['outcome'] == 'up').mean()
            error = actual - predicted

            calibration.append({
                'bin': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                'n': len(subset),
                'predicted': predicted,
                'actual': actual,
                'error': error,
            })

            print(f"{bins[i]:.1f}-{bins[i+1]:.1f}   {len(subset):>8} {predicted*100:>11.1f}% {actual*100:>11.1f}% {error*100:>+9.1f}%")

    results['calibration'] = calibration

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    for name in results['strategies']:
        strat = results['strategies'][name]
        tests = strat['tests']

        print(f"{name}:")
        print(f"  Win rate: {strat['win_rate']*100:.1f}% (n={strat['n_trades']})")

        # Count passed tests
        passed = 0
        total = 0

        if 'binomial' in tests:
            total += 1
            if tests['binomial']['significant_at_0.05']:
                passed += 1

        if 'bootstrap_ci' in tests:
            total += 1
            if tests['bootstrap_ci']['lower_above_50']:
                passed += 1

        if 'train_test' in tests:
            total += 1
            if not tests['train_test']['test_degrades']:
                passed += 1

        if 'permutation' in tests:
            total += 1
            if tests['permutation']['significant']:
                passed += 1

        if 'kfold' in tests:
            total += 1
            if not tests['kfold']['high_variance']:
                passed += 1

        if 'ev_ttest' in tests:
            total += 1
            if tests['ev_ttest']['significant']:
                passed += 1

        print(f"  Tests passed: {passed}/{total}")
        print()

    # Save results
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results = convert_types(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_analysis()
