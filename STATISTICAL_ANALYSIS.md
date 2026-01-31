# Statistical Analysis Guide

## Overview

Polynance now includes proper statistical testing to determine if Polymarket's 15-minute crypto predictions are better than random chance. This document explains the testing approach and how to interpret results.

## Key Principle: Sample Size Matters

**Minimum Sample Size**: 100 windows per asset
**Recommended for Strong Conclusions**: 200+ windows

With fewer than 100 samples, results are **descriptive only** and cannot support statistical conclusions.

## Automatic Hourly Analysis

The system now runs statistical analysis automatically **on the hour** (every 60 minutes).

### What Gets Analyzed

For each asset (BTC, ETH, SOL, XRP), the analyzer computes:

1. **Descriptive Statistics**
   - Total windows collected
   - Directional split (% up vs down)
   - Strong prediction count (YES > 0.6 or < 0.4)

2. **Prediction Accuracy** (N ≥ 10 strong predictions)
   - How often strong predictions were correct
   - Binomial test: Is accuracy significantly different from 50%?

3. **ROC/AUC Analysis** (N ≥ 50)
   - ROC curve and AUC score
   - Optimal prediction threshold
   - Ranges from 0.5 (random) to 1.0 (perfect)

4. **Timing Analysis** (N ≥ 100)
   - Which timepoint (t=2.5, t=5, t=7.5, t=10) is best predictor
   - Compares AUC scores across timepoints

## Statistical Tests Used

### 1. Binomial Test
**When**: You have ≥10 strong predictions (YES > 0.6 or < 0.4)
**Tests**: Is prediction accuracy significantly different from 50% (random chance)?

**Hypotheses**:
- H₀ (Null): Market predictions are random (p = 0.5)
- H₁ (Alternative): Market predictions are better/worse than random

**Interpretation**:
- **p < 0.05**: Reject null hypothesis → Market IS predictive ✓
- **p ≥ 0.05**: Cannot reject null → No evidence market is predictive ✗

**Example**:
```
Strong Predictions: 45
Correct: 32
Accuracy: 71.1%
p-value: 0.0023

✓ SIGNIFICANT - Market shows predictive power (p < 0.05)
```

### 2. ROC AUC (Receiver Operating Characteristic - Area Under Curve)
**When**: You have ≥50 windows with price data
**Tests**: How well do market prices discriminate between up/down outcomes?

**Score Range**:
- **0.5**: Random guessing (coin flip)
- **0.6-0.7**: Weak predictor
- **0.7-0.8**: Moderate predictor
- **0.8-0.9**: Strong predictor
- **0.9-1.0**: Excellent predictor

**Optimal Threshold**:
The analysis finds the best cutoff price (not necessarily 0.5) for making predictions.

**Example**:
```
ROC AUC Score: 0.678
Optimal Threshold: 0.58

Interpretation: Market is a weak-to-moderate predictor.
Using 0.58 instead of 0.50 as threshold may improve accuracy.
```

### 3. Mann-Whitney U Test (Future)
**When**: Comparing two groups (e.g., market prices when outcome was up vs down)
**Purpose**: Non-parametric test that doesn't assume normal distribution

Currently not used in hourly analysis but available in the `statistical_tests.py` module.

## Reports Generated

### 1. Hourly Report Files
**Location**: `data/reports/`

Files created:
- `{asset}_hourly_YYYYMMDD_HHMM.txt` - Timestamped report
- `{asset}_latest.txt` - Always contains most recent analysis
- `{asset}_analysis_log.csv` - CSV log of all analyses

### 2. Console Output
Every hour, you'll see:
```
======================================================================
Running hourly analysis at 2026-01-24 14:00 UTC
======================================================================

[BTC] Analyzing 127 windows...

======================================================================
MARKET PREDICTIVENESS ANALYSIS: BTC
======================================================================

Sample Size: 127 windows
  Up: 68 (53.5%)
  Down: 59 (46.5%)

Strong Predictions (YES > 0.6 or < 0.4): 42
  Correct: 29
  Accuracy: 69.0%

Binomial Test (Is market better than random?):
  Market is significantly predictive (69.0% accuracy, p=0.0156)
  ✓ SIGNIFICANT at p < 0.05

ROC AUC Score: 0.652
  (0.5 = random, 1.0 = perfect)
  Optimal Threshold: 0.547

Best Prediction Timepoint: t7.5

CONCLUSION:
  ✓ Market shows statistically significant predictive power
  Accuracy: 69.0% (p < 0.05)
======================================================================
```

## Dashboard Changes

The "Session Stats" panel is now "Data Collection Progress" and shows:

**Before 100 windows**:
```
BTC: 47 windows collected  [████████░░░░░░░░░░░░] 47%
  Need 53 more for statistical analysis
```

**After 100 windows**:
```
BTC: 127 windows, 54% directionally up
  42 windows with strong signal (>0.6 or <0.4)
  ✓ Ready for statistical analysis
```

This makes it clear that:
- Early stats are just counting, not conclusions
- You need to wait for sufficient data
- System tells you when analysis is meaningful

## Interpreting Results

### Scenario 1: Not Enough Data
```
⚠️  WARNING: Only 27 windows collected
Need at least 100 windows for reliable statistical analysis
Current results are DESCRIPTIVE ONLY, not statistically valid
```

**Action**: Keep collecting data. No conclusions can be drawn.

### Scenario 2: Significant Predictive Power
```
Binomial Test: p = 0.0023
✓ SIGNIFICANT at p < 0.05

ROC AUC Score: 0.734

CONCLUSION:
✓ Market shows statistically significant predictive power
Accuracy: 71% (p < 0.05)
```

**Action**: Market predictions appear useful! Consider:
- Building a trading strategy
- Testing with paper trading
- Collecting more data to confirm stability

### Scenario 3: No Significant Predictive Power
```
Binomial Test: p = 0.4521
✗ Not significant (p = 0.4521)

ROC AUC Score: 0.523

CONCLUSION:
✗ Market does not show significant predictive power
Cannot reject null hypothesis of random predictions
```

**Action**: Market doesn't appear to predict better than random.
- Possible reasons: Too much noise, market inefficiency, our sampling isn't capturing signal
- Continue collecting data to see if patterns emerge

### Scenario 4: Worse Than Random
```
Binomial Test: Market is significantly worse than random (34% accuracy, p=0.0087)
✓ SIGNIFICANT at p < 0.05

CONCLUSION:
✓ Market shows statistically significant predictive power
Accuracy: 34% (p < 0.05)
```

**Action**: Market is wrong consistently! This is actually valuable:
- Consider **inverse** strategy (bet against the market)
- Very unusual - investigate data quality issues first

## Using the Analysis Data

### CSV Analysis Log
Each asset has a growing CSV file: `data/reports/{asset}_analysis_log.csv`

Columns:
- `timestamp`: When analysis was run
- `total_windows`: Sample size
- `up_rate`: Fraction of windows that went up
- `predictions_made`: Strong predictions (>0.6 or <0.4)
- `accuracy`: Prediction accuracy
- `p_value`: Binomial test p-value
- `significant`: 1 if p < 0.05, else 0
- `auc_score`: ROC AUC score
- `optimal_threshold`: Best prediction threshold

**Example Analysis**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load analysis log
df = pd.read_csv('data/reports/btc_analysis_log.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot accuracy over time
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['accuracy'], marker='o')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random (50%)')
plt.xlabel('Time')
plt.ylabel('Prediction Accuracy')
plt.title('BTC Market Prediction Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.savefig('btc_accuracy_trend.png')

# Check if accuracy is improving
print(f"First 5 analyses: {df.head(5)['accuracy'].mean():.1%}")
print(f"Last 5 analyses: {df.tail(5)['accuracy'].mean():.1%}")
```

## Best Practices

### 1. Don't Rush to Conclusions
- Wait for N ≥ 100 before making any claims
- Look for consistency across multiple hourly analyses
- One significant result could be luck; repeated significance is evidence

### 2. Check for Stability
Compare early vs late accuracy:
```bash
# First analysis (N=100)
Accuracy: 68% (p=0.012)

# Later analysis (N=200)
Accuracy: 52% (p=0.421)
```
If significance disappears with more data, it was likely noise.

### 3. Consider Market Conditions
- Volatile markets may be harder to predict
- Check if accuracy correlates with crypto volatility
- Time of day, day of week might matter

### 4. Multiple Testing Correction
If you're testing 4 assets (BTC, ETH, SOL, XRP), you're running 4 parallel tests.

**Bonferroni correction**: Divide significance level by number of tests
- Instead of p < 0.05, use p < 0.0125 (0.05/4)
- More conservative but reduces false positives

## Future Enhancements

The statistical framework supports adding:

1. **Logistic Regression**
   - Multi-variable models: `outcome ~ pm_yes_t5 + pm_momentum + pm_spread`
   - Identify which features matter most

2. **Time Series Analysis**
   - Auto-correlation in predictions
   - Trend detection

3. **Bayesian Analysis**
   - Update beliefs as data accumulates
   - More nuanced than binary significant/not-significant

4. **Cross-Validation**
   - Train/test split to avoid overfitting
   - Out-of-sample accuracy

## Command Line Tools

### Run One-Time Analysis
```bash
# Analyze all assets with current data
python -m polynance.analyze
```

### Check Reports
```bash
# View latest report
cat data/reports/btc_latest.txt

# View analysis log
head -20 data/reports/btc_analysis_log.csv

# Count total windows
sqlite3 data/btc.db "SELECT COUNT(*) FROM windows"
```

## Summary

1. **Collect Data**: Run sampler for days/weeks to get N ≥ 100
2. **Monitor Progress**: Dashboard shows collection progress
3. **Hourly Analysis**: Automatic statistical testing every hour
4. **Read Reports**: Check `data/reports/{asset}_latest.txt`
5. **Look for Significance**: p < 0.05 and AUC > 0.6
6. **Verify Stability**: Results should hold across multiple analyses
7. **Make Decisions**: Only act on statistically validated signals

**Remember**: Statistics don't guarantee profits, but they prevent you from betting on noise!
