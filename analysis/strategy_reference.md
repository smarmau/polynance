# Momentum-Confirmed Strategy Reference

## Core Discovery: Momentum Confirmation

The single biggest improvement from backtesting: **pm_price_momentum_0_to_5** (price movement in the first 5 minutes of the window) is massively predictive.

**Definition:** `momentum = pm_yes_t5 - pm_yes_t0` (signed)

**Filter logic:**
- Bull trade (buying YES, expecting price up): requires `momentum >= +threshold`
- Bear trade (buying NO, expecting price down): requires `momentum <= -threshold`

This eliminates ~60% of trades but lifts win rate from 55.7% to 73.5%+ by filtering out the ~28% WR disaster trades where momentum runs against the position.

---

## Backtest Results Summary

All strategies below use the **contrarian_consensus** base with:
- `consensus_min_agree: 2` (broader diversification)
- `consensus_entry_time: "t5"` (evaluates after momentum data available)
- `recovery_sizing: "none"` (confirmed toxic in backtests)
- Exit at t12.5 (unless hold_to_resolution)

Baseline (no momentum): 55.7% WR, +$260, -47.2% MaxDD, 1737 trades

### Tier 1: Pure Signal (simplest)

| # | Strategy | Config File | Test WR | Test P&L | Test DD | Key Setting |
|---|----------|-------------|---------|----------|---------|-------------|
| 1 | A1_MOM_0.03 | `config_A1_mom_0.03.json` | 73.5% | +$5,112 | -6.7% | `momentum_min_threshold: 0.03` |
| 2 | A1_MOM_0.05 | `config_A1_mom_0.05.json` | 74.7% | +$4,959 | -7.7% | `momentum_min_threshold: 0.05` |
| 3 | A3_MOM+SWEET | `config_A3_mom_sweet.json` | 72.9% | +$4,997 | -6.7% | + `sweet_spot_band: 0.15` |

### Tier 2: Enhanced Sizing (more return, same base signal)

| # | Strategy | Config File | Test WR | Test P&L | Test DD | Key Setting |
|---|----------|-------------|---------|----------|---------|-------------|
| 4 | C4_CONF_1.5x | `config_C4_confidence_1.5x.json` | 73.5% | +$7,289 | -7.3% | `confidence_scaling_max: 1.5` |
| 5 | C4_CONF_2.0x | `config_C4_confidence_2.0x.json` | 73.5% | +$9,465 | -9.0% | `confidence_scaling_max: 2.0` |
| 6 | D5_ASYM+DAILY | `config_D5_asym_mom_daily.json` | 73.5% | +$5,842 | -10.2% | Bear=1.5x, Bull=0.75x + daily limit |

### Tier 3: Maximum Returns (more aggressive)

| # | Strategy | Config File | Test WR | Test P&L | Test DD | Key Setting |
|---|----------|-------------|---------|----------|---------|-------------|
| 7 | D1_ANTIMART+DAILY | `config_D1_antimart_mom_daily.json` | 73.5% | +$12,798 | -14.6% | `anti_mart_mult: 1.5` + daily limit |
| 8 | C4_CONF_3.0x | `config_C4_confidence_3.0x.json` | 73.5% | +$13,717 | -12.2% | `confidence_scaling_max: 3.0` |
| 9 | D6_KITCHEN_SINK | `config_D6_kitchen_sink.json` | 73.0% | +$12,089 | -15.6% | Everything combined |

### Tier 4: Resolution Hold (different exit approach)

| # | Strategy | Config File | Test WR | Test P&L | Test DD | Key Setting |
|---|----------|-------------|---------|----------|---------|-------------|
| 10 | B3_RESOL+MOM | `config_B3_resolution_mom.json` | 71.6% | +$5,171 | -25.8% | `hold_to_resolution: true` |
| 11 | D2_RESOL+DAILY | `config_D2_resolution_mom_daily.json` | 71.8% | +$4,990 | -17.3% | + `daily_loss_limit: 100` |

---

## Parameter Reference

### Momentum Confirmation (`momentum_min_threshold`)
- **What:** Minimum absolute pm_yes movement from t0 to t5 to confirm trade direction
- **Values:** 0 (disabled), 0.03 (recommended), 0.05 (stricter), 0.10 (very strict)
- **Effect:** Higher = fewer trades, slightly higher WR, lower total P&L
- **Requires:** `consensus_entry_time: "t5"` (needs both t0 and t5 data)

### Confidence Scaling (`confidence_scaling_max`, `confidence_scaling_ref`)
- **What:** Scale bet size proportional to momentum strength
- **Formula:** `scale = 1.0 + min(|momentum| / ref, max - 1.0)`
- **ref = 0.20:** momentum at which max scale is reached
- **Example (1.5x max):**
  - |momentum| = 0.03 -> 1.15x bet
  - |momentum| = 0.10 -> 1.50x bet (capped)
  - |momentum| = 0.20 -> 1.50x bet (capped)
- **Example (2.0x max):**
  - |momentum| = 0.03 -> 1.15x bet
  - |momentum| = 0.10 -> 1.50x bet
  - |momentum| = 0.20 -> 2.00x bet (capped)
- **Rationale:** Strongest momentum signals are the most reliable

### Daily Loss Limit (`daily_loss_limit`)
- **What:** Stop trading for the UTC day after cumulative losses exceed this amount
- **Values:** 0 (disabled), 100 (recommended)
- **Effect:** Caps daily downside, prevents drawdown cascades

### Asymmetric Direction Sizing (`bear_size_mult`, `bull_size_mult`)
- **What:** Direction-specific bet multipliers
- **D5 values:** bear=1.5, bull=0.75
- **Rationale:** Bear reversals (after strong up) historically more reliable

### Anti-Martingale (`anti_mart_mult`, `anti_mart_max`)
- **What:** Scale bet up after consecutive wins
- **Formula:** `scale = anti_mart_mult ^ win_streak` (capped at anti_mart_max)
- **Example (1.5x):**
  - 0 wins: 1.0x
  - 1 win: 1.5x
  - 2 wins: 2.25x
  - 3 wins: 3.375x
  - 4+ wins: 5.0x (capped)
- **Note:** Uses global win streak, not per-asset

### Hold to Resolution (`hold_to_resolution`)
- **What:** Hold position to binary resolution instead of early exit at t12.5
- **Payout:** Win = $1/contract, Lose = $0/contract (vs price spread at t12.5)
- **Effect:** Higher per-trade variance but momentum-confirmed trades have ~71.6% binary WR

### Sweet Spot Filter (`sweet_spot_band`)
- **What:** Require t0 pm_yes within this distance of 0.50
- **Example (0.15):** t0 must be in [0.35, 0.65]
- **Rationale:** Entries near neutral have more room for directional moves

---

## What Worked vs What Didn't

### Worked
- **Momentum confirmation:** THE single best improvement (+20pp WR)
- **Confidence-scaled sizing:** Bet more when momentum is strong -- strongest signals ARE most reliable
- **Daily loss limits:** Valuable insurance on top of momentum
- **Resolution exit:** Higher per-trade payout when momentum confirms (71.6% WR binary = great expectancy)

### Didn't Work
- **Stop-losses at t5/t7.5:** Harmful. With momentum filter, remaining trades rarely go underwater enough to trigger stops
- **Recovery/martingale sizing:** Confirmed toxic -- increasing after losses makes drawdowns worse
- **Kelly sizing:** Edge is real but compounds into absurd unrealistic numbers
- **t5 delayed entry without momentum filter:** Worse than t0

---

## Implementation Architecture

### Data Flow (contrarian_consensus with momentum)

```
t=0.0  -> on_sample_at_consensus_t0()     # Buffer pm_yes_t0 per asset
t=5.0  -> on_sample_at_consensus_entry()   # Buffer pm_yes_t5, evaluate when all 4 arrive
           -> _evaluate_consensus()
              Phase 1: prev window strong? (N assets with pm_t12.5 >= 0.80)
              Phase 2: current confirms reversal? (N assets with pm_t5 confirms dir)
              Phase 2.5: sweet spot? (t0 near 0.50, optional)
              Phase 3: momentum confirms? (pm_t5 - pm_t0 in correct direction)
              -> _enter_consensus_trade() per passing asset
                 Bet = base * confidence_scale * dir_mult * anti_mart_scale
t=12.5 -> on_sample_at_consensus_exit()    # Close position (unless hold_to_resolution)
           OR
window_complete -> _resolve_trade()         # Binary resolution (if hold_to_resolution)
```

### Config Files Location
All strategy configs are in `config/config_*.json`. Run with:
```bash
python -m polynance --config config/config_A1_mom_0.03.json
```

### Recommended Implementation Priority
1. **Start with A1_MOM_0.03** -- simplest, highest risk-adjusted returns (Sharpe)
2. **Graduate to C4_CONF_1.5x** -- same trades, better sizing
3. **Add daily limit** -- safety net for live trading
4. **Consider B3 resolution hold** -- for set-and-forget style
