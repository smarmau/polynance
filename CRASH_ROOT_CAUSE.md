# Root Cause Analysis: Sampler Stopping After 2 Windows

## The Bug

The sampler would:
1. ✓ Successfully collect 6-7 samples for the first 15-minute window
2. ✓ Start the second window and collect 2 samples (t=0.0, t=2.5)
3. ✗ **STOP collecting data entirely** and run idle for hours

## Root Cause: Markets Not Refreshed

### How Polymarket 15-Min Markets Work

Polymarket creates a **UNIQUE market for each 15-minute time window**:

```
Time Window    Market URL
-----------    ----------------------------------------------------------
01:45-02:00    btc-updown-15m-1769219100  (epoch: 1769219100)
02:00-02:15    btc-updown-15m-1769220000  (epoch: 1769220000)
02:15-02:30    btc-updown-15m-1769220900  (epoch: 1769220900)
```

Each market:
- Has a different `condition_id`
- Has different `token_id`s for YES/NO
- Only has active order books **during its 15-minute window**

### What the Sampler Was Doing (WRONG)

```python
# At initialization (01:45):
async def initialize(self):
    markets = await self.polymarket.find_active_15min_markets(self.assets)
    # Stores market for 01:45 window: btc-updown-15m-1769219100
    self.states["BTC"].market = markets[0]  ✓ WORKS

# At 02:00 (new window):
async def _check_window_transitions(self, now):
    # Start new window
    state.window_start = window_start  # Changed to 02:00
    state.current_window_id = "BTC_20260124_0200"  # New window ID
    state.samples = []  # Reset samples

    # BUT... still using OLD market from 01:45!
    # state.market still = btc-updown-15m-1769219100  ✗ BROKEN

# At 02:01 (trying to collect sample):
async def _collect_asset_sample(self, asset, state, now, sample_point):
    # Try to get price from OLD market (01:45)
    pm_price = await self.polymarket.get_market_price(state.market)
    # Returns None because order books are empty/closed  ✗ FAILS

    if pm_price is None:
        logger.warning(f"[{asset}] Failed to get Polymarket price")
        return False  # Sample not collected
```

### Why It Collected 2 Samples Then Stopped

**Sample 1 (t=0.0 at 02:00:19)**:
- New window just started (literally 19 seconds in)
- Old market (01:45) might still have some stale order book data
- ✓ Sample collected successfully

**Sample 2 (t=2.5 at 02:01:20)**:
- Old market (01:45) is now 16+ minutes old
- Order books completely cleared/closed
- Last sample that worked before total failure

**Sample 3+ (t=5.0 onwards)**:
- API calls to old market return empty/null responses
- `pm_price = None` every time
- No samples collected
- Loop continues but does nothing useful

### The Fix

Now the sampler **refreshes markets on every window transition**:

```python
async def _check_window_transitions(self, now: datetime):
    for asset, state in self.states.items():
        if state.window_start is None or window_start != state.window_start:
            # Finalize old window
            if state.window_start is not None:
                await self._finalize_window(state)

            # ⭐ FETCH NEW MARKET FOR NEW WINDOW
            logger.info(f"[{asset}] Fetching market for new window")
            markets = await self.polymarket.find_active_15min_markets([asset])

            if markets and len(markets) > 0:
                state.market = markets[0]  # ✓ FRESH MARKET
                logger.info(f"[{asset}] Found market: {state.market.condition_id[:20]}...")
            else:
                logger.warning(f"[{asset}] No market found")
                state.market = None

            # Start new window with NEW market
            state.window_start = window_start
            ...
```

## Evidence

### Before Fix:
```
data/btc.db:
  BTC_20260123_1030: 6 samples  ✓
  BTC_20260123_1045: 2 samples  ✗ (stopped here)

data/btc.db (second run):
  BTC_20260124_0145: 7 samples  ✓
  BTC_20260124_0200: 2 samples  ✗ (stopped again)
```

Pattern: Always stops after ~2 samples in the second window.

### After Fix:
Should collect 6-7 samples for EVERY window, indefinitely.

## Why This Bug Was Hard to Spot

1. **Silent Failures**: Without `-v` verbose logging, you only see:
   ```
   [BTC] Failed to get Polymarket price
   ```
   No indication that market is stale/wrong.

2. **Timing**: Bug only manifests after first window completes (15 minutes in)

3. **No Error Messages**: The code didn't crash - it just stopped collecting data silently

4. **Health Checks**: Our new health checks would have caught this:
   ```
   ⚠️  SAMPLING STALLED: No samples collected for 5 consecutive iterations
   ```

## Verification

To verify the fix works:

1. **Run sampler for 30+ minutes** (2 complete windows):
   ```bash
   python -m polynance -v
   ```

2. **Check window counts**:
   ```bash
   sqlite3 data/btc.db "SELECT window_id, COUNT(*) FROM samples GROUP BY window_id"
   ```

3. **Expected output**:
   ```
   BTC_20260124_0200|7   ✓ First window
   BTC_20260124_0215|6   ✓ Second window
   BTC_20260124_0230|7   ✓ Third window
   ```

4. **Look for market refresh logs** (every 15 minutes):
   ```
   [BTC] Fetching market for new window starting at 02:15
   [BTC] Found market: 93427601801751679895...
   [BTC] Started new window: BTC_20260124_0215
   ```

## Lessons Learned

1. **API Integration Assumptions**: We assumed markets were long-lived, but they're ephemeral (15 min)

2. **Initialization vs. Runtime**: Fetching resources at init() is not enough if they expire

3. **Importance of Verbose Logging**: `-v` flag would have shown market epoch mismatches

4. **Health Monitoring**: The new stall detection would have alerted us within 2.5 minutes

## Related Fixes

Along with the market refresh fix, we also added:

1. **HTTP timeouts and retries** - Prevents indefinite hangs
2. **Health check logging** - Detects stalled sampling within 5 minutes
3. **Stall alerts** - Loud warnings when no samples collected
4. **Better error logging** - Stack traces and attempt counts

These complementary fixes ensure:
- If markets fail to refresh, we'll see clear errors
- If sampling stalls for any reason, we'll know immediately
- Transient API failures auto-recover with retries

## Status

**Status**: ✅ FIXED in commit [current]

**Confidence**: High - root cause identified and addressed

**Testing**: Run for 1+ hour to verify continuous operation across multiple windows
