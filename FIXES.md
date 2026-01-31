# Polynance Crash Fix - Summary

## Problem

Your Polynance tracker crashed after collecting only 2 samples in the second window. The application kept running for ~575 minutes but stopped collecting data entirely.

### What Happened:
1. Successfully collected 6 samples for the 10:30-10:45 window
2. Started the 10:45-11:00 window and collected 2 samples (at t=0.0 and t=2.5)
3. Stopped collecting samples entirely around 10:46:27
4. Application kept running but with no new data

### Root Causes:
1. **Markets not refreshed on window transitions** - CRITICAL BUG: Each 15-minute window has a unique market, but the sampler was reusing the old market from initialization
2. **Silent API failures** - Network timeouts, rate limiting, or connection errors that weren't being logged properly without `-v` verbose mode
3. **No timeout handling** - HTTP requests had no timeout limits and could hang indefinitely
4. **No retry logic** - Single failures would cause sample collection to fail permanently
5. **No health monitoring** - No way to detect when sampling had stalled

## Fixes Applied

### 1. Market Refresh on Window Transitions ⭐ CRITICAL

**The Problem:**
Polymarket creates a **new market for each 15-minute window**. The market URL includes an epoch timestamp:
```
btc-updown-15m-1769220000  # 02:00 window
btc-updown-15m-1769220900  # 02:15 window
```

The sampler was fetching markets ONCE at initialization, then reusing that same market forever. When the 15-minute window changed, it tried to get prices from the old market, which no longer had active order books, causing all API calls to fail silently.

**The Fix:**
Now on every window transition, the sampler:
1. Finalizes the old window
2. **Fetches a fresh market for the new window**
3. Starts collecting samples with the new market

**Code Change** (`sampler.py:170-196`):
```python
# IMPORTANT: Fetch new market for this window!
logger.info(f"[{asset}] Fetching market for new window starting at {window_start.strftime('%H:%M')}")
markets = await self.polymarket.find_active_15min_markets([asset])

if markets and len(markets) > 0:
    state.market = markets[0]
    logger.info(f"[{asset}] Found market: {state.market.condition_id[:20]}...")
else:
    logger.warning(f"[{asset}] No market found for this window - will retry next iteration")
    state.market = None
```

This was the **PRIMARY cause** of the crash - the sampler would work for one window, then fail forever on the next.

### 2. HTTP Client Improvements (Polymarket & Binance)

**Connection Settings:**
- Added 10-second total timeout, 5-second connect timeout
- Connection pool limits: 100 total, 30 per host
- DNS caching (300s TTL)

**Retry Logic:**
- Up to 3 retry attempts for failed requests
- Exponential backoff for rate limits (429 errors)
- Proper timeout handling with retries
- Better error logging with attempt counts

**Files Modified:**
- `src/polynance/clients/polymarket.py` - Added timeouts, retries to `get_order_book()`
- `src/polynance/clients/binance.py` - Added timeouts, retries to `get_price()`

### 2. Sampler Health Monitoring

**Health Checks:**
- Logs health status every 10 iterations (~5 minutes)
- Tracks total samples collected
- Shows time since last successful sample per asset
- Warns about assets with no samples

**Stall Detection:**
- Counts consecutive failures
- After 5 consecutive failures (~2.5 minutes), logs error alert:
  ```
  ⚠️  SAMPLING STALLED: No samples collected for 5 consecutive iterations
  ```

**Better Error Tracking:**
- `_collect_samples()` now returns count of successful samples
- `_collect_asset_sample()` returns bool indicating success/failure
- Full exception stack traces with `exc_info=True`

**Files Modified:**
- `src/polynance/sampler.py` - Added health tracking, stall detection, improved error handling

## How to Use

### 1. Run the Diagnostic Script

Test your API connectivity before running the full sampler:

```bash
python diagnose.py
```

This will test:
- Polymarket API connectivity for all assets
- Binance API connectivity for all assets
- Concurrent load simulation (4 assets simultaneously)
- Response times and error rates

### 2. Always Use Verbose Mode During Testing

```bash
python -m polynance -v
```

This enables DEBUG-level logging so you can see:
- Every API request and response
- Timing information
- Connection issues
- Retry attempts

### 3. Monitor Health Checks

The sampler now logs health checks every ~5 minutes:

```
============================================================
Health Check - Iteration 10
Total samples collected: 40
Consecutive failures: 0
  BTC: Last sample 0.5m ago
  ETH: Last sample 0.5m ago
  SOL: Last sample 0.5m ago
  XRP: Last sample 0.5m ago
============================================================
```

### 4. Watch for Stall Alerts

If sampling stops working, you'll see:

```
⚠️  SAMPLING STALLED: No samples collected for 5 consecutive iterations (150s). Check API connectivity!
```

## Testing the Fixes

1. **Run diagnostics:**
   ```bash
   python diagnose.py
   ```

2. **Start sampler with verbose logging:**
   ```bash
   python -m polynance -v
   ```

3. **Let it run for at least 1 complete window (15 minutes)**

4. **Check the data:**
   ```bash
   sqlite3 data/btc.db "SELECT window_id, COUNT(*) FROM samples GROUP BY window_id"
   ```

5. **Look for health check logs** - Should appear every 5 minutes

## What Changed

### Resilience Improvements:
✅ HTTP timeouts prevent indefinite hangs
✅ Automatic retries recover from transient failures
✅ Rate limit handling with exponential backoff
✅ Connection pooling prevents connection exhaustion
✅ Health monitoring detects stalled sampling
✅ Better error logging with stack traces
✅ Success/failure tracking per asset

### New Monitoring:
✅ Health checks every ~5 minutes
✅ Consecutive failure counter
✅ Per-asset last sample time tracking
✅ Stall detection and alerting
✅ Sample collection success rate

## Expected Behavior Now

**Normal Operation:**
- Every 30s: Attempts to collect samples for all assets
- Every ~5 min: Logs health check with status
- Transient failures: Automatically retried up to 3 times
- Rate limits: Exponential backoff, then retry

**During Problems:**
- API timeouts: Warning logged, retry attempted
- Connection errors: Error logged with stack trace, retry attempted
- 5 consecutive failures: Stall alert logged
- Verbose mode: Full debug details of every request

**You Should Now See:**
- Clear error messages when APIs fail
- Automatic recovery from transient issues
- Early warning when sampling stalls
- Regular health status updates

## Next Steps

1. Run `python diagnose.py` to verify API connectivity
2. Start the sampler with `-v` and monitor for at least 30 minutes
3. Check that health logs appear every ~5 minutes
4. Verify samples are being collected consistently
5. Once stable, you can remove `-v` flag for cleaner logs (but stall alerts will still appear)
