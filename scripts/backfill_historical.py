#!/usr/bin/env python3
"""
Historical backfill: fetch resolved Polymarket 15-min crypto windows
from the last recorded window up to now, and insert into the DB.

Uses:
  - Polymarket Gamma API to find markets by epoch slug
  - Polymarket CLOB prices-history for 1-min PM price data
  - Binance public klines API for spot price data
  - Backfill script (scripts/backfill_db.py) logic to populate prev/prev2 columns

Run from project root:
  source venv/bin/activate && python scripts/backfill_historical.py
"""

import asyncio
import aiohttp
import sqlite3
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from polynance.clients.polymarket import PolymarketClient, CRYPTO_15MIN_SLUGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / 'data'
ASSETS = ['BTC', 'ETH', 'SOL', 'XRP']

BINANCE_BASE = 'https://api.binance.com'
BINANCE_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT', 'XRP': 'XRPUSDT',
}

SAMPLE_MINUTES = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]


def get_window_boundaries(dt: datetime):
    minute = (dt.minute // 15) * 15
    start = dt.replace(minute=minute, second=0, microsecond=0)
    end = start + timedelta(minutes=15)
    return start, end


def get_window_id(asset: str, window_start: datetime) -> str:
    return f"{asset}_{window_start.strftime('%Y%m%d_%H%M')}"


def get_window_time(window_start: datetime) -> str:
    return window_start.strftime('%Y%m%d_%H%M')


async def fetch_pm_prices(session: aiohttp.ClientSession, token_id: str,
                          start_ts: int, end_ts: int) -> dict:
    """Fetch price history for a token. Returns dict: unix_ts → price.

    startTs/endTs are required by the CLOB prices-history endpoint.
    """
    url = 'https://clob.polymarket.com/prices-history'
    params = {'market': token_id, 'fidelity': 1,
              'startTs': start_ts - 120, 'endTs': end_ts + 120}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
                result = {}
                for point in data.get('history', []):
                    ts = int(point['t'])
                    result[ts] = float(point['p'])
                return result
            else:
                logger.warning(f"PM prices-history {resp.status} for {token_id[:12]}...")
    except Exception as e:
        logger.warning(f"PM price fetch failed for {token_id}: {e}")
    return {}


async def fetch_binance_klines(session: aiohttp.ClientSession, symbol: str,
                                start_ms: int, end_ms: int) -> list:
    """Fetch 1-minute Binance klines for the window period."""
    url = f'{BINANCE_BASE}/api/v3/klines'
    params = {
        'symbol': symbol, 'interval': '1m',
        'startTime': start_ms, 'endTime': end_ms,
        'limit': 20,
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data
    except Exception as e:
        logger.warning(f"Binance klines failed for {symbol}: {e}")
    return []


def sample_pm_price_at(price_history: dict, window_start: datetime, t_minutes: float,
                        tolerance_secs: int = 90) -> float | None:
    """Get PM price nearest to window_start + t_minutes, within tolerance_secs."""
    if not price_history:
        return None
    target_ts = int((window_start + timedelta(minutes=t_minutes)).timestamp())
    best_ts = min(price_history.keys(), key=lambda ts: abs(ts - target_ts))
    if abs(best_ts - target_ts) <= tolerance_secs:
        return price_history[best_ts]
    return None


def compute_spot_data(klines: list, window_start: datetime):
    """Extract spot open/close/high/low from klines for the 15-min window."""
    if not klines:
        return None, None, None, None

    window_end = window_start + timedelta(minutes=15)
    ws_ms = int(window_start.timestamp() * 1000)
    we_ms = int(window_end.timestamp() * 1000)

    relevant = [k for k in klines if ws_ms <= k[0] < we_ms]
    if not relevant:
        # Fallback: use all provided klines
        relevant = klines

    try:
        spot_open = float(relevant[0][1])   # open of first candle
        spot_close = float(relevant[-1][4])  # close of last candle
        spot_high = max(float(k[2]) for k in relevant)
        spot_low = min(float(k[3]) for k in relevant)
        return spot_open, spot_close, spot_high, spot_low
    except (IndexError, ValueError):
        return None, None, None, None


def get_existing_windows(db_path: Path) -> set:
    """Get set of window_ids already in the DB."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT window_id FROM windows").fetchall()
    conn.close()
    return {r[0] for r in rows}


def insert_window(db_path: Path, asset: str, window_data: dict):
    """Insert a window record into the DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            INSERT OR REPLACE INTO windows (
                window_id, asset, window_start_utc, window_end_utc,
                resolved_at_utc, outcome, outcome_binary,
                spot_open, spot_close, spot_change_pct, spot_change_bps,
                spot_high, spot_low, spot_range_bps,
                pm_yes_t0, pm_yes_t2_5, pm_yes_t5, pm_yes_t7_5, pm_yes_t10, pm_yes_t12_5,
                pm_spread_t0, pm_spread_t5,
                pm_price_momentum_0_to_5, pm_price_momentum_5_to_10,
                volatility_regime, window_time
            ) VALUES (
                :window_id, :asset, :window_start_utc, :window_end_utc,
                :resolved_at_utc, :outcome, :outcome_binary,
                :spot_open, :spot_close, :spot_change_pct, :spot_change_bps,
                :spot_high, :spot_low, :spot_range_bps,
                :pm_yes_t0, :pm_yes_t2_5, :pm_yes_t5, :pm_yes_t7_5, :pm_yes_t10, :pm_yes_t12_5,
                :pm_spread_t0, :pm_spread_t5,
                :pm_price_momentum_0_to_5, :pm_price_momentum_5_to_10,
                :volatility_regime, :window_time
            )
        """, window_data)
        conn.commit()
    finally:
        conn.close()


def backfill_prev_columns(db_path: Path):
    """Recompute prev_pm_t12_5 and prev2_pm_t12_5 for all windows."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Add columns if missing
    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(windows)").fetchall()]
    for col in ['prev_pm_t12_5', 'prev2_pm_t12_5']:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE windows ADD COLUMN {col} REAL")

    rows = conn.execute(
        "SELECT window_id, pm_yes_t12_5 FROM windows ORDER BY window_start_utc ASC"
    ).fetchall()

    prev_pm = None
    prev2_pm = None
    for row in rows:
        wid = row['window_id']
        pm_12_5 = row['pm_yes_t12_5']
        conn.execute(
            "UPDATE windows SET prev_pm_t12_5 = ?, prev2_pm_t12_5 = ? WHERE window_id = ?",
            (prev_pm, prev2_pm, wid)
        )
        prev2_pm = prev_pm
        prev_pm = pm_12_5

    conn.commit()
    conn.close()


async def process_asset(session: aiohttp.ClientSession, pm_client: PolymarketClient,
                         asset: str, windows_to_fetch: list) -> int:
    """Fetch and insert windows for one asset. Returns count inserted."""
    db_path = DATA_DIR / f"{asset.lower()}.db"
    if not db_path.exists():
        logger.error(f"DB not found: {db_path}")
        return 0

    existing = get_existing_windows(db_path)
    slug_prefix = CRYPTO_15MIN_SLUGS.get(asset)
    if not slug_prefix:
        logger.error(f"No slug prefix for {asset}")
        return 0

    binance_symbol = BINANCE_SYMBOLS[asset]
    inserted = 0

    for window_start in windows_to_fetch:
        wid = get_window_id(asset, window_start)
        if wid in existing:
            continue

        window_end = window_start + timedelta(minutes=15)
        epoch = int(window_start.timestamp())

        # Find market
        market = await pm_client._get_market_by_epoch(slug_prefix, epoch, asset)
        if market is None:
            logger.debug(f"No market found for {asset} at {window_start}")
            continue

        # Fetch PM price history (startTs/endTs required by CLOB endpoint)
        pm_prices = {}
        if market.yes_token_id:
            pm_prices = await fetch_pm_prices(session, market.yes_token_id,
                                               epoch, epoch + 900)

        if not pm_prices:
            logger.debug(f"No PM price history for {asset} {window_start}")
            continue

        # Sample PM prices at window timepoints
        pm_t = {}
        for t_min in SAMPLE_MINUTES:
            pm_t[t_min] = sample_pm_price_at(pm_prices, window_start, t_min)

        # Fetch Binance spot klines
        start_ms = int(window_start.timestamp() * 1000)
        end_ms = int(window_end.timestamp() * 1000)
        klines = await fetch_binance_klines(session, binance_symbol, start_ms, end_ms)
        spot_open, spot_close, spot_high, spot_low = compute_spot_data(klines, window_start)

        # Compute derived fields
        outcome = None
        outcome_binary = None
        spot_change_pct = None
        spot_change_bps = None
        spot_range_bps = None

        if spot_open and spot_close and spot_open > 0:
            spot_change_pct = ((spot_close - spot_open) / spot_open) * 100
            spot_change_bps = spot_change_pct * 100
            outcome = 'up' if spot_close > spot_open else 'down'
            outcome_binary = 1 if outcome == 'up' else 0

        if spot_open and spot_high and spot_low and spot_open > 0:
            spot_range_bps = ((spot_high - spot_low) / spot_open) * 10000

        # Volatility regime
        volatility_regime = None
        if spot_range_bps is not None:
            if spot_range_bps < 15:
                volatility_regime = 'low'
            elif spot_range_bps < 40:
                volatility_regime = 'normal'
            elif spot_range_bps < 80:
                volatility_regime = 'high'
            else:
                volatility_regime = 'extreme'

        # PM momentum
        pm_mom_0_5 = None
        pm_mom_5_10 = None
        if pm_t.get(0.0) is not None and pm_t.get(5.0) is not None:
            pm_mom_0_5 = pm_t[5.0] - pm_t[0.0]
        if pm_t.get(5.0) is not None and pm_t.get(10.0) is not None:
            pm_mom_5_10 = pm_t[10.0] - pm_t[5.0]

        window_data = {
            'window_id': wid,
            'asset': asset,
            'window_start_utc': window_start.isoformat(),
            'window_end_utc': window_end.isoformat(),
            'resolved_at_utc': window_end.isoformat(),
            'outcome': outcome,
            'outcome_binary': outcome_binary,
            'spot_open': spot_open,
            'spot_close': spot_close,
            'spot_change_pct': spot_change_pct,
            'spot_change_bps': spot_change_bps,
            'spot_high': spot_high,
            'spot_low': spot_low,
            'spot_range_bps': spot_range_bps,
            'pm_yes_t0': pm_t.get(0.0),
            'pm_yes_t2_5': pm_t.get(2.5),
            'pm_yes_t5': pm_t.get(5.0),
            'pm_yes_t7_5': pm_t.get(7.5),
            'pm_yes_t10': pm_t.get(10.0),
            'pm_yes_t12_5': pm_t.get(12.5),
            'pm_spread_t0': None,
            'pm_spread_t5': None,
            'pm_price_momentum_0_to_5': pm_mom_0_5,
            'pm_price_momentum_5_to_10': pm_mom_5_10,
            'volatility_regime': volatility_regime,
            'window_time': get_window_time(window_start),
        }

        if outcome is None:
            logger.debug(f"Skipping {wid}: no outcome (missing spot data)")
            continue

        insert_window(db_path, asset, window_data)
        inserted += 1
        t0_str = f"{pm_t.get(0.0):.3f}" if pm_t.get(0.0) is not None else "N/A"
        t12_str = f"{pm_t.get(12.5):.3f}" if pm_t.get(12.5) is not None else "N/A"
        logger.info(f"  Inserted {wid}: {outcome} | PM t0={t0_str} t12.5={t12_str}")

    if inserted > 0:
        logger.info(f"{asset}: backfilling prev columns...")
        backfill_prev_columns(db_path)

    return inserted


async def main():
    # Find the latest window in any DB
    latest_starts = []
    for asset in ASSETS:
        db_path = DATA_DIR / f"{asset.lower()}.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        r = conn.execute("SELECT MAX(window_start_utc) FROM windows WHERE outcome IS NOT NULL").fetchone()
        conn.close()
        if r[0]:
            latest_starts.append(r[0])

    if not latest_starts:
        logger.error("No existing data found")
        return

    # Use the MIN across assets so all assets get fully caught up
    last_window_str = min(latest_starts)
    # Parse ISO format (may have timezone)
    last_window_str_clean = last_window_str.replace('+00:00', '').replace('Z', '')
    last_window = datetime.fromisoformat(last_window_str_clean).replace(tzinfo=timezone.utc)
    logger.info(f"Last recorded window (min across assets): {last_window}")

    # Generate list of 15-min windows from last+15min to now-30min (completed windows only)
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(minutes=30)  # don't try to fetch still-active windows

    windows_to_fetch = []
    t = last_window + timedelta(minutes=15)
    while t <= cutoff:
        windows_to_fetch.append(t)
        t += timedelta(minutes=15)

    if not windows_to_fetch:
        logger.info("No new windows to fetch (already up to date)")
        return

    logger.info(f"Windows to fetch: {len(windows_to_fetch)} "
                f"({windows_to_fetch[0]} → {windows_to_fetch[-1]})")

    async with aiohttp.ClientSession() as session:
        pm_client = PolymarketClient(session=session)

        total_inserted = 0
        for asset in ASSETS:
            logger.info(f"\nProcessing {asset}...")
            count = await process_asset(session, pm_client, asset, windows_to_fetch)
            logger.info(f"{asset}: inserted {count} windows")
            total_inserted += count
            # Small delay between assets to be polite to APIs
            await asyncio.sleep(1)

    logger.info(f"\nDone. Total inserted: {total_inserted}")
    if total_inserted > 0:
        logger.info("Run 'python scripts/backfill_db.py' if any prev/prev2 columns need recompute")


if __name__ == '__main__':
    asyncio.run(main())
