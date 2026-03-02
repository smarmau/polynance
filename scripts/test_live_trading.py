#!/usr/bin/env python3
"""Test Polymarket live trading via polymarket-cli.

Install CLI first (~/.local/bin or /usr/local/bin):
    curl -sSL https://raw.githubusercontent.com/Polymarket/polymarket-cli/main/install.sh | sh

Requires in .env:
    POLYMARKET_PRIVATE_KEY=0x...
    POLYMARKET_SIGNATURE_TYPE=1    (0=eoa, 1=proxy, 2=gnosis-safe)

Usage:
    python scripts/test_live_trading.py                      # Auth + balance + market check
    python scripts/test_live_trading.py --buy                # Buy YES tokens
    python scripts/test_live_trading.py --buy --sell         # Buy + sell round-trip
    python scripts/test_live_trading.py --redeem             # Redeem all settled positions
    python scripts/test_live_trading.py --redeem --condition 0xABC...
    python scripts/test_live_trading.py --buy --sell --redeem
"""

import argparse
import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_live")

CLOB_HOST = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"
MIN_ORDER_SIZE     = 5.0   # Polymarket minimum BUY order size in shares
MIN_SELL_ORDER_SIZE = 1.0  # Minimum SELL size — can be less than 5 (closing existing position)

# Int → CLI string for POLYMARKET_SIGNATURE_TYPE env var
SIG_INT_TO_STR = {"0": "eoa", "1": "proxy", "2": "gnosis-safe"}

# Auth flags injected into every CLI call (set in main after env is loaded)
_AUTH: list[str] = []


# ── CLI wrapper ──────────────────────────────────────────────────────────────

def _find_polymarket_cli() -> str:
    """Return the polymarket CLI executable path, checking common locations."""
    import shutil
    # 1. Already on PATH
    found = shutil.which("polymarket")
    if found:
        return found
    # 2. Common install locations
    for candidate in [
        Path.home() / ".local" / "bin" / "polymarket",
        Path("/usr/local/bin/polymarket"),
        Path("/usr/bin/polymarket"),
    ]:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "polymarket CLI not found.\n"
        "Install: curl -sSL https://raw.githubusercontent.com/Polymarket/polymarket-cli/main/install.sh | sh\n"
        "Then add ~/.local/bin to your PATH if needed."
    )


def poly(*args, check=True, auth=True):
    """Run: polymarket <args> [auth-flags] --output json → dict/list/str.

    Raises RuntimeError on non-zero exit when check=True.
    Set auth=False for commands that don't need --private-key (e.g. wallet create).
    """
    cli = _find_polymarket_cli()
    cmd = [cli] + list(args)
    if auth and _AUTH:
        cmd += _AUTH
    cmd += ["--output", "json"]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "polymarket CLI not found.\n"
            "Install: curl -sSL https://raw.githubusercontent.com/Polymarket/polymarket-cli/main/install.sh | sh\n"
            "Then add ~/.local/bin to your PATH if needed."
        )

    if check and r.returncode != 0:
        err = (r.stderr.strip() or r.stdout.strip())[:500]
        raise RuntimeError(f"polymarket {' '.join(str(a) for a in args)}: {err}")

    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return r.stdout.strip()


def parse_balance(raw) -> float:
    """Parse USDC balance from CLI response.

    CLI returns balance as a float string (e.g. "84.418887"), not micro-USDC.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        for key in ("balance", "amount", "usdc"):
            if key in raw:
                try:
                    return float(raw[key])
                except (ValueError, TypeError):
                    pass
    return 0.0


def parse_order_id(resp) -> str:
    """Extract order ID from create-order response."""
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, dict):
        for key in ("orderID", "order_id", "id", "orderId"):
            if resp.get(key):
                return str(resp[key])
    return ""


def poll_order(order_id: str, max_wait: int = 30, interval: int = 3) -> dict | None:
    """Poll order until filled/cancelled or timeout; cancel on timeout."""
    elapsed = 0
    while elapsed < max_wait:
        time.sleep(interval)
        elapsed += interval
        try:
            detail = poly("clob", "order", order_id)
            if not isinstance(detail, dict):
                print(f"  [{elapsed}s] raw: {detail}")
                continue
            status = detail.get("status", "?")
            matched = detail.get("size_matched", "0")
            print(f"  [{elapsed}s] status={status}, matched={matched}")
            if status.upper() in ("MATCHED", "CLOSED"):
                return detail
            if status.upper() in ("CANCELLED", "EXPIRED"):
                return detail
        except Exception as e:
            print(f"  [{elapsed}s] poll error: {e}")

    print(f"  Timed out after {max_wait}s — cancelling")
    poly("clob", "cancel", order_id, check=False)
    print(f"  Cancelled {order_id[:20]}...")
    return None


# ── HTTP helpers (no auth needed) ────────────────────────────────────────────

async def fetch_orderbook(token_id: str) -> dict:
    """Fetch orderbook from CLOB REST API (public, no auth)."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{CLOB_HOST}/book",
            params={"token_id": token_id},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            return await r.json()


async def fetch_positions(wallet: str) -> list:
    """Fetch all positions from data API."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{DATA_API_URL}/positions",
            params={"user": wallet.lower()},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r:
            data = await r.json()
            return data if isinstance(data, list) else data.get("data", [])


# ── UI ────────────────────────────────────────────────────────────────────────

def step(n, title):
    print(f"\n{'=' * 60}")
    print(f"STEP {n}: {title}")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Test Polymarket live trading via polymarket-cli")
    p.add_argument("--buy", action="store_true", help="Place a BUY order")
    p.add_argument("--sell", action="store_true", help="Place a SELL order (requires --buy)")
    p.add_argument("--redeem", action="store_true", help="Redeem settled positions")
    p.add_argument("--condition", type=str, default=None,
                   help="Specific condition ID to redeem (with --redeem)")
    p.add_argument("--asset", type=str, default="BTC",
                   help="Asset keyword for 15-min market search (default: BTC)")
    p.add_argument("--amount", type=float, default=2.0,
                   help="Dollar amount for test order (default: $2)")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args()


# ── Sell helper ───────────────────────────────────────────────────────────────

def _sell_position(token_id: str, announce_existing: bool = False, max_attempts: int = 5) -> float:
    """Exit a YES position using repeated FOK orders until flat or no liquidity.

    FOK fills whatever bids exist at the moment and cancels the rest instantly.
    We loop up to max_attempts times to drain the position completely.

    Returns total USDC received across all fills.
    """
    total_usdc = 0.0

    for attempt in range(1, max_attempts + 1):
        # Always re-query the live balance — previous FOK may have partially filled.
        try:
            cond_raw = poly("clob", "balance", "--asset-type", "conditional", "--token", token_id)
            remaining = parse_balance(cond_raw)
        except Exception as e:
            logger.warning(f"  Balance query failed: {e}")
            break

        # Floor to 2 dp so we never submit more than we hold.
        remaining = math.floor(remaining * 100) / 100

        if remaining < MIN_SELL_ORDER_SIZE:
            if attempt == 1 and announce_existing:
                print(f"  No position to sell (balance {remaining:.2f})")
            elif attempt > 1:
                print(f"  Remaining {remaining:.2f} shares below min — done")
            break

        if attempt == 1 and announce_existing:
            print(f"  Found existing position: {remaining:.2f} YES shares")

        print(f"  [{attempt}/{max_attempts}] FOK sell {remaining:.2f} shares...", end=" ", flush=True)

        try:
            resp = poly(
                "clob", "create-order",
                "--token", token_id,
                "--side", "sell",
                "--price", "0.01",          # floor price — fills at actual best bid
                "--size", f"{remaining:.2f}",
                "--order-type", "FAK",      # Fill And Kill: partial fills OK, rest cancelled
            )
        except RuntimeError as e:
            err_str = str(e)
            # FAK/FOK with zero liquidity returns a 400 — treat as no-fill, not a crash
            if "no liquidity" in err_str.lower() or "couldn't be" in err_str.lower() or "400" in err_str:
                print(f"no fill (no bids available)")
                print(f"  ⚠ No liquidity — {remaining:.2f} shares remain unsold")
            else:
                print(f"ERROR: {e}")
                logger.error(f"FAK sell attempt {attempt} failed: {e}")
            break

        status       = resp.get("status", "")          if isinstance(resp, dict) else ""
        # For a sell: making_amount = shares sold, taking_amount = USDC received
        usdc_recvd   = float(resp.get("taking_amount", 0)) if isinstance(resp, dict) else 0.0
        shares_sold  = float(resp.get("making_amount", 0)) if isinstance(resp, dict) else 0.0
        total_usdc  += usdc_recvd

        if shares_sold > 0:
            print(f"filled {shares_sold:.2f} shares → ${usdc_recvd:.4f} USDC")
        elif status in ("CANCELLED", "CANCELED") or usdc_recvd == 0:
            print(f"no fill (no bids at ≥0.01)")
            print(f"  ⚠ No liquidity — {remaining:.2f} shares remain unsold")
            break
        else:
            print(f"status={status} usdc={usdc_recvd:.4f}")

    if total_usdc > 0:
        print(f"  Total received: ${total_usdc:.4f} USDC")
    return total_usdc


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global _AUTH

    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        logger.error("POLYMARKET_PRIVATE_KEY not set. Check .env")
        return 1

    # Normalize sig type: "1" → "proxy", "proxy" → "proxy", etc.
    sig_env = os.getenv("POLYMARKET_SIGNATURE_TYPE", "1")
    sig_type = SIG_INT_TO_STR.get(sig_env, sig_env)  # pass through if already a string

    # Build auth flags — injected into every poly() call
    _AUTH = ["--private-key", pk, "--signature-type", sig_type]

    # ── Magic auth pre-login (if --redeem with magic) ───────────────
    # Do this eagerly so the OTP prompt appears before any trading,
    # not mid-run at step 8.
    if args.redeem and os.getenv("POLYMARKET_AUTH_TYPE", "").lower() == "magic":
        print("\n" + "=" * 60)
        print("MAGIC AUTH — pre-seeding relayer session")
        print("=" * 60)
        try:
            from polynance.clients.polymarket_relayer import cookies_valid, refresh_cookies
            if not cookies_valid():
                print("  No cached session — logging in via Magic (email + OTP)...")
                refresh_cookies()
            if cookies_valid():
                print("  Relayer session ready ✓")
            else:
                print("  ⚠ Magic login did not produce cookies — redeem will be skipped")
        except Exception as e:
            logger.warning(f"Magic pre-login failed: {e}")

    # ── Step 1: CLI check + wallet ─────────────────────────────────
    step(1, "CLI Check + Wallet")

    try:
        wallet_resp = poly("wallet", "address")
    except RuntimeError as e:
        logger.error(str(e))
        return 1

    wallet = wallet_resp if isinstance(wallet_resp, str) else wallet_resp.get("address", str(wallet_resp))
    # For proxy wallets, positions/balances are held by the funder (proxy wallet), not the EOA signer
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
    positions_wallet = funder or wallet  # use funder if set, else fall back to signer

    print(f"  Signer:    {wallet}")
    if funder:
        print(f"  Funder:    {funder}  (proxy wallet, used for positions)")
    print(f"  Sig type:  {sig_type}")
    print(f"  CLI ✓")

    # ── Step 2: Baseline balance ────────────────────────────────────
    step(2, "Wallet Balance (baseline)")

    bal_raw = poly("clob", "balance", "--asset-type", "collateral")
    baseline = parse_balance(bal_raw)
    print(f"  USDC: ${baseline:.4f}")
    print(f"  Raw:  {bal_raw}")

    # ── Step 3: Market discovery + orderbook ───────────────────────
    step(3, f"Find {args.asset} Market + Orderbook")

    from polynance.clients.polymarket import PolymarketClient
    import aiohttp

    async with aiohttp.ClientSession() as session:
        poly_client = PolymarketClient(session=session)
        markets = await poly_client.find_active_15min_markets([args.asset])

    if not markets:
        logger.error(f"No active 15-min {args.asset} market found")
        return 1

    market = markets[0]
    print(f"  Market:       {market.question}")
    print(f"  Condition ID: {market.condition_id}")
    print(f"  YES token:    {market.yes_token_id}")
    print(f"  NO token:     {market.no_token_id}")

    token_id = market.yes_token_id
    best_ask: float = 0.60
    best_bid: float = 0.40
    has_bids = False

    try:
        book = await fetch_orderbook(token_id)
        asks = book.get("asks", [])
        bids = book.get("bids", [])
        if asks:
            best_ask = min(float(a["price"]) for a in asks)
        if bids:
            best_bid = max(float(b["price"]) for b in bids)
            has_bids = True
        else:
            # No bids — derive a sell price one tick below best ask so limit orders can rest
            best_bid = max(round(best_ask - 0.01, 2), 0.01)
        print(f"  Best bid: {best_bid:.4f}  Best ask: {best_ask:.4f}  Spread: {best_ask - best_bid:.4f}")
        print(f"  Depth:    {len(asks)} asks, {len(bids)} bids" + ("" if has_bids else "  ⚠ no bids"))
    except Exception as e:
        print(f"  Orderbook error: {e} — using defaults bid={best_bid}, ask={best_ask}")

    if not args.buy and not args.sell and not args.redeem:
        print("\n" + "=" * 60)
        print("CHECKS COMPLETE — use --buy to trade, --sell to close, --redeem to claim settled positions")
        print("=" * 60)
        return 0

    # ── Step 4: BUY ────────────────────────────────────────────────
    order_id = None
    buy_fill = None
    size = None

    if args.buy:
        buy_price = round(best_ask, 2)  # CLOB tick size is 0.01 — max 2 decimal places
        size = max(args.amount / buy_price, MIN_ORDER_SIZE)
        cost = size * buy_price
        step(4, f"BUY {size:.2f} YES @ {buy_price:.2f} (≈${cost:.2f})")

        try:
            resp = poly(
                "clob", "create-order",
                "--token", token_id,
                "--side", "buy",
                "--price", f"{buy_price:.2f}",
                "--size", f"{size:.2f}",
            )
            print(f"  Response: {resp}")
            order_id = parse_order_id(resp)
            if not order_id:
                logger.error(f"No order ID in response: {resp}")
                return 1
            print(f"  Order ID: {order_id} ✓")
        except RuntimeError as e:
            logger.error(f"create-order failed: {e}")
            return 1

        print("  Polling for fill...")
        buy_fill = poll_order(order_id)

        # ── Step 5: Position check ──────────────────────────────────
        step(5, "Conditional Token Balance (position check)")

        try:
            cond_raw = poly("clob", "balance", "--asset-type", "conditional", "--token", token_id)
            cond_bal = parse_balance(cond_raw)
            print(f"  YES token: {cond_bal:.4f} shares")
            print(f"  Raw:       {cond_raw}")
            print("  ✓ Position confirmed!" if cond_bal > 0 else "  ⚠ No position (order may not have filled)")
        except Exception as e:
            logger.error(f"Balance check failed: {e}")

    # ── Step 6: SELL ───────────────────────────────────────────────
    if args.sell:
        step(6, "SELL YES position (FOK loop)")
        _sell_position(token_id, not args.buy)

        step(7, "Final Wallet Balance")
        time.sleep(2)

        bal2_raw = poly("clob", "balance", "--asset-type", "collateral")
        final = parse_balance(bal2_raw)
        delta = final - baseline
        print(f"  USDC:  ${final:.4f}")
        print(f"  Delta: ${delta:+.4f}  (fees + spread)")

    # ── Step 8: Redeem ─────────────────────────────────────────────
    if args.redeem:
        step(8, "Redeem Settled Positions")
        await redeem_all(positions_wallet, pk, sig_type)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    return 0


async def redeem_all(wallet: str, pk: str, sig_type: str):
    """Redeem settled positions via Polymarket gasless relayer (relayer-v2.polymarket.com).

    Signs relay requests locally with POLYMARKET_PRIVATE_KEY (no MATIC needed).
    Requires Polymarket session cookies — auto-loaded when POLYMARKET_AUTH_TYPE=magic,
    otherwise set via set_session_cookies() or visit polymarket.com/portfolio manually.
    """
    try:
        from polynance.clients.polymarket_claims import (
            fetch_positions,
            sweep_and_redeem_claimables,
        )
        from polynance.clients.polymarket_relayer import (
            RELAYER_ENABLED,
            cookies_valid,
            refresh_cookies,
            get_circuit_status,
        )
    except ImportError as e:
        logger.error(f"Gasless redeem missing dependency: {e}")
        logger.error("pip install web3 eth-abi eth-utils requests")
        return

    print(f"  Fetching positions for {wallet}...")
    try:
        positions = await fetch_positions(wallet)
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return

    redeemable  = [p for p in positions if p.get("redeemable") is True]
    total_value = sum(float(p.get("currentValue", 0) or 0) for p in redeemable)
    print(f"  {len(positions)} total, {len(redeemable)} redeemable (≈${total_value:.2f} USDC)")

    if not redeemable:
        print("  Nothing to redeem ✓")
        return

    for p in redeemable[:5]:
        val   = float(p.get("currentValue", 0) or 0)
        title = p.get("title", (p.get("conditionId", "?"))[:40])
        print(f"    ${val:.2f}  {title}")
    if len(redeemable) > 5:
        print(f"    ... and {len(redeemable) - 5} more")

    if not RELAYER_ENABLED:
        print("  ✗ Relayer disabled (POLYMARKET_PRIVATE_KEY not set in .env)")
        return

    circuit = get_circuit_status()
    if circuit["open"]:
        print(f"  ⚠ Rate limit circuit open — resets in {circuit['secondsRemaining']}s")
        return

    # Cookies: auto-loaded from Magic if POLYMARKET_AUTH_TYPE=magic, else must be injected
    if not cookies_valid():
        print("  Attempting to load session cookies...")
        refresh_cookies()

    if not cookies_valid():
        print("  ✗ No Polymarket session cookies — cannot use gasless relayer")
        print("  Options:")
        print("    1. Set POLYMARKET_AUTH_TYPE=magic and run with Magic wallet (OTP login)")
        print("    2. Redeem manually at polymarket.com/portfolio")
        return

    print("  Submitting gasless batch redeem via relayer-v2.polymarket.com...")
    try:
        await sweep_and_redeem_claimables(wallet, redeem_losers=False)
    except Exception as e:
        logger.error(f"Redeem sweep failed: {e}")
        return

    # Post-redeem balance
    time.sleep(3)
    try:
        bal_raw = poly("clob", "balance", "--asset-type", "collateral")
        print(f"  Post-redeem balance: ${parse_balance(bal_raw):.4f} USDC")
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
