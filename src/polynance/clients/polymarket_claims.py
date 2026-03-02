"""
polymarket_claims.py

Discover and redeem settled Polymarket positions via the gasless proxy relayer.
Positions are fetched from the Data API; redemption is submitted via
polymarket_relayer.redeem_batch_via_relayer (one proxy() call per sweep).
"""

import os
import asyncio
import logging
from typing import Set, Dict, Any, List, Optional

import aiohttp
from web3 import Web3

from .polymarket_relayer import (
    redeem_batch_via_relayer,
    RELAYER_ENABLED,
    get_circuit_status,
    get_polymarket_wallet_address,
)

logger = logging.getLogger(__name__)

DATA_API_BASE   = "https://data-api.polymarket.com"
TERMINAL_STATES = {"STATE_MINED", "STATE_CONFIRMED", "STATE_EXECUTED"}


async def _resolve_positions_wallet(start_address: str) -> str:
    """
    Find the actual proxy wallet that holds positions.
    Priority: proxyWallet field in positions → POLYMARKET_FUNDER_ADDRESS env → start_address.
    """
    positions = await fetch_positions(start_address)
    proxy = None
    for p in positions:
        proxy = p.get("proxyWallet") or p.get("proxy_address")
        if proxy:
            break
    if proxy:
        logger.info(
            "[claims] Using proxyWallet=%s (resolved from %s via /positions)",
            proxy, start_address,
        )
        return proxy

    env_funder = (
        os.getenv("POLYMARKET_FUNDER_ADDRESS")
        or os.getenv("POLYMARKET_PROXY_ADDRESS")
    )
    if env_funder:
        addr = Web3.to_checksum_address(env_funder)
        logger.info(
            "[claims] No proxyWallet in positions; using configured funder=%s "
            "(resolved from %s)", addr, start_address,
        )
        return addr

    logger.info(
        "[claims] No proxyWallet found in positions, using address=%s as positions wallet",
        start_address,
    )
    return start_address


def get_active_wallet_address() -> Optional[str]:
    """
    Return the wallet address whose positions should be swept.
    Priority: POLYMARKET_FUNDER_ADDRESS → POLYMARKET_PROXY_ADDRESS → derived proxy.
    """
    funder = os.getenv("POLYMARKET_FUNDER_ADDRESS")
    if funder:
        return funder
    proxy = os.getenv("POLYMARKET_PROXY_ADDRESS")
    if proxy:
        return proxy
    return get_polymarket_wallet_address()


async def fetch_positions(user_address: str) -> List[Dict[str, Any]]:
    """Fetch raw positions for a user from the Data API."""
    url    = f"{DATA_API_BASE}/positions"
    params = {"user": user_address}
    logger.info(f"[claims] Fetching positions for {user_address}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                text = await resp.text()
                logger.debug(f"[claims] Positions raw response: {text[:2000]}")
                resp.raise_for_status()
                positions = await resp.json()
        except Exception as e:
            logger.error("[claims] Error fetching positions", exc_info=True)
            return []

    if not isinstance(positions, list):
        logger.warning(f"[claims] Unexpected positions payload type: {type(positions)}")
        return []

    logger.info(f"[claims] Retrieved {len(positions)} positions")
    if positions:
        proxy_wallet = positions[0].get("proxyWallet") or positions[0].get("proxywallet")
        if proxy_wallet:
            logger.info(f"[claims] Positions report proxyWallet={proxy_wallet} (queried user={user_address})")
    return positions


def _indexset_for_outcome(outcome_index: Optional[int]) -> Optional[int]:
    if outcome_index is None:
        return None
    try:
        oi = int(outcome_index)
    except Exception:
        return None
    if oi < 0:
        return None
    return 2 ** oi


def _is_zero_value_loser(p: Dict[str, Any]) -> bool:
    try:
        cur_price     = float(p.get("curPrice", 0) or 0)
        current_value = float(p.get("currentValue", 0) or 0)
    except Exception:
        return False
    return cur_price == 0.0 and current_value == 0.0


async def fetch_claimable_conditions_with_indexsets(
    user_address: str,
    redeem_losers: bool = False,
) -> Dict[str, Any]:
    """
    Scan positions and return {conditionId: {"conditionId": ...}} for all
    redeemable positions. Zero-value losers are skipped unless redeem_losers=True.
    """
    positions = await fetch_positions(user_address)
    cond_to_indexsets: Dict[str, Set[int]] = {}
    total_redeemable = 0
    loser_count      = 0

    for p in positions:
        cid = p.get("conditionId") or p.get("conditionid")
        if not cid:
            continue

        redeemable_raw = p.get("redeemable")
        redeemable     = redeemable_raw is True or redeemable_raw == "true"
        if not redeemable:
            continue

        size_raw = p.get("size", 0)
        try:
            size = float(size_raw)
        except Exception:
            logger.debug(f"[claims] Skipping non-numeric size: {size_raw!r}")
            continue
        if size <= 0:
            continue

        total_redeemable += 1

        if _is_zero_value_loser(p):
            if not redeem_losers:
                logger.debug(
                    f"[claims] Skipping zero-value loser cid={cid} "
                    f"curPrice={p.get('curPrice', 0)} currentValue={p.get('currentValue', 0)}"
                )
                loser_count += 1
                continue
            else:
                logger.debug(f"[claims] Including zero-value loser cid={cid} (redeem_losers=True)")

        outcome_index = p.get("outcomeIndex")
        indexset      = _indexset_for_outcome(outcome_index)
        if indexset is None:
            logger.warning(
                f"[claims] Could not derive indexset from outcomeIndex={outcome_index} "
                f"for cid={cid}, falling back to [1, 2]"
            )
            cond_to_indexsets.setdefault(cid, set()).update([1, 2])
        else:
            cond_to_indexsets.setdefault(cid, set()).add(indexset)

    result = {cid: {"conditionId": cid} for cid in cond_to_indexsets}
    logger.info(
        f"[claims] Found {len(result)} redeemable conditionIds "
        f"(redeem_losers={redeem_losers}, {loser_count} zero-value losers "
        f"{'included' if redeem_losers else 'skipped'} of {total_redeemable} total redeemable)"
    )
    return result


async def fetch_redeemable_conditions(
    wallet: str,
    max_batch: int = 32,
    redeem_losers: bool = False,
) -> List[Dict[str, Any]]:
    """Return a list of condition dicts ready for redeem_batch_via_relayer."""
    cond_to_indexsets = await fetch_claimable_conditions_with_indexsets(
        wallet, redeem_losers=redeem_losers
    )
    conditions = [
        {"conditionId": cid, "indexsets": idxsets}
        for cid, idxsets in cond_to_indexsets.items()
    ]
    if len(conditions) > max_batch:
        logger.info(f"[claims] Capping batch from {len(conditions)} → {max_batch}")
        conditions = conditions[:max_batch]
    return conditions


async def sweep_and_redeem_claimables(
    eoa_address:   Optional[str] = None,
    redeem_losers: bool = False,
) -> None:
    """
    Full sweep: discover redeemable positions and submit a batch redeem via relayer.
    This is the main entry point for the periodic claim loop.
    """
    wallet = eoa_address or get_active_wallet_address()
    if not wallet:
        logger.error("[claims] Cannot sweep: no wallet address available.")
        return

    logger.info(f"[claims] Starting sweep for wallet={wallet} redeem_losers={redeem_losers}")

    if not RELAYER_ENABLED:
        logger.warning("[claims] Relayer disabled — skipping sweep.")
        return

    circuit = get_circuit_status()
    if circuit["open"]:
        logger.warning(f"[claims] Circuit open, skipping sweep — resets in {circuit['secondsRemaining']}s")
        return

    positions_wallet  = await _resolve_positions_wallet(wallet)
    cond_to_indexsets = await fetch_claimable_conditions_with_indexsets(
        positions_wallet, redeem_losers=redeem_losers
    )

    if not cond_to_indexsets:
        logger.info("[claims] No redeemable positions found")
        return

    total      = len(cond_to_indexsets)
    conditions = [{"conditionId": cid, "indexsets": indexsets}
                  for cid, indexsets in cond_to_indexsets.items()]

    logger.info(f"[claims] Attempting redemption for {total} conditions")
    for i, c in enumerate(conditions, start=1):
        logger.info(f"[claims] {i}/{total} {c['conditionId']} indexsets={c['indexsets']}")

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: redeem_batch_via_relayer(
            conditions,
            proxy_address=positions_wallet,
            description=f"CTF Redeem x{total}",
        ),
    )

    state     = result.get("state")
    txhash    = result.get("txhash")
    succeeded = result.get("succeededCount", 0)

    if state == "STATE_CONFIRMED":
        logger.info(f"[claims] Sweep complete: {succeeded}/{total} redeemed tx={txhash}")
    elif state == "SKIPPED_NO_COOKIES":
        logger.error("[claims] Sweep skipped: no valid session cookies.")
    elif state == "REVERTED":
        logger.error(f"[claims] Sweep tx reverted — https://polygonscan.com/tx/{txhash}")
    elif state == "TIMEOUT":
        logger.warning(f"[claims] Sweep timed out — tx may still confirm: txhash={txhash}")
    elif state in ("SUBMIT_FAILED", "ENCODE_FAILED"):
        logger.error(f"[claims] Sweep failed at stage: {state}")
    elif state in ("SKIPPED_NO_CREDS", "SKIPPED_QUOTA_EXCEEDED", "NO_CONDITIONS"):
        logger.info(f"[claims] Sweep skipped: {state}")
    else:
        logger.error(f"[claims] Unexpected state={state}, tx={txhash}")

    logger.info(f"[claims] Sweep done: {succeeded} succeeded, {total - succeeded} not confirmed of {total} total")
