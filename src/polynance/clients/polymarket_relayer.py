"""
polymarket_relayer.py

Polymarket relayer — submits proxy-wallet transactions via relayer-v2.
Supports two signing modes:
  - EOA:   POLYMARKET_PRIVATE_KEY set → local eth_account signing
  - Magic: POLYMARKET_AUTH_TYPE=magic → TEE signing via magic_auth

Session cookies:
  - For Magic: automatically sourced from magic_auth.get_polymarket_session
  - For EOA:   inject via set_session_cookies() from your app.
"""

import os
import re
import time
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, NamedTuple

import requests
from eth_abi.codec import ABICodec
from eth_abi.registry import registry
from eth_utils import function_signature_to_4byte_selector
from web3 import Web3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contract addresses
# ---------------------------------------------------------------------------
USDC_E               = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF                  = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
PROXY_WALLET_FACTORY = Web3.to_checksum_address("0xaB45c5A4B0c941a2F231C04C3f49182e1A254052")
RELAY_HUB            = Web3.to_checksum_address("0xD216153c06E857cD7f72665E0aF1d7D82172F494")
HASH_ZERO            = b"\x00" * 32

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CHAIN_ID        = int(os.getenv("CHAIN_ID", 137))
RELAYER_URL     = os.getenv("RELAYER_URL", "https://relayer-v2.polymarket.com")
POLYGON_RPC     = os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
PRIVATE_KEY     = os.getenv("POLYMARKET_PRIVATE_KEY")
AUTH_TYPE       = os.getenv("POLYMARKET_AUTH_TYPE", "").lower()  # "magic" or ""
RELAYER_ENABLED = bool(PRIVATE_KEY) or AUTH_TYPE == "magic"

# Default indexSets for binary (YES/NO) markets — always redeem both outcomes.
# The CTF contract returns 0 for any outcome the proxy doesn't hold,
# so passing [1, 2] is always safe and matches browser behaviour.
DEFAULT_INDEX_SETS = [1, 2]

# ---------------------------------------------------------------------------
# Signer / Web3
# ---------------------------------------------------------------------------
w3      = Web3(Web3.HTTPProvider(POLYGON_RPC))
account = None
if PRIVATE_KEY:
    from eth_account import Account
    account = Account.from_key(PRIVATE_KEY)

codec = ABICodec(registry)

# ── Proxy wallet derivation ──────────────────────────────────────────────────
# Storage slot = keccak256("implementation") from ProxyWalletLib
_IMPL_SLOT = 0x8ba0ed1f62da1d3048614c2c1feb566f041c8467eb00fb8294776a9179dc1643
_derived_proxy_cache: Optional[str] = None


def _build_creation_code(factory: str, impl: str) -> bytes:
    """
    Replicates ProxyWalletLib.computeCreationCode(deployer=factory, target=impl).
    EIP-1167 minimal proxy clone with cloneConstructor(bytes) consData appended.
    Total: 99 bytes clone + 68 bytes consData = 167 bytes.
    """
    f = bytes.fromhex(factory[2:].lower())   # 20 bytes
    t = bytes.fromhex(impl[2:].lower())      # 20 bytes
    # consData = abi.encodeWithSignature("cloneConstructor(bytes)", new bytes(0))
    #          = selector(4) + offset(32) + length(32)
    selector  = Web3.keccak(b"cloneConstructor(bytes)")[:4]
    cons_data = selector + (32).to_bytes(32, "big") + (0).to_bytes(32, "big")
    # Assembly: mstore chunks that build the 99-byte clone proxy
    clone = (
        bytes.fromhex("3d3d606380380380913d393d73")  # 13 bytes
        + f                                           # deployer (factory) 20 bytes
        + bytes.fromhex("5af4602a57600080fd5b602d8060366000396000f3")  # 21 bytes
        + bytes.fromhex("363d3d373d3d3d363d73")       # 10 bytes
        + t                                           # implementation 20 bytes
        + bytes.fromhex("5af43d82803e903d91602b57fd5bf3")  # 15 bytes
    )                                                 # = 99 bytes total
    return clone + cons_data                          # 99 + 68 = 167 bytes


def _derive_proxy_wallet(eoa: str) -> str:
    """
    Derive the Polymarket proxy wallet address for a given EOA via CREATE2.
    Result is cached — the derivation is deterministic and never changes.
    """
    global _derived_proxy_cache
    if _derived_proxy_cache:
        return _derived_proxy_cache
    # 1. Read implementation address from factory storage
    raw_impl = w3.eth.get_storage_at(PROXY_WALLET_FACTORY, _IMPL_SLOT)
    impl     = Web3.to_checksum_address(bytes(raw_impl)[-20:])
    logger.debug(f"[relayer] proxy derivation — impl={impl}")
    # 2. Build creation code and hash it
    creation_code      = _build_creation_code(PROXY_WALLET_FACTORY, impl)
    creation_code_hash = Web3.keccak(creation_code)
    # 3. salt = keccak256(abi.encodePacked(address)) = keccak256(raw 20-byte eoa)
    salt = Web3.keccak(bytes.fromhex(eoa[2:].lower()))
    # 4. EIP-1014: keccak256(0xff ++ factory ++ salt ++ keccak256(initCode))[12:]
    raw = (
        b"\xff"
        + bytes.fromhex(PROXY_WALLET_FACTORY[2:].lower())
        + salt
        + creation_code_hash
    )
    proxy = Web3.to_checksum_address(Web3.keccak(raw)[12:])
    logger.info(f"[relayer] derived proxy wallet {proxy} for EOA {eoa}")
    _derived_proxy_cache = proxy
    return proxy


# ---------------------------------------------------------------------------
# Magic auth — lazy-loaded
# ---------------------------------------------------------------------------
_magic_creds:      Optional[Dict]             = None
_magic_lock                                   = threading.Lock()
_magic_pm_session: Optional[requests.Session] = None


def _get_magic_creds() -> Optional[Dict]:
    global _magic_creds
    if AUTH_TYPE != "magic":
        return None
    with _magic_lock:
        if _magic_creds is None:
            try:
                from .magic_auth import get_session
                _magic_creds = get_session()
                logger.info("[relayer] Magic auth credentials loaded")
            except Exception as e:
                logger.error(f"[relayer] Failed to load Magic credentials: {e}")
                return None
    return _magic_creds


def _get_magic_pm_session() -> Optional[requests.Session]:
    global _magic_pm_session
    if AUTH_TYPE != "magic":
        return None
    with _magic_lock:
        try:
            from .magic_auth import get_polymarket_session
            _magic_pm_session = get_polymarket_session()
            return _magic_pm_session
        except Exception as e:
            logger.error(f"[relayer] Failed to get Polymarket session via Magic: {e}")
            return None


def _magic_sign_hex(message_hex: str) -> str:
    creds = _get_magic_creds()
    if not creds:
        raise RuntimeError("Magic credentials not available")
    from .magic_auth import sign_message
    result = sign_message(creds, message_hex)
    return result.get("signature", "")


def get_signer_address() -> str:
    if account:
        return account.address
    creds = _get_magic_creds()
    if creds:
        return creds.get("wallet_address", "")
    return ""


# ---------------------------------------------------------------------------
# In-memory session cookies
# ---------------------------------------------------------------------------
_cookie_lock     = threading.Lock()
_session_cookies: Dict[str, str] = {
    "polymarketnonce":    "",
    "polymarketsession":  "",
    "polymarketauthtype": "",
}


def set_session_cookies(nonce: str, session: str, authtype: str = "magic") -> None:
    global _session_cookies
    with _cookie_lock:
        _session_cookies = {
            "polymarketnonce":    nonce,
            "polymarketsession":  session,
            "polymarketauthtype": authtype,
        }
    logger.info("[relayer] Session cookies updated")


def get_session_cookies() -> Dict[str, str]:
    with _cookie_lock:
        return dict(_session_cookies)


def cookies_valid() -> bool:
    return bool(get_session_cookies().get("polymarketsession"))


def _try_load_magic_cookies() -> bool:
    if AUTH_TYPE != "magic":
        return False
    try:
        s = _get_magic_pm_session()
        if not s:
            return False
        jar     = s.cookies
        nonce   = jar.get("polymarketnonce")    or ""
        session = jar.get("polymarketsession")  or ""
        autht   = jar.get("polymarketauthtype") or "magic"
        if session:
            set_session_cookies(nonce, session, autht)
            logger.info("[relayer] Loaded cookies from Magic Polymarket session")
            return True
    except Exception as e:
        logger.debug(f"[relayer] Magic cookie load failed: {e}")
    return False


def refresh_cookies() -> bool:
    if _try_load_magic_cookies():
        return True
    if cookies_valid():
        return True
    logger.warning(
        "[relayer] No valid session cookies available. Options:\n"
        "  1. Ensure POLYMARKET_AUTH_TYPE=magic and magic_auth is configured, or\n"
        "  2. Call set_session_cookies(nonce, session) from your app."
    )
    return False


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
_circuit_lock        = threading.Lock()
_circuit_open_until: float = 0.0
_CIRCUIT_STATE_FILE  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".relayer_circuit_state"
)


def _load_circuit_state() -> None:
    global _circuit_open_until
    try:
        if os.path.exists(_CIRCUIT_STATE_FILE):
            val = float(open(_CIRCUIT_STATE_FILE).read().strip())
            if val > time.time():
                _circuit_open_until = val
                dt = datetime.fromtimestamp(val, tz=timezone.utc)
                logger.warning(
                    f"[relayer] Circuit still OPEN — resets at "
                    f"{dt.strftime('%H:%M:%S')} UTC (in {int(val - time.time())}s)"
                )
            else:
                os.remove(_CIRCUIT_STATE_FILE)
    except Exception as e:
        logger.debug(f"[relayer] Could not load circuit state: {e}")


def _save_circuit_state(reset_epoch: float) -> None:
    try:
        open(_CIRCUIT_STATE_FILE, "w").write(str(reset_epoch))
    except Exception as e:
        logger.debug(f"[relayer] Could not save circuit state: {e}")


def _clear_circuit_state() -> None:
    try:
        if os.path.exists(_CIRCUIT_STATE_FILE):
            os.remove(_CIRCUIT_STATE_FILE)
    except Exception as e:
        logger.debug(f"[relayer] Could not clear circuit state: {e}")


def is_circuit_open() -> bool:
    with _circuit_lock:
        return time.time() < _circuit_open_until


def _trip_circuit(reset_epoch: float) -> None:
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = reset_epoch
        _save_circuit_state(reset_epoch)
    dt = datetime.fromtimestamp(reset_epoch, tz=timezone.utc)
    logger.warning(
        f"[relayer] Circuit OPEN — resets at {dt.strftime('%H:%M:%S')} UTC "
        f"(in {int(reset_epoch - time.time())}s)"
    )


def close_circuit() -> None:
    global _circuit_open_until
    with _circuit_lock:
        _circuit_open_until = 0.0
        _clear_circuit_state()
    logger.info("[relayer] Circuit CLOSED")


def get_circuit_status() -> Dict[str, Any]:
    with _circuit_lock:
        open_until = _circuit_open_until
    now = time.time()
    if now >= open_until:
        return {"open": False, "resetsAt": None, "secondsRemaining": 0}
    dt = datetime.fromtimestamp(open_until, tz=timezone.utc)
    return {
        "open": True,
        "resetsAt": dt.isoformat(),
        "secondsRemaining": int(open_until - now),
    }


def _parse_quota_reset(error_message: str) -> float:
    m = re.search(r"resets at (\d{1,2}):(\d{2}):(\d{2})", error_message)
    if m:
        now   = datetime.now(tz=timezone.utc)
        reset = now.replace(
            hour=int(m.group(1)), minute=int(m.group(2)),
            second=int(m.group(3)), microsecond=0,
        )
        if reset <= now:
            reset += timedelta(days=1)
        return reset.timestamp()
    m = re.search(r"circuit open for (\d+)s", error_message)
    if m:
        return time.time() + int(m.group(1))
    now   = datetime.now(tz=timezone.utc)
    reset = now.replace(hour=11, minute=0, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    logger.warning(
        f"[relayer] Could not parse reset time from {error_message!r} — "
        f"defaulting to next 11:00 UTC"
    )
    return reset.timestamp()


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("quota", "429", "rate limit", "too many requests"))


def is_quota_blocked() -> bool:
    return get_circuit_status()["open"]


def quota_blocked_seconds_remaining() -> float:
    return float(get_circuit_status()["secondsRemaining"])


# Backwards-compat shims
is_cf_blocked                = is_quota_blocked
cf_blocked_seconds_remaining = quota_blocked_seconds_remaining

# ---------------------------------------------------------------------------
# ABI helpers
# ---------------------------------------------------------------------------


class ProxyTx(NamedTuple):
    to:    str
    data:  str
    value: int = 0
    op:    int = 1


def _normalise_index_sets(raw: Any) -> List[int]:
    """
    Normalise caller-supplied indexSets to always include both binary outcomes.
    For binary (YES/NO) markets the browser always sends [1, 2].
    """
    if raw and isinstance(raw, (list, tuple)) and len(raw) > 0:
        isets = sorted(set(int(x) for x in raw))
        if isets == [1] or isets == [2]:
            logger.warning(
                "[relayer] indexSets=%s covers only one outcome; upgrading to [1,2] "
                "to match browser behaviour. Pass a custom list explicitly to override.",
                isets,
            )
            return DEFAULT_INDEX_SETS
        return isets
    return list(DEFAULT_INDEX_SETS)


def _encode_redeem_calldata(condition_id: str, index_sets: List[int]) -> str:
    """Encode redeemPositions(address,bytes32,bytes32,uint256[]) calldata as 0x-hex."""
    if not condition_id.startswith("0x") or len(condition_id) != 66:
        raise ValueError(f"condition_id must be 0x+64 hex chars, got {condition_id!r}")
    selector = function_signature_to_4byte_selector(
        "redeemPositions(address,bytes32,bytes32,uint256[])"
    )
    tail = codec.encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [USDC_E, HASH_ZERO, bytes.fromhex(condition_id[2:]), index_sets],
    )
    return Web3.to_hex(selector + tail)


def _encode_execute_calldata(txns: List[ProxyTx]) -> bytes:
    selector = function_signature_to_4byte_selector(
        "proxy((uint8,address,uint256,bytes)[])"
    )
    tuples = [
        (int(t.op), Web3.to_checksum_address(t.to), int(t.value), bytes.fromhex(t.data[2:]))
        for t in txns
    ]
    encoded = codec.encode(["(uint8,address,uint256,bytes)[]"], [tuples])
    return selector + encoded


def _build_single_redeem_txn(condition_id: str, index_sets: List[int]) -> ProxyTx:
    return ProxyTx(to=CTF, data=_encode_redeem_calldata(condition_id, index_sets), op=1)


# ---------------------------------------------------------------------------
# Relay helpers
# ---------------------------------------------------------------------------

def _get_relay_payload(eoa: str) -> Dict[str, Any]:
    r = requests.get(
        f"{RELAYER_URL}/relay-payload",
        params={"address": eoa, "type": "PROXY"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _build_relay_message(eoa, calldata, nonce, gas_limit, relay_addr):
    def a20(a): return bytes.fromhex(a.lower().removeprefix("0x").zfill(40))
    def a32(n): return n.to_bytes(32, "big")
    return (
        b"rlx:"
        + a20(eoa)
        + a20(PROXY_WALLET_FACTORY)
        + calldata
        + a32(0)          # relayerFee
        + a32(0)          # gasPrice
        + a32(gas_limit)
        + a32(nonce)
        + a20(RELAY_HUB)
        + a20(relay_addr)
    )


def _sign_relay_request(eoa, encoded_function, nonce, gas_limit, relay_addr):
    packed   = _build_relay_message(eoa, encoded_function, nonce, gas_limit, relay_addr)
    msg_hash = Web3.keccak(packed)
    hex_hash = "0x" + msg_hash.hex()
    logger.info(
        "[relayer] sign fields eoa=%s relay=%s nonce=%d gas=%d hash=%s",
        eoa, relay_addr, nonce, gas_limit, hex_hash,
    )
    if account:
        from eth_account.messages import encode_defunct
        sig = account.sign_message(encode_defunct(msg_hash))
        return "0x" + sig.signature.hex()
    sig = _magic_sign_hex(hex_hash)
    # Verify signature is recoverable to the correct address
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        recovered = Account.recover_message(encode_defunct(msg_hash),
                                            signature=bytes.fromhex(sig[2:]))
        match = recovered.lower() == eoa.lower()
        logger.info("[relayer] ecrecover: recovered=%s expected=%s match=%s",
                    recovered, eoa, "OK" if match else "MISMATCH")
    except Exception as e:
        logger.warning(f"[relayer] ecrecover check failed: {e}")
    return sig


def _get_cookies_for_request() -> Any:
    """Return the best available cookie jar for a relayer request."""
    _pm_session = _get_magic_pm_session()
    if _pm_session is not None and AUTH_TYPE == "magic":
        return _pm_session.cookies
    return get_session_cookies()


# ---------------------------------------------------------------------------
# Core submit — BATCH (all conditions in one proxy() call)
# ---------------------------------------------------------------------------

def _sign_and_submit_batch(
    conditions: List[Dict[str, Any]],
    proxy_addr: str,
) -> Dict[str, Any]:
    eoa = get_signer_address()
    if not eoa:
        raise RuntimeError("No signer address available")

    txns     = [_build_single_redeem_txn(c["conditionId"], c["indexSets"]) for c in conditions]
    calldata = _encode_execute_calldata(txns)

    logger.info(
        "[relayer] execute selector: %s  txns=%d",
        ("0x" + calldata.hex())[:10], len(txns),
    )

    payload    = _get_relay_payload(eoa)
    relay_addr = payload["address"]
    nonce      = int(payload["nonce"])
    gas_limit  = 80_000 * len(txns) + 150_000
    auth_mode  = "Magic TEE" if AUTH_TYPE == "magic" else "EOA"
    signature  = _sign_relay_request(eoa, calldata, nonce, gas_limit, relay_addr)
    cookies    = _get_cookies_for_request()

    logger.info(
        "[relayer] Submitting [%s] %d conditions nonce=%d gas=%d cookies=%s",
        auth_mode, len(conditions), nonce, gas_limit,
        "OK" if cookies_valid() else "MISSING",
    )

    body = {
        "from":        eoa,
        "to":          PROXY_WALLET_FACTORY,
        "proxyWallet": Web3.to_checksum_address(proxy_addr),
        "data":        "0x" + calldata.hex(),
        "nonce":       str(nonce),
        "signature":   signature,
        "signatureParams": {
            "gasPrice":   "0",
            "gasLimit":   str(gas_limit),
            "relayerFee": "0",
            "relayHub":   RELAY_HUB,
            "relay":      relay_addr,
        },
        "type":     "PROXY",
        "metadata": "",
    }
    headers = {
        "Content-Type": "application/json",
        "Origin":       "https://polymarket.com",
        "Referer":      "https://polymarket.com",
    }

    resp = requests.post(
        f"{RELAYER_URL}/submit",
        json=body,
        cookies=cookies,
        headers=headers,
        timeout=30,
    )

    if resp.status_code == 401:
        logger.warning("[relayer] 401 — refreshing cookies and retrying")
        _try_load_magic_cookies()
        if cookies_valid():
            resp = requests.post(
                f"{RELAYER_URL}/submit",
                json=body,
                cookies=_get_cookies_for_request(),
                headers=headers,
                timeout=30,
            )

    if not resp.ok:
        try:
            err_json = resp.json()
        except Exception:
            err_json = {"raw": resp.text}
        logger.error(
            "[relayer] submit %d: %s  nonce=%d gas=%d cookies=%s",
            resp.status_code, json.dumps(err_json),
            nonce, gas_limit, cookies_valid(),
        )
        resp.raise_for_status()

    return resp.json()


# ---------------------------------------------------------------------------
# Transaction verification
# ---------------------------------------------------------------------------

_TRANSACTION_RELAYED_TOPIC = Web3.keccak(
    text="TransactionRelayed(address,address,address,bytes4,uint8,uint256)"
).hex()


def _inner_call_succeeded(tx_hash: str) -> bool:
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        for log in receipt["logs"]:
            if log["address"].lower() == RELAY_HUB.lower():
                if log["topics"] and log["topics"][0].hex() == _TRANSACTION_RELAYED_TOPIC:
                    raw    = bytes(log["data"])
                    status = int.from_bytes(raw[4 * 36: 4 * 36 + 32], "big")
                    if status != 0:
                        logger.warning(
                            f"[relayer] Inner call FAILED — "
                            f"TransactionRelayed.status={status} tx={tx_hash}"
                        )
                        return False
                    logger.debug(f"[relayer] Inner call OK (status=0) tx={tx_hash}")
                    return True
        logger.warning(f"[relayer] No TransactionRelayed log found in {tx_hash}")
        return False
    except Exception as e:
        logger.warning(f"[relayer] Could not verify inner call for {tx_hash}: {e}")
        return True


STATE_MINED     = "MINED"
STATE_CONFIRMED = "CONFIRMED"
STATE_EXECUTED  = "EXECUTED"
STATE_FAILED    = "FAILED"


def _poll_until_terminal(request_id: str, poll_timeout: float = 120.0) -> Dict[str, Any]:
    cookies  = _get_cookies_for_request()
    deadline = time.time() + poll_timeout
    last: Dict[str, Any] = {}
    last_state = ""
    elapsed = 0
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{RELAYER_URL}/transaction",
                params={"id": request_id},
                cookies=cookies,
                headers={"Origin": "https://polymarket.com"},
                timeout=10,
            )
            data  = r.json()
            d     = data[0] if isinstance(data, list) else data
            state = d.get("state", "")
            tx    = d.get("transactionHash") or d.get("txHash")
            last  = d
            if state != last_state:
                logger.info(f"[relayer] Poll {request_id[:8]}… state={state} tx={tx}")
                last_state = state
            else:
                logger.debug(f"[relayer] Poll {request_id[:8]}… state={state} ({elapsed}s)")
            if state in (STATE_MINED, STATE_CONFIRMED, STATE_EXECUTED):
                if tx and not _inner_call_succeeded(tx):
                    return {"txhash": tx, "state": STATE_FAILED, "raw": d}
                return {"txhash": tx, "state": STATE_CONFIRMED, "raw": d}
            if state in (STATE_FAILED, "FAILED", "REVERTED", "CANCELLED"):
                logger.error(f"[relayer] Transaction failed: {d}")
                return {"txhash": tx, "state": STATE_FAILED, "raw": d}
        except Exception as e:
            logger.warning(f"[relayer] Poll error: {e}")
        time.sleep(3)
        elapsed += 3
    logger.warning(f"[relayer] Poll TIMEOUT after {poll_timeout}s — last state={last_state!r} raw={last}")
    return {"txhash": None, "state": "TIMEOUT", "raw": last}


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RETRY_DELAYS = [2, 4, 8]


def _execute_with_retry_single(
    condition_id: str,
    index_sets:   List[int],
    proxy_addr:   str,
    description:  str,
) -> Dict[str, Any]:
    for attempt in range(1, 4 + 1):
        logger.info(f"[relayer] Submitting {description!r} attempt {attempt}/4")
        try:
            result = _sign_and_submit_batch(
                [{"conditionId": condition_id, "indexSets": index_sets}],
                proxy_addr,
            )
            close_circuit()
            return result
        except Exception as e:
            if _is_quota_error(e):
                reset = _parse_quota_reset(str(e))
                _trip_circuit(reset)
                raise RuntimeError("SKIPPED_QUOTA_EXCEEDED") from e
            if attempt < 4:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(f"[relayer] Attempt {attempt}/4 failed: {e}, retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"[relayer] All 4 attempts failed: {e}", exc_info=True)
                raise RuntimeError("SUBMIT_FAILED") from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redeem_winnings_via_relayer(
    condition_id:  str,
    index_sets:    Optional[List[int]] = None,
    proxy_address: Optional[str] = None,
    description:   str = "",
    poll_timeout:  float = 120.0,
) -> Dict[str, Any]:
    if not RELAYER_ENABLED:
        return {"txhash": None, "state": "SKIPPED_NO_CREDS", "raw": None}
    if is_circuit_open():
        status = get_circuit_status()
        logger.warning(f"[relayer] Circuit open — skipping (resets in {status['secondsRemaining']}s)")
        return {"txhash": None, "state": "SKIPPED_QUOTA_EXCEEDED", "raw": None}
    if not cookies_valid():
        refresh_cookies()
    if not cookies_valid():
        return {"txhash": None, "state": "SKIPPED_NO_COOKIES", "raw": None}

    proxy = proxy_address or _derive_proxy_wallet(get_signer_address())
    isets = _normalise_index_sets(index_sets)
    desc  = description or f"Redeem {condition_id}"

    try:
        submit_resp = _execute_with_retry_single(condition_id, isets, proxy, desc)
    except RuntimeError as e:
        return {"txhash": None, "state": str(e), "raw": None}

    request_id = submit_resp.get("transactionID") or submit_resp.get("id")
    logger.info(f"[relayer] Submitted txID={request_id}")
    result = _poll_until_terminal(request_id, poll_timeout)
    result["succeededCount"] = (
        1 if result["state"] in (STATE_MINED, STATE_CONFIRMED, STATE_EXECUTED) else 0
    )
    return result


def redeem_batch_via_relayer(
    conditions:    List[Dict[str, Any]],
    proxy_address: Optional[str] = None,
    description:   str = "",
    poll_timeout:  float = 180.0,
) -> Dict[str, Any]:
    """
    Redeem multiple conditions via relayer.
    All conditions are batched into ONE proxy() call — matching browser behaviour.
    indexSets are normalised: a single-side [1] or [2] is upgraded to [1, 2].
    """
    def _skipped(reason: str) -> Dict[str, Any]:
        return {"txhash": None, "state": reason, "raw": None, "succeededCount": 0}

    if not RELAYER_ENABLED:  return _skipped("SKIPPED_NO_CREDS")
    if not conditions:       return _skipped("NO_CONDITIONS")
    if is_circuit_open():    return _skipped("SKIPPED_QUOTA_EXCEEDED")
    if not cookies_valid():
        refresh_cookies()
    if not cookies_valid():  return _skipped("SKIPPED_NO_COOKIES")

    proxy = proxy_address or _derive_proxy_wallet(get_signer_address())

    # Normalise key names AND indexSets
    normalised = [
        {
            "conditionId": c.get("conditionId") or c.get("condition_id", ""),
            "indexSets":   _normalise_index_sets(
                               c.get("indexSets") or c.get("indexsets")
                           ),
        }
        for c in conditions
    ]

    logger.info("[relayer] Submitting batch of %d redeems in ONE proxy() call", len(normalised))

    submit_resp = None
    for attempt in range(1, 4 + 1):
        logger.info("[relayer] Batch attempt %d/4", attempt)
        try:
            submit_resp = _sign_and_submit_batch(normalised, proxy)
            close_circuit()
            break
        except Exception as e:
            if _is_quota_error(e):
                reset = _parse_quota_reset(str(e))
                _trip_circuit(reset)
                return _skipped("SKIPPED_QUOTA_EXCEEDED")
            if attempt < 4:
                delay = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "[relayer] Batch attempt %d/4 failed: %s, retrying in %ds",
                    attempt, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error("[relayer] All 4 batch attempts failed: %s", e, exc_info=True)
                return _skipped("SUBMIT_FAILED")

    if submit_resp is None:
        return _skipped("SUBMIT_FAILED")

    request_id = submit_resp.get("transactionID") or submit_resp.get("id")
    logger.info("[relayer] Batch submitted txID=%s", request_id)
    result    = _poll_until_terminal(request_id, poll_timeout)
    succeeded = len(normalised) if result["state"] in (STATE_MINED, STATE_CONFIRMED, STATE_EXECUTED) else 0
    result["succeededCount"] = succeeded
    logger.info(
        "[relayer] Batch complete: %d/%d confirmed — state=%s tx=%s",
        succeeded, len(normalised), result["state"], result.get("txhash"),
    )
    return result


# ---------------------------------------------------------------------------
# Misc helpers / shims
# ---------------------------------------------------------------------------

def ensure_safe_deployed() -> bool:
    logger.debug("[relayer] PROXY type — no Safe deployment needed")
    return RELAYER_ENABLED


def get_polymarket_wallet_address() -> Optional[str]:
    """Return derived proxy wallet for the configured EOA signer."""
    eoa = get_signer_address()
    if eoa:
        return _derive_proxy_wallet(eoa)
    return None


# Load persisted circuit state on import
_load_circuit_state()
