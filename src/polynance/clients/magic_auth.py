"""
magic_auth.py — Polymarket Magic wallet authentication + Polymarket session

Verified against HAR capture 2026-02-21.

DEPENDENCIES:
    pip install curl_cffi cryptography requests
"""

import base64
import json
import logging
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Optional

from curl_cffi import requests as cf_requests
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, NoEncryption, load_pem_private_key,
)
import requests  # plain requests for gamma-api / relayer

logger = logging.getLogger(__name__)

CREDS_FILE   = Path.home() / '.polymarket_magic_creds.pkl'
TOASTER_BASE = 'https://api.toaster.magic.link'
API_KEY      = 'pk_live_99ABD23F9F1C8266'
TEE_BASE     = 'https://tee.express.magiclabs.com'
IMPERSONATE  = 'firefox133'   # matches HAR Firefox/148 TLS fingerprint class

# Polymarket-specific
GAMMA_API = "https://gamma-api.polymarket.com"

# Default aud — learned from first DIDT capture
DEFAULT_POLYMARKET_MAGIC_AUD = "XyeN-lDYx6Ne959LdUfJJeTEwVhePXofbUD0cYgtkdQ="

POLYMARKET_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (X11; Linux x86_64; rv:148.0) Gecko/20100101 Firefox/148.0",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-AU,en-US;q=0.9",
    "Origin":          "https://polymarket.com",
    "Referer":         "https://polymarket.com/",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "same-site",
    "Sec-Fetch-Dest":  "empty",
    "Cache-Control":   "no-cache",
}


# ── DPoP + ua-sig (ES256, P-256) ──────────────────────────────────────────────

class DPoPSigner:
    """
    Ephemeral EC P-256 keypair used for:
      - DPoP proof JWTs (fresh per request, RFC 9449)
      - ua-sig: ES256.sign(UserAgent, this_key)
    """

    USER_AGENT = (
        'Mozilla/5.0 (X11; Linux x86_64; rv:148.0) '
        'Gecko/20100101 Firefox/148.0'
    )

    def __init__(self):
        self._key = ec.generate_private_key(ec.SECP256R1())
        pub = self._key.public_key().public_numbers()
        self._jwk = {
            'alg': 'ES256', 'crv': 'P-256', 'ext': True,
            'key_ops': [], 'kty': 'EC',
            'x': self._b64url(pub.x.to_bytes(32, 'big')),
            'y': self._b64url(pub.y.to_bytes(32, 'big')),
        }

    @staticmethod
    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

    def make_proof(self) -> str:
        """Fresh DPoP JWT — call once per request."""
        header  = {'typ': 'dpop+jwt', 'alg': 'ES256', 'jwk': self._jwk}
        payload = {'iat': int(time.time()), 'jti': str(uuid.uuid4())}
        h64 = self._b64url(json.dumps(header,  separators=(',', ':')).encode())
        p64 = self._b64url(json.dumps(payload, separators=(',', ':')).encode())
        msg = f'{h64}.{p64}'.encode()
        der = self._key.sign(msg, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return f'{h64}.{p64}.{self._b64url(r.to_bytes(32, "big") + s.to_bytes(32, "big"))}'

    def make_ua_sig(self) -> str:
        """
        Raw ES256 signature (r||s, 64 bytes, base64url) of the User-Agent string.
        Required on: email_otp/start, mfa/verify
        """
        der = self._key.sign(self.USER_AGENT.encode(), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        return self._b64url(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))


# ── Session factory ────────────────────────────────────────────────────────────

def _new_session() -> cf_requests.Session:
    """curl_cffi Session impersonating Firefox 133 TLS fingerprint."""
    return cf_requests.Session(impersonate=IMPERSONATE)


# ── Cloudflare cookie harvesting ───────────────────────────────────────────────

def _get_cf_cookies(session: cf_requests.Session, url: str) -> dict:
    """Fire OPTIONS preflight to harvest CF cookies."""
    try:
        resp = session.options(
            url,
            headers={
                'accept':                         '*/*',
                'accept-language':                'en-AU,en-US;q=0.9',
                'access-control-request-method':  'POST',
                'access-control-request-headers': (
                    'accept-language,content-type,dpop,ua-sig,'
                    'x-amzn-trace-id,x-magic-api-key,x-magic-bundle-id,'
                    'x-magic-chain,x-magic-meta,x-magic-network,'
                    'x-magic-referrer,x-magic-sdk,x-magic-trace-id'
                ),
                'origin':  'https://auth.magic.link',
                'referer': 'https://auth.magic.link/',
            },
            timeout=10,
        )
        cookies = dict(resp.cookies)
        logger.debug(f'[magic_auth] CF cookies harvested: {list(cookies.keys())}')
        return cookies
    except Exception as e:
        logger.debug(f'[magic_auth] OPTIONS preflight warning: {e}')
        return {}


# ── Headers ────────────────────────────────────────────────────────────────────

def _make_headers(signer: DPoPSigner, extra: dict = None, ua_sig: bool = False) -> dict:
    """
    ua_sig=True  → email_otp/start, mfa/verify
    ua_sig=False → email_otp/verify, session/refresh, core/user
    """
    trace_id = str(uuid.uuid4())
    h = {
        'accept':            'application/json, text/plain, */*',
        'accept-language':   'en_US',
        'content-type':      'application/json;charset=UTF-8',
        'origin':            'https://auth.magic.link',
        'referer':           'https://auth.magic.link/',
        'x-amzn-trace-id':   f'Root={trace_id}',
        'x-magic-trace-id':  trace_id,
        'x-magic-network':   'mainnet',
        'x-magic-chain':     'ETH',
        'x-magic-api-key':   API_KEY,
        'x-magic-bundle-id': 'BundleIDMissing',
        'x-magic-referrer':  'https://polymarket.com',
        'x-magic-sdk':       'magic-sdk',
        'x-magic-meta':      'None',
        'dpop':              signer.make_proof(),
    }
    if ua_sig:
        h['ua-sig'] = signer.make_ua_sig()
    if extra:
        h.update(extra)
    return h


# ── Core request helpers ───────────────────────────────────────────────────────

def _post(session: cf_requests.Session, url: str, headers: dict,
          json_body: dict, timeout: int = 15) -> cf_requests.Response:
    """OPTIONS → POST with Cloudflare cookies injected from preflight response."""
    cf_cookies = _get_cf_cookies(session, url)
    return session.post(
        url,
        headers=headers,
        cookies=cf_cookies,
        json=json_body,
        timeout=timeout,
    )


def _get(session: cf_requests.Session, url: str, headers: dict,
         timeout: int = 10) -> cf_requests.Response:
    """OPTIONS → GET with Cloudflare cookies injected from preflight response."""
    cf_cookies = _get_cf_cookies(session, url)
    return session.get(
        url,
        headers=headers,
        cookies=cf_cookies,
        timeout=timeout,
    )


# ── JWT helpers ────────────────────────────────────────────────────────────────

def _jwt_payload(token: str) -> dict:
    try:
        part = token.split('.')[1]
        part += '=' * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part))
    except Exception:
        return {}


def _is_valid(token: str, buffer: int = 300) -> bool:
    exp = _jwt_payload(token).get('exp', 0)
    return bool(token) and (exp - time.time()) > buffer


def _expires_in(token: str) -> int:
    return max(0, int(_jwt_payload(token).get('exp', 0) - time.time()))


# ── Credential persistence ─────────────────────────────────────────────────────

def _load_creds() -> Optional[dict]:
    if not CREDS_FILE.exists():
        return None
    try:
        with open(CREDS_FILE, 'rb') as f:
            creds = pickle.load(f)
    except Exception as e:
        logger.warning(f'[magic_auth] Could not load creds: {e}')
        return None
    # Reconstruct the DPoP signer from persisted PEM — same keypair, same JWK thumbprint.
    # Without this, every restart generates a fresh key, invalidating the refresh token.
    signer_pem = creds.pop('_signer_pem', None)
    if signer_pem:
        signer = DPoPSigner.__new__(DPoPSigner)
        signer._key = load_pem_private_key(signer_pem, password=None)
        pub = signer._key.public_key().public_numbers()
        signer._jwk = {
            'alg': 'ES256', 'crv': 'P-256', 'ext': True,
            'key_ops': [], 'kty': 'EC',
            'x': DPoPSigner._b64url(pub.x.to_bytes(32, 'big')),
            'y': DPoPSigner._b64url(pub.y.to_bytes(32, 'big')),
        }
        creds['_signer'] = signer
    else:
        creds.setdefault('_signer', DPoPSigner())  # first-run fallback

    creds.setdefault('_session', _new_session())

    fridge = creds.get('fridge', '')
    if _is_valid(fridge, buffer=300):
        exp = _expires_in(fridge)
        logger.info(
            f"[magic_auth] Loaded session for {creds.get('email')} "
            f"(expires in {exp // 3600}h {(exp % 3600) // 60}m)"
        )
        return creds
    if creds.get('refresh_token'):
        logger.info('[magic_auth] Fridge expired — attempting silent refresh...')
        if _refresh_session(creds):
            return creds
    logger.info('[magic_auth] Session expired — re-login required')
    return None


def _save_creds(creds: dict) -> None:
    to_save = {k: v for k, v in creds.items() if not k.startswith('_')}
    # Persist the DPoP signer's private key so refresh works after restart.
    # The same keypair must be used across sessions for the JWK thumbprint to match.
    signer = creds.get('_signer')
    if isinstance(signer, DPoPSigner):
        to_save['_signer_pem'] = signer._key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
    try:
        with open(CREDS_FILE, 'wb') as f:
            pickle.dump(to_save, f)
        os.chmod(CREDS_FILE, 0o600)
    except Exception as e:
        logger.warning(f'[magic_auth] Could not save creds: {e}')


# ── Session refresh ────────────────────────────────────────────────────────────

def _refresh_session(creds: dict) -> bool:
    """OPTIONS → POST /v1/auth/session/refresh"""
    signer  = creds.get('_signer')  or DPoPSigner()
    session = creds.get('_session') or _new_session()
    try:
        r = _post(
            session,
            f'{TOASTER_BASE}/v1/auth/session/refresh',
            headers=_make_headers(signer, ua_sig=False),
            json_body={'refresh_token': creds.get('refresh_token', '')},
        )
        if r.status_code == 200:
            data = r.json()
            if data.get('fridge_access_token'):
                creds['fridge']        = data['fridge_access_token']
            if data.get('session_token'):
                creds['session_token'] = data['session_token']
            if data.get('refresh_token'):
                creds['refresh_token'] = data['refresh_token']
            _save_creds(creds)
            logger.info('[magic_auth] Session refreshed successfully')
            return True
        logger.warning(f'[magic_auth] Refresh failed: {r.status_code} {r.text[:120]}')
    except Exception as e:
        logger.warning(f'[magic_auth] Refresh error: {e}')
    return False


# ── Login (OTP flow) ───────────────────────────────────────────────────────────

def _login() -> dict:
    print('\n[Polymarket] Magic wallet login required')
    email = input('  Email address: ').strip()
    if not email:
        raise ValueError('Email address is required')

    signer  = DPoPSigner()
    session = _new_session()

    # Step 1: Send OTP
    logger.info(f'[magic_auth] Requesting OTP for {email}')
    r = _post(
        session,
        f'{TOASTER_BASE}/v1/auth/email_otp/start',
        headers=_make_headers(signer, ua_sig=True),
        json_body={'email': email, 'overrides': {}},
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f'Failed to start OTP flow: {r.status_code} {r.text[:200]}')
    flow_id = r.json().get('flow_id', '')
    if not flow_id:
        raise RuntimeError(f'No flow_id in OTP start response: {r.text[:200]}')
    print(f'  OTP sent to {email}')

    # Step 2: Verify OTP
    otp = input('  Enter OTP from email: ').strip()
    r = _post(
        session,
        f'{TOASTER_BASE}/v1/auth/email_otp/verify',
        headers=_make_headers(signer, ua_sig=False),
        json_body={'challenge_response': otp, 'flow_id': flow_id},
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f'OTP verify failed: {r.status_code} {r.text[:200]}')
    otp_data      = r.json()
    next_factors  = otp_data.get('next_factors', [])
    session_token = otp_data.get('session_token')       or ''
    fridge        = otp_data.get('fridge_access_token') or ''
    refresh_token = otp_data.get('refresh_token')       or ''
    auth_user_id  = otp_data.get('auth_user_id')        or ''

    # Step 3: MFA / TOTP (if required)
    if next_factors:
        totp_factor = next(
            (f for f in next_factors if f.get('flow_type') == 'totp'),
            next_factors[0]
        )
        print('  MFA enabled — check your authenticator app.')
        totp = input('  Enter 6-digit TOTP: ').strip()
        r = _post(
            session,
            f'{TOASTER_BASE}/v1/auth/mfa/verify',
            headers=_make_headers(signer, ua_sig=True),
            json_body={'flow_id': totp_factor['flow_id'], 'challenge_response': totp},
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f'MFA verify failed: {r.status_code} {r.text[:200]}')
        mfa = r.json()
        session_token = mfa.get('session_token')       or session_token
        fridge        = mfa.get('fridge_access_token') or fridge
        refresh_token = mfa.get('refresh_token')       or refresh_token
        auth_user_id  = mfa.get('auth_user_id')        or auth_user_id

    if not fridge:
        raise RuntimeError('No fridge_access_token in auth response')

    # Step 4: Fetch wallet address
    wallet_address = ''
    try:
        r = _get(
            session,
            f'{TOASTER_BASE}/v1/core/user',
            headers=_make_headers(signer, extra={
                'authorization': f'Bearer {session_token}',
            }, ua_sig=False),
        )
        if r.status_code == 200:
            u = r.json()
            wallets = u.get('wallets', [])
            wallet_address = wallets[0].get('public_address', '') if wallets else ''
    except Exception as e:
        logger.warning(f'[magic_auth] Could not fetch wallet: {e}')

    creds = {
        'email':          email,
        'auth_user_id':   auth_user_id,
        'session_token':  session_token,
        'refresh_token':  refresh_token,
        'fridge':         fridge,
        'wallet_address': wallet_address,
        # polymarket fields filled later
        'polymarket_session':          '',
        'polymarket_nonce':            '',
        'polymarket_auth_type':        '',
        'polymarket_session_expires':  0,
        'didt':                        '',
        'didt_expires':                0,
        'aud':                         DEFAULT_POLYMARKET_MAGIC_AUD,
        '_signer':        signer,
        '_session':       session,
    }
    _save_creds(creds)
    print(f'  [OK] Logged in: {email}  (wallet: {wallet_address or "unknown"})\n')
    return creds


# ── Public API: Magic core ─────────────────────────────────────────────────────

def get_session() -> dict:
    """Load cached Magic session (auto-refreshes if expired) or prompt for login."""
    creds = _load_creds()
    if creds:
        return creds
    return _login()


def refresh_fridge(creds: dict) -> bool:
    """Force-refresh the fridge token. Returns True on success."""
    return _refresh_session(creds)


def sign_message(creds: dict, message: str) -> dict:
    if not _is_valid(creds.get('fridge', ''), buffer=30):
        if not _refresh_session(creds):
            raise RuntimeError('Fridge expired and refresh failed')

    signer  = creds.get('_signer')  or DPoPSigner()
    session = creds.get('_session') or _new_session()
    url     = f'{TEE_BASE}/v1/wallet/sign/message'

    # Relay hashes arrive as "0x" + 64 hex chars — sign the 32 raw bytes
    if isinstance(message, str) and message.startswith('0x') and len(message) == 66:
        raw = bytes.fromhex(message[2:])
        logger.info("[magic_auth] sign_message: RAW 32-byte hash path")
    else:
        raw = message.encode()
        logger.info("[magic_auth] sign_message: UTF-8 string path (DIDT)")

    def _do_sign():
        cf_cookies = _get_cf_cookies(session, url)
        return session.post(
            url,
            headers=_make_headers(signer, extra={
                'authorization': f'Bearer {creds["fridge"]}',
            }, ua_sig=False),
            cookies=cf_cookies,
            json={'message_base64': base64.b64encode(raw).decode(), 'chain': 'ETH'},
            timeout=15,
        )

    r = _do_sign()
    logger.info("[magic_auth] TEE sign/message -> %s | %s", r.status_code, r.text[:300])
    if r.status_code == 401:
        if _refresh_session(creds):
            r = _do_sign()
        else:
            raise RuntimeError('TEE 401 — re-login required')
    r.raise_for_status()
    return r.json()


def get_wallet_address(creds: dict) -> str:
    """Return the cached ETH wallet address."""
    return creds.get('wallet_address', '')


# ── DIDT construction ──────────────────────────────────────────────────────────

def _build_didt(creds: dict) -> str:
    """
    Construct a Magic DID Token (DIDT) by signing the claim via TEE.
    DIDT format: base64( [ proof, JSON.stringify(claim) ] )
    """
    wallet = creds['wallet_address']
    if not wallet:
        raise RuntimeError("No wallet_address in creds")
    iat = int(time.time())
    aud = creds.get('aud') or DEFAULT_POLYMARKET_MAGIC_AUD
    claim = {
        "iat": iat,
        "ext": iat + 604800,  # 7 days
        "iss": f"did:ethr:{wallet}",
        "sub": base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode(),
        "aud": aud,
        "nbf": iat,
        "tid": str(uuid.uuid4()),
    }
    claim_str = json.dumps(claim, separators=(",", ":"))
    sig_res   = sign_message(creds, claim_str)
    proof     = sig_res.get("signature", "")
    if not proof:
        raise RuntimeError(f"TEE signing failed: {sig_res}")
    didt_payload = json.dumps([proof, claim_str], separators=(",", ":")).encode()
    didt = base64.b64encode(didt_payload).decode()
    creds["didt"]         = didt
    creds["didt_expires"] = claim["ext"]
    creds["aud"]          = aud
    _save_creds(creds)
    logger.info("[magic_auth] DIDT built / refreshed")
    return didt


# ── Polymarket gamma-api login ────────────────────────────────────────────────

def _polymarket_login(creds: dict) -> bool:
    """Exchange DIDT for polymarketsession + polymarketnonce cookies."""
    now  = time.time()
    didt = creds.get("didt", "")
    if not didt or now > creds.get("didt_expires", 0) - 300:
        didt = _build_didt(creds)
        creds["didt"] = didt

    logger.info("[magic_auth] Starting Polymarket gamma login...")
    s = requests.Session()
    s.headers.update(POLYMARKET_HEADERS)

    # Step 1: /nonce
    r1 = s.get(f"{GAMMA_API}/nonce")
    logger.info("[magic_auth] /nonce -> %s", r1.status_code)
    if not r1.ok:
        logger.warning("[magic_auth] /nonce failed: %s %s", r1.status_code, r1.text[:300])
        return False

    # Step 2: /login with DIDT
    login_headers = dict(s.headers)
    login_headers["Authorization"] = f"Bearer {didt}"
    r2 = s.get(f"{GAMMA_API}/login", headers=login_headers)
    logger.info("[magic_auth] /login -> %s", r2.status_code)
    if not r2.ok:
        logger.warning("[magic_auth] /login failed: %s %s", r2.status_code, r2.text[:300])
        return False

    creds["polymarket_session"]         = s.cookies.get("polymarketsession", "")
    creds["polymarket_nonce"]           = s.cookies.get("polymarketnonce", "")
    creds["polymarket_auth_type"]       = s.cookies.get("polymarketauthtype", "")
    creds["polymarket_session_expires"] = time.time() + 518400  # ~6 days
    logger.info(
        "[magic_auth] Polymarket session obtained (session=%s... nonce=%s authtype=%s)",
        (creds["polymarket_session"] or "")[:16],
        bool(creds["polymarket_nonce"]),
        creds["polymarket_auth_type"],
    )
    _save_creds(creds)
    return bool(creds["polymarket_session"])


def get_polymarket_session() -> requests.Session:
    creds  = get_session()
    now    = time.time()
    pm_exp = creds.get("polymarket_session_expires", 0)
    if not creds.get("polymarket_session") or now > pm_exp - 300:
        logger.info("[magic_auth] Polymarket session expired — refreshing...")
        ok = _polymarket_login(creds)
        if not ok:
            raise RuntimeError("Polymarket login failed")

    s = requests.Session()
    s.headers.update(POLYMARKET_HEADERS)
    domain = "polymarket.com"
    s.cookies.set("polymarketsession",  creds.get("polymarket_session", ""),  domain=domain)
    s.cookies.set("polymarketnonce",    creds.get("polymarket_nonce", ""),    domain=domain)
    s.cookies.set("polymarketauthtype", creds.get("polymarket_auth_type", ""), domain=domain)
    return s
