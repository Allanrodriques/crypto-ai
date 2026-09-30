"""HTTP transport with an explicit, auditable TLS override.

Why this module exists
----------------------
Some corporate networks (and this project's development environment) sit behind
a TLS-intercepting proxy that presents a self-signed certificate in the chain.
``requests`` then refuses every request with ``CERTIFICATE_VERIFY_FAILED``.

The tempting fix is ``verify=False``, which silently removes transport security
and invites a man-in-the-middle.  This project will not do that by default.
Instead:

* verification is **on** unless explicitly disabled,
* disabling is opt-in via the ``CRYPTO_ML_TLS_VERIFY`` environment variable,
* a CA bundle path may be supplied instead of disabling verification, and
* whenever verification is off, a loud warning is logged naming the risk.

Supported values for ``CRYPTO_ML_TLS_VERIFY``:

===============================  ==========================================
value                            meaning
===============================  ==========================================
unset / ``1`` / ``true``         verify against the default CA bundle
``0`` / ``false``                **insecure** - do not verify, logs a warning
any other value                  treated as a path to a CA bundle file
===============================  ==========================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils import get_logger

logger = get_logger("data.transport")

#: Environment variable controlling certificate verification.
TLS_VERIFY_ENV = "CRYPTO_ML_TLS_VERIFY"

#: Default retry policy for market-data reads.
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 1.0
DEFAULT_TIMEOUT = 30.0

_warned_insecure = False


class InsecureTLSWarning(UserWarning):
    """Raised (as a warning class) when TLS verification has been disabled."""


def tls_verify_setting() -> bool | str:
    """Resolve the effective verification setting from the environment.

    Returns ``True`` (verify with default CAs), ``False`` (verification
    disabled) or a ``str`` path to a CA bundle.
    """
    raw = os.environ.get(TLS_VERIFY_ENV)
    if raw is None:
        return True
    text = raw.strip()
    if text == "":
        return True
    if text.lower() in {"0", "false", "no", "off"}:
        return False
    if text.lower() in {"1", "true", "yes", "on"}:
        return True
    bundle = Path(text).expanduser()
    if not bundle.exists():
        raise FileNotFoundError(
            f"{TLS_VERIFY_ENV}={text!r} is neither a boolean nor an existing CA bundle path"
        )
    return str(bundle)


def _apply_tls(session: requests.Session) -> bool | str:
    """Resolve the TLS setting and record it on the session.

    The resolved value is *stored on the session* rather than only logged:
    :func:`get_json` reads it back and passes it to ``requests`` per request.
    Merely warning here - while leaving the default in force - means an operator
    who sets ``CRYPTO_ML_TLS_VERIFY=0`` still gets certificate verification, and
    the failures they were trying to work around come back with no obvious cause.
    """
    setting = tls_verify_setting()
    session.crypto_ml_tls_verify = setting  # type: ignore[attr-defined]
    if setting is False:
        global _warned_insecure
        if not _warned_insecure:
            logger.warning(
                "TLS certificate verification DISABLED via %s. Traffic is not protected "
                "against interception. Use this only on a trusted network, and prefer "
                "supplying a CA bundle path instead.",
                TLS_VERIFY_ENV,
            )
            _warned_insecure = True
    return setting


def build_session(
    *,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    status_forcelist: tuple[int, ...] | None = None,
) -> requests.Session:
    """Create a :class:`requests.Session` with retries and the TLS override.

    Parameters
    ----------
    retries:
        Maximum retry attempts for connection errors and retryable statuses.
    backoff:
        Exponential backoff factor.
    status_forcelist:
        HTTP statuses to retry.  Defaults to Binance's throttle/outage codes.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=list(status_forcelist or (418, 429, 500, 502, 503, 504)),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "crypto-ml-predictor/2.0 (research)"})
    _apply_tls(session)
    return session


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """``GET`` a URL and decode JSON, honouring the TLS override.

    ``requests`` is given the resolved ``verify`` value explicitly because the
    setting is decided per-process here rather than per-session by urllib3.  A
    session built by :func:`build_session` carries the value; a session built
    elsewhere falls back to ``requests``' default (verification on), so an
    unconfigured session is never *less* safe than the library default.
    """
    verify = getattr(session, "crypto_ml_tls_verify", True)
    response = session.get(url, params=params, timeout=timeout, verify=verify)
    response.raise_for_status()
    return response.json()
