"""The UI server's security envelope (docs/design/phase3/ui.md §3.2).

- **No cookies.** Chrome sends a cookie set by one loopback port to every other one (C4; RFC 6265 §8.5), so the API
  takes ``Authorization: Bearer <secret>`` only: the **server token** (the 0600 runtime file; scripts and the CLI) or
  a **session secret** (server memory; a browser tab keeps it in its origin's ``localStorage``). Secrets are compared
  with ``hmac.compare_digest``; sessions, nonces and tickets are looked up by their SHA-256.
- **Nonces** (the redirect file's ``#once=``) and **WebSocket tickets** are single-use and expire after their TTL. The
  TTLs are constructor parameters, so that tests run them at 0.5 s (§8.2 P1).
- **Request checks**, in this order: Host (a loopback name, any port: DNS rebinding), ``Sec-Fetch-Site``, ``Origin``
  (unsafe API methods and the WebSocket upgrade), then the credential. :func:`check_request` does the first three.
- **Headers** on every response (:data:`RESPONSE_HEADERS`) and the content policy of run files (:func:`file_headers`).
- **Logs** never hold a secret: :class:`RedactingFormatter` replaces the values of ``token=``, ``once=`` and
  ``ticket=`` (and Bearer values) with ``…``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

#: Host names accepted in the Host header, with any port (``ssh -L`` may forward another local port).
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "[::1]"})
#: The addresses the server may bind (any other ``--bind`` is BS-UI-BIND).
BIND_ADDRESSES = ("127.0.0.1", "::1")
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
CSP = ("default-src 'self'; connect-src 'self'; img-src 'self' blob: data:; worker-src 'self' blob:; "
       "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
#: Protocol major version, on every response.
PROTOCOL_HEADER = ("X-BallaSim-UI", "1")
RESPONSE_HEADERS = {
    "Content-Security-Policy": CSP,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    PROTOCOL_HEADER[0]: PROTOCOL_HEADER[1],
}
#: Run files (``GET /runs/{rid}/files/…``): types by extension; anything else is a download.
FILE_TYPES = {".png": "image/png", ".json": "application/json", ".txt": "text/plain; charset=utf-8",
              ".log": "text/plain; charset=utf-8", ".jsonl": "text/plain; charset=utf-8", ".h5": "application/x-hdf5"}
FILE_CSP = "sandbox; default-src 'none'"


class Denied(Exception):
    """A refused request: HTTP status, BS-UI code and message (§6)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def host_name(host: str) -> str | None:
    """The name part of a Host header value (``[::1]`` keeps its brackets), or None when the value is malformed."""
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return None
        name, rest = host[:end + 1], host[end + 1:]
    else:
        name, _, port = host.partition(":")
        rest = ":" + port if _ else ""
    if rest and not (rest.startswith(":") and rest[1:].isdigit()):
        return None
    return name.lower()


def check_request(headers: Mapping[str, str], method: str, *, api: bool, websocket: bool = False) -> None:
    """The Host, ``Sec-Fetch-Site`` and ``Origin`` checks of §3.2 (rules 1–3); raises :class:`Denied`.

    Static files accept any ``Sec-Fetch-Site`` on a navigation (the redirect file's ``file://`` page arrives
    ``cross-site``, C4); other requests need it absent, ``same-origin`` or ``none``. ``Origin`` on an unsafe API
    method or the WebSocket upgrade must be absent (a non-browser client, which still needs a secret) or exactly
    ``http://<Host>``."""
    host = headers.get("Host", "")
    if host and host_name(host) not in LOOPBACK_NAMES:
        raise Denied(403, "BS-UI-HOST", "the Host header must name 127.0.0.1, localhost or [::1]")
    site = headers.get("Sec-Fetch-Site")
    navigation = not api and headers.get("Sec-Fetch-Mode") == "navigate"
    if site is not None and site not in ("same-origin", "none") and not navigation:
        raise Denied(403, "BS-UI-FETCHSITE", f"Sec-Fetch-Site {site} is not allowed here")
    if api and (websocket or method.upper() not in SAFE_METHODS):
        origin = headers.get("Origin")
        if origin is not None and origin != f"http://{host}":
            raise Denied(403, "BS-UI-ORIGIN", "a request from another origin")


def file_headers(name: str) -> dict[str, str]:
    """The content policy of a run file (§3.2): an allow-listed type by extension, else an octet-stream download;
    always the ``sandbox`` CSP, so a ``.js``, ``.html`` or ``.svg`` file in a root never runs on the origin."""
    dot = name.rfind(".")
    ext = name[dot:].lower() if dot > name.rfind("/") else ""
    out = {"Content-Security-Policy": FILE_CSP}
    if ext in FILE_TYPES:
        out["Content-Type"] = FILE_TYPES[ext]
    else:
        out["Content-Type"] = "application/octet-stream"
        out["Content-Disposition"] = "attachment"
    return out


@dataclass(frozen=True)
class Credential:
    """An authenticated caller: the server token's holder, or a browser session (``key``: its SHA-256)."""

    kind: str           # "server" | "session"
    key: bytes | None = None


def _digest(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8", "surrogatepass")).digest()


class Secrets:
    """The server token, the browser sessions, the single-use nonces and WebSocket tickets (§3.2)."""

    def __init__(self, *, nonce_ttl: float = 60.0, ticket_ttl: float = 10.0, redirect_ttl: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.nonce_ttl, self.ticket_ttl, self.redirect_ttl = nonce_ttl, ticket_ttl, redirect_ttl
        self._clock = clock
        self.server_token = secrets.token_urlsafe(32)
        self._sessions: set[bytes] = set()
        self._nonces: dict[bytes, float] = {}
        self._tickets: dict[bytes, tuple[float, Credential]] = {}

    def _purge(self) -> None:
        now = self._clock()
        self._nonces = {k: t for k, t in self._nonces.items() if t > now}
        self._tickets = {k: v for k, v in self._tickets.items() if v[0] > now}

    def new_nonce(self) -> str:
        """A single-use login nonce for a redirect file, valid for ``nonce_ttl``."""
        self._purge()
        nonce = secrets.token_urlsafe(24)
        self._nonces[_digest(nonce)] = self._clock() + self.nonce_ttl
        return nonce

    def new_session(self, *, once: str | None = None, token: str | None = None) -> str:
        """``POST /session``: a new session secret for an unused, unexpired nonce or for the server token."""
        self._purge()
        if isinstance(once, str) and once:
            if self._nonces.pop(_digest(once), None) is None:
                raise Denied(401, "BS-UI-AUTH", "the sign-in link was used or has expired: "
                                                "reopen the UI with python -m ballasim ui --open")
        elif not (isinstance(token, str) and token and self.is_server_token(token)):
            raise Denied(401, "BS-UI-AUTH", "wrong sign-in secret")
        session = secrets.token_urlsafe(32)
        self._sessions.add(_digest(session))
        return session

    def is_server_token(self, secret: str) -> bool:
        return hmac.compare_digest(secret.encode("utf-8", "surrogatepass"), self.server_token.encode())

    def authenticate(self, authorization: str | None, *, server_only: bool = False) -> Credential:
        """The credential of an ``Authorization: Bearer <secret>`` header; ``server_only`` endpoints (``/attach``,
        ``/once``, ``/shutdown``) refuse session secrets."""
        scheme, _, secret = (authorization or "").partition(" ")
        secret = secret.strip()
        if scheme.lower() == "bearer" and secret:
            if self.is_server_token(secret):
                return Credential("server")
            key = _digest(secret)
            if key in self._sessions:
                if server_only:
                    raise Denied(401, "BS-UI-AUTH", "this endpoint needs the server token")
                return Credential("session", key)
        raise Denied(401, "BS-UI-AUTH", "missing or wrong Bearer secret: reopen the UI with python -m ballasim ui "
                                        "--open")

    def drop_session(self, credential: Credential) -> None:
        """``DELETE /session``: forgets the caller's session (the server token has none)."""
        if credential.kind == "session":
            self._sessions.discard(credential.key)

    def new_ticket(self, credential: Credential) -> str:
        """``POST /ws-ticket``: a single-use WebSocket ticket for the caller, valid for ``ticket_ttl``."""
        self._purge()
        ticket = secrets.token_urlsafe(24)
        self._tickets[_digest(ticket)] = (self._clock() + self.ticket_ttl, credential)
        return ticket

    def redeem_ticket(self, ticket: str | None) -> Credential:
        """The credential behind an unused, unexpired ticket (whose session still exists); single use."""
        self._purge()
        entry = self._tickets.pop(_digest(ticket), None) if ticket else None
        if entry is None or (entry[1].kind == "session" and entry[1].key not in self._sessions):
            raise Denied(401, "BS-UI-AUTH", "missing, used or expired WebSocket ticket")
        return entry[1]


_REDACT = (
    (re.compile(r"\b(token|once|ticket)=[^&\s#\"'<>]+"), r"\1=…"),
    (re.compile(r"(\"(?:token|once|ticket|session)\"\s*:\s*)\"[^\"]*\""), r'\1"…"'),
    (re.compile(r"\b(Bearer)\s+[^\s\"']+", re.IGNORECASE), r"\1 …"),
)


def redact(text: str) -> str:
    """``text`` with the values of ``token=``, ``once=``, ``ticket=``, JSON secret fields and Bearer values replaced
    by ``…``."""
    for pattern, repl in _REDACT:
        text = pattern.sub(repl, text)
    return text


class RedactingFormatter(logging.Formatter):
    """A formatter whose output (message, arguments and traceback) passes :func:`redact`."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))
