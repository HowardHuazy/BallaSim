"""The UI server (docs/design/phase3/ui.md §3.1–3.3) and the ``python -m ballasim ui`` command.

The server is I/O glue on one asyncio loop (Tornado, the ``ui`` extra): it never maps libcuda (the device probe runs
in a subprocess, §3.1), serves the static front end publicly, and the API ``/api/v1`` behind the envelope of
:mod:`.security`. This module holds the application, the runtime directory (``server.json`` and the browser redirect
files, both 0600 in a 0700 directory), the device probe, the GPU memory reading of ``/meta`` and the command's modes:

- ``ui``: a foreground server; prints ``{"url": ".../login#token=…", "pid": …}`` and, unless ``--no-browser``, opens
  the browser on a redirect file;
- ``ui --open [PATH]``: reuses the live server named by the runtime file or starts one detached (settings forwarded;
  it prints no secret and logs, redacted, to ``$XDG_STATE_HOME/ballasim/ui/server.log``), then opens the browser on a
  redirect file from ``POST /attach`` (a PATH) or ``POST /once``;
- ``ui --status`` and ``ui --stop``.

The event streams (``/streams``, their rate overview and blocks: UI-1) live in :mod:`.streams`; ``/colors`` (UI-2)
serves the viewer ``ballasim.viz``'s colours; real vs sim at the viewer level (``/compare/region``, its BSMK masks and
``/compare/stats``: UI-8a) lives in :mod:`.compare`; the roots, ``/attach``, ``/fs`` and the run pages (UI-6) in
:mod:`.fs`, the jobs (``/jobs``: queue, lanes, spawn, cancel, reattach) in :mod:`.jobs`, the WebSocket hub with
its tailers in :mod:`.live` and the forms (``/form``, ``/resolve``, ``/configs``: UI-4a) in :mod:`.api_params`.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import functools
import hashlib
import importlib.machinery
import inspect
import json
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import tornado.httpserver
import tornado.httputil
import tornado.netutil
import tornado.web
import tornado.websocket

import ballasim
from ballasim import _core

from .api_params import Params
from .compare import Compare
from .fs import Fs, Root
from .jobs import Jobs, job_id_of
from .live import Hub
from .security import (
    BIND_ADDRESSES,
    RESPONSE_HEADERS,
    Credential,
    Denied,
    RedactingFormatter,
    Secrets,
    check_request,
    file_headers,
    redact,
)
from .streams import IMAGE_KINDS, MAX_BLOCKS, MAX_SLICE_EVENTS, StreamError, Streams, _path_errors, resolve_path

log = logging.getLogger("ballasim.ui")

STATIC_DIR = Path(__file__).with_name("static")
API = "/api/v1"
PROTOCOL = [1, 0]
MIN_CHROME = 147
RUNTIME_FILE = "server.json"
#: The device probe (§3.1): ``device_available`` maps libcuda, so the server asks a subprocess once.
PROBE_CODE = "from ballasim import _engine; print(int(_engine.device_available()))"
BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
DEFAULT_BIND = "127.0.0.1"
#: The exit code of a detached server that stops because the runtime file names another live server.
EXIT_UNRECORDED = 3
DETACHED_START_S = 10.0
_REDIRECT_NAME = re.compile(r"open-[0-9a-f]{16}\.html")
#: A resource id in a URL: base64url without padding (§3.3).
RID = r"[A-Za-z0-9_-]{1,4096}"
_ROOT_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
_PROFILE_FILE = re.compile(r"[^/]+/[^/]+@v\d+\.yaml")
_STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8"}


class UIError(Exception):
    """A refused ``ui`` command line or start (exit 2), e.g. BS-UI-BIND."""


# ---------------------------------------------------------------------------------------------------------------
# Runtime and state directories
# ---------------------------------------------------------------------------------------------------------------

def _private_dir(path: Path) -> Path:
    """``path`` as a directory of this user's with mode 0700 (created, or its group and other bits cleared)."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = path.stat()
    if st.st_uid != os.getuid():
        raise UIError(f"{path} belongs to another user")
    if st.st_mode & 0o077:
        os.chmod(path, 0o700)
    return path


def runtime_dir() -> Path:
    """``$XDG_RUNTIME_DIR/ballasim-ui``, else ``~/.cache/ballasim/ui`` (0700): ``server.json`` and redirect files."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    return _private_dir(Path(base) / "ballasim-ui" if base else Path.home() / ".cache" / "ballasim" / "ui")


def state_dir() -> Path:
    """``$XDG_STATE_HOME/ballasim/ui`` (default ``~/.local/state``; 0700): ``server.log`` and the jobs root."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return _private_dir(Path(base) / "ballasim" / "ui")


def write_private(path: Path, text: str) -> None:
    """Writes ``text`` to ``path`` atomically with mode 0600."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, text.encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)


def open_log(path: Path) -> int:
    """``server.log`` opened for appending, mode 0600 (a file descriptor)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(fd, 0o600)
    return fd


def _pid_alive(pid: int) -> bool:
    """A running process (an exited, unreaped one keeps ``/proc/<pid>/stat`` in state Z)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


@contextlib.contextmanager
def _flock(path: Path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Settings:
    """The server's settings (the ``ui`` options, §3.1)."""

    bind: str = DEFAULT_BIND
    port: int = 0
    roots: list[Root] = field(default_factory=list)
    jobs_root: Path | None = None
    repo: Path | None = None
    gpu_lock: Path | None = None
    cache_mb: int = 512
    resim_device: str = "auto"
    dev: bool = False

    @property
    def host(self) -> str:
        return "[::1]" if self.bind == "::1" else self.bind


def find_repo() -> Path | None:
    """The nearest directory above the package holding ``tools/evimo2_bench/evb.py`` (a source checkout)."""
    for d in Path(ballasim.__file__).resolve().parents:
        with contextlib.suppress(OSError):
            if (d / "tools" / "evimo2_bench" / "evb.py").is_file():
                return d
    return None


def _evimo2_default() -> Path | None:
    """The machine's EVIMO2 root when it is a directory this user can reach, else None: the default is optional, so a
    root behind a directory this user cannot traverse (``is_dir`` raises EACCES on Python 3.12) is simply absent."""
    from ballasim.evimo2.dataset import evimo2_root

    try:
        root = evimo2_root()
        return root.resolve() if root.is_dir() else None
    except OSError:
        return None


def _test_path(test, path: Path, what: str) -> bool:
    """``test(path)`` (``Path.is_dir``, ``is_file`` or ``exists``), which raises for errors other than "not found",
    e.g. EACCES on a parent directory: refused as a :class:`UIError` naming the option (exit 2)."""
    try:
        return test(path)
    except OSError as e:
        raise UIError(f"{what}: {e.strerror}") from None


def _directory(text: str, what: str) -> Path:
    path = Path(text).expanduser().resolve()
    if not _test_path(Path.is_dir, path, f"{what} {text}"):
        raise UIError(f"{what} {text}: not a directory")
    return path


def check_options(args) -> tuple[list[Root], Path | None]:
    """Checks the server options (BS-UI-BIND, the port, the cache size, root names and directories, the jobs root and
    the repo) without side effects, so that ``--open`` refuses a bad option itself before it reuses or starts a
    server. Returns the roots before ``jobs`` and the repo."""
    bind = DEFAULT_BIND if args.bind is None else args.bind
    if bind not in BIND_ADDRESSES:
        raise UIError(f"BS-UI-BIND: --bind {bind}: the UI binds only {' or '.join(BIND_ADDRESSES)}")
    if not 0 <= args.port <= 65535:
        raise UIError(f"--port {args.port}: not a port number")
    if args.cache_mb <= 0:
        raise UIError(f"--cache-mb {args.cache_mb}: must be positive")
    roots, seen = [], set()
    for item in args.root or [f"out={os.getcwd()}"]:
        name, sep, path = item.partition("=")
        if not sep or not _ROOT_NAME.fullmatch(name) or name in ("evimo2", "jobs") or name.startswith("ext-"):
            raise UIError(f"--root {item}: expected NAME=PATH with NAME of [A-Za-z0-9_.-] (not evimo2, jobs, ext-*)")
        if name in seen:
            raise UIError(f"--root {name}: given twice")
        seen.add(name)
        p = _directory(path, "--root")
        roots.append(Root(name, p, not os.access(p, os.W_OK)))
    evimo2 = _directory(args.evimo2_root, "--evimo2-root") if args.evimo2_root else _evimo2_default()
    if evimo2 is not None:
        roots.append(Root("evimo2", evimo2, True))
    if args.jobs_root:
        jobs = Path(args.jobs_root).expanduser().resolve()
        if _test_path(Path.exists, jobs, f"--jobs-root {args.jobs_root}") and not jobs.is_dir():
            raise UIError(f"--jobs-root {args.jobs_root}: not a directory")
    if args.repo:
        repo = _directory(args.repo, "--repo")
        if not _test_path(Path.is_file, repo / "tools" / "evimo2_bench" / "evb.py", f"--repo {args.repo}"):
            raise UIError(f"--repo {args.repo}: no tools/evimo2_bench/evb.py (a BallaSim source checkout)")
    else:
        repo = find_repo()
    return roots, repo


def settings_from_args(args) -> Settings:
    """Checks the options (:func:`check_options`) and builds the settings; creates the jobs root. A jobs root this
    user cannot write is a read-only root, and its server never dispatches (:meth:`UIServer._take_dispatch_lock`)."""
    roots, repo = check_options(args)
    jobs = Path(args.jobs_root).expanduser().resolve() if args.jobs_root else state_dir() / "jobs"
    try:
        jobs.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise UIError(f"jobs root {jobs}: {e.strerror}") from None
    roots.append(Root("jobs", jobs, not os.access(jobs, os.W_OK)))
    gpu_lock = Path(args.gpu_lock).expanduser().absolute() if args.gpu_lock else None
    return Settings(bind=DEFAULT_BIND if args.bind is None else args.bind, port=args.port, roots=roots,
                    jobs_root=jobs, repo=repo, gpu_lock=gpu_lock, cache_mb=args.cache_mb,
                    resim_device=args.resim_device, dev=args.dev)


# ---------------------------------------------------------------------------------------------------------------
# Device probe and GPU memory
# ---------------------------------------------------------------------------------------------------------------

def interpreter_flags() -> list[str]:
    """This interpreter's flags that decide what it imports (-I, -E, -s, -S, -P), for a child that must import the
    same ballasim (e.g. ``-S`` for a source tree on ``PYTHONPATH``)."""
    f = sys.flags
    return [flag for flag, on in (("-I", f.isolated), ("-E", f.ignore_environment), ("-s", f.no_user_site),
                                  ("-S", f.no_site), ("-P", f.safe_path)) if on]


def _package_on_sys_path() -> bool:
    """Whether ``sys.path`` alone finds the ``ballasim`` this process imported and its compiled modules (a source
    tree on ``PYTHONPATH``, tools/dev/pytest_build.py's temporary package, a plain install), rather than an import
    finder that ``site`` installs (an editable install's: ``PathFinder`` finds the source package there too, but
    maps ``_core`` and the other extension modules from site-packages). The modules are looked up where a child
    without ``site`` would: the package directory and ``BALLASIM_BUILD_DIR`` (python/ballasim/__init__.py)."""
    spec = importlib.machinery.PathFinder.find_spec("ballasim", sys.path)
    if (spec is None or spec.origin is None or spec.submodule_search_locations is None
            or Path(spec.origin).resolve() != Path(ballasim.__file__).resolve()):
        return False
    build_dirs = os.environ.get("BALLASIM_BUILD_DIR", "").split(os.pathsep)
    where = [*spec.submodule_search_locations, *filter(None, build_dirs)]
    return importlib.machinery.PathFinder.find_spec("ballasim._core", where) is not None


def probe_argv(code: str = PROBE_CODE) -> list[str]:
    """The device probe's command (§3.1): ``python -I -c``, importing the ballasim the server imports. When
    ``sys.path`` alone finds the server's package and its compiled modules, the child runs without ``site``
    (``-I -S``) on the server's ``sys.path``: ``-I`` drops ``PYTHONPATH``, and ``site`` would install an editable
    install's finder, which can map ``ballasim`` to another checkout. A package that needs such a finder (the
    editable install's maps the compiled modules, which ``-S`` drops) is imported again by plain ``-I``."""
    if _package_on_sys_path():
        return [sys.executable, "-I", "-S", "-c", f"import sys; sys.path[:] = {[p for p in sys.path if p]!r}; {code}"]
    return [sys.executable, "-I", "-c", code]


class GpuMemory:
    """Used and total GPU memory in MiB (device 0) for ``/meta``: NVML through ctypes (no CUDA context), else
    ``nvidia-smi``, cached for ``max_age`` seconds; None when both fail."""

    def __init__(self, libraries: tuple[str, ...] = ("libnvidia-ml.so.1",), smi: tuple[str, ...] = ("nvidia-smi",),
                 max_age: float = 1.0):
        self.libraries, self.smi, self.max_age = libraries, smi, max_age
        self._lock = threading.Lock()
        self._at, self._value = -1e9, None
        self._nvml = None          # (library, device handle), or False after a failed init

    def query(self) -> dict | None:
        with self._lock:
            if time.monotonic() - self._at > self.max_age:
                self._value = self._read_nvml() or self._read_smi()
                self._at = time.monotonic()
            return self._value

    def _read_nvml(self) -> dict | None:
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        if self._nvml is None:
            self._nvml = False
            for name in self.libraries:
                try:
                    lib = ctypes.CDLL(name)
                    handle = ctypes.c_void_p()
                    if lib.nvmlInit_v2() == 0 and lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
                        self._nvml = (lib, handle)
                        break
                except (OSError, AttributeError):
                    continue
        if not self._nvml:
            return None
        lib, handle = self._nvml
        mem = Memory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            return None
        return {"used_mib": mem.used >> 20, "total_mib": mem.total >> 20}

    def _read_smi(self) -> dict | None:
        try:
            out = subprocess.run([*self.smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            used, total = (int(v) for v in out.splitlines()[0].split(","))
            return {"used_mib": used, "total_mib": total}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------------------------------------------

class ApiError(tornado.web.HTTPError):
    """An error response ``{"error": {"code", "message"}}`` with its HTTP status (§3.3, §6)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(status, reason=None)
        self.code, self.message = code, message


#: Tornado's own errors: 404 (no route), 403 (a static path outside the static directory), else a refused request.
_STATUS_CODES = {404: "BS-UI-NOT-FOUND", 403: "BS-UI-PATH"}


def _error_body(status: int, exc: BaseException | None) -> dict:
    if isinstance(exc, ApiError):
        return {"error": {"code": exc.code, "message": exc.message}}
    if status < 500:
        return {"error": {"code": _STATUS_CODES.get(status, "BS-UI-REQUEST"),
                          "message": tornado.httputil.responses.get(status, "request refused")}}
    return {"error": {"code": "BS-UI-INTERNAL", "message": "internal error; see the server log"}}


def _guarded(method):
    """Maps Denied to its error and a SystemExit or KeyboardInterrupt to 500 BS-UI-INTERNAL, so that no handler can
    stop the loop (RV r1, §6)."""

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        try:
            result = method(self, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except (Denied, StreamError) as e:
            raise ApiError(e.status, e.code, e.message) from None
        except (SystemExit, KeyboardInterrupt) as e:
            log.error("%s %s raised %s", self.request.method, redact(self.request.path), type(e).__name__)
            raise ApiError(500, "BS-UI-INTERNAL", f"internal error ({type(e).__name__})") from None

    return wrapper


class _Envelope:
    """Response headers, request checks and error bodies shared by every handler (§3.2)."""

    api = False

    def initialize(self, ui: UIServer, **kwargs):
        self.srv = ui

    @staticmethod
    def checked(check, *args, **kwargs):
        """``check(*args, **kwargs)``, with a :class:`Denied` raised as its error response."""
        try:
            return check(*args, **kwargs)
        except Denied as e:
            raise ApiError(e.status, e.code, e.message) from None

    def prepare(self):
        self.checked(check_request, self.request.headers, self.request.method, api=self.api)

    def set_default_headers(self):
        for k, v in RESPONSE_HEADERS.items():
            self.set_header(k, v)
        if self.api:
            self.set_header("Cache-Control", "no-store")

    def write_error(self, status_code: int, **kwargs):
        exc = kwargs.get("exc_info", (None, None, None))[1]
        body = _error_body(status_code, exc)
        if self.api:
            self.set_header("Content-Type", "application/json; charset=utf-8")
            self.finish(json.dumps(body))
        else:
            self.set_header("Content-Type", "text/plain; charset=utf-8")
            self.finish(f"{status_code} {body['error']['code']}: {body['error']['message']}\n")


class ApiHandler(_Envelope, tornado.web.RequestHandler):
    """An ``/api/v1`` endpoint. ``auth``: ``"any"`` (server token or session), ``"server"`` (server token only) or
    None (no credential); per method through :meth:`auth_for`."""

    api = True
    auth: str | None = "any"
    credential: Credential | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in ("get", "post", "delete", "put"):
            if name in cls.__dict__:
                setattr(cls, name, _guarded(cls.__dict__[name]))

    def auth_for(self, method: str) -> str | None:
        return self.auth

    def prepare(self):
        super().prepare()
        auth = self.auth_for(self.request.method)
        if auth is not None:
            self.credential = self.checked(self.srv.secrets.authenticate, self.request.headers.get("Authorization"),
                                           server_only=auth == "server")

    def compute_etag(self):
        return None

    def json_body(self) -> dict:
        try:
            body = json.loads(self.request.body or b"{}")
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    def write_json(self, value, status: int = 200):
        self.set_status(status)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.finish(json.dumps(value, ensure_ascii=False))


class SessionHandler(ApiHandler):
    """``POST /session`` ``{once}`` or ``{token}`` (no Bearer) → ``{session}``; ``DELETE /session`` forgets it."""

    def auth_for(self, method):
        return None if method == "POST" else "any"

    def post(self):
        body = self.json_body()
        self.write_json({"session": self.srv.secrets.new_session(once=body.get("once"), token=body.get("token"))})

    def delete(self):
        self.srv.secrets.drop_session(self.credential)
        self.set_status(204)


class OnceHandler(ApiHandler):
    """``POST /once`` (server token) → ``{once, redirect_file}``: a new nonce in a new redirect file."""

    auth = "server"

    def post(self):
        once = self.srv.secrets.new_nonce()
        self.write_json({"once": once, "redirect_file": str(self.srv.write_redirect(once))})


class TicketHandler(ApiHandler):
    """``POST /ws-ticket`` → ``{ticket}``: single-use, valid for the ticket TTL (§3.4)."""

    def post(self):
        self.write_json({"ticket": self.srv.secrets.new_ticket(self.credential)})


class ShutdownHandler(ApiHandler):
    """``POST /shutdown`` (server token) → 202; the server stops (jobs keep running)."""

    auth = "server"

    def post(self):
        self.write_json({}, 202)
        asyncio.get_running_loop().call_soon(self.srv.request_stop)


class MetaHandler(ApiHandler):
    async def get(self):
        self.write_json(await self.srv.meta())


class ParamsHandler(ApiHandler):
    """``GET /params``: the ``ballasim.params/1`` document; ``ETag`` = SHA-256 of the body."""

    def get(self):
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.finish(self.srv.params_body())

    def compute_etag(self):
        return '"' + hashlib.sha256(b"".join(self._write_buffer)).hexdigest() + '"'


class ProfilesHandler(ApiHandler):
    def get(self):
        self.write_json(profiles())


class FormHandler(ApiHandler):
    """``GET /form``: the form model ``ballasim.ui.form/1`` (§3.6); ``ETag`` = SHA-256 of the body."""

    def get(self):
        self.set_header("Cache-Control", "no-cache")
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.finish(self.srv.params.form_body())

    def compute_etag(self):
        return '"' + hashlib.sha256(b"".join(self._write_buffer)).hexdigest() + '"'


class ResolveHandler(ApiHandler):
    """``POST /resolve`` ``{base, sets, profiles, compare_to?, explain?}`` → 200 the resolved record, 422
    BS-UI-CONFIG with diagnostics and missing keys, or 202 ``{needs_bundle}`` (§3.6, §4.12)."""

    async def post(self):
        status, doc = await self.srv.params.resolve(self.json_body())
        self.write_json(doc, status)


class ConfigsHandler(ApiHandler):
    """``POST /configs`` ``{path, base, sets, profiles}`` → 201 ``{path, sha256}``: a format-2 YAML in a writable
    root (§3.3)."""

    async def post(self):
        status, doc = await self.srv.params.save_config(self.json_body())
        self.write_json(doc, status)


_FLOAT = re.compile(r"[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?|\.[0-9]+([eE][-+]?[0-9]+)?")
#: The largest clip whose exact count table (2⌈clip⌉ + 1 colours) ``/colors`` builds.
MAX_TABLE_CLIP = 65536.0
#: The largest K = ⌈clip⌉ whose exact polarity table ((K + 1)^2 colours) ``/colors`` builds; above, the browser
#: mixes in f32 (§4.5).
MAX_POLARITY_TABLE_K = 255
COLOR_KINDS = ("events", "polarity", "counts")


def float_argument(handler: ApiHandler, name: str, default: float, lo: float, hi: float) -> float:
    """A decimal float query argument in (lo, hi]; 400 BS-UI-REQUEST otherwise."""
    text = handler.get_query_argument(name, None)
    if text is None:
        return default
    if not _FLOAT.fullmatch(text) or not lo < float(text) <= hi:
        raise ApiError(400, "BS-UI-REQUEST", f"{name}: a decimal number in ({lo:g}, {hi:g}]")
    return float(text)


@functools.lru_cache(maxsize=32)
def colors_body(theme: str, clip: float, gamma: float, kind: str = "events") -> bytes:
    """The ``/colors`` document: ``ballasim.viz.render``'s theme, its 511-entry diverging LUT and 256-entry
    sequential LUT, and the exact colours of the kind for integer counts, the very maps of ``viz.render`` (§4.5), with
    K = ceil(clip):

    - ``events``: ``colorize_signed(n / clip)`` for nets n in [-K, K] (``event_frame``);
    - ``polarity``: ``polarity_image`` of (ON, OFF) = (i, j) at index i (K + 1) + j, i, j in [0, K] (counts from K
      on saturate), or null for K > MAX_POLARITY_TABLE_K;
    - ``counts``: ``colorize_magnitude(n, clip)`` for n in [0, K] (from K on, the top of the ramp).

    Colours are rgba8 bytes (alpha 255), base64."""
    import base64

    import numpy as np

    from ballasim.viz import render

    def rgba(rgb) -> str:
        rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
        return base64.b64encode(np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1).tobytes()).decode()

    th, k = render.theme(theme), math.ceil(clip)
    if kind == "events":
        table = render.colorize_signed(np.arange(-k, k + 1) / max(clip, 1e-9), theme, gamma)
    elif kind == "polarity":
        n = np.arange(k + 1)
        table = None if k > MAX_POLARITY_TABLE_K else render.polarity_image(n[:, None], n[None, :], clip, theme, gamma)
    else:
        table = render.colorize_magnitude(np.arange(k + 1), clip, theme)
    return json.dumps({"theme": theme, "kind": kind, "background": list(th.background), "on": list(th.on),
                       "off": list(th.off), "ink": list(th.ink), "muted": list(th.muted), "contour": list(th.contour),
                       "clip": clip, "gamma": gamma, "k": k, "lut_rgba": rgba(render.diverging_lut(theme)),
                       "sequential_rgba": rgba(render.sequential_lut(theme)),
                       "table_rgba": None if table is None else rgba(table)}).encode()


def choice_argument(handler: ApiHandler, name: str, choices: tuple[str, ...]) -> str:
    """A query argument among ``choices`` (the first by default); 400 BS-UI-REQUEST otherwise."""
    value = handler.get_query_argument(name, choices[0])
    if value not in choices:
        raise ApiError(400, "BS-UI-REQUEST", f"{name}: one of {', '.join(choices)}")
    return value


class ColorsHandler(ApiHandler):
    """``GET /colors?theme=dark|white&clip&gamma&kind=events|polarity|counts`` → the theme, LUTs and the kind's exact
    table (§4.5), from ``ballasim.viz.render`` (imported at the first request: it brings cv2 and matplotlib, §3.1)."""

    async def get(self):
        theme = choice_argument(self, "theme", ("dark", "white"))
        clip = float_argument(self, "clip", 3.0, 0.0, MAX_TABLE_CLIP)
        gamma = float_argument(self, "gamma", 0.5, 0.0, 16.0)
        kind = choice_argument(self, "kind", COLOR_KINDS)
        body = await asyncio.get_running_loop().run_in_executor(self.srv.pool, colors_body, theme, clip, gamma, kind)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.finish(body)


_INT = re.compile(r"-?[0-9]{1,19}")


def int_argument(handler: ApiHandler, name: str, default: int | None = None) -> int:
    """An int64 query argument in decimal (times are decimal strings, §3.3); 400 BS-UI-REQUEST otherwise."""
    text = handler.get_query_argument(name, None)
    if text is None and default is not None:
        return default
    if text is None or not _INT.fullmatch(text) or not -(1 << 63) <= int(text) < (1 << 63):
        raise ApiError(400, "BS-UI-REQUEST", f"{name}: a decimal int64")
    return int(text)


class StreamsHandler(ApiHandler):
    """``POST /streams`` ``{source: {run, camera} | {path} | {evimo2}, polarity, label?}`` → the stream (§3.3)."""

    async def post(self):
        stream = await self.srv.streams.open(self.json_body())
        self.write_json(stream.info())


class StreamHandler(ApiHandler):
    """``DELETE /streams/{sid}`` → 204: closes the source and drops its cached blocks."""

    def delete(self, sid):
        self.srv.streams.delete(sid)
        self.set_status(204)


class RateHandler(ApiHandler):
    """``GET /streams/{sid}/rate?t0_ns&t1_ns&bin_ms=1`` → BSRT (§3.5); the span defaults to the stream's, from its
    1 ms grid bin."""

    async def get(self, sid):
        streams = self.srv.streams
        stream = streams.get(sid)
        r = stream.reader
        start = r.t_offset_ns + (r.t_start_ns - r.t_offset_ns) // 1_000_000 * 1_000_000
        body = await streams.rate(stream, int_argument(self, "t0_ns", start), int_argument(self, "t1_ns", r.t_end_ns),
                                  int_argument(self, "bin_ms", 1))
        self.set_header("Content-Type", "application/octet-stream")
        self.finish(body)


class BlocksHandler(ApiHandler):
    """``GET /streams/{sid}/blocks?k0&k1`` → the heads of blocks [k0, k1) (built as needed)."""

    async def get(self, sid):
        streams = self.srv.streams
        stream = streams.get(sid)
        k0 = int_argument(self, "k0", stream.k_first)
        k1 = int_argument(self, "k1", min(stream.k_end, k0 + MAX_BLOCKS))
        if not 0 <= k1 - k0 <= MAX_BLOCKS:
            raise ApiError(400, "BS-UI-REQUEST", f"need 0 <= k1 - k0 <= {MAX_BLOCKS}")
        self.write_json(await streams.heads(stream, k0, k1))


class SliceHandler(ApiHandler):
    """``GET /streams/{sid}/blocks/{k}?i0&i1&cols=ev[,…]&enc=raw|deflate-raw`` → one BSEV slice (§3.5); closed files'
    slices are immutable."""

    async def get(self, sid, k):
        streams = self.srv.streams
        stream = streams.get(sid)
        enc = self.get_query_argument("enc", "raw")
        if enc not in ("raw", "deflate-raw"):
            raise ApiError(400, "BS-UI-REQUEST", "enc: raw or deflate-raw")
        cols = [c for c in self.get_query_argument("cols", "ev").split(",") if c]
        body = await streams.slice(stream, int(k), int_argument(self, "i0"), int_argument(self, "i1"), cols,
                                   enc == "deflate-raw")
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Cache-Control", "private, max-age=31536000, immutable")
        self.finish(body)


class ImageHandler(ApiHandler):
    """``GET /streams/{sid}/image?t0_ns&t1_ns&kind&theme&clip&gamma&tau_ms`` → a PNG panel of the window rendered by
    ``ballasim.viz.compare.Renderer``: the viewer's fallback without WebGPU (§4.11). Without ``clip``, viz's
    ``auto_clip`` over the stream's timeline of windows of this length (one clip per timeline, §4.5);
    ``X-BallaSim-Clip`` gives the clip used (absent for the time surface, which uses none)."""

    async def get(self, sid):
        stream = self.srv.streams.get(sid)
        kind = choice_argument(self, "kind", IMAGE_KINDS)
        theme = choice_argument(self, "theme", ("dark", "white"))
        clip = float_argument(self, "clip", 0.0, 0.0, MAX_TABLE_CLIP) or None
        gamma = float_argument(self, "gamma", 0.5, 0.0, 16.0)
        tau_ms = float_argument(self, "tau_ms", 20.0, 0.0, 1e9)
        t0, t1 = int_argument(self, "t0_ns"), int_argument(self, "t1_ns")
        if t1 <= t0:
            raise ApiError(400, "BS-UI-REQUEST", "need t0_ns < t1_ns")
        png, used = await self.srv.streams.image(stream, t0, t1, kind, theme, clip, gamma, tau_ms)
        self.set_header("Content-Type", "image/png")
        if used is not None:
            self.set_header("X-BallaSim-Clip", repr(used))
        self.finish(png)


class RegionHandler(ApiHandler):
    """``POST /compare/region`` ``{ref, test, mode, span?, crop?, hot, allow_bad?}`` → the region on the reference's
    clock, the clock rule's δ_stream proposal and its BS-UI-CLOCK warnings (§3.3, §4.5, §4.6)."""

    async def post(self):
        self.write_json(await self.srv.compare.region(self.json_body()))


def optional_int(handler: ApiHandler, name: str) -> int | None:
    return None if handler.get_query_argument(name, None) is None else int_argument(handler, name)


class MasksHandler(ApiHandler):
    """``GET /compare/region/{id}/masks?m0&m1`` → BSMK of masks [m0, m1), at most 256 (§3.5). A region id is the
    digest of its content, so the answer never changes."""

    def get(self, region_id):
        body = self.srv.compare.masks_message(region_id, optional_int(self, "m0"), optional_int(self, "m1"))
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Cache-Control", "private, max-age=31536000, immutable")
        self.finish(body)


class CompareStatsHandler(ApiHandler):
    """``POST /compare/stats`` ``{streams, t0_ns, t1_ns, region_id?, offsets?, options?}`` → viewer statistics per
    stream from ``ballasim.viz.stats`` (§4.5)."""

    async def post(self):
        self.write_json(await self.srv.compare.stats(self.json_body()))


class AttachHandler(ApiHandler):
    """``POST /attach`` (server token) ``{path}`` → ``{root_id, kind, id, once, redirect_file}``: the job or run
    directory becomes a read-only root (recorded in the runtime file), and a redirect file opens its page."""

    auth = "server"

    async def post(self):
        path = self.json_body().get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ApiError(400, "BS-UI-REQUEST", "path: an absolute path")
        kind, d = await asyncio.get_running_loop().run_in_executor(self.srv.pool, self.srv.fs.attach_dir, Path(path))
        root_id = self.srv.fs.add_attached(d)
        await self.srv.record_attached()
        ident = job_id_of(d) if kind == "job" else self.srv.fs.rid(d)
        once = self.srv.secrets.new_nonce()
        redirect = self.srv.write_redirect(once, f"/{'jobs' if kind == 'job' else 'runs'}/{ident}")
        self.write_json({"root_id": root_id, "kind": kind, "id": ident, "once": once, "redirect_file": str(redirect)})


class FsHandler(ApiHandler):
    """``GET /fs?root=&path=`` → one directory level (§3.3)."""

    async def get(self):
        root, rel = self.get_query_argument("root", ""), self.get_query_argument("path", "")
        srv = self.srv
        directory, entries = await asyncio.get_running_loop().run_in_executor(srv.pool, srv.fs.listing, root, rel)
        for e in entries:
            if e["kind"] == "job":
                e["job"] = srv.jobs.summary(directory / e["name"])
        self.write_json(entries)


class RunHandler(ApiHandler):
    """``GET /runs/{rid}`` → run.json without ``resolved``, ``live``, ``files`` and ``capabilities`` (§4.11)."""

    async def get(self, rid):
        repo = self.srv.settings.repo is not None
        self.write_json(await asyncio.get_running_loop().run_in_executor(self.srv.pool, functools.partial(
            self.srv.fs.run_page, rid, repo=repo)))


class RunRecordHandler(ApiHandler):
    async def get(self, rid):
        self.write_json(await asyncio.get_running_loop().run_in_executor(self.srv.pool, self.srv.fs.record, rid))


class RunProgressHandler(ApiHandler):
    def get(self, rid):
        self.write_json(self.srv.fs.progress(rid))


class RunPerfHandler(ApiHandler):
    """``GET /runs/{rid}/perf?since=<byte>`` → NDJSON, complete lines only; ``X-Next-Offset``."""

    def get(self, rid):
        data, end = self.srv.fs.perf(rid, max(int_argument(self, "since", 0), 0))
        self.set_header("Content-Type", "application/x-ndjson")
        self.set_header("X-Next-Offset", str(end))
        self.finish(data)


class RunTracksHandler(ApiHandler):
    """``GET /runs/{rid}/tracks`` → the run's timeline tracks (§3.3; UI-7: a Blender job's preview track)."""

    async def get(self, rid):
        self.write_json(await asyncio.get_running_loop().run_in_executor(self.srv.pool, self.srv.fs.tracks, rid))


class RunTrackHandler(ApiHandler):
    """``GET /runs/{rid}/tracks/{id}?t0_ns&t1_ns`` → BSTK (§3.5) on the run's sensor clock; ``no-cache``."""

    async def get(self, rid, track_id):
        t0, t1 = (int_argument(self, k) if self.get_query_argument(k, None) is not None else None
                  for k in ("t0_ns", "t1_ns"))
        body = await asyncio.get_running_loop().run_in_executor(self.srv.pool, self.srv.fs.track, rid, track_id, t0,
                                                                t1)
        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Cache-Control", "no-cache")
        self.finish(body)


_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RunFileHandler(ApiHandler):
    """``GET /runs/{rid}/files/{relpath}`` (``Range``: one byte range) → the file's bytes, read-only, under the run
    files' content policy (§3.2: an allow-listed type, else a download; always the ``sandbox`` CSP)."""

    CHUNK = 1 << 20

    async def get(self, rid, rel):
        run = self.srv.fs.resolve(rid)
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise ApiError(403, "BS-UI-PATH", "the path leaves the run directory")
        path = resolve_path(self.srv.settings.roots, str(run / rel))
        if not (str(path) + "/").startswith(str(run) + "/") or not path.is_file():
            raise ApiError(404 if path.is_dir() else 403, "BS-UI-NOT-FOUND" if path.is_dir() else "BS-UI-PATH",
                           "not a file of the run")
        loop = asyncio.get_running_loop()
        with _path_errors():                        # §6: a file this user may not read is 403 BS-UI-PATH
            size = path.stat().st_size
            f = await loop.run_in_executor(None, open, path, "rb")
        try:
            start, end, status = 0, size, 200
            m = _RANGE.fullmatch(self.request.headers.get("Range", "").strip())
            if m and (m.group(1) or m.group(2)):
                a, b = m.group(1), m.group(2)
                start, end = (int(a), min(int(b) + 1, size) if b else size) if a else (max(size - int(b), 0), size)
                if start >= end:
                    self.set_header("Content-Range", f"bytes */{size}")
                    raise ApiError(416, "BS-UI-SLICE", "the range is outside the file")
                status = 206
                self.set_header("Content-Range", f"bytes {start}-{end - 1}/{size}")
            for k, v in file_headers(path.name).items():
                self.set_header(k, v)
            self.set_header("Accept-Ranges", "bytes")
            self.set_header("Content-Length", str(end - start))
            self.set_status(status)
            f.seek(start)
            while start < end:
                chunk = await loop.run_in_executor(None, f.read, min(self.CHUNK, end - start))
                if not chunk:
                    break
                start += len(chunk)
                self.write(chunk)
                await self.flush()
        finally:
            f.close()
        self.finish()


class JobsHandler(ApiHandler):
    """``POST /jobs`` ``{kind, args, label, overwrite}`` → 201 ``{job_id, state, position, gpu}``; ``GET /jobs`` →
    the records of this jobs root's and the attached jobs (§3.7)."""

    async def post(self):
        srv = self.srv
        prepared = await asyncio.get_running_loop().run_in_executor(srv.pool, srv.jobs.prepare, self.json_body())
        job, rec = srv.jobs.commit(prepared)
        self.write_json({"job_id": job.job_id, "state": rec["state"], "position": rec["position"], "gpu": rec["gpu"],
                         "warnings": rec["warnings"]}, 201)
        srv.hub.tick_soon()

    def get(self):
        self.write_json(self.srv.jobs.records())


class JobHandler(ApiHandler):
    def get(self, job_id):
        self.write_json(self.srv.jobs.record(self.srv.jobs.get(job_id)))


class JobCancelHandler(ApiHandler):
    def post(self, job_id):
        self.write_json(self.srv.jobs.cancel(job_id, asyncio.get_running_loop()), 202)
        self.srv.hub.tick_soon()


class JobKillHandler(ApiHandler):
    def post(self, job_id):
        self.write_json(self.srv.jobs.kill(job_id), 202)


class QueueResumeHandler(ApiHandler):
    def post(self):
        self.write_json({"resumed": self.srv.jobs.resume()}, 202)
        self.srv.hub.tick_soon()


class JobLogHandler(ApiHandler):
    """``GET /jobs/{id}/log?since=<byte>&stream=stdout|stderr`` → text; ``X-Next-Offset``."""

    def get(self, job_id):
        stream = choice_argument(self, "stream", ("stdout", "stderr"))
        data, end = self.srv.jobs.log(job_id, stream, max(int_argument(self, "since", 0), 0))
        self.set_header("Content-Type", "text/plain; charset=utf-8")
        self.set_header("X-Next-Offset", str(end))
        self.finish(data)


class ApiNotFound(ApiHandler):
    def get(self, *args):
        raise ApiError(404, "BS-UI-NOT-FOUND", f"no API endpoint {self.request.path}")

    post = delete = put = get


class SocketHandler(_Envelope, tornado.websocket.WebSocketHandler):
    """``/api/v1/ws?ticket=…`` (§3.4): the Host, Fetch-Site and Origin checks, then a single-use ticket; the messages
    go through the hub (:mod:`.live`: ``sub``, ``unsub``, ``ping``)."""

    api = True

    def prepare(self):
        self.checked(check_request, self.request.headers, self.request.method, api=True, websocket=True)
        self.credential = self.checked(self.srv.secrets.redeem_ticket, self.get_query_argument("ticket", None))

    def check_origin(self, origin):
        return True                 # checked in prepare (exactly http://<Host>)

    def open(self):
        self.srv.sockets.add(self)
        self.srv.hub.add(self)

    def on_close(self):
        self.srv.sockets.discard(self)
        self.srv.hub.remove(self)

    def on_message(self, message):
        try:
            msg = json.loads(message)
        except ValueError:
            return
        self.srv.hub.on_message(self, msg)


class PageHandler(_Envelope, tornado.web.RequestHandler):
    """``/`` (the shell) and ``/login``: public pages without secrets."""

    def initialize(self, ui, page: str):
        self.srv, self.page = ui, page

    def get(self):
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.set_header("Cache-Control", "no-store" if self.srv.settings.dev else "no-cache")
        self.finish((STATIC_DIR / self.page).read_bytes())


class StaticHandler(_Envelope, tornado.web.StaticFileHandler):
    """``/static/**``: public, ``no-cache`` (``no-store`` with ``--dev``)."""

    def initialize(self, ui, path: str):
        self.srv = ui
        tornado.web.StaticFileHandler.initialize(self, path)

    def get_content_type(self):
        return _STATIC_TYPES.get(Path(self.absolute_path).suffix) or super().get_content_type()

    def set_extra_headers(self, path):
        self.set_header("Cache-Control", "no-store" if self.srv.settings.dev else "no-cache")

    def get_cache_time(self, path, modified, mime_type):
        return 0


class NotFound(_Envelope, tornado.web.RequestHandler):
    def prepare(self):
        super().prepare()
        raise ApiError(404, "BS-UI-NOT-FOUND", "no such page")


def profiles() -> list[dict]:
    """``GET /profiles``: the profile files of ``PROFILES_DIR`` with their SHA-256. Every one is frozen: a committed
    profile is immutable (P0_PARAMS §2.5; a change needs a new version), and there is one search location (Q14)."""
    from ballasim.config import PROFILES_DIR

    out = []
    for path in sorted(PROFILES_DIR.rglob("*.yaml")):
        rel = path.relative_to(PROFILES_DIR).as_posix()
        if _PROFILE_FILE.fullmatch(rel):
            out.append({"id": rel[:-len(".yaml")], "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frozen": True})
    return out


# ---------------------------------------------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------------------------------------------

class UIServer:
    """The application and its state: secrets, redirect files, the runtime file, the device probe, the dispatcher
    lock. ``secrets``, ``probe`` and ``gpu_memory`` are parameters, so that tests run short TTLs and stubs."""

    def __init__(self, settings: Settings, *, secrets_: Secrets | None = None, probe: list[str] | None = None,
                 gpu_memory: GpuMemory | None = None, jobs_options: dict | None = None, tick_s: float | None = None):
        self.settings = settings
        self.secrets = secrets_ or Secrets()
        self.probe = probe_argv() if probe is None else probe
        self.gpu_memory = gpu_memory or GpuMemory()
        self.pool = ThreadPoolExecutor(2, thread_name_prefix="ballasim-ui")
        self.streams = Streams(settings.roots, settings.cache_mb)
        self.compare = Compare(self.streams, self.pool, settings.repo)
        self.fs = Fs(settings.roots, owned_runs=lambda: self.jobs.owned_runs())
        self.jobs = Jobs(self.fs, settings, device=lambda: self.device_available, **(jobs_options or {}))
        self.hub = Hub(self.fs, self.jobs, self.server_message, **({} if tick_s is None else {"tick_s": tick_s}))
        self.params = Params(self)
        self.sockets: set[SocketHandler] = set()
        self.device_available: bool | None = None
        self.port: int | None = None
        self.started = datetime.now(UTC).isoformat(timespec="seconds")
        self.dispatcher = False
        self.runtime_file: Path | None = None
        self._dispatch_fd: int | None = None
        self._redirects: dict[Path, asyncio.TimerHandle] = {}
        self._params: bytes | None = None
        self._git: dict | None = None
        self._probe_task: asyncio.Task | None = None
        self._probe_done = asyncio.Event()
        self._http: tornado.httpserver.HTTPServer | None = None
        self._stopped = asyncio.Event()
        self._stopping = False
        self._stop_task: asyncio.Future | None = None

    # -- application -------------------------------------------------------------------------------------------

    def make_app(self) -> tornado.web.Application:
        a = {"ui": self}
        return tornado.web.Application([
            (rf"{API}/session", SessionHandler, a),
            (rf"{API}/once", OnceHandler, a),
            (rf"{API}/ws-ticket", TicketHandler, a),
            (rf"{API}/shutdown", ShutdownHandler, a),
            (rf"{API}/meta", MetaHandler, a),
            (rf"{API}/params", ParamsHandler, a),
            (rf"{API}/profiles", ProfilesHandler, a),
            (rf"{API}/form", FormHandler, a),
            (rf"{API}/resolve", ResolveHandler, a),
            (rf"{API}/configs", ConfigsHandler, a),
            (rf"{API}/colors", ColorsHandler, a),
            (rf"{API}/ws", SocketHandler, a),
            (rf"{API}/streams", StreamsHandler, a),
            (rf"{API}/streams/([0-9a-f]{{16}})", StreamHandler, a),
            (rf"{API}/streams/([0-9a-f]{{16}})/rate", RateHandler, a),
            (rf"{API}/streams/([0-9a-f]{{16}})/image", ImageHandler, a),
            (rf"{API}/streams/([0-9a-f]{{16}})/blocks", BlocksHandler, a),
            (rf"{API}/streams/([0-9a-f]{{16}})/blocks/(-?[0-9]{{1,12}})", SliceHandler, a),
            (rf"{API}/compare/region", RegionHandler, a),
            (rf"{API}/compare/region/([0-9a-f]{{16}})/masks", MasksHandler, a),
            (rf"{API}/compare/stats", CompareStatsHandler, a),
            (rf"{API}/attach", AttachHandler, a),
            (rf"{API}/fs", FsHandler, a),
            (rf"{API}/runs/({RID})", RunHandler, a),
            (rf"{API}/runs/({RID})/record", RunRecordHandler, a),
            (rf"{API}/runs/({RID})/progress", RunProgressHandler, a),
            (rf"{API}/runs/({RID})/perf", RunPerfHandler, a),
            (rf"{API}/runs/({RID})/tracks", RunTracksHandler, a),
            (rf"{API}/runs/({RID})/tracks/([a-z0-9_-]{{1,64}})", RunTrackHandler, a),
            (rf"{API}/runs/({RID})/files/(.*)", RunFileHandler, a),
            (rf"{API}/jobs", JobsHandler, a),
            (rf"{API}/jobs/queue/resume", QueueResumeHandler, a),
            (rf"{API}/jobs/([0-9a-f]{{16}})", JobHandler, a),
            (rf"{API}/jobs/([0-9a-f]{{16}})/cancel", JobCancelHandler, a),
            (rf"{API}/jobs/([0-9a-f]{{16}})/kill", JobKillHandler, a),
            (rf"{API}/jobs/([0-9a-f]{{16}})/log", JobLogHandler, a),
            (rf"{API}/.*", ApiNotFound, a),
            (r"/", PageHandler, {**a, "page": "index.html"}),
            (r"/login", PageHandler, {**a, "page": "login.html"}),
            (r"/static/(.*)", StaticHandler, {**a, "path": str(STATIC_DIR)}),
        ], default_handler_class=NotFound, default_handler_args=a, log_function=_log_request,
            websocket_ping_interval=20, websocket_max_message_size=1 << 20)

    @property
    def origin(self) -> str:
        return f"http://{self.settings.host}:{self.port}"

    def params_body(self) -> bytes:
        """The ``ballasim.params/1`` document (tools/config/gen_params_doc.py's fields), built once."""
        if self._params is None:
            doc = {"format": "ballasim.params/1", "generated_by": "ballasim.ui", "params": _core.config.schema(),
                   "families": _core.config.families(), "accuracy_profiles": _core.config.accuracy_profiles()}
            self._params = json.dumps(doc, ensure_ascii=False, allow_nan=False).encode()
        return self._params

    async def git(self) -> dict:
        if self._git is None:
            def read():
                from ballasim.engine.runlog import git_commit

                return git_commit()

            self._git = await asyncio.get_running_loop().run_in_executor(self.pool, read)
        return self._git

    async def meta(self) -> dict:
        loop = asyncio.get_running_loop()
        nvml = await loop.run_in_executor(self.pool, self.gpu_memory.query)
        s = self.settings
        return {"protocol": PROTOCOL, "version": ballasim.__version__, "git": await self.git(),
                "build": dict(_core.build_info()), "roots": [r.as_json() for r in s.roots],
                "jobs_root": str(s.jobs_root), "dispatcher": self.dispatcher, "repo": s.repo is not None,
                "dev": s.dev, "min_chrome": MIN_CHROME,
                "limits": {"max_slice_events": MAX_SLICE_EVENTS, "cache_mb": s.cache_mb},
                "gpu": {"available": self.device_available, "lock": str(s.gpu_lock) if s.gpu_lock else None,
                        "busy": self.jobs.gpu_busy(), "nvml": nvml},
                "workers": {"gpu": None, "cpu": None}}

    def server_message(self) -> dict:
        return {"op": "server", "gpu": {"busy": self.jobs.gpu_busy(), "lock_holder": None},
                "workers": {"gpu": None, "cpu": None}, "dispatcher": self.dispatcher}


    # -- device probe ------------------------------------------------------------------------------------------

    async def _run_probe(self) -> None:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*self.probe, stdin=subprocess.DEVNULL,
                                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
            self.device_available = proc.returncode == 0 and out.strip() == b"1"
            if proc.returncode != 0:
                tail = err.decode(errors="replace").strip().splitlines()[-1:] or [""]
                log.warning("device probe failed (exit %s): %s", proc.returncode, tail[0])
        except (OSError, TimeoutError) as e:
            log.warning("device probe failed: %s", e)
            self.device_available = False
        finally:
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
            self._probe_done.set()

    async def wait_device(self) -> bool:
        """The probe's answer, waiting for it if it has not answered yet (the first ``/resolve``, UI-4a)."""
        await self._probe_done.wait()
        return bool(self.device_available)

    # -- redirect files and the runtime file -------------------------------------------------------------------

    def write_redirect(self, once: str, target: str = "/") -> Path:
        """A 0600 redirect file to ``/login#once=…`` (§3.2): the browser's command line holds only its path. It is
        deleted after the redirect TTL."""
        url = f"{self.origin}/login#once={once}"
        if target not in ("", "/"):
            url += "&to=" + urllib.parse.quote(target, safe="/")
        path = runtime_dir() / f"open-{secrets.token_hex(8)}.html"
        write_private(path, '<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer">'
                            f'<title>BallaSim UI</title><script>location.replace({json.dumps(url)})</script>\n')
        self._redirects[path] = asyncio.get_running_loop().call_later(self.secrets.redirect_ttl,
                                                                      self._drop_redirect, path)
        return path

    def _drop_redirect(self, path: Path) -> None:
        handle = self._redirects.pop(path, None)
        if handle is not None:
            handle.cancel()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

    def _sweep_redirects(self) -> None:
        """Deletes redirect files older than the TTL (left by a server that died)."""
        cutoff = time.time() - self.secrets.redirect_ttl
        for path in runtime_dir().glob("open-*.html"):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()

    def login_url(self) -> str:
        return f"{self.origin}/login#token={self.secrets.server_token}"

    async def _write_runtime_file(self) -> None:
        """Records this server in the runtime file, unless the file names another live server (checked off the loop:
        its ``GET /meta`` may go to a server on this very loop, e.g. a second one in a test process)."""
        path = runtime_dir() / RUNTIME_FILE
        other = await asyncio.get_running_loop().run_in_executor(self.pool, live_server, path.parent)
        if other is not None and (other["pid"], other["port"]) != (os.getpid(), self.port):
            log.warning("the runtime file names the live server pid %s port %s; this server (pid %s port %s) is not "
                        "recorded", other["pid"], other["port"], os.getpid(), self.port)
            return
        if other is None and self.runtime_file is None:     # a server that died: its attached roots come back
            self.fs.restore((read_runtime_file(path.parent) or {}).get("attached"))
        self.runtime_file = path
        self._write_runtime()

    def _write_runtime(self) -> None:
        write_private(self.runtime_file, json.dumps({
            "pid": os.getpid(), "port": self.port, "token": self.secrets.server_token, "url": self.login_url(),
            "started": self.started, "jobs_root": str(self.settings.jobs_root), "dispatcher": self.dispatcher,
            "attached": self.fs.attached(), "repo": str(self.settings.repo) if self.settings.repo else None},
            indent=1) + "\n")

    async def record_attached(self) -> None:
        """Records the attached roots in the runtime file (when it names this server)."""
        if self.runtime_file is not None:
            await asyncio.get_running_loop().run_in_executor(self.pool, self._write_runtime)

    def _take_dispatch_lock(self) -> None:
        """The jobs root's dispatcher lock (§3.1): the holder is the only server that spawns its jobs. A server that
        cannot take it (another server holds it, or this user cannot open or lock ``.dispatch.lock``, e.g. in a
        read-only jobs root) still serves and lists the jobs, but never spawns them (``dispatcher`` stays False)."""
        jobs = self.settings.jobs_root
        try:
            fd = os.open(jobs / ".dispatch.lock", os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            log.warning("cannot open %s (%s): jobs are listed, never spawned", jobs / ".dispatch.lock", e.strerror)
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if isinstance(e, BlockingIOError):
                log.warning("another server dispatches %s: jobs are listed, never spawned", jobs)
            else:
                log.warning("cannot lock %s (%s): jobs are listed, never spawned", jobs / ".dispatch.lock", e.strerror)
            return
        self._dispatch_fd, self.dispatcher = fd, True
        self.jobs.dispatcher = True

    # -- lifetime ----------------------------------------------------------------------------------------------

    async def start(self) -> None:
        """Binds, takes the dispatcher lock, starts the device probe and writes the runtime file (``runtime_file``
        stays None when the file names another live server)."""
        try:
            sockets = tornado.netutil.bind_sockets(self.settings.port, address=self.settings.bind)
        except OSError as e:
            raise UIError(f"cannot listen on {self.settings.host}:{self.settings.port}: {e.strerror}") from None
        self.port = sockets[0].getsockname()[1]
        self._http = tornado.httpserver.HTTPServer(self.make_app(), max_body_size=16 << 20)
        self._http.add_sockets(sockets)
        self._take_dispatch_lock()
        self._sweep_redirects()
        self._probe_task = asyncio.create_task(self._run_probe())
        await self._write_runtime_file()
        self.jobs.scan()
        self.hub.start()

    async def stop(self) -> None:
        """Stops serving; removes the runtime file (when it names this server) and the redirect files."""
        if self._stopping:
            return
        self._stopping = True
        if self._http is not None:
            self._http.stop()
        await self.hub.stop()
        self.jobs.close()
        for ws in list(self.sockets):
            ws.close(1001, "server shutdown")
        if self._probe_task is not None and not self._probe_task.done():
            self._probe_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._probe_task
        for path in list(self._redirects):
            self._drop_redirect(path)
        if self.runtime_file is not None:
            with contextlib.suppress(OSError, ValueError):
                if json.loads(self.runtime_file.read_text()).get("pid") == os.getpid():
                    self.runtime_file.unlink()
        if self._dispatch_fd is not None:
            os.close(self._dispatch_fd)
            self._dispatch_fd, self.dispatcher = None, False
            self.jobs.dispatcher = False
        if self._http is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._http.close_all_connections(), 2.0)
        self.streams.close()
        self.pool.shutdown(wait=False)
        self._stopped.set()

    def request_stop(self) -> None:
        """Starts :meth:`stop` from a callback (``/shutdown``, a signal); the task is kept until it ends."""
        if self._stop_task is None:
            self._stop_task = asyncio.ensure_future(self.stop())

    async def wait_stopped(self) -> None:
        await self._stopped.wait()


def _log_request(handler: tornado.web.RequestHandler) -> None:
    """The access log, with ``token=``, ``once=`` and ``ticket=`` values redacted."""
    status = handler.get_status()
    level = logging.INFO if status < 400 else logging.WARNING if status < 500 else logging.ERROR
    r = handler.request
    logging.getLogger("tornado.access").log(level, "%d %s %s (%s) %.1fms", status, r.method, redact(r.uri or ""),
                                            r.remote_ip, 1000.0 * r.request_time())


def setup_logging(verbose: bool = False) -> None:
    """Every log line through the redacting formatter, on stderr (a detached server's stderr is ``server.log``)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


# ---------------------------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------------------------

def main(args) -> int:
    """``python -m ballasim ui`` (engine/cli.py parses the options)."""
    try:
        if args.status:
            return status_command()
        if args.stop:
            return stop_command()
        if args.open is not None:
            return open_command(args)
        return serve(args)
    except UIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def serve(args) -> int:
    """Runs a server until ``/shutdown`` or SIGINT/SIGTERM/SIGHUP."""
    settings = settings_from_args(args)
    setup_logging(args.verbose)
    return asyncio.run(_serve(settings, detached=args.detached, browser=None if args.no_browser or args.detached
                              else find_browser(args.browser, log.warning) or []))


async def _serve(settings: Settings, *, detached: bool, browser: list[str] | None) -> int:
    ui = UIServer(settings)
    await ui.start()
    if detached and ui.runtime_file is None:
        # --open cannot reach an unrecorded server (nor can --status or --stop): it stops, and --open reuses the
        # recorded one (a foreground server that started meanwhile)
        log.error("the runtime file names another live server; this detached server (pid %s) stops", os.getpid())
        await ui.stop()
        return EXIT_UNRECORDED
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, ui.request_stop)
    if detached:        # no secret: the URL is only in the 0600 runtime file
        print(json.dumps({"pid": os.getpid(), "port": ui.port, "runtime_file": str(ui.runtime_file)}), flush=True)
    else:
        print(json.dumps({"url": ui.login_url(), "pid": os.getpid()}), flush=True)
        if browser:
            redirect = ui.write_redirect(ui.secrets.new_nonce())
            try:
                proc = await asyncio.create_subprocess_exec(*browser, str(redirect), stdin=subprocess.DEVNULL,
                                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                            start_new_session=True)
                reaper = asyncio.ensure_future(proc.wait())         # noqa: F841  (kept until the server stops)
            except OSError as e:
                log.warning("cannot start the browser %s: %s; open the URL printed above", browser[0], e.strerror)
        elif browser is not None:
            log.warning("no browser found (%s, $BALLASIM_BROWSER or --browser): open the URL printed above",
                        ", ".join(BROWSERS))
    log.info("serving %s/ (pid %s)", ui.origin, os.getpid())
    await ui.wait_stopped()
    log.info("stopped")
    return 0


def sandboxed_browser(path: str) -> str | None:
    """``"snap"`` or ``"flatpak"`` when ``path`` launches a confined browser, else None. A confined browser cannot read
    the redirect file: e.g. the Chromium snap's AppArmor profile allows only ``$XDG_RUNTIME_DIR/snap.chromium/`` of
    the runtime directory and only non-hidden files of the home. Ubuntu's ``chromium`` is a link to ``/usr/bin/snap``;
    a wrapper script that runs ``/snap/bin/…`` or ``flatpak run`` counts too."""
    real = os.path.realpath(path)
    # snap may itself be a symlink to the snapd multiplexer (Ubuntu 26.04); compare resolved launchers too.
    if real == os.path.realpath("/usr/bin/snap") or real.startswith(("/snap/", "/var/lib/snapd/")):
        return "snap"
    if "/flatpak/" in real or "/flatpak/" in path:
        return "flatpak"
    try:
        with open(real, "rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if head.startswith(b"#!"):
        if b"/snap/" in head or b"snap run" in head:
            return "snap"
        if b"flatpak run" in head:
            return "flatpak"
    return None


def find_browser(command: str | None, warn=None) -> list[str] | None:
    """``--browser CMD``, ``$BALLASIM_BROWSER``, else the first of :data:`BROWSERS` on PATH that is not a confined
    (snap, flatpak) launcher (never extra flags). ``warn(message)`` hears of each skipped launcher and of a named
    confined one, which is still used."""
    warn = warn or log.warning
    text = command or os.environ.get("BALLASIM_BROWSER")
    if text:
        argv = shlex.split(text)
        if argv and (kind := sandboxed_browser(shutil.which(argv[0]) or argv[0])):
            warn(f"{argv[0]} is a {kind} browser, whose sandbox may not read the redirect file: if the page does not "
                 "open, sign in with the URL in the runtime file")
        return argv or None
    for name in BROWSERS:
        if (path := shutil.which(name)) and (kind := sandboxed_browser(path)):
            warn(f"skipped {path}: a {kind} browser, whose sandbox cannot read the redirect file")
        elif path:
            return [path]
    return None


def checked_redirect(body: dict, directory: Path) -> Path:
    """The ``redirect_file`` of a server's ``/once`` or ``/attach`` reply, accepted only as a redirect file of this
    user's runtime directory ``directory`` (a regular file of this user named ``open-<16 hex>.html``). The command
    trusts a server only as far as its recorded pid and port: a reply from another listener must never put a browser
    switch or a URL on the browser's command line."""
    text = body.get("redirect_file") if isinstance(body, dict) else None
    path = Path(text) if isinstance(text, str) and text.startswith("/") else None
    try:
        st = path.lstat() if path is not None else None
    except OSError:
        st = None
    if (path is None or st is None or path.parent != directory or not _REDIRECT_NAME.fullmatch(path.name)
            or not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid()):
        raise UIError(f"the UI server's reply names no redirect file in {directory}; is the server in the runtime file "
                      "this user's? (python -m ballasim ui --status)")
    return path


class Client:
    """The command's HTTP client: the server token, loopback, never through a proxy."""

    def __init__(self, info: dict):
        parts = urllib.parse.urlsplit(info["url"])
        self.base, self.token = f"{parts.scheme}://{parts.netloc}{API}", info["token"]
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, method: str, path: str, body: dict | None = None, timeout: float = 5.0) -> tuple[int, dict]:
        """``(status, JSON object)``; a body that is not a JSON object reads as ``{}``."""
        data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=timeout) as r:
                return r.status, _json_object(r.read())
        except urllib.error.HTTPError as e:
            return e.code, _json_object(e.read())
        except OSError as e:
            raise UIError(f"the UI server at {self.base} did not answer: {e}") from None


def _json_object(text: bytes) -> dict:
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _error_of(body: dict) -> dict:
    error = body.get("error")
    return error if isinstance(error, dict) else {}


def read_runtime_file(directory: Path) -> dict | None:
    try:
        info = json.loads((directory / RUNTIME_FILE).read_text())
    except (OSError, ValueError):
        return None
    types = {"pid": int, "port": int, "token": str, "url": str}
    ok = isinstance(info, dict) and all(isinstance(info.get(k), t) for k, t in types.items())
    return info if ok else None


def live_server(directory: Path) -> dict | None:
    """The runtime file's server when it is live: its pid runs and ``GET /meta`` answers with its token. The
    result holds the runtime file's fields and ``meta``."""
    info = read_runtime_file(directory)
    if info is None or not _pid_alive(info["pid"]):
        return None
    try:
        status, meta = Client(info).call("GET", "/meta", timeout=3.0)
    except UIError:
        return None
    return {**info, "meta": meta} if status == 200 else None


def _base_url(info: dict) -> str:
    parts = urllib.parse.urlsplit(info["url"])
    return f"{parts.scheme}://{parts.netloc}/"


def status_command() -> int:
    """``--status``: the running server without its secret; exit 1 if none."""
    info = live_server(runtime_dir())
    if info is None:
        print("no UI server is running", file=sys.stderr)
        return 1
    print(json.dumps({"url": _base_url(info), "pid": info["pid"], "port": info["port"],
                      "roots": info["meta"].get("roots"), "dispatcher": info["meta"].get("dispatcher")}))
    return 0


def stop_command() -> int:
    """``--stop``: ``POST /shutdown`` with the server token; waits up to 10 s for the server to exit."""
    info = live_server(runtime_dir())
    if info is None:
        print("no UI server is running", file=sys.stderr)
        return 1
    status, _ = Client(info).call("POST", "/shutdown")
    if status != 202:
        raise UIError(f"the server refused to stop (HTTP {status})")
    deadline = time.monotonic() + 10
    while _pid_alive(info["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    print(f"stopped the UI server (pid {info['pid']})")
    return 0


def attach_dir(path: Path) -> Path:
    """The job or run directory ``path`` names: itself or its nearest parent holding ``job.json`` or ``run.json``,
    else ``path`` (or its directory, for a file)."""
    start = path if path.is_dir() else path.parent
    for d in (start, *start.parents):
        if (d / "job.json").is_file() or (d / "run.json").is_file():
            return d
    return start


SERVER_OPTIONS = (("root", "--root"), ("evimo2_root", "--evimo2-root"), ("jobs_root", "--jobs-root"),
                  ("gpu_lock", "--gpu-lock"), ("repo", "--repo"), ("cache_mb", "--cache-mb"),
                  ("resim_device", "--resim-device"))


def _forwarded(args, target: Path | None) -> list[str]:
    """The server options of this command line, for a detached server; without ``--root`` the attached path's
    parent (or the working directory) is its root ``out``."""
    out = ["--port", str(args.port), "--bind", args.bind or DEFAULT_BIND]
    roots = args.root or [f"out={attach_dir(target).parent if target is not None else Path.cwd()}"]
    for dest, flag in SERVER_OPTIONS:
        value = roots if dest == "root" else getattr(args, dest)
        for v in value if isinstance(value, list) else [value] if value is not None else []:
            out += [flag, str(v)]
    return out + (["--dev"] if args.dev else [])


def detached_argv(args, target: Path | None) -> list[str]:
    """The detached server's command: this interpreter, its import flags, ``-m ballasim ui --detached``."""
    return [sys.executable, *interpreter_flags(), "-m", "ballasim", *(["-v"] if args.verbose else []), "ui",
            "--detached", *_forwarded(args, target)]


def _end(proc: subprocess.Popen) -> None:
    """Stops a detached server that ``--open`` gives up on, with its process group (it leads its own session; the
    device probe is in it): SIGTERM, then SIGKILL after 5 s. A reaped child is left alone (its pid may be reused)."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def _log_error(log_path: Path, offset: int) -> str:
    """The last ``error:`` line that ``server.log`` gained after ``offset`` (a refused start prints one), in
    parentheses with a leading space, or ""."""
    try:
        with open(log_path, "rb") as f:
            f.seek(offset)
            lines = f.read(1 << 16).decode(errors="replace").splitlines()
    except OSError:
        return ""
    errors = [line for line in lines if line.startswith("error: ")]
    return f" ({errors[-1][len('error: '):]})" if errors else ""


def start_detached(args, target: Path | None, timeout: float = DETACHED_START_S) -> dict:
    """Starts a detached server (own session, stdin closed, output to the 0600 ``server.log``) and waits up to
    ``timeout`` seconds for its runtime file. A server that finds another live one recorded stops with
    :data:`EXIT_UNRECORDED`, and that one is returned (with ``meta``) instead. On a timeout or any other failure the
    child is stopped, so that no unrecorded server (perhaps holding the dispatcher lock) is left behind."""
    rdir, log_path = runtime_dir(), state_dir() / "server.log"
    fd = open_log(log_path)
    offset = os.fstat(fd).st_size
    try:
        proc = subprocess.Popen(detached_argv(args, target), stdin=subprocess.DEVNULL, stdout=fd, stderr=fd,
                                start_new_session=True, close_fds=True)
    finally:
        os.close(fd)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = read_runtime_file(rdir)
            if info is not None and info["pid"] == proc.pid:
                return info
            if proc.poll() is not None:
                if proc.returncode == EXIT_UNRECORDED and (other := live_server(rdir)) is not None:
                    return other
                raise UIError(f"the UI server exited with code {proc.returncode}{_log_error(log_path, offset)}; "
                              f"see {log_path}")
            time.sleep(0.05)
        raise UIError(f"the UI server wrote no runtime file within {timeout:g} s and was stopped; see {log_path}")
    except BaseException:
        _end(proc)
        raise


def _mismatches(args, info: dict) -> list[str]:
    """The options of this command that the running server does not have."""
    meta, out = info.get("meta", {}), []
    roots = {r["id"]: r["path"] for r in meta.get("roots", [])}
    for item in args.root or []:
        name, _, path = item.partition("=")
        if roots.get(name) != str(Path(path).expanduser().resolve()):
            out.append(f"--root {item}")
    if args.evimo2_root and roots.get("evimo2") != str(Path(args.evimo2_root).expanduser().resolve()):
        out.append("--evimo2-root")
    if args.jobs_root and meta.get("jobs_root") != str(Path(args.jobs_root).expanduser().resolve()):
        out.append("--jobs-root")
    if args.gpu_lock and (meta.get("gpu") or {}).get("lock") != str(Path(args.gpu_lock).expanduser().absolute()):
        out.append("--gpu-lock")
    if args.repo and info.get("repo") != str(Path(args.repo).expanduser().resolve()):
        out.append("--repo")
    if args.bind is not None and args.bind != urllib.parse.urlsplit(info["url"]).hostname:
        out.append(f"--bind {args.bind}")
    if args.port and args.port != info["port"]:
        out.append(f"--port {args.port}")
    return out


def open_command(args) -> int:
    """``--open [PATH]``: check the server options, reuse or start a server, attach PATH (``POST /attach``) or ask
    for a sign-in link (``POST /once``), and open the browser on the redirect file. Prints no secret."""
    check_options(args)                         # a bad option is refused here, not only in a detached server's log
    target = None
    if args.open:
        target = Path(args.open).expanduser().resolve()
        if not target.exists():
            raise UIError(f"{args.open}: no such file or directory")
    rdir = runtime_dir()
    with _flock(rdir / ".open.lock"):           # two --open at once start one server
        info = live_server(rdir)
        if info is None:
            info = start_detached(args, target)  # with "meta" only when another server won the start
        if "meta" in info and (differ := _mismatches(args, info)):
            print(f"warning: the running UI server (pid {info['pid']}) keeps its settings; ignored: "
                  f"{', '.join(differ)} (stop it with python -m ballasim ui --stop)", file=sys.stderr)
    client = Client(info)
    status, body = client.call("POST", "/attach", {"path": str(target)}) if target is not None else (None, {})
    if status == 404 and _error_of(body).get("code") == "BS-UI-NOT-FOUND":
        print(f"warning: this server cannot attach {args.open} (no /attach): opening its start page", file=sys.stderr)
        status = None
    if status is None:
        status, body = client.call("POST", "/once")
    if status not in (200, 201):
        error = _error_of(body)
        raise UIError(f"{error.get('code', 'HTTP')} {status}: {error.get('message', 'the server refused the request')}")
    redirect, runtime = str(checked_redirect(body, rdir)), str(rdir / RUNTIME_FILE)
    browser = None if args.no_browser else find_browser(args.browser, lambda m: print(f"warning: {m}", file=sys.stderr))
    if browser:
        try:
            subprocess.Popen([*browser, redirect], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        except OSError as e:
            raise UIError(f"cannot start the browser {browser[0]}: {e.strerror}") from None
    elif not args.no_browser:
        print(f"no browser found ({', '.join(BROWSERS)}, $BALLASIM_BROWSER or --browser): open {_base_url(info)} "
              f"and sign in with the URL in {runtime}", file=sys.stderr)
    print(json.dumps({"pid": info["pid"], "port": info["port"], "runtime_file": runtime,
                      **({"redirect_file": redirect} if args.no_browser else {})}))
    return 0
