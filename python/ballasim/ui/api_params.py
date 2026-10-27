"""Forms and resolve (docs/design/phase3/ui.md §3.6, §4.9, §4.12; UI-4a): ``GET /form``, ``POST /resolve`` and
``POST /configs``.

- **Form model** (``ballasim.ui.form/1``), built once from ``_core.config.{schema, families, accuracy_profiles,
  derived_from}``: per row its widget, slider track (§3.6: the soft range, else a closed finite hard range; a log track
  starts three decades below its top when its lower end is 0 or open; a "0" stop when the hard range admits 0 and the
  track does not reach it), default (literal, the family priors, the derivation's formula or required), editability
  (the class), a tier hint (the stage; E for the event-tier keys of §4.7) and ``feeds``, the derived accuracy keys each
  key may move under each accuracy profile (the inverse of ``derived_from``).
- **Resolve.** A base (``{config_path}``, whose relative paths start from the parent of the path as given, as
  ``ballasim run`` takes it; ``{config, base_dir}``; or ``{run: rid}``, whose specification is rebuilt from run.json
  ``inputs`` as one item list: the recorded configuration without ``output.dir`` and ``output.overwrite``, the
  recorded CLI layer as ``KEY=<json>`` items without them, then the user's) and the user's ``sets`` as ``KEY=VALUE``
  items (a path-typed value absolute: the path widget sends one, and a relative one would start from another directory
  here, in a saved file and in a job), through ``config.load`` and the C++ resolver on a bundle that is already
  cached: an EVIMO2 bundle under its cache key, a Blender exchange's converted bundle, a bundle directory, or a scene
  YAML whose ``bundle`` job completed (``<jobs_root>/bundles.json``, UI-6's record) and whose assets are all cached
  (``assets.cached_key``). Otherwise 202 ``{needs_bundle}``: this module never builds a bundle or imports an asset
  (§4.12). Errors are 422 with the resolver's diagnostics and every missing required key (``missing_keys``, collect
  mode); the OS's refusals map as §6 says (403 BS-UI-PATH, 422 BS-UI-SOURCE); a success lists the warnings as
  diagnostics, the stage digests and ``relevant`` (each conditional row's visibility, §4.9). The tier
  (``compare_to``, ``config::classify``) comes with UI-4b.
- **Save** (``/configs``): a format-2 YAML of the base with the edits merged in (layer ``config``), every path-typed key
  absolute against the base's directory, inside a writable root and never in ``PROFILES_DIR``; a run base's own
  directory as ``output.dir`` with ``output.overwrite: false`` unless an edit sets them.

Python here is glue: every check, resolution, diagnostic and derivation table is the C++ resolver's. Blocking work
(loading YAML and bundles, resolving) runs on the server's 2-thread pool.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import copy
import hashlib
import io
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ballasim import _core, config

from .streams import StreamError, _inside, _path_errors, decode_rid, resolve_path

FORM_FORMAT = "ballasim.ui.form/1"
BUNDLES_FORMAT = "ballasim.ui.bundles/1"
BUNDLES_FILE = "bundles.json"
#: The event-tier keys (§4.7): a pure function of a session's raw events, so their hint is E.
E_KEYS = frozenset({"sensor.readout_delay_s", "sensor.readout_quantum_s", "sensor.clock_offset_ns",
                    "sensor.clock_drift_ppm", "output.label_cut"})
EDITABLE = {"N-accuracy": "accuracy", "D-measured": "data", "E-heuristic": "heuristic"}
#: The stage's hint (§3.6); a schedule edit is R until UI-4b's S~ table (config::classify) refines it.
STAGE_TIER = {"sensor": "S", "output": "O", "schedule": "R", "render": "R", "scene": "R", "none": "none"}
#: A log track's span below its top when the range's lower end is 0 or open (§3.6).
LOG_DECADES = 3
BUNDLE_LRU = 2
#: A run's output keys: the ``{run}`` rebuild drops them (a recorded ``--overwrite`` must never reach a new run).
OUTPUT_KEYS = ("output.dir", "output.overwrite")
MAX_EXPLAIN = 64
#: JSON numbers above this lose integer precision in a browser: the record carries such integers as decimal strings.
_SAFE_INT = (1 << 53) - 1


# ---------------------------------------------------------------------------------------------------------------
# The form model (§3.6)
# ---------------------------------------------------------------------------------------------------------------

def track(row: dict) -> dict | None:
    """A numeric row's slider track ``{lo, hi, from, zero_stop}``, or None (a ``number`` widget)."""
    if row["type"] not in ("float", "int"):
        return None
    log = row["scale"] == "log"
    hard = row["hard"] or {}
    for name in ("soft", "hard"):
        rng = row[name]
        if not rng or rng["lo"] is None or rng["hi"] is None or rng["hi_open"]:
            continue                                       # an open bound is never a track end
        if name == "hard" and rng["lo_open"]:
            continue                                       # the hard range serves only when closed
        lo, hi = rng["lo"], rng["hi"]
        if log:
            if not hi > 0:
                continue
            if lo <= 0 or rng["lo_open"]:
                lo = hi / 10 ** LOG_DECADES
        elif rng["lo_open"]:
            continue
        if not lo < hi:
            continue
        zero = hard.get("lo") == 0 and not hard.get("lo_open") and lo > 0
        return {"lo": lo, "hi": hi, "from": name, "zero_stop": bool(zero)}
    return None


def widget(row: dict, trk: dict | None) -> str:
    t = row["type"]
    if t.endswith("_list"):
        return "vector"
    return {"bool": "toggle", "enum": "select", "path": "path", "string": "text"}.get(
        t, "slider" if trk is not None else "number")


def tier_hint(row: dict) -> str:
    return "E" if row["key"] in E_KEYS else STAGE_TIER[row["stage"]]


def group_of(key: str) -> str:
    """The form group of a key: its section, two levels for scene.* and dataset.* (as PARAMS_REFERENCE.md)."""
    parts = key.split(".")
    return ".".join(parts[:2]) if parts[0] in ("scene", "dataset") and len(parts) > 2 else parts[0]


def _default(row: dict, priors: dict[str, dict[str, dict]]) -> dict:
    kind = row["default"]
    if kind == "literal":
        return {"kind": "literal", "value": row["value"], "status": row["status"]}
    if kind == "family":
        return {"kind": "family", "by_family": {
            fam: {k: e[k] for k in ("value", "status", "source", "evimo2_derived", "no_literature", "inherited")}
            for fam, table in priors.items() if (e := table.get(row["key"])) is not None}}
    if kind == "derived":
        return {"kind": "derived", "formula": row["formula"]}
    return {"kind": "required"}


def feeds_table() -> dict[str, dict[str, list[str]]]:
    """key -> {accuracy profile: the derived accuracy keys it feeds} (the inverse of ``derived_from``)."""
    out: dict[str, dict[str, list[str]]] = {}
    for ap in _core.config.accuracy_profiles():
        for derived, inputs in _core.config.derived_from(ap["name"]).items():
            for k in inputs:
                out.setdefault(k, {}).setdefault(ap["name"], []).append(derived)
    return out


def form_model(schema_sha256: str) -> dict:
    """The ``ballasim.ui.form/1`` document (§3.6)."""
    schema = _core.config.schema()
    families = _core.config.families()
    profiles = _core.config.accuracy_profiles()
    priors = {f["name"]: {e["key"]: e for e in f["priors"]} for f in families}
    feeds = feeds_table()
    groups: dict[str, list[str]] = {}
    rows = []
    for r in schema:
        trk = track(r)
        groups.setdefault(group_of(r["key"]), []).append(r["key"])
        rows.append({
            "key": r["key"], "scope": r["scope"], "type": r["type"], "length": r["length"], "nullable": r["nullable"],
            "unit": r["unit"], "choices": r["choices"] or None, "hard": r["hard"], "soft": r["soft"],
            "scale": r["scale"], "widget": widget(r, trk), "track": trk, "default": _default(r, priors),
            "class": r["class"], "editable": EDITABLE.get(r["class"], "tunable"), "stage": r["stage"],
            "tier_hint": tier_hint(r), "feeds": {ap["name"]: feeds.get(r["key"], {}).get(ap["name"], [])
                                                 for ap in profiles},
            "when": r["when"], "also_when": r["also_when"], "help": r["help"], "source": r["source"],
            "status": r["status"]})
    return {"format": FORM_FORMAT, "schema_sha256": schema_sha256,
            "groups": [{"id": g, "title": " / ".join([g.split(".")[0].capitalize(), *g.split(".")[1:]]), "keys": keys}
                       for g, keys in groups.items()],
            "rows": rows, "families": families, "accuracy_profiles": profiles}


# ---------------------------------------------------------------------------------------------------------------
# Bundles recorded by `bundle` jobs (§4.12)
# ---------------------------------------------------------------------------------------------------------------

def _stamp(path: Path) -> dict:
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _file_stamp(path: str) -> tuple[int, int] | None:
    """(size, mtime_ns) of a file, None when it cannot be read: an LRU entry's freshness."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def read_bundles(jobs_root: Path | None) -> dict[str, dict]:
    """The scene YAMLs whose ``bundle`` job completed, ``{real path: {size, mtime_ns}}``, from
    ``<jobs_root>/bundles.json``. Its one writer is UI-6's job completion (``Jobs._record_bundle``); its layout is
    ``{format: "ballasim.ui.bundles/1", entries: [{path: real path, size, mtime_ns: decimal string}]}`` (ui.md §4.12).
    Empty without one, or when it is not that layout; a malformed entry is skipped."""
    if jobs_root is None:
        return {}
    try:
        doc = json.loads((Path(jobs_root) / BUNDLES_FILE).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(doc, dict) or doc.get("format") != BUNDLES_FORMAT or not isinstance(doc.get("entries"), list):
        return {}
    out = {}
    for e in doc["entries"]:
        if not isinstance(e, dict) or not isinstance(e.get("path"), str):
            continue
        try:
            out[e["path"]] = {"size": int(e["size"]), "mtime_ns": int(e["mtime_ns"])}
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------------------------------------------

def _bad(message: str) -> StreamError:
    return StreamError(400, "BS-UI-REQUEST", message)


def set_items(sets: Any) -> list[str]:
    """The ``sets`` mapping as ``KEY=VALUE`` items for ``config.load`` (§3.6: a mapping would bypass the YAML parse;
    an empty text is null; the split is at the first ``=``, so a value may hold one). A path-typed value must be
    absolute (or start with ``$`` or ``~``): ``config.load`` would expand a relative one against the server's working
    directory, a saved file against the base's and a job against its own, so it is refused (400 BS-UI-REQUEST; the
    path widget sends absolute paths). A value that does not parse is left to ``config.load``'s diagnostic."""
    if sets is None:
        return []
    if not isinstance(sets, dict):
        raise _bad("sets: a mapping of full keys to value texts")
    items = []
    paths = config.path_keys()
    for k, v in sets.items():
        if not isinstance(k, str) or not k.strip() or "=" in k or not isinstance(v, str):
            raise _bad("sets: a mapping of full keys to value texts (strings)")
        item = f"{k}={v}"
        if k.strip() in paths:
            try:
                value = config.parse_sets([item])[k.strip()]
            except Exception:  # noqa: BLE001  config.load reports it as a diagnostic
                value = None
            if isinstance(value, str) and not _absolute_text(value):
                raise _bad(f"sets: {k.strip()} = {value!r}: a path value is absolute (a relative one would start "
                           f"from another directory here, in a saved file and in a job)")
        items.append(item)
    return items


def _profile_ids(profiles: Any) -> list[str]:
    if profiles is None:
        return []
    if not isinstance(profiles, list) or not all(isinstance(p, str) for p in profiles):
        raise _bad("profiles: a list of profile ids")
    return list(profiles)


def _flat(doc: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """The leaves of a run configuration's sections as (full key, value); camera blocks are not walked."""
    out = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if key in ("format", "profiles", "cameras"):
                continue
            out += _flat(v, key) if isinstance(v, dict) else [(key, v)]
    return out


def _absolute_text(value: str) -> bool:
    """A path value that is absolute as written or after expansion of its leading variable or ``~``."""
    return value.startswith(("/", "$", "~"))


def _put(m: dict, parts: list[str], value) -> None:
    """Sets the key ``parts`` in a mapping where it is written nested or dotted (the longest existing prefix wins);
    a new key is written dotted."""
    for i in range(1, len(parts) + 1):
        head = ".".join(parts[:i])
        if head not in m:
            continue
        if i == len(parts):
            m[head] = value
            return
        if m[head] is None:
            m[head] = {}
        if isinstance(m[head], dict):
            _put(m[head], parts[i:], value)
            return
    m[".".join(parts)] = value


def merge_key(doc: dict, full: str, value) -> None:
    """Sets a full dotted key (``cameras.<camera>.<key>`` or a run key) in a run configuration."""
    parts = full.split(".")
    if parts[0] == "cameras":
        if len(parts) < 3:
            raise _bad(f"sets: {full}: expected cameras.<camera>.<key>")
        if not isinstance(doc.get("cameras"), dict):
            doc["cameras"] = {}
        if not isinstance(doc["cameras"].get(parts[1]), dict):
            doc["cameras"][parts[1]] = {}
        _put(doc["cameras"][parts[1]], parts[2:], value)
        return
    if len(parts) < 2 or parts[0] in ("format", "profiles"):
        raise _bad(f"sets: {full}: not a key")
    if not isinstance(doc.get(parts[0]), dict):
        doc[parts[0]] = {}
    _put(doc[parts[0]], parts[1:], value)


def _drop(doc: dict, key: str) -> None:
    """Removes a run key from a configuration, written nested or dotted (at the top or in its section)."""
    parts = key.split(".")
    for i in range(len(parts)):
        node = doc
        for p in parts[:i]:
            node = node.get(p) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(".".join(parts[i:]), None)


def _jsonable(x):
    """Integers beyond 2^53 as decimal strings (§3.3), recursively."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x if -_SAFE_INT <= x <= _SAFE_INT else str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _error_row(message: str, key: str | None = None) -> dict:
    """An error the C++ resolver did not raise (YAML, ``--set`` items, path variables, scene files): one diagnostic of
    kind ``error``."""
    return {"kind": "error", "severity": "error", "key": key, "camera": None, "layer": None,
            "keys": [key] if key else [], "message": message, "nearest": None}


def _diagnostic(e: Exception, key: str | None = None) -> dict:
    d = getattr(e, "diagnostic", None)
    return d if isinstance(d, dict) else _error_row(str(e) or type(e).__name__, key)


@contextlib.contextmanager
def _os_refusals(verb: str = "read"):
    """The OS's refusals while reading a configuration, a scene source or its files (or writing a saved file) as §6's
    codes, as ``streams._path_errors`` maps them: PermissionError 403 BS-UI-PATH, any other OSError 422 BS-UI-SOURCE.
    FileNotFoundError passes: the caller reports it (a diagnostic naming the key, or 404)."""
    try:
        yield
    except FileNotFoundError:
        raise
    except PermissionError:
        raise StreamError(403, "BS-UI-PATH", f"this user may not {verb} the path") from None
    except OSError as e:
        raise StreamError(422, "BS-UI-SOURCE", e.strerror or type(e).__name__) from None


@dataclass
class Base:
    """A resolve or save base: the configuration (a file path or a dict), its directory, the items that precede the
    user's (a run's recorded CLI layer, also as typed values) and the profile ids they append, warnings about the base
    and, for a run, its directory (a saved configuration's output.dir unless an edit sets one)."""

    source: Any
    base_dir: Path
    items: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recorded_cli: dict = field(default_factory=dict)
    run_dir: str | None = None


class Unresolved(Exception):
    """A 422 BS-UI-CONFIG answer: the diagnostics and the missing keys."""

    def __init__(self, diagnostics: list[dict], missing: list[dict] | None = None):
        super().__init__(diagnostics[0]["message"] if diagnostics else "the configuration does not resolve")
        self.diagnostics, self.missing = diagnostics, missing or []


class NeedsBundle(Exception):
    def __init__(self, kind: str, key: str, source: str):
        super().__init__(f"the {kind} bundle {key} is not built")
        self.doc = {"kind": kind, "key": key, "source": source}


# ---------------------------------------------------------------------------------------------------------------
# The endpoints
# ---------------------------------------------------------------------------------------------------------------

class Params:
    """``/form``, ``/resolve`` and ``/configs`` of one server: the form model (built once), the bundle LRU (two
    bundles, keyed by the cache key of the source's entry or by the source's path, real path and mtime; a scene YAML's
    also by its assets' cache keys, and reused only while the other files it names are unchanged) and the rebuild
    rules."""

    def __init__(self, ui):
        self.ui = ui
        self._form: bytes | None = None
        self._bundles: collections.OrderedDict = collections.OrderedDict()
        self._bundles_lock = threading.Lock()   # the pool's two threads share the LRU (a load runs outside it)
        #: The rows with when/also_when: the keys of /resolve's ``relevant``.
        self.conditional = [r["key"] for r in _core.config.schema() if r["when"] or r["also_when"]]

    @property
    def roots(self):
        return self.ui.settings.roots

    def default_root(self) -> Path:
        """The default output root: the first writable root (§3.1)."""
        root = next((r for r in self.roots if not r.readonly), None)
        if root is None:
            raise StreamError(403, "BS-UI-PATH", "no writable root")
        return Path(root.path)

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.ui.pool, fn, *args)

    # -- /form ---------------------------------------------------------------------------------------------------

    def form_body(self) -> bytes:
        if self._form is None:
            sha = hashlib.sha256(self.ui.params_body()).hexdigest()
            self._form = json.dumps(form_model(sha), ensure_ascii=False, allow_nan=False).encode()
        return self._form

    # -- bases (§3.6) --------------------------------------------------------------------------------------------

    def base(self, spec: Any) -> Base:
        if not isinstance(spec, dict) or sum(k in spec for k in ("config_path", "config", "run", "cache")) != 1:
            raise _bad("base: give exactly one of config_path, config (with base_dir), run or cache")
        if "config_path" in spec:
            ref = spec["config_path"]
            real = resolve_path(self.roots, ref)        # containment on the real path
            if not real.is_file():
                raise StreamError(422, "BS-UI-SOURCE", "base.config_path: not a file")
            # read from the path as given, so relative paths start from its parent as `ballasim run` takes it
            # (config.load: path.parent.absolute()), not from the real path's: a symlinked configuration names the
            # same scene here, on the CLI and in UI-6's jobs
            if ref.startswith("/"):
                given = Path(ref)
            else:
                root_id, rel = decode_rid(ref)
                given = (Path(next(r for r in self.roots if r.id == root_id).path) / rel).absolute()
            return Base(str(given), given.parent)
        if "config" in spec:
            base_dir = spec.get("base_dir")
            if not isinstance(spec["config"], dict) or not isinstance(base_dir, str) or not os.path.isabs(base_dir):
                raise _bad("base: config is a mapping and base_dir an absolute path")
            return Base(copy.deepcopy(spec["config"]), Path(base_dir))
        if "cache" in spec:
            raise StreamError(404, "BS-UI-CACHE-NONE", "no signal cache with this id (signal caches come with UI-9)")
        return self.rebuild(resolve_path(self.roots, spec["run"]))

    @_path_errors()
    def rebuild(self, run_dir: Path) -> Base:
        """A run's specification from its run.json ``inputs`` (§3.6 ``{run}``): the recorded configuration (its file's
        directory as the base) without ``output.dir`` and ``output.overwrite`` (a placeholder ``output.dir`` under the
        default output root takes the first's place: a stage-none row; a recorded ``--overwrite`` must never reach a
        saved file), the recorded CLI layer as ``KEY=<json>`` items (JSON is YAML flow: null, '1e3' and '#'
        round-trip) without those two keys, and its profiles. Refused: format 1, a supplied bundle, a dict source with
        a relative source path."""
        try:
            doc = json.loads((run_dir / "run.json").read_text())
        except FileNotFoundError:
            raise StreamError(404, "BS-UI-NOT-FOUND", "not a run directory (no run.json)") from None
        except ValueError:
            raise StreamError(422, "BS-UI-SOURCE", "run.json is not JSON") from None
        inputs = doc.get("inputs") if isinstance(doc, dict) else None
        if not isinstance(inputs, dict) or "resolved" not in doc or not isinstance(inputs.get("config"), dict):
            raise StreamError(422, "BS-UI-RUN-FORMAT1", "a format-1 run (no resolved block); convert it with "
                                                        "python tools/config/convert_v1.py --runs")
        if (inputs.get("bundle") or {}).get("supplied"):
            raise StreamError(422, "BS-UI-RUN-SUPPLIED", "the run's bundle was supplied: its configuration holds no "
                                                         "scene keys")
        raw = copy.deepcopy(inputs["config"])
        for key in OUTPUT_KEYS:
            _drop(raw, key)
        file = inputs.get("file")
        if file:
            base_dir = Path(file).parent
        else:
            base_dir = self.default_root()
            paths = config.path_keys() - {"output.dir"}
            for key, value in _flat(raw):
                if key in paths and isinstance(value, str) and not _absolute_text(value):
                    raise StreamError(422, "BS-UI-RUN-RELPATH", f"a dict-source run with the relative source path "
                                                                f"{key} = {value!r}")
        if not isinstance(raw.get("output"), dict):
            raw["output"] = {}
        raw["output"]["dir"] = str(self.default_root() / ".ui-resolve")   # never created: resolve writes nothing
        cli = inputs.get("cli") or {}
        recorded = {k: v for k, v in (cli.get("set") or {}).items() if k.strip() not in OUTPUT_KEYS}
        items = [f"{k}={json.dumps(v)}" for k, v in recorded.items()]
        warnings = []
        for p in inputs.get("profiles") or []:
            pid = p.get("id") if isinstance(p, dict) else None
            now = None
            if isinstance(pid, str):
                with contextlib.suppress(OSError):
                    now = hashlib.sha256(config.profile_path(pid).read_bytes()).hexdigest()
            if now is None or now != p.get("sha256"):
                warnings.append(f"profile {pid} changed since the run")
        return Base(raw, base_dir, items, list(cli.get("profiles") or []), warnings, recorded, str(run_dir))

    # -- bundles (§4.12) -----------------------------------------------------------------------------------------

    def _lru(self, key, load, fresh=None):
        """``load()``'s value under ``key``; with ``fresh``, a cached value is reused only while ``fresh(value)``."""
        with self._bundles_lock:
            if key in self._bundles and (fresh is None or fresh(self._bundles[key])):
                self._bundles.move_to_end(key)
                return self._bundles[key]
        value = load()
        with self._bundles_lock:
            self._bundles[key] = value
            while len(self._bundles) > BUNDLE_LRU:
                self._bundles.popitem(last=False)
        return value

    def _readonly(self, path: Path) -> bool:
        real = os.path.realpath(path)
        return any(r.readonly and _inside(real, os.path.realpath(r.path)) for r in self.roots)

    def source_bundle(self, values: dict) -> tuple[Any, dict]:
        """The scene source's bundle and its facts (digest, low_light) from the LRU or a cache; NeedsBundle
        otherwise (runs on the pool)."""
        from ballasim.engine.scene import (
            _GROUP_KEYS,
            SIDECAR,
            bundle_cache_root,
            bundle_digest,
            evimo2_cache_key,
            evimo2_sequence_dir,
        )
        from ballasim.scene.bundle import SceneBundle

        if values.get("scene.evimo2.sequence") is not None:
            from ballasim.evimo2.adapter import Evimo2Lighting, bundle_low_light, relight
            from ballasim.scene.bundle import BundleError

            key = evimo2_cache_key(values)
            entry = bundle_cache_root() / key
            if not ((entry / "bundle.json").is_file() and (entry / SIDECAR).is_file()):
                raise NeedsBundle("evimo2", key, str(values["scene.evimo2.sequence"]))
            unlit, sidecar = self._lru(("evimo2", key), lambda: (SceneBundle.load(entry),
                                                                  json.loads((entry / SIDECAR).read_text())))
            lighting = Evimo2Lighting.from_values(values) if values["scene.evimo2.lights"] else None
            try:
                bundle = relight(unlit, sidecar, lighting)
            except BundleError as e:                   # the light groups' checks, as load_evimo2 reports them
                seq = evimo2_sequence_dir(values["scene.evimo2.sequence"], values.get("scene.evimo2.camera"))
                raise config.ConfigError(f"{_GROUP_KEYS[bool(sidecar['low_light'])]}: refused for {seq}: {e}") from None
            return bundle, {"digest": key, "low_light": bundle_low_light(bundle)}
        if values.get("scene.path") is None:
            raise Unresolved([{**_error_row("scene.path or scene.evimo2.sequence is required (or a bundle given to "
                                            "the run)", "scene.path"), "kind": "required",
                               "keys": ["scene.path", "scene.evimo2.sequence"]}])
        path = Path(values["scene.path"])
        if path.is_dir() and (path / "exchange.json").is_file():
            from ballasim.blender.convert import blender_cache_key
            from ballasim.blender.exchange import read_exchange

            ex = read_exchange(path)
            key = blender_cache_key(values, ex)
            entry = ex.root / "cache" / key
            if not (entry / "bundle.json").is_file():
                if self._readonly(ex.root):
                    raise StreamError(422, "BS-UI-EXT-READONLY", "this edit needs a new bundle inside a read-only "
                                                                 "exchange: change scene.blender.* in Blender, or copy "
                                                                 "the job into a writable root")
                raise NeedsBundle("blender", key, str(path))
            bundle, digest = self._lru(("blender", key), lambda: _with_digest(SceneBundle.load(entry), bundle_digest))
            return bundle, {"digest": digest, "low_light": None}
        if path.is_dir():
            stamp = _stamp(path / "bundle.json") if (path / "bundle.json").is_file() else _stamp(path)
            bundle, digest = self._lru(("dir", os.path.realpath(path), stamp["mtime_ns"]),
                                       lambda: _with_digest(SceneBundle.load(path), bundle_digest))
            return bundle, {"digest": digest, "low_light": None}
        if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
            from ballasim.scene import assets
            from ballasim.scene.authoring import asset_specs, load_scene

            real = os.path.realpath(path)
            stamp = _stamp(Path(real))
            if read_bundles(self.ui.settings.jobs_root).get(real) != stamp:
                raise NeedsBundle("yaml", real, str(path))
            keys = tuple(assets.cached_key(p, **opts) for p, opts in asset_specs(path))
            if None in keys:
                raise NeedsBundle("yaml", real, str(path))

            def load():
                files: list[Path] = []
                bundle = load_scene(path, files=files)
                stamps = tuple((str(f), _file_stamp(str(f))) for f in dict.fromkeys(files))
                return bundle, bundle_digest(bundle), stamps

            # keyed by the assets' cache keys too: an edited mesh re-imported by its bundle job (or by a CLI run) is
            # a new key; a texture the YAML names is read by every load, so the entry lives while its files are
            # unchanged. Loaded from the path as given, as load_source does (the meta records the asset paths).
            bundle, digest, _ = self._lru(("yaml", str(path), real, stamp["mtime_ns"], keys), load,
                                          lambda v: all(_file_stamp(f) == st for f, st in v[2]))
            return bundle, {"digest": digest, "low_light": None}
        if not path.exists():
            raise FileNotFoundError(f"scene.path {path}: no such file or directory")
        raise ValueError(f"scene.path {path}: expected a bundle directory, a Blender exchange or a scene YAML file")

    # -- /resolve ------------------------------------------------------------------------------------------------

    async def resolve(self, body: dict) -> tuple[int, dict]:
        base = self.base(body.get("base"))
        items = base.items + set_items(body.get("sets"))
        profiles = base.profiles + _profile_ids(body.get("profiles"))
        explain = body.get("explain") or []
        if not isinstance(explain, list) or len(explain) > MAX_EXPLAIN or not all(isinstance(k, str) for k in explain):
            raise _bad(f"explain: a list of at most {MAX_EXPLAIN} keys")
        device = await self.ui.wait_device()
        try:
            return 200, await self._run(self._resolve, base, items, profiles, explain, device, body.get("compare_to"))
        except Unresolved as e:
            return 422, {"ok": False, "error": {"code": "BS-UI-CONFIG", "message": str(e)},
                         "diagnostics": e.diagnostics, "missing": e.missing}
        except NeedsBundle as e:
            return 202, {"needs_bundle": e.doc}

    def _resolve(self, base: Base, items: list[str], profiles: list[str], explain: list[str], device: bool,
                 comparison: dict | None = None) -> dict:
        from ballasim.engine.scene import render_bundle

        t0 = time.perf_counter()
        try:
            with _os_refusals():
                spec = config.load(base.source, sets=items, profiles=profiles, base_dir=base.base_dir)
        except (config.ConfigError, FileNotFoundError) as e:
            raise Unresolved([_diagnostic(e)]) from None
        values = spec.scene.values()["run"]
        t1 = time.perf_counter()
        try:
            with _os_refusals():
                source, facts = self.source_bundle(values)
                bundle = render_bundle(source, values)
        except (config.ConfigError, FileNotFoundError, ValueError) as e:
            key = "scene.evimo2.sequence" if values.get("scene.evimo2.sequence") is not None else "scene.path"
            raise Unresolved([_diagnostic(e, key)]) from None
        t2 = time.perf_counter()
        info = config.bundle_info(bundle, supplied=False, device_available=device, **facts)
        try:
            resolved = _core.config.resolve(spec.inputs, info)
        except ValueError as e:
            first = _diagnostic(e)
            if first["kind"] not in ("required", "family_missing"):
                raise Unresolved([first]) from None
            try:                                    # every unresolved required key at once (collect mode)
                missing = _core.config.missing_keys(spec.inputs, info, diagnostics=True)
            except ValueError:
                missing = []
            if not missing:
                raise Unresolved([first]) from None
            raise Unresolved([m.pop("diagnostic") for m in missing], missing) from None
        t3 = time.perf_counter()
        warnings = list(resolved.warnings) + base.warnings
        diagnostics = list(resolved.diagnostics) + [{**_error_row(w), "kind": "profile", "severity": "warning"}
                                                    for w in base.warnings]
        relevant = {cam: {k: _core.config.relevant(k, resolved, cam) for k in self.conditional}
                    for cam in resolved.cameras}
        out = {"ok": True, "cameras": resolved.cameras, "span_ns": [str(t) for t in resolved.span],
               "record": _jsonable(resolved.record()), "warnings": warnings, "missing": [],
               "diagnostics": diagnostics, "stage_digests": resolved.digests(), "tier": self._classify(resolved, comparison),
               "relevant": relevant,
               "timing_ms": {"load": round(1e3 * (t1 - t0), 3), "bundle": round(1e3 * (t2 - t1), 3),
                             "resolve": round(1e3 * (t3 - t2), 3)}}
        if explain:
            out["explain"] = self._explain(spec, resolved, explain)
        return out

    @_path_errors()
    def _classify(self, resolved, comparison: dict | None) -> dict | None:
        try:
            return self._classify_record(resolved, comparison)
        except (ValueError, KeyError, TypeError) as error:
            raise StreamError(422, "BS-UI-SOURCE", f"comparison run record is invalid: {error}") from None

    def _classify_record(self, resolved, comparison: dict | None) -> dict | None:
        """Authoritative tiers against a writer run or a cache (ui.md §4.7); cache lookup arrives with UI-9."""
        if comparison is None:
            return None
        if not isinstance(comparison, dict) or sum(k in comparison for k in ("run", "cache")) != 1:
            raise _bad("compare_to: exactly one of run or cache")
        cache = None
        if "cache" in comparison:
            manager = getattr(self.ui, "caches", None)
            if manager is None:
                raise StreamError(404, "BS-UI-CACHE-NONE", "no signal cache with this id")
            cache = manager.get(comparison["cache"])
            run_dir = Path(cache["run_dir"])
        else:
            run_dir = resolve_path(self.roots, comparison["run"])
        with _os_refusals():
            doc = json.loads((run_dir / "run.json").read_text())
        if not isinstance(doc, dict) or not isinstance(doc.get("resolved"), dict) or not isinstance(
                doc.get("stage_digests"), dict):
            raise StreamError(422, "BS-UI-RUN-FORMAT1", "comparison needs a format-2 run record")
        old = _core.config.from_record(doc["resolved"], doc["stage_digests"])
        summaries = {c["name"]: c for c in doc.get("cameras", [])}
        out = {}
        named = comparison.get("camera")
        if named is not None and (not isinstance(named, str) or named not in resolved.cameras):
            raise _bad("compare_to.camera: a camera of the resolved configuration")
        for camera in resolved.cameras:
            if named is not None and camera != named:
                continue
            if camera not in old.cameras:
                out[camera] = {"tier": "R", "moved": ["scene"], "keys": [],
                               "reasons": ["camera absent from the writer run"]}
                continue
            summary = summaries.get(camera, {})
            facts = {"photometry_mode": old.get("photometry.mode", camera),
                     "scale": summary.get("illuminance_scale", 1.0),
                     "reference_current_fA": old.get("photometry.reference_current_fA", camera) or 1.0,
                     "precision": "fp32" if summary.get("sensor_device") == "cuda" else
                                  old.get("accuracy.sensor_precision")}
            if cache is not None:
                ph = cache["photometry"]
                facts.update(photometry_mode=ph["mode"], scale=ph["scale"],
                             reference_current_fA=ph.get("reference_current_fA") or 1.0,
                             precision=cache["sensor_precision"])
            out[camera] = _core.config.classify(old, resolved, camera, facts)
        return out

    @staticmethod
    def _explain(spec, resolved, keys: list[str]) -> dict:
        rows = {r["key"]: r for r in _core.config.schema()}
        out: dict = {"run": {}, "cameras": {cam: {} for cam in resolved.cameras}}
        for k in keys:
            row = rows.get(k)
            if row is None:
                raise _bad(f"explain: unknown key {k}")
            if row["scope"] == "camera":
                for cam in resolved.cameras:
                    out["cameras"][cam][k] = _jsonable(_core.config.explain(spec.inputs, resolved, k, cam))
            else:
                out["run"][k] = _jsonable(_core.config.explain(spec.inputs, resolved, k))
        return out

    # -- /configs ------------------------------------------------------------------------------------------------

    @_path_errors()
    def target(self, ref: Any) -> Path:
        """The real path a new or replaced configuration file takes: a ``rid`` or an absolute path, a ``.yaml`` or
        ``.yml`` name in an existing directory inside a writable root, never in a read-only root or ``PROFILES_DIR``
        (403 BS-UI-PATH)."""
        if not isinstance(ref, str) or not ref:
            raise _bad("path: a resource id or an absolute path")
        if ref.startswith("/"):
            candidate = Path(ref)
        else:
            root_id, rel = decode_rid(ref)
            root = next((r for r in self.roots if r.id == root_id), None)
            if root is None:
                raise StreamError(404, "BS-UI-NOT-FOUND", f"no root {root_id!r}")
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise StreamError(403, "BS-UI-PATH", "the path leaves its root")
            candidate = Path(root.path) / rel
        if candidate.suffix.lower() not in (".yaml", ".yml"):
            raise _bad("path: a .yaml or .yml file")
        parent = os.path.realpath(candidate.parent)
        real = os.path.join(parent, candidate.name)
        if os.path.lexists(real):
            real = os.path.realpath(real)
        roots = [os.path.realpath(r.path) for r in self.roots if not r.readonly]
        if not any(_inside(real, r) for r in roots) or self._readonly(Path(real)):
            raise StreamError(403, "BS-UI-PATH", "the path lies outside the writable roots")
        if _inside(real, os.path.realpath(config.PROFILES_DIR)):
            raise StreamError(403, "BS-UI-PATH", "profiles are never written")
        if not os.path.isdir(parent):
            raise StreamError(404, "BS-UI-NOT-FOUND", "no such directory")
        if os.path.isdir(real):
            raise _bad("path: a directory")
        return Path(real)

    async def save_config(self, body: dict) -> tuple[int, dict]:
        target = self.target(body.get("path"))
        base = self.base(body.get("base"))
        items = set_items(body.get("sets"))
        profiles = _profile_ids(body.get("profiles"))
        try:
            return 201, await self._run(self._save, target, base, items, profiles)
        except Unresolved as e:
            return 422, {"ok": False, "error": {"code": "BS-UI-CONFIG", "message": str(e)},
                         "diagnostics": e.diagnostics, "missing": []}

    def _save(self, target: Path, base: Base, items: list[str], profiles: list[str]) -> dict:
        from ruamel.yaml import YAML

        try:
            with _os_refusals():
                if isinstance(base.source, str):
                    doc = YAML(typ="safe").load(Path(base.source).read_text())
                else:
                    doc = copy.deepcopy(base.source)
            if not isinstance(doc, dict):
                raise config.ConfigError(f"{base.source}: a run configuration is a YAML mapping")
            edits = {**base.recorded_cli, **config.parse_sets(items)}
        except (config.ConfigError, ValueError) as e:
            raise Unresolved([_diagnostic(e)]) from None
        except FileNotFoundError as e:
            raise Unresolved([_error_row(f"{base.source}: {e.strerror or e}")]) from None
        if base.run_dir is not None:
            # a run's own directory, never overwritten (the rebuild dropped the recorded output.overwrite), unless an
            # edit names another
            merge_key(doc, "output.dir", base.run_dir)
            merge_key(doc, "output.overwrite", False)
        for k, v in edits.items():
            merge_key(doc, k, v)
        listed = list(doc.get("profiles") or [])
        for p in base.profiles + profiles:
            if p not in listed:
                listed.append(p)
        if listed:
            doc["profiles"] = listed
        paths = config.path_keys()
        for key, value in _flat(doc):
            if key in paths and isinstance(value, str) and not _absolute_text(value):
                merge_key(doc, key, str(base.base_dir / value))
        out = {"format": doc.pop("format", "ballasim.run/2"), **doc}
        try:
            with _os_refusals():
                config.load(out, base_dir=target.parent)
        except (config.ConfigError, FileNotFoundError) as e:
            raise Unresolved([_diagnostic(e)]) from None
        buf = io.StringIO()
        yaml = YAML(typ="safe")
        yaml.default_flow_style = False
        yaml.representer.sort_base_mapping_type_on_output = False   # format first, then the base's order
        yaml.dump(out, buf)
        data = buf.getvalue().encode()
        try:
            with _os_refusals("write"):
                _write_atomic(target, data)
        except FileNotFoundError:                       # the directory went away since target()
            raise StreamError(404, "BS-UI-NOT-FOUND", "no such directory") from None
        return {"path": str(target), "sha256": hashlib.sha256(data).hexdigest()}


def _with_digest(bundle, digest_fn) -> tuple[Any, str]:
    return bundle, digest_fn(bundle)


def _write_atomic(target: Path, data: bytes) -> None:
    """Writes ``data`` to ``target`` through a temporary file of its own (created exclusively, a random name: two saves
    of one target on the pool's two threads never share one) and an atomic replace; the temporary is removed on a
    failure."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

