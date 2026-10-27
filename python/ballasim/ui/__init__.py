"""The browser UI (docs/design/phase3/ui.md): a loopback Tornado server (``python -m ballasim ui``) and a Chrome-only
front end (``ui/static``, plain web code, no third-party libraries).

- :mod:`.security`: the security envelope (§3.2): Bearer secrets, single-use nonces and WebSocket tickets with TTLs,
  the Host, Fetch-Metadata and Origin checks, the response headers, log redaction;
- :mod:`.server`: the application, the runtime directory and its redirect files, the device probe, ``/meta``,
  ``/params``, ``/profiles``, and the ``ui`` command's start, ``--open``, ``--status`` and ``--stop``.
- :mod:`.streams`: event streams (``/streams``, ``/rate``, ``/blocks``; §3.3, §3.5, §4.2): sources in roots, a reader
  thread per source, the block LRU and the BSEV/BSRT messages over ``_core.wire`` (C++, GIL released).

Tornado is the ``ui`` extra's dependency: this package imports it only in :mod:`.server`, which ``python -m ballasim
ui`` imports at its first use.
"""
