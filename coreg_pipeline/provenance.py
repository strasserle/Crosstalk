#!/usr/bin/env python
"""
Provenance recording for the coregulatory pipeline.

Every input the pipeline consumes is registered here with its origin, a
checksum and its dimensions. The point is that "this pipeline fetches its own
data" becomes a checkable property of manifest.json rather than a claim in a
methods section: anything read from disk that was not fetched by stage 01 shows
up as origin="local-supplied" and is listed separately by `audit()`.

Two origin classes:
  fetched         -- retrieved from a public URL by this pipeline; carries the
                     URL, HTTP date and checksum, so a reviewer can re-fetch.
  local-supplied  -- a file provided by the user rather than fetched. Allowed,
                     but never silently: it is flagged in the audit so the
                     dependency is visible.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MANIFEST = BASE / "manifest.json"


def _relpath(path) -> str | None:
    """
    Record paths relative to the repository root, so the manifest carries no
    machine-specific layout. A path outside the repository (e.g. under
    $SPONGE_RESOURCES) is reduced to `<external>/<filename>`: the filename is
    what a reviewer needs, the directory it sat in is not.
    """
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return f"<external>/{p.name}"


def _load() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "inputs": {}, "outputs": {}}


def _save(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))


def sha256(path: str | Path, limit_mb: int | None = None) -> str:
    """
    Checksum a file. limit_mb hashes only the leading bytes, for files where a
    full hash costs more than the provenance is worth (multi-GB matrices); the
    returned digest is then prefixed so it can never be mistaken for a full one.
    """
    h = hashlib.sha256()
    cap = None if limit_mb is None else limit_mb * 1024 * 1024
    read = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            if cap is not None and read + len(chunk) > cap:
                h.update(chunk[: cap - read])
                return f"partial-{limit_mb}MB:{h.hexdigest()}"
            h.update(chunk)
            read += len(chunk)
    return h.hexdigest()


def record_input(key: str, *, path=None, url=None, origin="fetched",
                 rows=None, cols=None, note="", checksum_limit_mb=256) -> None:
    """Register one pipeline input."""
    m = _load()
    entry = {
        "origin": origin,
        "url": url,
        "path": _relpath(path),
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": rows,
        "cols": cols,
        "note": note,
    }
    if path and os.path.exists(path):
        entry["bytes"] = os.path.getsize(path)
        entry["sha256"] = sha256(path, limit_mb=checksum_limit_mb)
    m["inputs"][key] = entry
    _save(m)


def record_output(key: str, *, path=None, rows=None, cols=None, note="",
                  params=None) -> None:
    m = _load()
    entry = {"path": _relpath(path),
             "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "rows": rows, "cols": cols, "note": note, "params": params or {}}
    if path and os.path.exists(path):
        entry["bytes"] = os.path.getsize(path)
    m["outputs"][key] = entry
    _save(m)


def audit(verbose=True) -> dict:
    """
    Summarise provenance. Returns counts and the list of local-supplied inputs,
    which is the set a reviewer needs to scrutinise.
    """
    m = _load()
    fetched = {k: v for k, v in m["inputs"].items() if v["origin"] == "fetched"}
    supplied = {k: v for k, v in m["inputs"].items() if v["origin"] != "fetched"}
    if verbose:
        print(f"=== provenance audit ({MANIFEST.name}) ===")
        print(f"  fetched from public URLs : {len(fetched)}")
        for k, v in sorted(fetched.items()):
            print(f"      {k:28s} {v['url']}")
        print(f"  locally supplied         : {len(supplied)}")
        for k, v in sorted(supplied.items()):
            print(f"      {k:28s} {v['path']}  <- {v['note']}")
        print(f"  outputs written          : {len(m['outputs'])}")
    return {"fetched": fetched, "supplied": supplied, "outputs": m["outputs"]}


if __name__ == "__main__":
    audit()
