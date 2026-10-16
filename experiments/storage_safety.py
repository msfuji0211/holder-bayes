"""Storage-only guards; no changes to posterior targets or numerical caches.

Installed by the workflow at runtime so the existing inference source hashes,
cache identities, JSON bytes and array contents stay compatible.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

VERSION = "storage_io_v1_lossless_cache"
GIB = 1024**3
_ORIGINAL_SAVEZ = np.savez_compressed
_NUMPY_PICKLE_ARGUMENT = "allow_pickle" in inspect.signature(_ORIGINAL_SAVEZ).parameters


class StorageSpaceError(OSError):
    """An output write could not fit on its destination filesystem."""


def reserve_bytes():
    value = float(os.environ.get("HBAYES_STORAGE_RESERVE_GIB", "1"))
    if not np.isfinite(value) or value < 0:
        raise ValueError("HBAYES_STORAGE_RESERVE_GIB must be finite and nonnegative")
    return int(value*GIB)


def disk_folder(path):
    folder = Path(path)
    if not folder.is_dir():
        folder = folder.parent
    while not folder.exists():
        folder = folder.parent
    return folder


def is_space_error(exc):
    return isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}


def space_error(path, required=None, available=None):
    detail = ""
    if required is not None and available is not None:
        detail = f" Need {required/GIB:.3f} GiB including reserve; available {available/GIB:.3f} GiB."
    return StorageSpaceError(errno.ENOSPC,
        "Insufficient storage for a verified output."+detail+
        " Free disk space or move the complete outputs directory to a larger disk,"
        " preserving the old path with a symlink. Do not delete pilot reports or fit caches.", str(path))


def ensure_capacity(path, payload_bytes=0, *, reserve=None):
    keep = reserve_bytes() if reserve is None else int(reserve)
    free = shutil.disk_usage(disk_folder(path)).free
    required = max(0, int(payload_bytes))+keep
    if free < required:
        raise space_error(path, required, free)


@contextmanager
def closing_without_masking(resource):
    try:
        yield resource
    except BaseException:
        try:
            resource.close()
        except Exception:
            pass  # The first write failure is the useful error.
        raise
    else:
        resource.close()


def _replace_write(path, writer, payload_bytes, reserve):
    path = Path(path)
    ensure_capacity(path, payload_bytes, reserve=reserve)
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=path.name+".storage-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with closing_without_masking(os.fdopen(fd, "wb")) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if is_space_error(exc):
            raise space_error(path) from exc
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Never mask the write error with a cleanup error.


def atomic_bytes(path, content):
    path = Path(path)
    # Rewriting identical cache metadata changes neither bytes nor dependencies.
    if path.is_file() and path.read_bytes() == content:
        return
    _replace_write(path, lambda stream: stream.write(content), len(content), reserve=0)


def atomic_text(path, text):
    atomic_bytes(path, text.encode("utf-8"))


def atomic_json(path, value):
    from hbayes.paper_sampling import clean_json
    # Identical to the existing cache serialization, including its final newline.
    atomic_text(path, json.dumps(clean_json(value), indent=2, allow_nan=False)+"\n")


def _arrays(args, values):
    named = dict(values)
    for i, value in enumerate(args):
        key = f"arr_{i}"
        if key in named:
            raise ValueError(f"Cannot use un-named variables and keyword {key}")
        named[key] = value
    return {key: np.asanyarray(value) for key, value in named.items()}


def _archive(stream, values, allow_pickle):
    # Close the archive while its file is still open; preserve the first failure
    # even if writing the ZIP footer also fails on a full disk.
    with closing_without_masking(zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED)) as archive:
        for key, value in values.items():
            with closing_without_masking(archive.open(key+".npy", "w", force_zip64=True)) as member:
                np.lib.format.write_array(member, value, allow_pickle=allow_pickle)


def _save_archive(path, values, allow_pickle=True):
    # A conservative bound before sampling outputs are written. Compression
    # usually uses less; headers/ZIP framing and the reserve remain available.
    upper_bound = sum(int(value.nbytes) for value in values.values())
    upper_bound += upper_bound//1000+65536*(len(values)+1)
    _replace_write(path, lambda stream: _archive(stream, values, allow_pickle), upper_bound, reserve_bytes())


def atomic_npz(path, **values):
    _save_archive(Path(path), _arrays((), values))


def savez_compressed(file, *args, **values):
    allow_pickle = values.pop("allow_pickle", True) if _NUMPY_PICKLE_ARGUMENT else True
    named = _arrays(args, values)
    if hasattr(file, "write"):
        _archive(file, named, allow_pickle)
        return
    path = Path(os.fsdecode(file))
    if not str(path).endswith(".npz"):
        path = Path(str(path)+".npz")
    _save_archive(path, named, allow_pickle)


def install_storage_safety():
    from hbayes import paper_sampling
    paper_sampling.atomic_json = atomic_json
    paper_sampling.atomic_npz = atomic_npz
    # Existing experiment exporters refer to np.savez_compressed dynamically.
    # This covers their direct writes without altering numerical source hashes.
    np.savez_compressed = savez_compressed
    from experiments.compact_cache import install
    install(paper_sampling, raw_writer=atomic_npz, raw_json_writer=atomic_json)


def metadata():
    return {"version": VERSION, "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "npz_reserve_bytes": reserve_bytes(), "temporary_cleanup": "current_write_only",
            "inference_sources_changed": False,
            "compact_cache_sha256": hashlib.sha256(Path(__file__).with_name("compact_cache.py").read_bytes()).hexdigest(),
            "lossless_cache_enabled": os.environ.get("HBAYES_COMPACT_CACHE", "1") != "0"}


def save_error_report(writer, path, report, original):
    try:
        writer(path, report)
    except Exception as secondary:
        message = f"Could not save error report {path}: {secondary}. The original exception is preserved."
        if hasattr(original, "add_note"):
            original.add_note(message)
        print(message, file=sys.stderr, flush=True)
