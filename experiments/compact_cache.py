"""Lossless checkpoint deduplication; posterior and inference contracts stay intact.

Only chain checkpoints change format. Every decoded array is checked against
its dtype, shape and SHA256 of the original bytes. No inverse coordinate
transform, downcast, thinning, target change or sampler change is performed.
"""
from __future__ import annotations

import hashlib
import errno
import io
import json
import os
from pathlib import Path
import re
import socket
import shutil
import tempfile

import numpy as np

VERSION = "lossless_chain_references_v1"
MARKER = "__cache_compact_manifest__"
CHAIN = re.compile(r"(?:nuts|mh)_chain([0-9]+)\.npz\Z")
METHODS = {"DPD", "ExpSatDPD", "RatSatDPD", "NLL", "StudentT", "GammaDiv", "DownstreamNLL"}
_INSTALLED = False
_COMPACT_WRITER = None
_COMPACT_JSON_WRITER = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError("Object arrays cannot be compacted")
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + np.ascontiguousarray(value).tobytes()).hexdigest()


def identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def assert_payload(a, b):
    if list(a) != list(b) or any(not identical(a[k], b[k]) for k in a):
        raise ValueError("Compaction did not preserve every array byte, key, dtype and shape")


def raw_load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def decode(payload, path):
    if MARKER not in payload:
        return payload
    manifest = json.loads(np.asarray(payload[MARKER], dtype=np.uint8).tobytes())
    if manifest.get("version") != VERSION:
        raise ValueError(f"Unknown compact checkpoint format at {path}")
    decoded, final = {}, None
    for spec in manifest["arrays"]:
        key, kind = spec["key"], spec["kind"]
        if kind == "stored":
            value = payload[key]
        elif kind == "posterior_chain":
            reference = manifest.get("posterior", {})
            if reference.get("name") != "posterior.npz":
                raise ValueError("Compact checkpoint reference must be the local posterior.npz")
            canonical = Path(path).parent/"posterior.npz"
            if final is None:
                if sha(canonical) != reference["sha256"]:
                    raise ValueError(f"Compact checkpoint posterior hash differs: {canonical}")
                final = raw_load(canonical)
                if MARKER in final:
                    raise ValueError("The canonical posterior must remain an ordinary NPZ")
            index = reference["chain"]
            if not isinstance(index, int) or index < 0 or index >= final[key].shape[0]:
                raise ValueError("Invalid referenced chain index")
            value = final[key][index].copy()
        elif kind == "column_patch":
            base = decoded.get(spec["base"])
            if base is None or base.ndim != 2:
                raise ValueError("Missing physical-coordinate base for a column patch")
            columns = payload[spec["columns"]]
            values = payload[spec["values"]]
            if columns.ndim != 1 or columns.dtype.kind not in "iu" or len(np.unique(columns)) != len(columns):
                raise ValueError("Invalid compact column indices")
            if np.any(columns < 0) or np.any(columns >= base.shape[1]):
                raise ValueError("Compact column index is out of range")
            if values.dtype != base.dtype or values.shape != (base.shape[0], len(columns)):
                raise ValueError("Invalid compact column values")
            value = base.copy()
            value[:, columns] = values
        else:
            raise ValueError(f"Unknown compact array encoding: {kind}")
        if value.dtype.str != spec["dtype"] or list(value.shape) != spec["shape"] or array_sha(value) != spec["sha256"]:
            raise ValueError(f"Decoded compact array bytes differ: {path}, {key}")
        decoded[key] = value
    order = manifest["key_order"]
    if set(order) != set(decoded) or len(order) != len(decoded):
        raise ValueError("Invalid compact checkpoint key order")
    return {key: decoded[key] for key in order}


def load(path):
    return decode(raw_load(path), path)


def encode(payload, path, posterior=None, posterior_sha=None):
    payload = {key: np.asarray(value) for key, value in payload.items()}
    if any(k.startswith("__cache_compact_") for k in payload):
        raise ValueError("Reserved compact checkpoint key")
    physical = payload.get("samples_physical")
    sampling = payload.get("samples_sampling")
    if physical is None or sampling is None or physical.ndim != 2 or physical.dtype.hasobject:
        return payload
    if sampling.shape != physical.shape or sampling.dtype != physical.dtype:
        return payload
    # Compare the original bits, preserving signed zero and every float64 bit.
    bits = physical.dtype.itemsize
    a = np.ascontiguousarray(physical).view(np.uint8).reshape(*physical.shape, bits)
    b = np.ascontiguousarray(sampling).view(np.uint8).reshape(*sampling.shape, bits)
    changed = np.flatnonzero(np.any(a != b, axis=(0, 2))).astype(np.int32)
    if len(changed) == physical.shape[1] and posterior is None:
        return payload
    encoded, specs = {}, []
    manifest = {"version": VERSION, "key_order": list(payload)}
    match = CHAIN.fullmatch(Path(path).name)
    chain = int(match[1]) if match else -1
    referenced = set()
    if posterior is not None and chain >= 0:
        for key in ["samples_physical", "log_posterior_physical"]:
            if key in payload and key in posterior and chain < len(posterior[key]) and identical(payload[key], posterior[key][chain]):
                referenced.add(key)
        if referenced:
            manifest["posterior"] = {"name": "posterior.npz", "sha256": posterior_sha, "chain": chain}
    # Decode physical coordinates first, even if old NPZ keys used another order.
    order = ["samples_physical"] + [k for k in payload if k not in {"samples_physical", "samples_sampling"}] + ["samples_sampling"]
    for key in order:
        value = payload[key]
        spec = {"key": key, "dtype": value.dtype.str, "shape": list(value.shape), "sha256": array_sha(value)}
        if key in referenced:
            spec["kind"] = "posterior_chain"
        elif key == "samples_sampling":
            spec.update(kind="column_patch", base="samples_physical", columns="__cache_compact_columns__", values="__cache_compact_values__")
            encoded[spec["columns"]] = changed
            encoded[spec["values"]] = value[:, changed].copy()
        else:
            spec["kind"] = "stored"
            encoded[key] = value
        specs.append(spec)
    manifest["arrays"] = specs
    encoded[MARKER] = np.frombuffer(json.dumps(manifest, separators=(",", ":")).encode(), dtype=np.uint8).copy()
    assert_payload(payload, decode(encoded, path))
    return encoded


def archive_bytes(payload):
    from experiments.storage_safety import _archive
    stream = io.BytesIO()
    _archive(stream, payload, False)
    return stream.getvalue()


def _sync_folder(folder):
    fd = os.open(folder, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, getattr(errno, "ENOTSUP", -1)}:
                raise
    finally:
        os.close(fd)


def _write_bytes(path, content, reserve=0):
    from experiments.storage_safety import _replace_write
    _replace_write(path, lambda stream: stream.write(content), len(content), reserve)
    _sync_folder(Path(path).parent)


def _write_json(path, value):
    _write_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+"\n").encode())


def transaction_path(path):
    return Path(path).with_name(Path(path).name+".compaction.json")


def recover_transaction(path):
    """Finish an interrupted NPZ/receipt pair without accepting unverified bytes."""
    path = Path(path)
    journal = transaction_path(path)
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    if record.get("version") != VERSION or record.get("name") != path.name or not CHAIN.fullmatch(path.name):
        raise ValueError(f"Invalid compaction transaction: {journal}")
    stage_name = record["stage_name"]
    if Path(stage_name).name != stage_name or not stage_name.startswith(path.name+".compact-"):
        raise ValueError("Invalid compaction stage path")
    stage = path.parent/stage_name
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    if receipt not in [record["old_receipt"], record["new_receipt"]]:
        raise ValueError(f"Compaction receipt changed unexpectedly: {receipt_path}")
    actual = sha(path)
    if actual == record["old_sha256"]:
        if receipt != record["old_receipt"]:
            raise ValueError("New receipt paired with the old checkpoint")
        if not stage.exists():
            journal.unlink()
            _sync_folder(path.parent)
            return  # Nothing was replaced; all old data remain intact.
        if sha(stage) != record["new_sha256"]:
            raise ValueError("Compaction staged checkpoint checksum differs")
        assert_payload(load(path), load(stage))
        os.replace(stage, path)
        _sync_folder(path.parent)
    elif actual != record["new_sha256"]:
        raise ValueError(f"Neither old nor new verified checkpoint at {path}")
    load(path)  # Verify every encoded array before committing its new receipt.
    _write_json(receipt_path, record["new_receipt"])
    stage.unlink(missing_ok=True)
    journal.unlink()
    _sync_folder(path.parent)


def verified(path):
    path = Path(path)
    recover_transaction(path)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        return None
    record = json.loads(path.with_suffix(".json").read_text())
    if sha(path) != record["sha256"]:
        raise ValueError(f"Checkpoint hash differs: {path}")
    for name, digest in record.get("dependencies", {}).items():
        if Path(name).name != name or sha(path.parent/name) != digest:
            raise ValueError(f"Checkpoint metadata hash differs: {path.parent/name}")
    return load(path)


def replace_checkpoint(path, content, old_payload, *, reserve=64*1024**2):
    """One bounded temporary file; recovery journal bridges both atomic replaces."""
    path = Path(path)
    recover_transaction(path)
    old_sha = sha(path)
    old_receipt = json.loads(path.with_suffix(".json").read_text())
    if old_receipt["sha256"] != old_sha:
        raise ValueError("Checkpoint changed before compaction")
    new_sha = hashlib.sha256(content).hexdigest()
    if new_sha == old_sha:
        return
    from experiments.storage_safety import ensure_capacity
    ensure_capacity(path, len(content)+65536, reserve=reserve)
    fd, name = tempfile.mkstemp(prefix=path.name+".compact-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    stage = Path(name)
    journal = transaction_path(path)
    try:
        _write_bytes(stage, content)
        assert_payload(old_payload, load(stage))
        new_receipt = {**old_receipt, "sha256": new_sha, "storage": {"version": VERSION,
            "previous_checkpoint_sha256": old_sha, "arrays_bitwise_preserved": True}}
        record = {"version": VERSION, "name": path.name, "stage_name": stage.name,
            "old_sha256": old_sha, "new_sha256": new_sha, "old_receipt": old_receipt, "new_receipt": new_receipt}
        _write_json(journal, record)
        recover_transaction(path)
    finally:
        # If a journal exists, leave its verified stage for restart recovery.
        if not journal.exists():
            stage.unlink(missing_ok=True)


def completed_fit(folder):
    folder = Path(folder)
    contract_path = folder/"contract.json"
    if not contract_path.is_file():
        return None
    contract = json.loads(contract_path.read_text())
    if contract.get("model") not in {"regression", "density"} or contract.get("method") not in METHODS:
        return None
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if digest != folder.name:
        raise ValueError(f"Fit contract digest differs: {folder}")
    final = verified(folder/"posterior.npz")
    if final is None or not (folder/"summary.json").is_file():
        return None
    samples = final["samples_physical"]
    if samples.ndim != 3 or samples.shape[:2] != (contract["chains"], contract["samples"]):
        raise ValueError("Canonical posterior layout differs from its contract")
    if final["log_posterior_physical"].shape != samples.shape[:2]:
        raise ValueError("Canonical log density layout differs")
    if not np.isfinite(samples).all() or not np.isfinite(final["log_posterior_physical"]).all():
        raise ValueError("Non-finite canonical posterior")
    return final


def compact_fit(folder, *, apply=False, reserve=64*1024**2, protected=frozenset()):
    folder = Path(folder)
    result = {"folder": str(folder), "status": "incomplete_or_unsupported", "before_bytes": 0, "after_bytes": 0, "files": []}
    if folder.name in protected:
        result["status"] = "protected_mh_extension_parent"
        return result
    if list(folder.glob("*.compaction.json")):
        if not apply:
            result["status"] = "interrupted_transaction_run_apply_to_recover"
            return result
        for journal in sorted(folder.glob("*.compaction.json")):
            recover_transaction(journal.with_name(journal.name.removesuffix(".compaction.json")))
    final = completed_fit(folder)
    if final is None:
        return result
    posterior_sha = sha(folder/"posterior.npz")
    preserved = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and not CHAIN.fullmatch(p.name) and not re.fullmatch(r"(?:nuts|mh)_chain[0-9]+\.json", p.name)}
    paths = [p for p in sorted(folder.glob("*_chain*.npz")) if CHAIN.fullmatch(p.name)]
    # Detect an invalid later chain before changing any earlier chain in the fit.
    checked_sha = {}
    for path in paths:
        if verified(path) is None:
            raise ValueError(f"Missing verified chain receipt: {path}")
        checked_sha[path.name] = sha(path)
    for path in paths:
        if sha(path) != checked_sha[path.name]:
            raise ValueError(f"Chain changed during compaction: {path}")
        part = verified(path)
        if part is None:
            raise ValueError(f"Missing verified chain receipt: {path}")
        encoded = encode(part, path, final, posterior_sha)
        content = archive_bytes(encoded)
        old_size = path.stat().st_size
        old_receipt_size = path.with_suffix(".json").stat().st_size
        # A small fixture can grow because of the validation manifest. Keep it.
        useful = len(content)+1024 < old_size
        entry = {"name": path.name, "before_bytes": old_size, "candidate_bytes": len(content), "changed": False}
        if useful:
            if apply:
                replace_checkpoint(path, content, part, reserve=reserve)
                assert_payload(part, verified(path))
                entry["changed"] = True
            new_size = len(content)
        else:
            new_size = old_size
        entry["after_bytes"] = new_size
        result["files"].append(entry)
        result["before_bytes"] += old_size+old_receipt_size
        result["after_bytes"] += new_size+(path.with_suffix(".json").stat().st_size if apply else old_receipt_size+ (240 if useful else 0))
    if preserved != {name: sha(folder/name) for name in preserved}:
        raise ValueError("A canonical posterior, proposal, contract, summary or other preserved file changed")
    result["status"] = "compacted" if apply and any(p["changed"] for p in result["files"]) else "verified"
    result["saved_bytes"] = result["before_bytes"]-result["after_bytes"]
    return result


def extension_parents(roots):
    """Do not invalidate an existing extension's recorded parent-file hashes."""
    protected = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.glob("**/mh_extensions/**/contract.json"):
            record = json.loads(path.read_text())
            if record.get("parent_contract_sha256"):
                protected.add(record["parent_contract_sha256"])
    return protected


def _check_compaction_lock(path):
    for folder in Path(path).parents:
        lock = folder/".cache_compaction.lock"
        if not lock.exists():
            continue
        record = json.loads(lock.read_text())
        if record.get("host") == socket.gethostname():
            try:
                os.kill(int(record["pid"]), 0)
            except ProcessLookupError:
                continue
        raise RuntimeError(f"Cache compaction is active; restart the experiment after it finishes: {lock}")


def install(paper_sampling, raw_writer, raw_json_writer):
    """Runtime-only adapter: the original inference source files stay byte-identical."""
    global _INSTALLED, _COMPACT_WRITER, _COMPACT_JSON_WRITER
    if _INSTALLED:
        paper_sampling.atomic_npz = _COMPACT_WRITER
        paper_sampling.atomic_json = _COMPACT_JSON_WRITER
        return
    _INSTALLED = True
    restore = paper_sampling._restore
    paper_sampling._load_npz = load

    def restored(path):
        _check_compaction_lock(path)
        recover_transaction(path)
        return restore(path)

    def write(path, **values):
        path = Path(path)
        _check_compaction_lock(path)
        enabled = os.environ.get("HBAYES_COMPACT_CACHE", "1") != "0"
        if enabled and CHAIN.fullmatch(path.name) and (path.parent/"contract.json").is_file():
            packed = encode(values, path)
            # Encoding metadata can outweigh deduplication on very short runs.
            if sum(np.asarray(v).nbytes for v in packed.values()) < sum(np.asarray(v).nbytes for v in values.values()):
                values = packed
        return raw_writer(path, **values)

    def write_json(path, value):
        path = Path(path)
        _check_compaction_lock(path)
        raw_json_writer(path, value)
        if (path.name == "posterior.json" and isinstance(value, dict) and value.get("dependencies")
                and os.environ.get("HBAYES_COMPACT_CACHE", "1") != "0"):
            # Extension folders have a different contract and are excluded.
            protected = extension_parents([path.parent.parents[2], Path("outputs")])
            compact_fit(path.parent, apply=True, reserve=64*1024**2, protected=protected)

    paper_sampling._restore = restored
    _COMPACT_WRITER = write
    _COMPACT_JSON_WRITER = write_json
    paper_sampling.atomic_npz = write
    paper_sampling.atomic_json = write_json
