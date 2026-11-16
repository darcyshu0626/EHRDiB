"""File storage, output locations, random seeds, and device helpers."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch


def log_stage(title):
    logging.getLogger("ehrdib").info("\n%s\n%s\n%s", "=" * 60, title, "=" * 60)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as handle:
        value = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: str | Path, writer: Callable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def private_directory(path: str | Path) -> None:
    # POSIX mode 0700 is not a portable Windows ACL; inherit the authorized parent there.
    Path(path).mkdir(parents=True, mode=0o777 if os.name == "nt" else 0o700, exist_ok=False)


def write_json(path: str | Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda handle: handle.write(encoded))


def save_tensors(path: str | Path, value: object) -> None:
    atomic_write(path, lambda handle: torch.save(value, handle))


def load_tensors(path: str | Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Tensor artifact must contain a dictionary")
    return value


def relative_path(path, root):
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    logs: Path
    reports: Path

    @classmethod
    def load(cls, run):
        root = Path(run).resolve()
        config = root / "config.json"
        saved = read_json(config).get("paths", {}) if config.is_file() else {}
        destinations = []
        for name in ("checkpoints", "logs", "reports"):
            key = {"checkpoints": "checkpoint_dir", "logs": "log_dir", "reports": "report_dir"}[
                name
            ]
            value = saved.get(key, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid {name} path in run configuration")
            destinations.append((root / value).resolve())
        result = cls(root, *destinations)
        result.validate()
        return result

    def validate(self):
        for destination in (self.checkpoints, self.logs, self.reports):
            if self.root == destination or self.root.is_relative_to(destination):
                raise ValueError("Artifact directories cannot contain the run directory")
        for a, b in (
            (self.checkpoints, self.logs),
            (self.checkpoints, self.reports),
            (self.logs, self.reports),
        ):
            if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                raise ValueError("Logs, reports, and run artifacts need separate directories")

    def to_config(self):
        self.validate()
        return {
            "checkpoint_dir": relative_path(self.checkpoints, self.root),
            "log_dir": relative_path(self.logs, self.root),
            "report_dir": relative_path(self.reports, self.root),
        }

    @property
    def has_weights(self):
        return all((self.stage(name) / "best.pt").is_file() for name in ("vae", "bridge"))

    @property
    def prepared_data(self):
        filename = read_json(self.root / "config.json").get("paths", {}).get("data_file")
        if filename:
            return (self.root / filename).resolve()
        directory = read_json(self.root / "config.json").get("paths", {}).get("data_dir")
        if not isinstance(directory, str) or not directory:
            raise ValueError("Run configuration must identify its prepared data directory")
        return (self.root / directory / "data.pt").resolve()

    @property
    def latents(self):
        return self.root / "data/latents.pt"

    @property
    def probabilities(self):
        return self.root / "data/probabilities.pt"

    @property
    def synthesis(self):
        return self.root / "data/synthesis.pt"

    def stage(self, name):
        return self.checkpoints / name


def derived_seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "little") % (
        2**63
    )


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; choose device=cpu explicitly")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
        if device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
    return device


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def read_table(path: str | Path, member: str | None = None) -> list[dict[str, str]]:
    path = Path(path)

    def parse(handle) -> list[dict[str, str]]:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("CSV requires a nonempty unique header")
        rows = []
        for index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV row {index + 2} does not match the header width")
            rows.append(row)
        if not rows:
            raise ValueError("CSV contains no records")
        return rows

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                i
                for i in archive.infolist()
                if not i.is_dir() and i.filename.lower().endswith(".csv")
            ]
            selected = [i for i in candidates if member is None or i.filename == member]
            if len(selected) != 1:
                raise ValueError("ZIP requires one CSV or an explicit unique member name")
            with (
                archive.open(selected[0]) as raw,
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle,
            ):
                return parse(handle)
    if member is not None:
        raise ValueError("member is only valid for ZIP inputs")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return parse(handle)


def write_table(path: str | Path, rows: Sequence[Mapping], names: Sequence[str]) -> None:

    def write(handle):
        wrapper = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        try:
            writer = csv.DictWriter(wrapper, fieldnames=list(names), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            wrapper.flush()
        finally:
            wrapper.detach()

    atomic_write(path, write)
