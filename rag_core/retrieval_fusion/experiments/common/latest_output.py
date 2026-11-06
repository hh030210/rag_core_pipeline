"""Coordinate publication and evaluation of the main experiment's latest run."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def lock_output(output_dir: Path):
    """Serialize publication and metric generation for a stable output path."""
    lock_path = output_dir.parent / f".{output_dir.name}.lock"
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def stage_latest(output_dir: Path):
    """Publish a complete export and discard the preceding result directory."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".staging-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "ready"
        previous = Path(temporary) / "previous"
        staging.mkdir()
        yield staging

        with lock_output(output_dir):
            if output_dir.is_symlink():
                raise FileExistsError(f"Refusing to replace a symlink: {output_dir}")
            try:
                if output_dir.exists():
                    manifest_path = output_dir / "details_manifest.json"
                    if not manifest_path.is_file():
                        raise FileExistsError(f"No experiment manifest in {output_dir}")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("experiment") != "00_main_experiment":
                        raise FileExistsError(f"Output belongs to a different experiment: {output_dir}")
                    output_dir.rename(previous)
                staging.rename(output_dir)
            except BaseException:
                if previous.exists() and not output_dir.exists():
                    previous.rename(output_dir)
                raise
