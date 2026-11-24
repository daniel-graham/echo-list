"""SafeWriter — the single chokepoint for all filesystem mutations."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


class UnsafeWriteError(Exception):
    ...


class SafeWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, rel_or_abs) -> Path:
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        if not p.is_relative_to(self.root):
            raise UnsafeWriteError(f"outside workspace: {p}")
        return p

    def makedirs(self, rel) -> Path:
        p = self._resolve(rel)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_text(self, rel, text) -> Path:
        p = self._resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, text.encode("utf-8"))
        return p

    def write_bytes(self, rel, data) -> Path:
        p = self._resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, data)
        return p

    def copy_in(self, src, rel, progress_cb=None, compute_hash=False):
        """Copy src into the workspace at rel, atomically and durably.

        The bytes go to a hidden ``.part`` temp file in the destination
        directory, are fsynced, and only then renamed over ``rel``. A failure
        part-way (ENOSPC, card pulled, progress_cb raising) removes the temp
        file and leaves nothing at ``rel``. The ``.part`` suffix is not an
        audio extension, so a temp file stranded by a hard crash is never
        adopted as a track by ``rescan_playlist``.

        Only timestamps are carried over from src — never permission bits or
        BSD flags, which would make the copy read-only (un-taggable) or fail
        outright on FAT32 (see review finding #10).
        """
        dest = self._resolve(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(src)
        st = src.stat()
        total = st.st_size
        h = hashlib.blake2b() if compute_hash else None
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".echolist-copy-",
                                   suffix=".part")
        try:
            with open(src, "rb") as fsrc, os.fdopen(fd, "wb") as fdst:
                fd = -1  # now owned by fdst
                copied = 0
                while True:
                    buf = fsrc.read(1024 * 256)
                    if not buf:
                        break
                    fdst.write(buf)
                    if h:
                        h.update(buf)
                    copied += len(buf)
                    if progress_cb:
                        progress_cb(copied, total)
                fdst.flush()
                os.fsync(fdst.fileno())
            os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
            os.replace(tmp, dest)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            _unlink_quiet(tmp)
            raise
        return dest, h.hexdigest() if h else None

    def delete(self, rel) -> None:
        p = self._resolve(rel)
        if p == self.root:
            raise UnsafeWriteError("refusing to delete workspace root")
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def rename(self, old_rel, new_rel) -> Path:
        old = self._resolve(old_rel)
        new = self._resolve(new_rel)
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return new

    def resolve(self, rel) -> Path:
        return self._resolve(rel)


def _unlink_quiet(path) -> None:
    """Remove a temp file during error cleanup; it may already be gone."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _write_all(fd: int, data: bytes) -> None:
    """os.write may write fewer bytes than asked; loop until all are written."""
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _atomic_write(dest: Path, data: bytes) -> None:
    """Write data to dest atomically via temp-file + fsync + os.replace."""
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp, dest)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        _unlink_quiet(tmp)
        raise


def atomic_write_text(dest: Path, text: str) -> None:
    """Write text to dest atomically. Safe for use outside the workspace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(dest, text.encode("utf-8"))
