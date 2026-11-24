"""SafeWriter durability and error-cleanup fixes (A24, N16)."""

from __future__ import annotations

import errno
import os
import stat

import pytest

import echolist.safe_write as safe_write
from echolist.safe_write import SafeWriter, atomic_write_text


def _spy_fsync_and_replace(monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
        events.append(f"fsync-{kind}")
        return real_fsync(fd)

    def replace(a, b):
        events.append("replace")
        return real_replace(a, b)

    monkeypatch.setattr(safe_write.os, "fsync", fsync)
    monkeypatch.setattr(safe_write.os, "replace", replace)
    return events


@pytest.mark.skipif(os.name == "nt", reason="directories cannot be fsynced on Windows")
def test_copy_in_fsyncs_directory_after_replace(tmp_path, monkeypatch):
    events = _spy_fsync_and_replace(monkeypatch)
    src = tmp_path / "a.flac"
    src.write_bytes(b"x" * 100)
    w = SafeWriter(tmp_path / "ws")
    w.makedirs("Mix")
    events.clear()
    w.copy_in(src, "Mix/a.flac")
    assert events == ["fsync-file", "replace", "fsync-dir"]


@pytest.mark.skipif(os.name == "nt", reason="directories cannot be fsynced on Windows")
def test_atomic_write_fsyncs_directory_after_replace(tmp_path, monkeypatch):
    events = _spy_fsync_and_replace(monkeypatch)
    atomic_write_text(tmp_path / "state" / "x.json", "{}")
    assert events == ["fsync-file", "replace", "fsync-dir"]


@pytest.mark.skipif(os.name == "nt", reason="directories cannot be fsynced on Windows")
def test_rename_fsyncs_directory(tmp_path, monkeypatch):
    w = SafeWriter(tmp_path / "ws")
    w.write_text("Mix/a.txt", "a")
    events = _spy_fsync_and_replace(monkeypatch)
    w.rename("Mix/a.txt", "Mix/b.txt")
    assert "fsync-dir" in events


def test_directory_fsync_unsupported_is_tolerated(tmp_path, monkeypatch):
    """Some filesystems refuse fsync on a directory (EINVAL): not an error."""
    real_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_fsync(fd)

    monkeypatch.setattr(safe_write.os, "fsync", fsync)
    atomic_write_text(tmp_path / "x.json", "{}")
    assert (tmp_path / "x.json").read_text() == "{}"


def test_directory_fsync_io_error_propagates(tmp_path, monkeypatch):
    real_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "I/O error")
        return real_fsync(fd)

    monkeypatch.setattr(safe_write.os, "fsync", fsync)
    if os.name == "nt":
        pytest.skip("no directory fsync on Windows")
    with pytest.raises(OSError) as e:
        atomic_write_text(tmp_path / "x.json", "{}")
    assert e.value.errno == errno.EIO


# ── N16: cleanup failure never hides the original error ──

def _failing_unlink(monkeypatch):
    def unlink(path, *a, **kw):
        raise OSError(errno.EIO, "card pulled", str(path))
    monkeypatch.setattr(safe_write.os, "unlink", unlink)


def test_copy_in_keeps_original_error_when_cleanup_fails(tmp_path, monkeypatch):
    src = tmp_path / "a.flac"
    src.write_bytes(b"x" * 100)
    w = SafeWriter(tmp_path / "ws")
    w.makedirs("Mix")
    _failing_unlink(monkeypatch)

    class TagError(Exception):
        pass

    def bad_tags(tmp):
        raise TagError("tag write failed")

    with pytest.raises(TagError):
        w.copy_in(src, "Mix/a.flac", before_commit=bad_tags)


def test_atomic_write_keeps_original_error_when_cleanup_fails(tmp_path, monkeypatch):
    def enospc(fd, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(safe_write.os, "write", enospc)
    _failing_unlink(monkeypatch)
    with pytest.raises(OSError) as e:
        atomic_write_text(tmp_path / "x.json", "{}")
    assert e.value.errno == errno.ENOSPC
