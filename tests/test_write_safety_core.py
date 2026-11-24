"""Regression tests for write-safety defects in SafeWriter, naming, m3u and
rescan_playlist (review findings #4, #5, #8, #9, #10, #11, #17)."""

import errno
import os
import stat
import unicodedata
from pathlib import Path

import pytest

import echolist.safe_write as safe_write
from echolist.m3u import parse_m3u
from echolist.naming import playlist_id, sanitize
from echolist.safe_write import SafeWriter
from conftest import TINY_FLAC, _make_flac


# ── #4: copy_in must be atomic and durable ──


def _fail_after_first_chunk(copied, total):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_copy_in_failure_leaves_no_partial_file_at_final_name(tmp_path):
    src = tmp_path / "big.flac"
    src.write_bytes(os.urandom(1024 * 1024))
    w = SafeWriter(tmp_path / "ws")
    with pytest.raises(OSError):
        w.copy_in(src, "Mix/01 - Big.flac", progress_cb=_fail_after_first_chunk,
                  compute_hash=True)
    mix = tmp_path / "ws" / "Mix"
    assert not (mix / "01 - Big.flac").exists()
    assert list(mix.iterdir()) == [], "temp file was not cleaned up"


def test_failed_copy_is_not_adopted_by_rescan(manager, source):
    pid = manager.create_playlist("Mix")
    with pytest.raises(OSError):
        manager.add_track(pid, source / "ArtistA" / "Album1" / "01 Song One.flac",
                          progress_cb=_fail_after_first_chunk)
    manager.rescan_playlist(pid)
    assert manager.store.playlists[pid]["tracks"] == []


def test_stranded_copy_temp_is_not_adopted_by_rescan(manager, source):
    pid = manager.create_playlist("Mix")
    folder = manager.writer.root / manager.store.playlists[pid]["folder"]
    # What a power cut mid-copy leaves behind: only the temp file.
    (folder / ".echolist-copy-abc123.part").write_bytes(TINY_FLAC[:20])
    manager.rescan_playlist(pid)
    assert manager.store.playlists[pid]["tracks"] == []


def test_copy_in_fsyncs_before_rename(tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(safe_write.os, "fsync",
                        lambda fd: (events.append("fsync"), real_fsync(fd)))
    monkeypatch.setattr(safe_write.os, "replace",
                        lambda a, b: (events.append("replace"), real_replace(a, b)))
    src = tmp_path / "a.flac"
    src.write_bytes(b"x" * 1000)
    (tmp_path / "ws").mkdir()
    w = SafeWriter(tmp_path / "ws")
    dest, _ = w.copy_in(src, "a.flac")
    assert dest.read_bytes() == b"x" * 1000
    assert events[:2] == ["fsync", "replace"]


def test_atomic_write_fsyncs_and_handles_short_writes(tmp_path, monkeypatch):
    real_write = os.write
    fsyncs = []
    real_fsync = os.fsync
    # A short write: the kernel accepts at most 7 bytes per call.
    monkeypatch.setattr(safe_write.os, "write", lambda fd, b: real_write(fd, bytes(b[:7])))
    monkeypatch.setattr(safe_write.os, "fsync",
                        lambda fd: (fsyncs.append(fd), real_fsync(fd)))
    w = SafeWriter(tmp_path / "ws")
    payload = '{"schema": 1, "playlists": {}}'
    p = w.write_text(".echolist/playlists.json", payload)
    assert p.read_text(encoding="utf-8") == payload
    assert fsyncs, "atomic write never fsynced"


# ── #10: copies must not inherit read-only mode / flags ──


def test_copy_in_of_read_only_source_is_writable(tmp_path):
    src = tmp_path / "ro.flac"
    src.write_bytes(TINY_FLAC)
    os.chmod(src, 0o444)
    try:
        w = SafeWriter(tmp_path / "ws")
        for compute_hash in (False, True):
            dest, _ = w.copy_in(src, f"Mix/{compute_hash}.flac", compute_hash=compute_hash)
            assert os.stat(dest).st_mode & stat.S_IWUSR, "copy inherited read-only mode"
    finally:
        os.chmod(src, 0o644)


def test_add_track_from_read_only_source_tags_copy(manager, source):
    src = source / "ArtistA" / "Album1" / "01 Song One.flac"
    os.chmod(src, 0o444)
    try:
        pid = manager.create_playlist("Mix")
        manager.add_track(pid, src)
    finally:
        os.chmod(src, 0o644)
    assert len(manager.store.playlists[pid]["tracks"]) == 1


def test_copy_in_keeps_source_mtime(tmp_path):
    src = tmp_path / "a.flac"
    src.write_bytes(b"x")
    os.utime(src, (1_600_000_000, 1_600_000_000))
    dest, _ = SafeWriter(tmp_path / "ws").copy_in(src, "a.flac")
    assert int(dest.stat().st_mtime) == 1_600_000_000


# ── #9: device track count must stay on the device ──


def test_expensive_stats_do_not_scan_sibling_volumes(tmp_path, source):
    from echolist.manager import PlaylistManager

    volumes = tmp_path / "Volumes"
    card = volumes / "ECHO MINI"
    card.mkdir(parents=True)
    _make_flac(card / "Music" / "on_device.flac", "A", "On device")
    _make_flac(volumes / "Backup" / "elsewhere.flac", "B", "Other volume")
    mgr = PlaylistManager.init(source, card)
    try:
        device_tracks, _ = mgr.compute_expensive_stats()
    finally:
        mgr.release_lock()
    assert device_tracks == 1


# ── #8: playlist ids must be safe path segments and match folders ──


@pytest.mark.parametrize("name", ["AC/DC", "a\\b", "..", "../../etc", "what?", "a:b", "x*y"])
def test_playlist_id_is_a_safe_path_segment(name):
    pid = playlist_id(name)
    assert pid
    assert not any(c in pid for c in '\\/:*?"<>|')
    assert ".." not in pid


def test_sanitize_strips_control_characters():
    assert sanitize("a\x00b\x1fc\x7f") == "a_b_c_"


def test_create_playlist_rejects_folder_collision(manager, source):
    first = manager.create_playlist("AC/DC")
    manager.add_track(first, source / "ArtistA" / "Album1" / "01 Song One.flac")
    with pytest.raises(ValueError):
        manager.create_playlist("AC_DC")
    with pytest.raises(ValueError):
        manager.create_playlist("ac_dc")


def test_create_playlist_rejects_folder_used_by_legacy_pid(manager):
    # A store written before playlist_id() sanitized names keeps pid "ac/dc".
    manager.store.add_playlist("ac/dc", "AC/DC", "AC_DC")
    with pytest.raises(ValueError):
        manager.create_playlist("ac_dc")


def test_slash_playlist_backups_are_listed(manager, source):
    from echolist.config import list_all_backup_pids

    pid = manager.create_playlist("AC/DC")
    manager.add_track(pid, source / "ArtistA" / "Album1" / "01 Song One.flac")
    manager.backup_playlist_metadata(pid)
    assert pid in list_all_backup_pids(manager.writer.root)


# ── #11: NFC/NFD ──


def test_sanitize_truncation_keeps_accent_on_nfd_input():
    name = "A" * 59 + unicodedata.normalize("NFD", "é")
    assert sanitize(name) == "A" * 59 + "é"


def test_rescan_matches_nfd_disk_name_to_nfc_store_entry(manager, source):
    pid = manager.create_playlist("Mix")
    manager.add_track(pid, source / "ArtistA" / "Album1" / "01 Song One.flac")
    pl = manager.store.playlists[pid]
    folder = manager.writer.root / pl["folder"]
    nfc_name = unicodedata.normalize("NFC", "01 - Café.flac")
    nfd_name = unicodedata.normalize("NFD", nfc_name)
    (folder / pl["tracks"][0]["copy_name"]).rename(folder / nfd_name)
    pl["tracks"][0]["copy_name"] = nfc_name
    assert [f.name for f in folder.iterdir()] == [nfd_name]

    manager.rescan_playlist(pid)

    track = manager.store.playlists[pid]["tracks"][0]
    assert track["src_path"] == "ArtistA/Album1/01 Song One.flac"
    assert track["src_hash"].startswith("blake2b:")
    assert "original_tags" in track


def test_rescan_title_match_survives_nfd(manager, source):
    pid = manager.create_playlist("Mix")
    manager.add_track(pid, source / "ArtistA" / "Album1" / "01 Song One.flac")
    pl = manager.store.playlists[pid]
    folder = manager.writer.root / pl["folder"]
    (folder / pl["tracks"][0]["copy_name"]).rename(
        folder / unicodedata.normalize("NFD", "07 - Café.flac"))
    pl["tracks"][0]["copy_name"] = unicodedata.normalize("NFC", "01 - Café.flac")

    manager.rescan_playlist(pid)

    assert manager.store.playlists[pid]["tracks"][0]["src_path"] != ""


# ── #5: rescan order must follow the numeric prefix ──


def test_rescan_sorts_mixed_width_prefixes_numerically(manager):
    pid = manager.create_playlist("Big")
    folder = manager.writer.root / manager.store.playlists[pid]["folder"]
    names = ["01 - a.flac", "09 - b.flac", "10 - c.flac", "11 - d.flac",
             "99 - e.flac", "100 - f.flac", "101 - g.flac"]
    for n in names:
        (folder / n).write_bytes(TINY_FLAC)
    manager.rescan_playlist(pid)
    got = [t["copy_name"] for t in manager.store.playlists[pid]["tracks"]]
    assert got == names


# ── #17: m3u8 BOM and file:// URIs ──


def _lib(tmp_path):
    lib = tmp_path / "library"
    _make_flac(lib / "My Artist" / "01 Song.flac", "A", "Song")
    return lib


def test_m3u8_with_bom_parses_header(tmp_path):
    lib = _lib(tmp_path)
    m = tmp_path / "list.m3u8"
    m.write_bytes(b"\xef\xbb\xbf#EXTM3U\n" + str(lib / "My Artist" / "01 Song.flac").encode() + b"\n")
    result = parse_m3u(m)
    assert result["missing"] == []
    assert len(result["tracks"]) == 1


def test_m3u8_with_bom_and_no_header_resolves_first_track(tmp_path):
    lib = _lib(tmp_path)
    m = lib / "list.m3u8"
    m.write_bytes(b"\xef\xbb\xbfMy Artist/01 Song.flac\n")
    assert len(parse_m3u(m)["tracks"]) == 1


def test_m3u_file_uri_is_resolved(tmp_path):
    lib = _lib(tmp_path)
    track = lib / "My Artist" / "01 Song.flac"
    m = tmp_path / "vlc.m3u8"
    m.write_text("#EXTM3U\n" + track.resolve().as_uri() + "\n", encoding="utf-8")
    assert "%20" in m.read_text(encoding="utf-8")
    result = parse_m3u(m)
    assert result["missing"] == []
    assert result["tracks"] == [track.resolve()]


def test_m3u_file_localhost_uri_is_resolved(tmp_path):
    lib = _lib(tmp_path)
    track = (lib / "My Artist" / "01 Song.flac").resolve()
    m = tmp_path / "x.m3u"
    uri = "file://localhost" + track.as_uri()[len("file://"):]
    m.write_text(uri + "\n", encoding="utf-8")
    assert parse_m3u(m)["tracks"] == [track]
