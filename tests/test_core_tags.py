"""Tag fixes: adoption restore points keep every field the tagger changes
(A12), raw ID3 sort frames are stripped on every Mutagen version (A20), one
format registry (A18), and unreadable files are not taggable (N18)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TDRC, TIT2, TPE1, TRCK, TSO2, TXXX
from mutagen.wave import WAVE

from conftest import TINY_FLAC
from test_tags_numbering import _make_aiff, _make_wav
from echolist import tags as tagmod
from echolist.manager import AUDIO_EXTS
from echolist.tags import tags_supported

# A valid MPEG-1 Layer III frame header + padding (silent, 128 kbps, 44.1 kHz).
_MP3_FRAME = bytes.fromhex("fffb9064") + b"\x00" * 413


def _make_mp3(path: Path, frames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_MP3_FRAME * 8)
    t = ID3()
    for f in frames:
        t.add(f)
    t.save(path)


def _id3(path: Path):
    if path.suffix == ".mp3":
        return ID3(path)
    return WAVE(path).tags


def _frames(tags) -> dict:
    return {k: [str(x) for x in v.text] for k, v in tags.items() if hasattr(v, "text")}


ORIGINAL_ID3 = [
    TIT2(encoding=3, text=["Song"]),
    TPE1(encoding=3, text=["Artist"]),
    TRCK(encoding=3, text=["4/20"]),
    TDRC(encoding=3, text=["2020-02-03"]),
    TSO2(encoding=3, text=["sort"]),
]


@pytest.mark.parametrize("ext", [".wav", ".mp3"])
def test_adoption_restore_point_restores_id3_exactly(manager, ext):
    folder = manager.writer.root / "Old"
    p = folder / f"Song{ext}"
    if ext == ".wav":
        _make_wav(p, with_id3=False)
        f = WAVE(p)
        f.add_tags()
        for fr in ORIGINAL_ID3:
            f.tags.add(fr)
        f.save()
    else:
        _make_mp3(p, ORIGINAL_ID3)
    before = _frames(_id3(p))

    pid = manager.adopt_playlist("Old")
    copy = folder / manager.store.playlists[pid]["tracks"][0]["copy_name"]
    assert _frames(_id3(copy)) != before  # adoption really did change tags

    assert manager.restore_playlist_metadata(pid, "before_echolist") == 1
    assert _frames(_id3(copy)) == before


def test_adoption_restore_point_restores_flac_exactly(manager):
    folder = manager.writer.root / "Old"
    folder.mkdir()
    p = folder / "Song.flac"
    p.write_bytes(TINY_FLAC)
    f = FLAC(p)
    f["TITLE"] = "Song"
    f["ARTIST"] = ["A", "B"]
    f["TRACKNUMBER"] = "4/20"
    f["DATE"] = "2020-02-03"
    f["ALBUMARTISTSORT"] = "sort"
    f["MUSICBRAINZ_ALBUMID"] = "abc"
    f.save()
    before = dict(FLAC(p).tags.as_dict())

    pid = manager.adopt_playlist("Old")
    copy = folder / manager.store.playlists[pid]["tracks"][0]["copy_name"]
    assert manager.restore_playlist_metadata(pid, "before_echolist") == 1
    assert dict(FLAC(copy).tags.as_dict()) == before


@pytest.mark.parametrize("frame", [
    TSO2(encoding=3, text=["Earlier Artist"]),
    TXXX(encoding=3, desc="ALBUMARTISTSORT", text=["Earlier Artist"]),
])
def test_mp3_album_artist_sort_removed_in_both_representations(manager, source, frame):
    src = source / "Mp3" / "song.mp3"
    _make_mp3(src, [TIT2(encoding=3, text=["Song"]), frame])
    pid = manager.create_playlist("Mix")
    rel = manager.add_track(pid, src)
    left = ID3(manager.writer.root / rel)
    assert not left.getall("TSO2")
    assert not left.getall("TXXX:ALBUMARTISTSORT")
    assert left["TPE2"].text == ["* PLAYLISTS *"]


# ── A18 ──

def test_format_registry_covers_every_taggable_extension():
    for ext in (".mp4", ".m4a", ".aif", ".aiff", ".wav", ".flac", ".mp3", ".aac"):
        assert ext in AUDIO_EXTS


def test_mp4_track_survives_rescan(manager, source):
    pid = manager.create_playlist("Mix")
    folder = manager.writer.root / "Mix"
    (folder / "01 - Clip.mp4").write_bytes(b"\x00" * 32)
    manager.store.add_track(pid, {"index": 1, "src_path": "x/Clip.mp4",
                                  "copy_name": "01 - Clip.mp4", "src_hash": ""})
    manager.rescan_playlist(pid)
    assert [t["copy_name"] for t in manager.store.playlists[pid]["tracks"]] == ["01 - Clip.mp4"]


def test_aif_alias_is_tagged(manager, source):
    src = source / "Aif" / "song.aif"
    _make_aiff(src)
    assert tags_supported(src)
    pid = manager.create_playlist("Mix")
    rel = manager.add_track(pid, src)
    copy = manager.writer.root / rel
    assert copy.suffix == ".aif"
    assert tagmod.read_playlist_tags(copy)["albumartist"] == "* PLAYLISTS *"


# ── N18 ──

@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions, not root")
def test_tags_supported_is_false_for_unreadable_aac(tmp_path):
    p = tmp_path / "broken.aac"
    p.write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 16)
    p.chmod(0)
    try:
        assert tags_supported(p) is False
    finally:
        p.chmod(0o644)
