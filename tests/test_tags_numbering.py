"""WAV/AIFF tagging, track-number rewrites on renumber, consistent padding,
and untaggable formats."""

import struct
import wave
from pathlib import Path

import pytest
from mutagen.aiff import AIFF
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1, TRCK, TSO2, TXXX
from mutagen.wave import WAVE

from conftest import _make_flac
from echolist.manager import PlaylistManager
from echolist.tags import _EASY_STRIP_TAGS, read_playlist_tags, restore_tags

TITLE = "What A Miracle Feels Like"


# ── Fixture builders ──

def _pcm_bytes(n_frames: int = 64) -> bytes:
    return b"".join(struct.pack("<h", (i * 517) % 32768 - 16384) for i in range(n_frames))


def _make_wav(path: Path, with_id3: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(_pcm_bytes())
    if with_id3:
        f = WAVE(path)
        f.add_tags()
        _add_source_frames(f.tags)
        f.save()


def _make_aiff(path: Path, with_id3: bool = True) -> None:
    """Minimal AIFF (big-endian PCM) built by hand; stdlib aifc is gone in 3.13."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"".join(struct.pack(">h", (i * 517) % 32768 - 16384) for i in range(64))
    # 44100 Hz as an 80-bit IEEE extended float
    rate = bytes.fromhex("400EAC44000000000000")
    comm = struct.pack(">hIh", 1, 64, 16) + rate
    ssnd = struct.pack(">II", 0, 0) + pcm
    body = b"AIFF"
    body += b"COMM" + struct.pack(">I", len(comm)) + comm
    body += b"SSND" + struct.pack(">I", len(ssnd)) + ssnd
    path.write_bytes(b"FORM" + struct.pack(">I", len(body)) + body)
    if with_id3:
        f = AIFF(path)
        f.add_tags()
        _add_source_frames(f.tags)
        f.save()


def _add_source_frames(tags: ID3) -> None:
    """Frames a real ripped source WAV carries (seen on the user's library)."""
    tags.add(TIT2(encoding=3, text=[TITLE]))
    tags.add(TPE1(encoding=3, text=["Some Artist"]))
    tags.add(TRCK(encoding=3, text=["1/20"]))
    tags.add(TALB(encoding=3, text=["Original Album"]))
    tags.add(TSO2(encoding=3, text=["Sort Album Artist"]))
    tags.add(TXXX(encoding=3, desc="MusicBrainz Album Id", text=["abc"]))
    tags.add(APIC(encoding=3, mime="image/png", type=3, desc="", data=b"\x89PNG fake"))


def _aiff_ssnd(path: Path) -> bytes:
    data = path.read_bytes()
    pos = 12
    while pos < len(data):
        cid, size = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        if cid == b"SSND":
            return data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
    raise AssertionError("no SSND chunk")


def _wav_frames(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.getparams(), w.readframes(w.getnframes())


def _adts_aac(path: Path, n_frames: int = 8) -> None:
    """Raw ADTS AAC-LC stream (44.1 kHz mono), as produced by many encoders."""
    payload = bytes.fromhex("2110050020a01933")
    frame_len = 7 + len(payload)
    header = bytes([
        0xFF, 0xF1,
        (1 << 6) | (4 << 2),            # profile LC, 44100 Hz, chan bit 2 = 0
        (1 << 6) | ((frame_len >> 11) & 3),  # mono
        (frame_len >> 3) & 0xFF,
        ((frame_len & 7) << 5) | 0x1F,
        0xFC,
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((header + payload) * n_frames)


@pytest.fixture
def wav_src(source):
    p = source / "Some Artist" / "Original Album" / f"01 - {TITLE}.wav"
    _make_wav(p)
    return p


@pytest.fixture
def aiff_src(source):
    p = source / "Some Artist" / "Original Album" / f"01 - {TITLE}.aiff"
    _make_aiff(p)
    return p


# ── Item 1: WAV / AIFF ──

@pytest.mark.parametrize("fixture", ["wav_src", "aiff_src"])
def test_add_id3_chunk_file_names_copy_from_tit2(manager, request, fixture):
    src = request.getfixturevalue(fixture)
    pid = manager.create_playlist("Ceremony")
    rel = manager.add_track(pid, src)
    assert Path(rel).name == f"01 - {TITLE}{src.suffix}"


@pytest.mark.parametrize("fixture,cls", [("wav_src", WAVE), ("aiff_src", AIFF)])
def test_add_id3_chunk_file_gets_playlist_tags(manager, request, fixture, cls):
    src = request.getfixturevalue(fixture)
    pid = manager.create_playlist("Ceremony")
    rel = manager.add_track(pid, src)
    tags = cls(manager.writer.root / rel).tags
    assert tags["TPE2"].text == ["* PLAYLISTS *"]
    assert tags["TALB"].text == ["Ceremony"]
    assert tags["TRCK"].text == ["1"]
    assert tags["TPOS"].text == ["1"]
    # Title, artist and cover art survive; sort / MusicBrainz frames go.
    assert tags["TIT2"].text == [TITLE]
    assert tags["TPE1"].text == ["Some Artist"]
    assert tags.getall("APIC")
    assert not tags.getall("TSO2")
    assert not tags.getall("TXXX:MusicBrainz Album Id")
    assert manager.audit_playlist_metadata(pid) == []


@pytest.mark.parametrize("make,ext", [(_make_wav, ".wav"), (_make_aiff, ".aiff")])
def test_read_playlist_tags_id3_chunk(tmp_path, make, ext):
    p = tmp_path / f"x{ext}"
    make(p)
    tags = read_playlist_tags(p)
    assert tags["title"] == TITLE
    assert tags["artist"] == "Some Artist"
    assert tags["album"] == "Original Album"
    assert tags["tracknumber"] == "1"


@pytest.mark.parametrize("make,cls,ext", [(_make_wav, WAVE, ".wav"), (_make_aiff, AIFF, ".aiff")])
def test_untagged_id3_chunk_file_gets_new_chunk(manager, source, make, cls, ext):
    src = source / "Bare" / f"Bare Song{ext}"
    make(src, with_id3=False)
    pid = manager.create_playlist("Ceremony")
    rel = manager.add_track(pid, src)
    assert Path(rel).name == f"01 - Bare Song{ext}"
    tags = cls(manager.writer.root / rel).tags
    assert tags is not None
    assert tags["TPE2"].text == ["* PLAYLISTS *"]
    assert tags["TRCK"].text == ["1"]


def test_wav_tagging_leaves_pcm_untouched(manager, wav_src):
    params_before, frames_before = _wav_frames(wav_src)
    pid = manager.create_playlist("Ceremony")
    rel = manager.add_track(pid, wav_src)
    params_after, frames_after = _wav_frames(manager.writer.root / rel)
    assert params_after == params_before
    assert frames_after == frames_before
    assert WAVE(manager.writer.root / rel).tags["TALB"].text == ["Ceremony"]


def test_aiff_tagging_leaves_pcm_untouched(manager, aiff_src):
    before = _aiff_ssnd(aiff_src)
    pid = manager.create_playlist("Ceremony")
    rel = manager.add_track(pid, aiff_src)
    assert _aiff_ssnd(manager.writer.root / rel) == before
    assert AIFF(manager.writer.root / rel).tags["TALB"].text == ["Ceremony"]


def test_restore_tags_wav(tmp_path):
    p = tmp_path / "x.wav"
    _make_wav(p)
    restore_tags(p, {"albumartist": "AA", "album": "Back", "tracknumber": "7/9",
                     "title": "New Title", "artist": ""})
    tags = WAVE(p).tags
    assert tags["TPE2"].text == ["AA"]
    assert tags["TALB"].text == ["Back"]
    assert tags["TRCK"].text == ["7"]
    assert tags["TIT2"].text == ["New Title"]
    assert not tags.getall("TPE1")


def test_wav_strip_frames_match_mp3_strip_list(manager, source, tmp_path):
    """Every easy key the MP3 path strips is stripped from WAV ID3 too."""
    scratch = tmp_path / "scratch.mp3"
    easy = EasyID3()
    for key in sorted(_EASY_STRIP_TAGS):
        easy[key] = "1" if key == "compilation" else "x"
    easy.save(scratch)
    strip_frames = ID3(scratch)

    src = source / "Strip" / "Strip.wav"
    _make_wav(src)
    f = WAVE(src)
    for frame in strip_frames.values():
        f.tags.add(frame)
    f.save()

    pid = manager.create_playlist("Ceremony")
    rel = manager.add_track(pid, src)
    left = set(WAVE(manager.writer.root / rel).tags.keys()) & set(strip_frames.keys())
    assert left == set()


# ── Item 2: renumbering rewrites TRACKNUMBER and keeps names ──

def test_remove_track_rewrites_tracknumber(manager, source):
    pid = manager.create_playlist("Mix")
    for src in sorted(source.rglob("*.flac")):
        manager.add_track(pid, src)
    manager.remove_track(pid, 1)

    folder = manager.writer.root / "Mix"
    for t in manager.store.playlists[pid]["tracks"]:
        f = FLAC(folder / t["copy_name"])
        assert f["TRACKNUMBER"] == [str(t["index"])], t["copy_name"]
        assert f["ECHOLIST_INDEX"] == [str(t["index"])], t["copy_name"]
    assert manager.audit_playlist_metadata(pid) == []


def test_remove_then_add_has_no_duplicate_tracknumbers(manager, source):
    pid = manager.create_playlist("Mix")
    srcs = sorted(source.rglob("*.flac"))
    for src in srcs:
        manager.add_track(pid, src)
    manager.remove_track(pid, 1)
    manager.add_track(pid, srcs[0])

    folder = manager.writer.root / "Mix"
    numbers = [FLAC(folder / t["copy_name"])["TRACKNUMBER"][0]
               for t in manager.store.playlists[pid]["tracks"]]
    assert numbers == ["1", "2", "3"]


def test_adopt_keeps_names_without_index_prefix(manager):
    folder = manager.writer.root / "Old Mix"
    _make_flac(folder / "01. Song.flac", "X", "Song")
    p = folder / "Band - Title.flac"
    _make_flac(p, "Band", "Title")
    f = FLAC(p)
    del f["TITLE"]
    f.save()

    pid = manager.adopt_playlist("Old Mix")
    names = [t["copy_name"] for t in manager.store.playlists[pid]["tracks"]]
    assert names == ["01 - Song.flac", "02 - Band - Title.flac"]


def test_renumber_never_overwrites_another_track(manager):
    """'001 - Y' renamed to '01 - Y' must not clobber the existing '01 - Y'."""
    folder = manager.writer.root / "Legacy"
    _make_flac(folder / "001 - Y.flac", "First", "Y")
    _make_flac(folder / "01 - Y.flac", "Second", "Y")

    pid = manager.adopt_playlist("Legacy")
    tracks = manager.store.playlists[pid]["tracks"]
    artists = [FLAC(folder / t["copy_name"])["ARTIST"][0] for t in tracks]
    assert sorted(artists) == ["First", "Second"]
    assert len(list(folder.glob("*.flac"))) == 2


# ── Item 3: consistent padding across the 99 → 100 boundary ──

@pytest.fixture
def big_source(tmp_path):
    lib = tmp_path / "biglib"
    for i in range(1, 102):
        _make_flac(lib / f"{i:03d}.flac", f"Artist{i}", f"Track {i}")
    return lib


def _assert_consistent(mgr, pid, width):
    tracks = mgr.store.playlists[pid]["tracks"]
    names = [t["copy_name"] for t in tracks]
    assert all(n.split(" - ", 1)[0].isdigit() and len(n.split(" - ", 1)[0]) == width
               for n in names), names
    on_disk = sorted(p.name for p in (mgr.writer.root / mgr.store.playlists[pid]["folder"]).iterdir()
                     if p.suffix == ".flac")
    assert on_disk == names  # lexicographic order on disk == playlist order


def test_crossing_100_tracks_repads_whole_playlist(big_source, tmp_path):
    dest = tmp_path / "card"
    dest.mkdir()
    mgr = PlaylistManager.init(big_source, dest)
    pid = mgr.create_playlist("Big")
    srcs = sorted(big_source.glob("*.flac"))
    for src in srcs[:100]:
        mgr.add_track(pid, src)
    _assert_consistent(mgr, pid, 3)
    assert mgr.rescan_playlist(pid) is False
    assert [t["copy_name"] for t in mgr.store.playlists[pid]["tracks"]][:2] == [
        "001 - Track 1.flac", "002 - Track 2.flac"]

    mgr.remove_track(pid, 50)
    _assert_consistent(mgr, pid, 2)
    assert mgr.rescan_playlist(pid) is False
    mgr.release_lock()


# ── Item 4: formats EchoList can't tag ──

def test_raw_adts_aac_add_does_not_crash(manager, source):
    src = source / "Raw" / "Raw Song.aac"
    _adts_aac(src)
    pid = manager.create_playlist("Mix")
    rel = manager.add_track(pid, src)
    assert (manager.writer.root / rel).read_bytes() == src.read_bytes()


@pytest.mark.parametrize("ext", [".ogg", ".wma", ".aac", ".dsf"])
def test_untaggable_formats_not_reported_as_fixable(manager, source, ext):
    src = source / "Odd" / f"Odd Song{ext}"
    if ext == ".aac":
        _adts_aac(src)
    else:
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"\x00" * 256)
    pid = manager.create_playlist("Mix")
    manager.add_track(pid, src)
    manager.add_track(pid, source / "ArtistA" / "Album1" / "01 Song One.flac")

    assert manager.audit_playlist_metadata(pid) == []
    assert manager.fix_playlist_metadata(pid) == 0
