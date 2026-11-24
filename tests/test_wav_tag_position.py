"""WAV tags must sit before the audio: players that stop parsing at the
``data`` chunk never see a tag chunk appended after it."""
import struct
import wave

import pytest

from echolist.tags import apply_playlist_tags, read_playlist_tags, restore_tags


def _chunks(path):
    b = path.read_bytes()
    assert b[:4] == b"RIFF" and struct.unpack("<I", b[4:8])[0] == len(b) - 8
    out, off = [], 12
    while off < len(b):
        cid, size = struct.unpack("<4sI", b[off:off + 8])
        out.append((cid, b[off + 8:off + 8 + size]))
        off += 8 + size + (size & 1)
    assert off == len(b)
    return out


def _untagged_wav(path, frames=44100):
    pcm = bytes(range(256)) * (frames * 4 // 256)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(pcm)
    return pcm


def _assert_tag_first(path, pcm):
    ids = [c for c, _ in _chunks(path)]
    assert b"id3 " in ids, ids
    assert ids.index(b"id3 ") < ids.index(b"data"), ids
    assert dict(_chunks(path))[b"data"] == pcm, "audio bytes changed"
    with wave.open(str(path), "rb") as w:  # still a valid WAV
        assert w.readframes(w.getnframes()) == pcm


def test_tagging_an_untagged_wav_puts_the_tag_before_the_audio(tmp_path):
    p = tmp_path / "01 - Song.wav"
    pcm = _untagged_wav(p)
    apply_playlist_tags(p, "* PLAYLISTS *", "Mix", 1, "src.wav", "mix")
    _assert_tag_first(p, pcm)
    tags = read_playlist_tags(p)
    assert tags["album"] == "Mix" and tags["tracknumber"] == "1"


def test_retagging_keeps_the_tag_before_the_audio(tmp_path):
    p = tmp_path / "01 - Song.wav"
    pcm = _untagged_wav(p)
    apply_playlist_tags(p, "* PLAYLISTS *", "Mix", 1, "src.wav", "mix")
    apply_playlist_tags(p, "* PLAYLISTS *", "A much longer playlist name " * 4, 12, "src.wav", "mix")
    _assert_tag_first(p, pcm)


def test_a_tag_already_after_the_audio_is_moved_ahead(tmp_path):
    # Files tagged elsewhere with Mutagen carry the chunk at the end.
    from mutagen.id3 import TIT2
    from mutagen.wave import WAVE
    p = tmp_path / "01 - Song.wav"
    pcm = _untagged_wav(p)
    f = WAVE(p)
    f.add_tags()
    f.tags.add(TIT2(encoding=3, text=["Original Title"]))
    f.save()
    assert [c for c, _ in _chunks(p)][-1] == b"id3 "
    apply_playlist_tags(p, "* PLAYLISTS *", "Mix", 3, "src.wav", "mix")
    _assert_tag_first(p, pcm)
    assert read_playlist_tags(p)["title"] == "Original Title"


def test_restoring_tags_on_an_untagged_wav_puts_them_first(tmp_path):
    p = tmp_path / "01 - Song.wav"
    pcm = _untagged_wav(p)
    restore_tags(p, {"title": "T", "artist": "A", "album": "Al"})
    _assert_tag_first(p, pcm)


def _odd_wav(path, frames, padded):
    """Mono 8-bit WAV with an odd-length data chunk, with or without the pad
    byte RIFF requires (Python's wave module omits it)."""
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 8000, 8000, 1, 8)
    data = struct.pack("<4sI", b"data", frames) + b"\x80" * frames + (b"\0" if padded else b"")
    body = b"WAVE" + fmt + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


@pytest.mark.parametrize("frames", [1, 3])
def test_odd_sized_chunks_stay_word_aligned(tmp_path, frames):
    p = tmp_path / "odd.wav"
    _odd_wav(p, frames, padded=True)
    apply_playlist_tags(p, "* PLAYLISTS *", "Odd", 1, "src.wav", "mix")
    ids = [c for c, _ in _chunks(p)]
    assert ids.index(b"id3 ") < ids.index(b"data")
    assert dict(_chunks(p))[b"data"] == b"\x80" * frames


def test_malformed_chunk_layout_is_not_rearranged(tmp_path):
    # Without the pad byte the chunks don't tile the file; moving anything
    # could shift audio, so the layout Mutagen wrote is kept as is.
    p = tmp_path / "unpadded.wav"
    _odd_wav(p, 1, padded=False)
    before = p.read_bytes()
    apply_playlist_tags(p, "* PLAYLISTS *", "Odd", 1, "src.wav", "mix")
    after = p.read_bytes()
    assert after[:45][8:] == before[8:], "audio or header bytes moved"

def test_adding_an_untagged_wav_to_a_playlist_tags_it_before_the_audio(manager, source):
    src = source / "Untagged.wav"
    pcm = _untagged_wav(src)
    pid = manager.create_playlist("Mix")
    manager.add_track(pid, src)
    copy = manager.writer.root / "Mix" / manager.store.playlists[pid]["tracks"][0]["copy_name"]
    _assert_tag_first(copy, pcm)
    assert [c for c, _ in _chunks(src)] == [b"fmt ", b"data"]
