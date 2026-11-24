"""M3U import fixes: long-name disambiguation (A25/N24) and URL case (A26)."""

from __future__ import annotations

from echolist.m3u import curate_playlist_name, parse_m3u
from echolist.naming import playlist_id


def test_long_name_gets_a_distinct_id():
    name = "A" * 60
    existing = {playlist_id(name)}
    curated = curate_playlist_name(name, existing)
    assert playlist_id(curated) not in existing
    existing.add(playlist_id(curated))
    third = curate_playlist_name(name, existing)
    assert playlist_id(third) not in existing


def test_long_name_disambiguation_survives_100_collisions():
    name = "B" * 60
    existing = {playlist_id(name)}
    for _ in range(105):
        c = curate_playlist_name(name, existing)
        assert playlist_id(c) not in existing
        existing.add(playlist_id(c))


def test_dot_playlist_name_is_importable():
    curated = curate_playlist_name(".echolist", set())
    assert not curated.startswith(".")


def test_file_uri_host_is_case_insensitive(tmp_path):
    track = tmp_path / "lib" / "a b.flac"
    track.parent.mkdir()
    track.write_bytes(b"x")
    m = tmp_path / "x.m3u"
    uri = "file://LOCALHOST" + track.resolve().as_uri()[len("file://"):]
    m.write_text(uri + "\n", encoding="utf-8")
    assert parse_m3u(m)["tracks"] == [track.resolve()]


def test_uppercase_stream_urls_are_skipped(tmp_path):
    m = tmp_path / "x.m3u"
    m.write_text("HTTP://example.com/stream.mp3\nHttps://example.com/b.mp3\n",
                 encoding="utf-8")
    assert parse_m3u(m) == {"name": "x", "tracks": [], "missing": []}
