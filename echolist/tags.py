"""Write playlist tags to copies (never originals)."""

from __future__ import annotations

from pathlib import Path

import mutagen
from mutagen.aiff import AIFF
from mutagen.flac import FLAC
from mutagen.easyid3 import EasyID3
from mutagen.easymp4 import EasyMP4
from mutagen.mp4 import MP4
from mutagen.id3 import TALB, TDRC, TIT2, TPE1, TPE2, TPOS, TRCK
from mutagen.wave import WAVE

# Containers that carry their tags in an embedded ID3v2 chunk.
_ID3_CHUNK_TYPES = {".wav": WAVE, ".aiff": AIFF}
_MP4_EXTS = (".m4a", ".mp4")

# Errors mutagen raises for unreadable or malformed files (it wraps I/O
# errors in MutagenError, but OSError can escape from size/seek checks).
_TAG_IO_ERRORS = (mutagen.MutagenError, OSError)


def _is_mp4_container(path: Path) -> bool:
    """True if the file is an ISO-BMFF/MP4 container ('ftyp' box first).

    A ``.aac`` file may be either an MP4 container or a raw ADTS stream;
    raw ADTS has nowhere to store tags.
    """
    return isinstance(mutagen.File(path), MP4)


def _tag_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".flac":
        return "flac"
    if suffix == ".mp3":
        return "mp3"
    if suffix in _MP4_EXTS:
        return "mp4"
    if suffix == ".aac":
        return "mp4" if _is_mp4_container(path) else None
    if suffix in _ID3_CHUNK_TYPES:
        return "id3chunk"
    return None


def tags_supported(path: Path) -> bool:
    """True if EchoList can write playlist tags into this file.

    Files that return False (OGG, WMA, DSF, raw ADTS AAC, ...) are still
    copied, but are left untagged and must not be reported as fixable.
    """
    return _tag_kind(Path(path)) is not None


def read_title(path: Path) -> str | None:
    """The file's title tag, or None if it has none / can't be read."""
    path = Path(path)
    try:
        if _tag_kind(path) == "id3chunk":
            tags = _ID3_CHUNK_TYPES[path.suffix.lower()](path).tags
            title = _id3_text(tags, "TIT2") if tags is not None else ""
            return title or None
        m = mutagen.File(path, easy=True)
    except _TAG_IO_ERRORS:
        return None
    if m is not None and m.tags is not None and "title" in m:
        return m["title"][0] or None
    return None


def _normalize_tracknumber(raw: str) -> str:
    """Strip 'N/M' or 'N\\M' total-tracks suffix, returning just the track number."""
    for sep in ("/", "\\"):
        if sep in raw:
            return raw.split(sep, 1)[0].strip()
    return raw.strip()


def apply_playlist_tags(
    path: Path, node_name: str, album: str, index: int, src_rel: str, playlist_id: str
) -> None:
    kind = _tag_kind(path)
    if kind == "flac":
        _tag_flac(path, node_name, album, index, src_rel, playlist_id)
    elif kind == "mp3":
        _tag_easy(EasyID3, path, node_name, album, index)
    elif kind == "mp4":
        _tag_easy(EasyMP4, path, node_name, album, index)
    elif kind == "id3chunk":
        _tag_id3_chunk(path, node_name, album, index)


def read_playlist_tags(path: Path) -> dict:
    """Read the metadata fields EchoList cares about from a track file."""
    result = {"albumartist": "", "album": "", "tracknumber": "", "title": "", "artist": ""}
    try:
        kind = _tag_kind(path)
        if kind == "flac":
            f = FLAC(path)
            result["albumartist"] = (f.get("ALBUMARTIST") or [""])[0]
            result["album"] = (f.get("ALBUM") or [""])[0]
            result["tracknumber"] = _normalize_tracknumber((f.get("TRACKNUMBER") or [""])[0])
            result["title"] = (f.get("TITLE") or [""])[0]
            result["artist"] = (f.get("ARTIST") or [""])[0]
            result["echolist_role"] = (f.get("ECHOLIST_ROLE") or [""])[0]
        elif kind in ("mp3", "mp4"):
            tags = (EasyID3 if kind == "mp3" else EasyMP4)(path)
            result["albumartist"] = (tags.get("albumartist") or [""])[0]
            result["album"] = (tags.get("album") or [""])[0]
            result["tracknumber"] = _normalize_tracknumber((tags.get("tracknumber") or [""])[0])
            result["title"] = (tags.get("title") or [""])[0]
            result["artist"] = (tags.get("artist") or [""])[0]
        elif kind == "id3chunk":
            tags = _ID3_CHUNK_TYPES[path.suffix.lower()](path).tags
            if tags is not None:
                for field, frame_id in _ID3_FIELD_FRAMES.items():
                    result[field] = _id3_text(tags, frame_id)
                result["tracknumber"] = _normalize_tracknumber(result["tracknumber"])
    except _TAG_IO_ERRORS:
        pass
    return result


def restore_tags(path: Path, tags: dict) -> None:
    """Write saved tag values back to a file (for metadata restore points)."""
    cleaned = dict(tags)
    if cleaned.get("tracknumber"):
        cleaned["tracknumber"] = _normalize_tracknumber(cleaned["tracknumber"])
    try:
        kind = _tag_kind(path)
        if kind == "flac":
            f = FLAC(path)
            for field in ("albumartist", "album", "tracknumber", "title", "artist"):
                flac_key = field.upper()
                if cleaned.get(field):
                    f[flac_key] = cleaned[field]
                elif flac_key in f:
                    del f[flac_key]
            f.save()
        elif kind in ("mp3", "mp4"):
            t = (EasyID3 if kind == "mp3" else EasyMP4)(path)
            for field in ("albumartist", "album", "tracknumber", "title", "artist"):
                if cleaned.get(field):
                    t[field] = cleaned[field]
                elif field in t:
                    del t[field]
            t.save(path)
        elif kind == "id3chunk":
            f = _load_id3_chunk(path)
            for field, frame_id in _ID3_FIELD_FRAMES.items():
                if cleaned.get(field):
                    _id3_set(f.tags, frame_id, cleaned[field])
                else:
                    f.tags.delall(frame_id)
            f.save()
    except _TAG_IO_ERRORS:
        pass


def _extract_year(raw: str) -> str:
    """Extract a 4-digit year from any date string (e.g. '2023-10-19', '10/19/2023', '2023')."""
    import re
    m = re.search(r'\b(\d{4})\b', raw)
    return m.group(1) if m else raw


# TODO: ALBUMARTISTSORT overrides device grouping — the device uses it instead
# of ALBUMARTIST. We strip it now, but this could be a feature: setting
# ALBUMARTISTSORT intentionally could let us control how playlists appear
# in the device's artist/album browser. Needs testing on Echo Mini.
_FLAC_STRIP_TAGS = frozenset({
    "ALBUMARTISTSORT", "ARTISTSORT", "ALBUMARTISTS", "ALBUMARTISTS_SORT",
    "ALBUMARTISTS_CREDIT", "ALBUMARTIST_CREDIT",
    "ARTISTS", "ARTISTS_SORT", "ARTISTS_CREDIT", "ARTIST_CREDIT",
    "TOTALTRACKS", "TOTALDISCS", "TRACKTOTAL", "DISCTOTAL",
    "TRACKC", "DISCC", "TRACK", "DISC",
    "RELEASETYPE", "RELEASESTATUS", "RELEASECOUNTRY",
    "ALBUMARTISTSORT", "COMPOSERSORT",
    "MAIN_ARTIST", "ALBUM ARTIST", "ALBUM_ARTIST", "ALBUM_ARTISTS",
    "ORIGINALDATE", "ORIGINALYEAR",
    "MUSICBRAINZ_ALBUMARTISTID", "MUSICBRAINZ_ALBUMID",
    "MUSICBRAINZ_ARTISTID", "MUSICBRAINZ_TRACKID",
    "MUSICBRAINZ_RELEASETRACKID", "MUSICBRAINZ_RELEASEGROUPID",
    "MUSICBRAINZ_ALBUMSTATUS", "MUSICBRAINZ_ALBUMTYPE",
    "MUSICBRAINZ_ALBUMCOMMENT", "MUSICBRAINZ_DISCID",
    "MUSICBRAINZ_WORKID",
    "COMPILATION", "MEDIA", "SCRIPT", "BARCODE", "CATALOGNUMBER",
    "ASIN", "LABEL", "PUBLISHER", "ORGANIZATION",
    "AUTHOR", "LYRICIST", "ARRANGER", "CONDUCTOR",
    "ENCODEDBY", "ENCODER", "LANGUAGE", "BPM",
    "ACOUSTID_ID", "ACOUSTID_FINGERPRINT",
    "ITUNESADVISORY", "LENGTH",
    "DISCSUBTITLE", "GROUPING", "DESCRIPTION",
})


def _tag_flac(
    path: Path, node_name: str, album: str, index: int, src_rel: str, pid: str
) -> None:
    f = FLAC(path)
    for tag in _FLAC_STRIP_TAGS:
        if tag in f:
            del f[tag]
    f["ALBUMARTIST"] = node_name
    f["ALBUM"] = album
    f["TRACKNUMBER"] = str(index)
    f["DISCNUMBER"] = "1"
    date = (f.get("DATE") or [""])[0]
    if date:
        f["DATE"] = _extract_year(date)
    f["ECHOLIST_ROLE"] = "playlist-copy"
    f["ECHOLIST_PLAYLIST"] = pid
    f["ECHOLIST_INDEX"] = str(index)
    f["ECHOLIST_SRC"] = src_rel
    f.save()


_EASY_STRIP_TAGS = frozenset({
    "albumartistsort", "artistsort", "composersort",
    "compilation", "media", "barcode", "catalognumber",
    "asin", "organization",
    "musicbrainz_albumartistid", "musicbrainz_albumid",
    "musicbrainz_artistid", "musicbrainz_trackid",
    "musicbrainz_releasetrackid", "musicbrainz_releasegroupid",
    "musicbrainz_albumstatus", "musicbrainz_albumtype",
})


def _tag_easy(easy_cls, path: Path, node_name: str, album: str, index: int) -> None:
    try:
        tags = easy_cls(path)
    except mutagen.MutagenError:
        tags = easy_cls()
        tags.filename = str(path)
    for tag in _EASY_STRIP_TAGS:
        if tag in tags:
            del tags[tag]
    tags["albumartist"] = node_name
    tags["album"] = album
    tags["tracknumber"] = str(index)
    tags["discnumber"] = "1"
    date = (tags.get("date") or [""])[0]
    if date:
        tags["date"] = _extract_year(date)
    tags.save(path)


# ── WAV / AIFF: tags live in an embedded ID3v2 chunk ──

# Field -> ID3 frame, matching what EasyID3 uses for the same keys.
_ID3_FIELD_FRAMES = {
    "albumartist": "TPE2",
    "album": "TALB",
    "tracknumber": "TRCK",
    "title": "TIT2",
    "artist": "TPE1",
}
_ID3_FRAME_CLASSES = {
    "TPE2": TPE2, "TALB": TALB, "TRCK": TRCK, "TIT2": TIT2, "TPE1": TPE1,
    "TPOS": TPOS, "TDRC": TDRC,
}

# The ID3 frames EasyID3 maps _EASY_STRIP_TAGS to, so WAV/AIFF copies are
# stripped exactly like MP3 copies. Kept in sync by
# tests/test_tags_numbering.py::test_wav_strip_frames_match_mp3_strip_list.
_ID3_STRIP_FRAMES = frozenset({
    "TSO2", "TSOP", "TSOC", "TCMP", "TMED", "TPUB",
    "TXXX:BARCODE", "TXXX:CATALOGNUMBER", "TXXX:ASIN",
    "TXXX:MusicBrainz Album Artist Id", "TXXX:MusicBrainz Album Id",
    "TXXX:MusicBrainz Artist Id", "UFID:http://musicbrainz.org",
    "TXXX:MusicBrainz Release Track Id", "TXXX:MusicBrainz Release Group Id",
    "TXXX:MusicBrainz Album Status", "TXXX:MusicBrainz Album Type",
})


def _id3_text(tags, frame_id: str) -> str:
    frame = tags.get(frame_id)
    return str(frame.text[0]) if frame is not None and frame.text else ""


def _id3_set(tags, frame_id: str, value: str) -> None:
    tags.setall(frame_id, [_ID3_FRAME_CLASSES[frame_id](encoding=3, text=[value])])


def _load_id3_chunk(path: Path):
    """Open a WAV/AIFF file, adding an empty ID3 chunk if it has none."""
    f = _ID3_CHUNK_TYPES[path.suffix.lower()](path)
    if f.tags is None:
        f.add_tags()
    return f


def _tag_id3_chunk(path: Path, node_name: str, album: str, index: int) -> None:
    """Tag a WAV/AIFF copy. mutagen rewrites only the ID3 chunk (and the
    RIFF/FORM size); the audio chunk bytes are untouched."""
    f = _load_id3_chunk(path)
    tags = f.tags
    for key in [k for k in tags.keys() if k in _ID3_STRIP_FRAMES]:
        del tags[key]
    _id3_set(tags, "TPE2", node_name)
    _id3_set(tags, "TALB", album)
    _id3_set(tags, "TRCK", str(index))
    _id3_set(tags, "TPOS", "1")
    date = _id3_text(tags, "TDRC")
    if date:
        _id3_set(tags, "TDRC", _extract_year(date))
    f.save()
