"""Regression tests for the GUI findings of the final review (synthesis.md).

Each test names its finding ID (A# = astra.md, N# = neon.md, T# = test
defects). They drive real App methods, run background work to completion with
pump() (no sleeps), and reload persisted state where the finding is about
what survives a restart.
"""

import errno
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import echolist.config as _config
import echolist.gui as gui
import echolist.journal as _journal
from echolist.gui import App
from conftest import _make_flac
from test_gui import (app, gui_env, isolated_home, callback_errors,  # noqa: F401  (fixtures)
                      pump)
from test_gui_sync_safety import (_sources, _titles, _disk_names, _synced_playlist,
                                  _reopen, Crash)


# ── T2 / T6 / N13: test harness and callback loop ──

def test_fixture_app_has_every_attribute_init_sets(app):
    """T2: the shared fixture built App.__new__ by hand and missed
    _exit_after_sync / _callback_queue, so callbacks died on AttributeError."""
    with patch.object(App, "_set_icon"), patch.object(App, "_center_window"), \
            patch.object(App, "_show_setup"), patch.object(App, "_poll_callbacks"):
        real = App()
    try:
        missing = set(vars(real)) - set(vars(app))
    finally:
        real.root.destroy()
    assert not missing


def test_opening_a_workspace_writes_only_the_isolated_home(app, gui_env, tmp_path):
    """T6: _open_workspace wrote the real ~/.echolist/default.json and read the
    real legacy journal/pending file from the shared fixture."""
    for path in (_config.DEFAULT_FILE, _config.BACKUPS_ROOT, gui.PENDING_FILE,
                 _journal.JOURNAL_FILE):
        assert Path(path).is_relative_to(tmp_path), path
    _reopen(app, gui_env)
    assert _config.DEFAULT_FILE.exists()


def test_failing_callback_does_not_stop_the_poll_loop(app, callback_errors):
    """N13: one raising callback ended _poll_callbacks for good (no reschedule),
    hanging every later sync, playlist op and Ctrl+C."""
    ran = []

    def bad():
        raise RuntimeError("boom")

    app._schedule_callback(bad)
    app._schedule_callback(lambda: ran.append("next"))
    with patch.object(app.root, "after") as after:
        app._poll_callbacks()

    assert ran == ["next"]
    after.assert_called_once_with(16, app._poll_callbacks)
    assert [type(e[1]) for e in callback_errors] == [RuntimeError]
    callback_errors.clear()


# ── A6 / N1: one writer at a time; store mutated on the Tk thread ──

class _Gate:
    """Hold a worker inside ``target`` until release(), so the test can act
    on the Tk thread while the background operation is running."""

    def __init__(self, obj, name):
        self.entered = threading.Event()
        self.go = threading.Event()
        real = getattr(obj, name)

        def held(*a, **kw):
            self.entered.set()
            assert self.go.wait(30), "gate never released"
            return real(*a, **kw)
        self._patch = patch.object(obj, name, side_effect=held)

    def __enter__(self):
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self.go.set()
        self._patch.stop()

    def wait_entered(self):
        assert self.entered.wait(30), "worker never reached the gate"


def _quiet():
    return patch.multiple("echolist.gui.messagebox",
                          showinfo=lambda *a, **k: None,
                          showwarning=lambda *a, **k: None,
                          showerror=lambda *a, **k: None,
                          askyesno=lambda *a, **k: True)


def test_sync_is_refused_while_a_playlist_operation_runs(app, gui_env):
    """A6/N1: Sync started while an offload worker was mid-way, so two threads
    rewrote playlists.json from different Store states."""
    one = _synced_playlist(app, gui_env, 1)
    app._create_playlist()
    two = app.current_pid
    app._stage_add_files(_sources(gui_env)[1:2])

    with _quiet(), _Gate(app.mgr, "backup_playlist_metadata") as gate:
        app._offload_playlist(one)
        gate.wait_entered()
        app._do_sync()
        assert not app._syncing
        gate.go.set()
        pump(app)

    assert app.mgr.store.playlists[one]["offloaded"] is True
    assert app.mgr.store.playlists[two]["tracks"] == []
    assert len(app.staging.pending_adds) == 1


def test_store_changing_actions_wait_for_a_running_operation(app, gui_env):
    """A6/N1: refresh, create, rename, delete, drag-reorder, settings and the
    playlist menus ignored _busy_playlists and changed the Store meanwhile."""
    one = _synced_playlist(app, gui_env, 2)
    app._create_playlist()
    two = app.current_pid
    pids_before = set(app.mgr.store.playlists)

    with _quiet(), _Gate(app.mgr, "backup_playlist_metadata") as gate:
        app._offload_playlist(one)
        gate.wait_entered()
        with patch.object(app.mgr, "rescan_playlist") as rescan:
            app._refresh_all()
            app.current_pid = two
            app._refresh_tracks()
        rescan.assert_not_called()
        app._create_playlist()
        assert not app._rename_playlist(two, "Other")
        app._delete_playlist()
        app._reindex_playlist(one)
        app._create_restore_point(two)
        app._show_settings()
        assert app.playlist_tree.winfo_exists()
        assert set(app.mgr.store.playlists) == pids_before
        assert not app.staging.has_pending
        gate.go.set()
        pump(app)

    assert app.mgr.store.playlists[one]["offloaded"] is True
    assert set(app.mgr.store.playlists) == pids_before
    assert app.mgr.list_metadata_backups(two) == []


def test_playlist_operations_are_refused_during_sync(app, gui_env):
    """A6/N1: menu and context-menu operations ignored _syncing."""
    one = _synced_playlist(app, gui_env, 1)
    app._stage_add_files(_sources(gui_env)[1:2])

    with _quiet(), _Gate(app.mgr, "add_track") as gate:
        app._do_sync()
        gate.wait_entered()
        app._offload_playlist(one)
        app._create_restore_point(one)
        assert not app._busy_playlists
        gate.go.set()
        pump(app)

    assert not app.mgr.store.playlists[one].get("offloaded")
    assert _titles(app, one) == ["Song One", "Song Two"]


def test_delete_and_offload_change_the_store_on_the_tk_thread(app, gui_env):
    """A6/N1 preferred design: workers do the file I/O; the Store mutation and
    store.save() run in on_done on the Tk thread."""
    one = _synced_playlist(app, gui_env, 1)
    two = _synced_playlist(app, gui_env, 1)
    threads = []
    real_save = app.mgr.store.save

    def spy():
        threads.append(threading.current_thread() is threading.main_thread())
        return real_save()

    with _quiet(), patch.object(app.mgr.store, "save", side_effect=spy):
        app._offload_playlist(one)
        pump(app)
        app.current_pid = two
        app._delete_playlist()
        pump(app)

    assert threads and all(threads)
    assert app.mgr.store.playlists[one]["offloaded"] is True
    assert two not in app.mgr.store.playlists


def test_close_during_a_playlist_operation_keeps_the_lock_until_it_ends(app, gui_env):
    """A6: closing released the workspace lock while a worker still wrote."""
    one = _synced_playlist(app, gui_env, 1)

    with _quiet(), _Gate(app.mgr, "backup_playlist_metadata") as gate, \
            patch.object(app.root, "destroy") as destroy:
        app._offload_playlist(one)
        gate.wait_entered()
        app._on_close()
        destroy.assert_not_called()
        assert app.mgr._lock_fd is not None
        gate.go.set()
        pump(app)
        destroy.assert_called_once()
    assert app.mgr._lock_fd is None
    assert app.mgr.store.playlists[one]["offloaded"] is True


# ── A22 / N11: a failed sync preflight must not lock the UI ──

def test_sync_preflight_failure_unlocks_the_ui(app, gui_env):
    """A22/N11: _backup_before_sync raising (read-only card) left _syncing set
    forever; Close then waited for a sync that never ran."""
    app._create_playlist()
    app._stage_add_files(_sources(gui_env)[:1])

    with patch.object(app.mgr.config, "increment_sync",
                      side_effect=OSError(errno.EROFS, "Read-only file system")), \
            patch("echolist.gui.messagebox.showerror") as err:
        app._do_sync()
    pump(app)

    err.assert_called_once()
    assert not app._syncing
    assert app.sync_btn.winfo_manager() == "pack"
    assert len(app.staging.pending_adds) == 1
    with patch.object(app.root, "destroy") as destroy, \
            patch("echolist.gui.messagebox.askyesnocancel", return_value=False):
        app._on_close()
    destroy.assert_called_once()


# ── N21: Settings must not drop the lock before the new workspace opens ──

def test_failed_workspace_switch_keeps_the_current_lock_and_config(app, gui_env, tmp_path):
    """N21: Settings released the lock (and saved the new backup interval into
    the old card's config) before opening the new workspace; when that failed
    the app kept editing the old card unlocked."""
    from echolist.manager import PlaylistManager, WorkspaceLockError
    other = tmp_path / "card2"
    other.mkdir()
    holder = PlaylistManager.init(gui_env["src"], other)  # another instance
    old_interval = app.mgr.config.backup_interval
    try:
        app._show_settings()
        app._dest_var.set(str(other))
        app._backup_var.set(str(old_interval + 7))
        with patch("echolist.gui.messagebox.showerror") as err, \
                patch("echolist.gui.messagebox.showwarning"):
            app._on_settings_open()
        err.assert_called_once()
        assert app.mgr.writer.root == (gui_env["dest"] / "Playlists").resolve()
        with pytest.raises(WorkspaceLockError):
            PlaylistManager.open(gui_env["dest"])
        import json
        cfg = json.loads((gui_env["dest"] / "Playlists" / ".echolist" / "config.json")
                         .read_text(encoding="utf-8"))
        assert cfg["backup_interval"] == old_interval
    finally:
        holder.release_lock()


def test_workspace_switch_releases_the_old_lock_after_the_new_one_opens(app, gui_env, tmp_path):
    """N21: the successful path still ends with exactly one workspace locked."""
    from echolist.manager import PlaylistManager
    other = tmp_path / "card2"
    other.mkdir()
    old_mgr = app.mgr
    app._show_settings()
    app._dest_var.set(str(other))
    with patch("echolist.gui.messagebox.showwarning"), \
            patch("echolist.gui.messagebox.askyesno", return_value=False):
        app._on_settings_open()
    pump(app)
    assert app.mgr is not old_mgr
    assert old_mgr._lock_fd is None
    assert app.mgr._lock_fd is not None
    PlaylistManager.open(gui_env["dest"]).release_lock()
    app.mgr.release_lock()


# ── N26: unreadable legacy pending.json must not abort opening ──

def test_unreadable_legacy_pending_file_does_not_abort_open(app, gui_env):
    """N26: _bind_legacy_staging ran outside the StagingLoadError handling, so
    a truncated ~/.echolist/pending.json stopped the workspace from opening."""
    staging_path = gui.pending_file(app.mgr.writer.root)
    staging_path.unlink(missing_ok=True)
    gui.PENDING_FILE.write_text('{"adds": [', encoding="utf-8")

    app.mgr.release_lock()
    with patch("echolist.gui.messagebox.showwarning") as warn:
        app._open_workspace(str(gui_env["src"]), str(gui_env["dest"]))
    pump(app)

    warn.assert_called_once()
    assert app.playlist_tree.winfo_exists()
    assert gui.PENDING_FILE.read_text(encoding="utf-8") == '{"adds": ['


# ── A11 (GUI) / A17: offload and delete must not lose the restore point ──

def test_delete_is_aborted_when_the_restore_point_cannot_be_saved(app, gui_env):
    """A11: _delete_playlist swallowed a backup failure and deleted anyway,
    leaving a playlist that 'can be brought back' with no restore point."""
    pid = _synced_playlist(app, gui_env, 2)
    with patch.object(app.mgr, "backup_playlist_metadata",
                      side_effect=OSError(errno.ENOSPC, "No space left")), \
            patch("echolist.gui.messagebox.askyesno", return_value=True), \
            patch("echolist.gui.messagebox.showerror") as err:
        app._delete_playlist()
        pump(app)
    err.assert_called_once()
    assert len(_disk_names(app, pid)) == 2
    _reopen(app, gui_env)
    assert _titles(app, pid) == ["Song One", "Song Two"]


def test_offload_keeps_the_playlist_when_the_device_delete_fails(app, gui_env):
    """A11: offload swallowed writer.delete errors, then marked the playlist
    offloaded with tracks=[] although the files were still on the card."""
    pid = _synced_playlist(app, gui_env, 2)
    with _quiet(), patch.object(app.mgr.writer, "delete",
                                side_effect=OSError(errno.EIO, "I/O error")):
        app._offload_playlist(pid)
        pump(app)
    _reopen(app, gui_env)
    pl = app.mgr.store.playlists[pid]
    assert not pl.get("offloaded")
    assert _titles(app, pid) == ["Song One", "Song Two"]


def test_onload_uses_the_restore_point_the_offload_saved(app, gui_env):
    """A17: backup_playlist_metadata dedupes to an older identical point, but
    onload took the newest backup, restoring a different track list."""
    pid = _synced_playlist(app, gui_env, 2)
    app.mgr.backup_playlist_metadata(pid, timestamp="20000101_000000")
    app._stage_add_files(_sources(gui_env)[2:3])
    app._do_sync()
    pump(app)
    app.mgr.backup_playlist_metadata(pid, timestamp="20990101_000000")
    app._stage_remove_track(pid, 3, app.mgr.store.playlists[pid]["tracks"][2]["copy_name"])
    app._do_sync()
    pump(app)
    assert _titles(app, pid) == ["Song One", "Song Two"]

    with _quiet():
        app._offload_playlist(pid)
        pump(app)
        _reopen(app, gui_env)
        app._onload_playlist(pid)
        pump(app)
    assert [a["title"] for a in app.staging.pending_adds] == ["Song One", "Song Two"]


# ── A1 / N4 / N23: staged removes are matched by stable identity ──

def _track(app, pid, index):
    return next(t for t in app.mgr.store.playlists[pid]["tracks"] if t["index"] == index)


def test_rejected_remove_never_retargets_the_replacement_track(app, gui_env):
    """A1: a remove rejected as stale had its copy_name rewritten by rebase to
    whatever now sat at that index, so the next sync deleted that track."""
    import shutil
    pid = _synced_playlist(app, gui_env, 2)
    first = _track(app, pid, 1)
    app._stage_remove_track(pid, 1, first["copy_name"])
    folder = app.mgr.writer.root / app.mgr.store.playlists[pid]["folder"]
    (folder / first["copy_name"]).unlink()  # replaced outside EchoList
    shutil.copy(_sources(gui_env)[2], folder / "01 - Song Three.flac")
    app.mgr.rescan_playlist(pid)

    first_errors = app._do_sync_blocking()
    second_errors = app._do_sync_blocking()

    assert first_errors
    assert second_errors == []
    assert _titles(app, pid) == ["Song Three", "Song Two"]
    assert not app.staging.pending_removes


def test_removing_first_and_last_of_100_tracks_survives_the_pad_change(app, gui_env):
    """N4: removing #100 re-pads 001.. to 01..; the staged remove of #1 still
    named '001 - ...' and was rejected as stale."""
    app._create_playlist()
    pid = app.current_pid
    for i in range(1, 101):
        src = gui_env["src"] / "Many" / f"{i:03d}.flac"
        _make_flac(src, "Many", f"Tune {i:03d}")
        app.mgr.add_track(pid, src)
    app._stage_remove_track(pid, 1, _track(app, pid, 1)["copy_name"])
    app._stage_remove_track(pid, 100, _track(app, pid, 100)["copy_name"])

    assert app._do_sync_blocking() == []

    titles = _titles(app, pid)
    assert len(titles) == 98
    assert titles[0] == "Tune 002" and titles[-1] == "Tune 099"
    assert _disk_names(app, pid)[0] == "01 - Tune 002.flac"


def test_remove_matches_a_name_stored_in_a_different_unicode_form(app, gui_env):
    """N23: the staged copy_name (NFC) was compared byte-for-byte with the
    store's name, which a rescan on macOS takes from disk in NFD."""
    import unicodedata
    app._create_playlist()
    pid = app.current_pid
    src = gui_env["src"] / "Cafe" / "cafe.flac"
    _make_flac(src, "Cafe", "Café")
    app.mgr.add_track(pid, src)
    nfc = _track(app, pid, 1)["copy_name"]
    app._stage_remove_track(pid, 1, nfc)
    _track(app, pid, 1)["copy_name"] = unicodedata.normalize("NFD", nfc)

    assert app._do_sync_blocking() == []
    assert app.mgr.store.playlists[pid]["tracks"] == []


# ── A9 / A10: a reorder's commit is recorded, not inferred ──

def test_reorder_store_save_failure_does_not_double_shuffle_on_retry(app, gui_env):
    """A9/N6: store.save() failing in the reorder commit discarded the journal
    and left the reorder staged, so the retry shuffled the files again."""
    pid = _synced_playlist(app, gui_env, 3)
    app.staging.set_reorder(pid, [{"key": "c:3"}, {"key": "c:1"}, {"key": "c:2"}])
    real_save = app.mgr.store.save
    calls = []

    def full_once():
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_save()

    with patch.object(app.mgr.store, "save", side_effect=full_once):
        assert app._do_sync_blocking()
    assert app._do_sync_blocking() == []
    _reopen(app, gui_env)

    assert _titles(app, pid) == ["Song Three", "Song One", "Song Two"]
    assert _disk_names(app, pid) == ["01 - Song Three.flac", "02 - Song One.flac",
                                     "03 - Song Two.flac"]
    assert pid not in app.staging.pending_reorders
    assert not gui.journal_file(app.mgr.writer.root).exists()


def test_reorder_store_save_failure_is_committed_on_next_open(app, gui_env):
    """A9: the same failure followed by a restart (no retry in between)."""
    pid = _synced_playlist(app, gui_env, 3)
    app.staging.set_reorder(pid, [{"key": "c:3"}, {"key": "c:1"}, {"key": "c:2"}])
    with patch.object(app.mgr.store, "save",
                      side_effect=OSError(errno.ENOSPC, "No space left on device")):
        assert app._do_sync_blocking()
    _reopen(app, gui_env)
    assert _titles(app, pid) == ["Song Three", "Song One", "Song Two"]
    assert pid not in app.staging.pending_reorders
    assert not gui.journal_file(app.mgr.writer.root).exists()


def _same_title_pair(app, gui_env):
    a = gui_env["src"] / "X1" / "x.flac"
    b = gui_env["src"] / "X2" / "x.flac"
    _make_flac(a, "Band A", "Same")
    _make_flac(b, "Band B", "Same")
    app._create_playlist()
    pid = app.current_pid
    app.mgr.add_track(pid, a)
    app.mgr.add_track(pid, b)
    return pid, "X1/x.flac", "X2/x.flac"


def _src_order(app, pid):
    tracks = sorted(app.mgr.store.playlists[pid]["tracks"], key=lambda t: t["index"])
    return [t["src_path"] for t in tracks]


def test_same_title_swap_recovered_before_the_store_save_keeps_identities(app, gui_env):
    """A10: recovery decided 'already committed' because the store's names
    equalled the plan's new names, which is always true for a same-title swap,
    so each entry kept the other file's src_path."""
    pid, a, b = _same_title_pair(app, gui_env)
    app.staging.set_reorder(pid, [{"key": "c:2"}, {"key": "c:1"}])
    with patch.object(app.mgr.store, "save", side_effect=Crash()):
        with pytest.raises(Crash):
            app._do_sync_blocking()
    _reopen(app, gui_env)

    assert _src_order(app, pid) == [b, a]
    folder = app.mgr.writer.root / app.mgr.store.playlists[pid]["folder"]
    from echolist.tags import read_playlist_tags
    assert read_playlist_tags(folder / "01 - Same.flac")["artist"] == "Band B"


def test_recovery_records_its_rename_phase(app, gui_env):
    """A15/N15.2: recovery ran the final rename pass without recording phase
    'final'; a second crash then made the next recovery treat parked files as
    unmoved and restore them under a taken name."""
    pid, a, b = _same_title_pair(app, gui_env)
    app.staging.set_reorder(pid, [{"key": "c:2"}, {"key": "c:1"}])
    from echolist.safe_write import SafeWriter
    real_rename = SafeWriter.rename

    def crash_on(n):
        calls = []

        def rename(self, old, new):
            calls.append(old)
            if len(calls) == n:
                raise Crash()
            return real_rename(self, old, new)
        return patch.object(SafeWriter, "rename", rename)

    with crash_on(2), pytest.raises(Crash):
        app._do_sync_blocking()          # one file parked, phase "tmp"
    with crash_on(3), pytest.raises(Crash):
        _reopen(app, gui_env)            # recovery: parks the other, moves one back
    _reopen(app, gui_env)

    assert _disk_names(app, pid) == ["01 - Same.flac", "02 - Same.flac"]
    assert _src_order(app, pid) == [b, a]


# ── A15 / N15.1: recovery keeps what never started and what it couldn't fix ──

def test_reorder_that_never_started_survives_an_interrupted_sync(app, gui_env):
    """A15/N15.1: every pid with an unfinished journal step lost its staged
    reorder, including reorders the crash never reached."""
    one = _synced_playlist(app, gui_env, 2)
    app.staging.set_reorder(one, [{"key": "c:2"}, {"key": "c:1"}])
    app._create_playlist()
    two = app.current_pid
    app._stage_add_files(_sources(gui_env)[2:3])

    with patch.object(app.mgr, "add_track", side_effect=Crash()):
        with pytest.raises(Crash):
            app._do_sync_blocking()
    _reopen(app, gui_env)

    assert app.staging.pending_reorders.get(one) == [{"key": "c:2"}, {"key": "c:1"}]
    assert app._do_sync_blocking() == []
    assert _titles(app, one) == ["Song Two", "Song One"]
    assert _titles(app, two) == ["Song Three"]


def test_journal_is_kept_when_recovery_cannot_finish(app, gui_env):
    """A15: the journal was discarded even when finishing the reorder failed,
    so the half-renamed playlist could never be completed."""
    pid = _synced_playlist(app, gui_env, 3)
    app.staging.set_reorder(pid, [{"key": "c:3"}, {"key": "c:1"}, {"key": "c:2"}])
    real_rename = app.mgr.writer.rename
    calls = []

    def crash_on_fourth(old, new):
        calls.append(old)
        if len(calls) == 4:
            raise Crash()
        return real_rename(old, new)

    with patch.object(app.mgr.writer, "rename", side_effect=crash_on_fourth):
        with pytest.raises(Crash):
            app._do_sync_blocking()
    jpath = gui.journal_file(app.mgr.writer.root)

    from echolist.safe_write import SafeWriter
    with patch.object(SafeWriter, "rename", side_effect=OSError(errno.EIO, "I/O error")):
        warn = _reopen(app, gui_env)["warning"]
    assert warn.called
    assert jpath.exists()

    app._refresh_all()  # must not rescan the parked files out of the store
    _reopen(app, gui_env)
    assert _titles(app, pid) == ["Song Three", "Song One", "Song Two"]
    assert [t["src_path"] for t in sorted(app.mgr.store.playlists[pid]["tracks"],
                                          key=lambda t: t["index"])] == [
        "ArtistC/Album3/05 Song Three.flac", "ArtistA/Album1/01 Song One.flac",
        "ArtistB/Album2/03 Song Two.flac"]
    assert not jpath.exists()


# ── A8 / N15.3: an add published before the store save is adopted ──

def test_add_published_before_store_save_is_not_copied_twice(app, gui_env):
    """A8: a crash after copy_in published the file but before store.save()
    left the add staged; rescan saw an unknown file, and the retry copied the
    track a second time."""
    pid = _synced_playlist(app, gui_env, 1)
    app._stage_add_files(_sources(gui_env)[1:2])
    with patch.object(app.mgr.store, "save", side_effect=Crash()):
        with pytest.raises(Crash):
            app._do_sync_blocking()
    _reopen(app, gui_env)

    assert not app.staging.pending_adds
    assert app._do_sync_blocking() == []
    assert _titles(app, pid) == ["Song One", "Song Two"]
    assert _track(app, pid, 2)["src_path"] == "ArtistB/Album2/03 Song Two.flac"


# ── A21 / N19: restores stage under the playlist they actually created ──

def test_snapshot_restore_stages_under_the_created_playlist_id(app, gui_env, tmp_path):
    """A21/N19: snapshot restore staged adds under the snapshot's (legacy) pid
    although create_playlist() returned a different one, so every restored
    track failed to sync."""
    card = tmp_path / "card2"
    card.mkdir()
    snapshot = {
        "config": {"source_root": str(gui_env["src"]),
                   "playlist_folder": gui.DEFAULT_PLAYLIST_FOLDER},
        "store": {"schema": 1, "playlists": {"ac/dc": {
            "name": "AC/DC", "folder": "AC_DC",
            "tracks": [{"index": 1, "copy_name": "01 - Song One.flac",
                        "src_path": "ArtistA/Album1/01 Song One.flac"}]}}},
    }
    app.mgr.release_lock()
    with patch.object(gui.PlaylistManager, "find_snapshot", return_value=snapshot), \
            patch("echolist.gui.messagebox.askyesno", return_value=True), \
            patch("echolist.gui.messagebox.showwarning"), \
            patch("echolist.gui.messagebox.showinfo"):
        app._open_workspace(str(gui_env["src"]), str(card))
    pump(app)

    [pid] = list(app.mgr.store.playlists)
    assert [a["pid"] for a in app.staging.pending_adds] == [pid]
    assert app._do_sync_blocking() == []
    assert _titles(app, pid) == ["Song One"]


def test_restore_deleted_stages_under_the_recreated_playlist_id(app, gui_env):
    """A21/N19: a backup kept under an old pid ('old_mix') recreated the
    playlist as playlist_id(name) but staged its tracks under 'old_mix'."""
    from echolist.config import save_backup
    from echolist.naming import playlist_id
    save_backup(app.mgr.writer.root, "old_mix", "20200101_000000", {
        "playlist_name": "Mix", "folder": "Mix",
        "tracks": [{"copy_name": "01 - Song One.flac", "index": 1,
                    "src_path": "ArtistA/Album1/01 Song One.flac"}]})
    with _quiet():
        app._restore_deleted("old_mix", "20200101_000000")
        pump(app)

    pid = playlist_id("Mix")
    assert [a["pid"] for a in app.staging.pending_adds] == [pid]
    assert app._do_sync_blocking() == []
    assert _titles(app, pid) == ["Song One"]


# ── A13 / N2 (GUI part): restores and staged changes ──

def test_restore_point_replaces_the_playlists_staged_changes(app, gui_env):
    """A13/N2 (CORE handoff 2): restoring rewrote the playlist under staged
    removes/reorders whose indices then pointed at other tracks. The restore
    now discards that playlist's staged changes (and undo) first; other
    playlists keep theirs."""
    pid = _synced_playlist(app, gui_env, 2)
    app.mgr.backup_playlist_metadata(pid, timestamp="20200101_000000")
    app._create_playlist()
    other = app.current_pid
    app._stage_add_files(_sources(gui_env)[2:3])
    app._stage_remove_track(pid, 1, _track(app, pid, 1)["copy_name"])
    app.staging.set_reorder(pid, [{"key": "c:2"}, {"key": "c:1"}])

    with _quiet():
        app._restore_from_point(pid, "20200101_000000")
        pump(app)

    assert not app.staging.pending_removes
    assert pid not in app.staging.pending_reorders
    assert [a["pid"] for a in app.staging.pending_adds] == [other]
    assert not app._undo_stack
    assert _titles(app, pid) == ["Song One", "Song Two"]


def test_restore_point_puts_restaged_tracks_back_in_their_place(app, gui_env):
    """CORE handoff 1: to_stage entries carry the position they had in the
    restore point; staging them as plain adds put them at the end."""
    pid = _synced_playlist(app, gui_env, 3)
    app.mgr.backup_playlist_metadata(pid, timestamp="20200101_000000")
    app._stage_remove_track(pid, 1, _track(app, pid, 1)["copy_name"])
    assert app._do_sync_blocking() == []
    assert _titles(app, pid) == ["Song Two", "Song Three"]

    with _quiet():
        app.current_pid = pid
        app._restore_from_point(pid, "20200101_000000")
        pump(app)
    assert [r["title"] for r in app._track_data] == ["Song One", "Song Two", "Song Three"]
    assert app._do_sync_blocking() == []
    assert _titles(app, pid) == ["Song One", "Song Two", "Song Three"]


def test_restore_point_reports_what_it_could_not_do(app, gui_env):
    """CORE handoff 1: files the card refused to delete come back in
    "failed"; the summary never mentioned them."""
    pid = _synced_playlist(app, gui_env, 1)
    app.mgr.backup_playlist_metadata(pid, timestamp="20200101_000000")
    app._stage_add_files(_sources(gui_env)[1:2])
    assert app._do_sync_blocking() == []
    extra = _track(app, pid, 2)["copy_name"]

    with patch.object(app.mgr.writer, "delete",
                      side_effect=PermissionError(errno.EACCES, "Permission denied")), \
            patch("echolist.gui.messagebox.askyesno", return_value=True), \
            patch("echolist.gui.messagebox.showinfo") as info, \
            patch("echolist.gui.messagebox.showwarning") as warn:
        app._restore_from_point(pid, "20200101_000000")
        pump(app)
    text = " ".join(str(c) for c in info.call_args_list + warn.call_args_list)
    assert extra in text


def test_undo_after_a_restore_does_not_unstage_its_tracks(app, gui_env):
    """A13/N27: undo snapshots taken before a structural operation survived
    it; undoing then silently dropped the tracks the restore had staged."""
    pid = _synced_playlist(app, gui_env, 2)
    app.mgr.backup_playlist_metadata(pid, timestamp="20200101_000000")
    app._stage_remove_track(pid, 2, _track(app, pid, 2)["copy_name"])
    assert app._do_sync_blocking() == []
    app._create_playlist()
    other = app.current_pid
    app._stage_add_files(_sources(gui_env)[2:3])   # leaves an undo entry
    app.current_pid = pid

    with _quiet():
        app._restore_from_point(pid, "20200101_000000")
        pump(app)
    assert any(a["pid"] == pid for a in app.staging.pending_adds)
    app._do_undo()
    assert any(a["pid"] == pid for a in app.staging.pending_adds)
    assert any(a["pid"] == other for a in app.staging.pending_adds)


# ── N7 / A19: WAV / AIFF titles ──

def _make_tagged_wav(path, title, artist):
    import wave
    from mutagen.id3 import TIT2, TPE1
    from mutagen.wave import WAVE
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 800)
    f = WAVE(path)
    f.add_tags()
    f.tags.add(TIT2(encoding=3, text=title))
    f.tags.add(TPE1(encoding=3, text=artist))
    f.save()


def test_wav_titles_are_shown_for_staged_and_synced_tracks(app, gui_env):
    """N7/A19: titles were read with mutagen easy=True, which has no title key
    for WAV/AIFF ID3 chunks, so these tracks showed blank."""
    wav = gui_env["src"] / "Wav" / "take.wav"
    _make_tagged_wav(wav, "Wave Song", "Wave Band")
    app._create_playlist()
    pid = app.current_pid
    app._stage_add_files([wav])
    assert (app.staging.pending_adds[0]["title"],
            app.staging.pending_adds[0]["artist"]) == ("Wave Song", "Wave Band")
    assert app._do_sync_blocking() == []
    app._tag_cache.clear()
    app._refresh_tracks()
    pump(app)
    rows = [app.track_tree.item(i, "values") for i in app.track_tree.get_children()]
    assert [tuple(r[1:]) for r in rows] == [("Wave Song", "Wave Band")]


def test_untagged_file_falls_back_to_its_name(app, gui_env):
    """N7: an empty title displays the file stem, not a blank row."""
    wav = gui_env["src"] / "Wav" / "untitled take.wav"
    _make_tagged_wav(wav, "", "")
    app._create_playlist()
    app._stage_add_files([wav])
    assert app.staging.pending_adds[0]["title"] == "untitled take"


# ── N20: file-name search treats names literally ──

def test_moved_source_with_brackets_in_its_name_is_found(app, gui_env):
    """N20: the filename fallback used rglob(name) unescaped, so '[Remix]'
    became a character class and the moved file was never found."""
    moved = gui_env["src"] / "Moved" / "Song [Remix].flac"
    _make_flac(moved, "ArtistR", "Song Remix")
    root = Path(gui_env["src"]).resolve()
    assert gui._resolve_source_file("Old/Song [Remix].flac", root) == moved.resolve()


# ── N27: stale rows and stale reorders ──

def test_remove_during_loading_does_not_use_the_previous_playlists_rows(app, gui_env):
    """N27: _track_data kept the previous playlist's rows while 'Loading...'
    was shown, so Delete staged a remove of the wrong track."""
    app._create_playlist()
    b = app.current_pid
    app.mgr.add_track(b, _sources(gui_env)[2])
    _synced_playlist(app, gui_env, 2)          # A's rows are on screen
    assert len(app._track_data) == 2
    app.current_pid = b
    app._tag_cache.clear()                      # B's tags must load in the background
    app._refresh_tracks()
    assert app.track_tree.get_children() == ("_loading",)
    app.track_tree.selection_set("_loading")
    app._remove_track()
    assert not app.staging.pending_removes
    pump(app)


def test_declining_an_orphan_remove_still_clears_the_shifted_reorder(app, gui_env):
    """N27: answering No to the 'no source' prompt returned before
    unstage_reorder, leaving a reorder whose p:N keys now named other adds."""
    import shutil
    pid = _synced_playlist(app, gui_env, 1)
    folder = app.mgr.writer.root / app.mgr.store.playlists[pid]["folder"]
    shutil.copy(_sources(gui_env)[3], folder / "02 - Stray.flac")  # no source
    app._refresh_tracks()
    pump(app)
    app._stage_add_files(_sources(gui_env)[1:3])
    app.staging.set_reorder(pid, [{"key": "p:1"}, {"key": "p:0"},
                                  {"key": "c:1"}, {"key": "c:2"}])
    app._refresh_tracks()
    pump(app)
    rows = {r["key"]: i for r, i in zip(app._track_data, app.track_tree.get_children())}
    app.track_tree.selection_set([rows["p:0"], rows["c:2"]])
    with patch("echolist.gui.messagebox.askyesno", return_value=False):
        app._remove_track()

    assert [a["title"] for a in app.staging.pending_adds] == ["Song Three"]
    assert pid not in app.staging.pending_reorders


# ── CORE handoff 3: a folder already on the card blocks one restored playlist ──

def test_snapshot_restore_reports_a_playlist_whose_folder_exists(app, gui_env, tmp_path):
    """create_playlist now refuses a folder that already exists on the card;
    the snapshot restore must say which playlist was not restored and why,
    instead of counting its tracks as missing sources."""
    card = tmp_path / "card2"
    (card / gui.DEFAULT_PLAYLIST_FOLDER / "AC_DC").mkdir(parents=True)
    snapshot = {
        "config": {"source_root": str(gui_env["src"]),
                   "playlist_folder": gui.DEFAULT_PLAYLIST_FOLDER},
        "store": {"schema": 1, "playlists": {"ac_dc": {
            "name": "AC/DC", "folder": "AC_DC",
            "tracks": [{"index": 1, "copy_name": "01 - Song One.flac",
                        "src_path": "ArtistA/Album1/01 Song One.flac"}]}}},
    }
    app.mgr.release_lock()
    with patch.object(gui.PlaylistManager, "find_snapshot", return_value=snapshot), \
            patch("echolist.gui.messagebox.askyesno", return_value=True), \
            patch("echolist.gui.messagebox.showwarning") as warn, \
            patch("echolist.gui.messagebox.showinfo"):
        app._open_workspace(str(gui_env["src"]), str(card))
    pump(app)
    text = " ".join(str(c) for c in warn.call_args_list)
    assert "AC/DC" in text and "not restored" in text
    assert not app.staging.pending_adds


# ── CORE handoff 4: rename checks the folder name the way the card does ──

def test_rename_refuses_a_dot_folder(app, gui_env):
    pid = _synced_playlist(app, gui_env, 1)
    folder = app.mgr.store.playlists[pid]["folder"]
    with patch("echolist.gui.messagebox.showerror") as err:
        assert not app._rename_playlist(pid, ".Hidden")
    err.assert_called_once()
    assert app.mgr.store.playlists[pid]["folder"] == folder
    assert (app.mgr.writer.root / folder).is_dir()


def test_rename_refuses_a_folder_differing_only_in_unicode_form(app, gui_env):
    """macOS (and so the card as macOS sees it) treats NFC and NFD names as
    one; the casefold-only compare let two playlists share a folder."""
    app.mgr.create_playlist("Café")
    pid = _synced_playlist(app, gui_env, 1)
    with patch("echolist.gui.messagebox.showerror") as err:
        assert not app._rename_playlist(pid, "CAFÉ")
    err.assert_called_once()


# ── CORE handoff 5 / 6: reorder retags through the writer; one TMP_PREFIX ──

def test_reorder_retags_through_writer_edit_in_place(app, gui_env):
    """Tag writes re-create macOS ._ sidecars on FAT32; edit_in_place drops
    them. _commit_reorder wrote the tags directly."""
    pid = _synced_playlist(app, gui_env, 2)
    app.staging.set_reorder(pid, [{"key": "c:2"}, {"key": "c:1"}])
    real = app.mgr.writer.edit_in_place
    edited = []

    def spy(rel, edit):
        edited.append(Path(rel).name)
        return real(rel, edit)

    with patch.object(app.mgr.writer, "edit_in_place", side_effect=spy):
        assert app._do_sync_blocking() == []
    assert sorted(edited) == ["01 - Song Two.flac", "02 - Song One.flac"]


def test_gui_does_not_redefine_tmp_prefix():
    """The parking prefix is defined once, in manager.py (rescan ignores it,
    recovery restores it); gui.py must import it, not keep a copy."""
    import inspect
    import re
    from echolist import manager
    assert not re.search(r"^TMP_PREFIX\s*=", inspect.getsource(gui), re.M), (
        "gui.py defines TMP_PREFIX; use `from .manager import TMP_PREFIX`")
    assert gui.TMP_PREFIX == manager.TMP_PREFIX
