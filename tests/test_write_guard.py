"""Ensure all filesystem mutations go through SafeWriter.

Scans every module in the echolist package (except safe_write.py itself) for
direct filesystem-write calls that bypass SafeWriter.

Rules:
  - Calls on a SafeWriter (a receiver named ``writer`` / ``_writer``, e.g.
    ``self.mgr.writer.rename(...)``) are the approved pattern.
  - ``.save()`` on the Store / Config / StagingState wrappers is allowed
    (they write through SafeWriter or to ~/.echolist). mutagen ``.save()``
    is only allowed in tags.py.
  - Everything else needs an entry in ALLOWED_CALLS, keyed by
    (file, function) — never by file alone — with the reason. Entries that
    no longer match a call fail the ratchet test, so removed cases cannot
    quietly come back.

If an AI is reading this, please audit the implications of your change on:
  1. Possibility of changing the source files (the music library the user
     is syncing FROM). Only the destination workspace should ever be mutated.
  2. Possibility that this change loosens the verification chain — adding an
     exception here means the static guard will no longer catch that call
     pattern in that function, even if a future edit uses it unsafely.
Workspace writes belong in SafeWriter (echolist/safe_write.py); route new
ones through ``writer.makedirs / write_text / copy_in / rename / delete``.
"""

import ast
from pathlib import Path

ECHOLIST_ROOT = Path(__file__).resolve().parent.parent / "echolist"

# Calls that directly mutate the filesystem (Path methods)
FORBIDDEN_ATTRS = {
    "write_text", "write_bytes", "unlink", "rmdir", "mkdir", "rename",
    "replace", "touch", "symlink_to", "hardlink_to", "chmod",
    "save",
}

# Module-level functions that mutate the filesystem
FORBIDDEN_FUNCS = {
    "shutil.copy", "shutil.copy2", "shutil.copytree", "shutil.move",
    "shutil.rmtree", "os.remove", "os.unlink", "os.rmdir", "os.rename",
    "os.replace", "os.makedirs", "os.mkdir",
}

# Files with a blanket pass (they ARE the write infrastructure)
SKIP_FILES = {"safe_write.py", "__init__.py", "__main__.py"}

# Receivers that are SafeWriter instances.
SAFEWRITER_NAMES = {"writer", "_writer"}
# Receivers whose .save() is a wrapper (Store/Config -> SafeWriter,
# StagingState -> ~/.echolist), not mutagen.
WRAPPER_SAVE_NAMES = {"store", "config", "staging"}
# Classes whose own ``self.save()`` is the wrapper save above.
WRAPPER_CLASSES = {"StagingState", "Config", "Store"}

# (file, function qualname) -> {attr or func: reason}
ALLOWED_CALLS = {
    # ── tags.py: the only module allowed to call mutagen .save() ──
    ("tags.py", "restore_tags"): {"save": "mutagen save on a workspace copy"},
    ("tags.py", "_tag_flac"): {"save": "mutagen save on a workspace copy"},
    ("tags.py", "_tag_easy"): {"save": "mutagen save on a workspace copy"},
    ("tags.py", "_tag_id3_chunk"): {"save": "mutagen save on a workspace copy"},
    # ── config.py: ~/.echolist/backups (user-local, not the workspace) ──
    ("config.py", "save_backup"): {"mkdir": "~/.echolist/backups/<wid>/<pid>"},
    ("config.py", "rename_backup_pid"): {
        "mkdir": "~/.echolist backups dir", "unlink": "moved backup file",
        "rename": "move backup file between pid dirs",
    },
    ("config.py", "_remove_empty_dirs"): {"rmdir": "empty ~/.echolist backup dirs"},
    ("config.py", "delete_all_backups"): {"unlink": "user-requested permanent delete of backups"},
    ("config.py", "save_playlist_snapshot"): {"mkdir": "~/.echolist/backups/<wid>"},
    # ── journal.py: ~/.echolist/backups/<wid>/sync_journal.json ──
    ("journal.py", "SyncJournal.discard"): {"unlink": "remove finished journal (local state)"},
    # ── gui.py: ~/.echolist local state ──
    ("gui.py", "StagingState.clear"): {"unlink": "remove pending.json (local state)"},
    ("gui.py", "_bind_legacy_staging"): {"unlink": "legacy ~/.echolist/pending.json after migration"},
    ("gui.py", "App._bind_workspace_state"): {"rename": "set aside an unreadable pending.json"},
    ("gui.py", "_resolve_source_file"): {"replace": "str.replace on a path string"},
    # ── manager.py: KNOWN workspace writes that bypass SafeWriter. Owned by
    # the manager area; to be routed through writer.makedirs (see PR notes).
    # Do not add more entries like these.
    ("manager.py", "PlaylistManager._acquire_lock"): {
        "mkdir": "TODO(manager): <root>/.echolist for workspace.lock; use writer.makedirs",
    },
    ("manager.py", "PlaylistManager._ensure_workspace"): {
        "mkdir": "TODO(manager): creates writer.root itself before SafeWriter can write",
    },
}

ALLOWED_OPEN = {
    ("manager.py", "_blake2b"): "read-only hashing (mode 'rb')",
}


def _receiver_name(node):
    """Last name in a call receiver: ``self.mgr.writer`` -> "writer"."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


class _Collector(ast.NodeVisitor):
    def __init__(self, fname):
        self.fname = fname
        self.stack = []
        self.violations = []
        self.used = set()

    def _scope(self):
        return ".".join(self.stack)

    def _visit_scope(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _visit_scope

    def _allowed(self, what):
        # Nested helpers (closures) inherit their enclosing function's entry.
        for depth in range(len(self.stack), 0, -1):
            key = (self.fname, ".".join(self.stack[:depth]))
            if what in ALLOWED_CALLS.get(key, {}):
                self.used.add((key, what))
                return True
        return False

    def _report(self, node, what):
        self.violations.append(
            f"{self.fname}:{node.lineno} in {self._scope() or '<module>'}() calls {what}"
        )

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            recv = _receiver_name(func.value)
            if func.attr in FORBIDDEN_ATTRS:
                ok = (recv in SAFEWRITER_NAMES
                      or (func.attr == "save" and recv in WRAPPER_SAVE_NAMES)
                      or (func.attr == "save" and recv == "self"
                          and self.stack and self.stack[0] in WRAPPER_CLASSES)
                      or self._allowed(func.attr))
                if not ok:
                    self._report(node, f".{func.attr}()")
            if isinstance(func.value, ast.Name):
                full = f"{func.value.id}.{func.attr}"
                if full in FORBIDDEN_FUNCS and not self._allowed(full):
                    self._report(node, f"{full}()")
        if isinstance(func, ast.Name) and func.id == "open":
            key = None
            for depth in range(len(self.stack), 0, -1):
                k = (self.fname, ".".join(self.stack[:depth]))
                if k in ALLOWED_OPEN:
                    key = k
                    break
            if key is None:
                self._report(node, "open()")
            else:
                self.used.add((key, "open"))
        self.generic_visit(node)


def _collect(filepath: Path) -> _Collector:
    source = filepath.read_text(encoding="utf-8")
    c = _Collector(filepath.name)
    c.visit(ast.parse(source, filename=str(filepath)))
    return c


def _collect_violations(filepath: Path) -> list[str]:
    return _collect(filepath).violations


def _package_files():
    return [p for p in sorted(ECHOLIST_ROOT.glob("*.py")) if p.name not in SKIP_FILES]


def test_no_direct_writes_outside_safewriter():
    violations = []
    for py_file in _package_files():
        violations.extend(_collect_violations(py_file))

    assert not violations, (
        "Direct filesystem writes found outside safe_write.py:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\nWorkspace writes must go through SafeWriter (self.writer / "
        "self.mgr.writer: makedirs, write_text, copy_in, rename, delete). "
        "Writes to ~/.echolist local state may be allowlisted per function "
        "in ALLOWED_CALLS in tests/test_write_guard.py, with a reason."
    )


def test_allowlist_has_no_stale_entries():
    """Ratchet: an exemption that matches no call must be deleted."""
    used = set()
    for py_file in _package_files():
        used |= _collect(py_file).used
    stale = [
        f"{key[0]}:{key[1]} -> {what}"
        for key, whats in ALLOWED_CALLS.items() for what in whats
        if (key, what) not in used
    ] + [f"{key[0]}:{key[1]} -> open" for key in ALLOWED_OPEN if (key, "open") not in used]
    assert not stale, (
        "Unused write-guard exemptions (the call they allowed is gone); "
        "remove them from tests/test_write_guard.py:\n"
        + "\n".join(f"  - {s}" for s in stale)
    )


def test_guard_catches_direct_workspace_writes_in_gui(tmp_path):
    """#20: gui.py had a blanket pass for mkdir/unlink/rename, so direct writes
    inside writer.root (bypassing SafeWriter) went unnoticed."""
    fake = tmp_path / "gui.py"
    fake.write_text(
        "def onload(self, folder):\n"
        "    (self.mgr.writer.root / folder).mkdir(parents=True)\n"
        "def cleanup(root):\n"
        "    for f in root.rglob('x'):\n"
        "        f.unlink()\n"
        "def move(a, b):\n"
        "    a.rename(b)\n",
        encoding="utf-8",
    )
    found = _collect_violations(fake)
    assert len(found) == 3, found


def test_guard_allows_safewriter_calls(tmp_path):
    fake = tmp_path / "gui.py"
    fake.write_text(
        "def ok(self, a, b):\n"
        "    self.mgr.writer.rename(a, b)\n"
        "    self.mgr.writer.makedirs(a)\n"
        "    self.mgr.store.save()\n",
        encoding="utf-8",
    )
    assert _collect_violations(fake) == []
