"""File safety: output containment, collision rules, atomic writes, safe reads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from mcp_solveit_workflows.files import (
    MANIFEST_NAME,
    MAX_READ_BYTES,
    FileSafetyError,
    OutputDir,
    read_workflow_file,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def out(root: Path) -> OutputDir:
    return OutputDir(root)


def listing(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


# ---- resolve_target ----


def test_root_created_lazily(root: Path, out: OutputDir) -> None:
    assert not root.exists()
    out.write(out.resolve_target("a.json", None), "{}")
    assert (root / "a.json").read_text() == "{}"


def test_title_slug(out: OutputDir, root: Path) -> None:
    assert out.resolve_target(None, "Copy files by name!") == root.resolve() / (
        "Copy-files-by-name.json"
    )
    assert out.resolve_target(None, "  a   b__c-d  ").name == "a-b__c-d.json"
    assert out.resolve_target(None, "../../etc/passwd").name == "etcpasswd.json"


def test_title_slug_fallback_and_length(out: OutputDir) -> None:
    assert out.resolve_target(None, None).name == "workflow.json"
    assert out.resolve_target(None, "!!!///").name == "workflow.json"
    assert out.resolve_target(None, "").name == "workflow.json"
    long = out.resolve_target(None, "x" * 300).name
    assert long == "x" * 80 + ".json"


def test_json_suffix_added(out: OutputDir) -> None:
    assert out.resolve_target("plan", None).name == "plan.json"
    assert out.resolve_target("plan.JSON", None).name == "plan.JSON"


def test_other_extension_refused(out: OutputDir) -> None:
    with pytest.raises(FileSafetyError):
        out.resolve_target("plan.txt", None)


def test_subdirectory_inside_root_allowed(out: OutputDir, root: Path) -> None:
    assert out.resolve_target("sub/plan.json", None) == root.resolve() / "sub" / "plan.json"


@pytest.mark.parametrize("name", ["../../x.json", "sub/../../x.json", "..", "a/../b.json"])
def test_dotdot_refused(out: OutputDir, root: Path, name: str) -> None:
    with pytest.raises(FileSafetyError) as exc:
        out.resolve_target(name, None)
    assert str(root.resolve()) in str(exc.value)
    assert not root.exists()


def test_absolute_outside_refused(out: OutputDir, root: Path, tmp_path: Path) -> None:
    for name in [str(tmp_path / "x.json"), "/etc/x.json", "~/.zshrc"]:
        with pytest.raises(FileSafetyError) as exc:
            out.resolve_target(name, None)
        assert str(root.resolve()) in str(exc.value)
    assert not root.exists()


def test_absolute_inside_root_refused(out: OutputDir, root: Path) -> None:
    # Absolute paths are refused outright, even when they point inside the root.
    with pytest.raises(FileSafetyError):
        out.resolve_target(str(root / "x.json"), None)


def test_symlink_inside_root_pointing_out_refused(
    out: OutputDir, root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "file.json").symlink_to(outside / "target.json")
    for name in ["link/x.json", "file.json"]:
        with pytest.raises(FileSafetyError) as exc:
            out.resolve_target(name, None)
        assert str(root.resolve()) in str(exc.value)
    assert listing(outside) == []


def test_write_rechecks_containment(out: OutputDir, tmp_path: Path) -> None:
    target = tmp_path / "escape.json"
    with pytest.raises(FileSafetyError):
        out.write(target, "{}")
    assert not target.exists()


@pytest.mark.parametrize("name", [MANIFEST_NAME, MANIFEST_NAME.upper(), "sub/" + MANIFEST_NAME])
def test_manifest_name_refused(out: OutputDir, name: str) -> None:
    with pytest.raises(FileSafetyError):
        out.resolve_target(name, None)


def test_title_cannot_produce_manifest_name(out: OutputDir) -> None:
    # The slug drops the leading dot, so a title can never collide with the manifest.
    name = out.resolve_target(None, MANIFEST_NAME.removesuffix(".json")).name
    assert name.lower() != MANIFEST_NAME.lower()


# ---- write / collision rules ----


def test_write_records_manifest(out: OutputDir, root: Path) -> None:
    result = out.write(out.resolve_target("a.json", None), '{"x":1}')
    assert result.path == root.resolve() / "a.json"
    assert result.replaced is False
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    assert manifest["a.json"] == hashlib.sha256(b'{"x":1}').hexdigest()
    assert out.is_server_created(result.path)


def test_server_file_is_replaced(out: OutputDir, root: Path) -> None:
    target = out.resolve_target("a.json", None)
    out.write(target, "{}")
    result = out.write(target, '{"v":2}')
    assert result.path == target
    assert result.replaced is True
    assert target.read_text() == '{"v":2}'
    assert listing(root) == sorted([MANIFEST_NAME, "a.json"])


def test_server_file_not_replaced_when_disallowed(out: OutputDir) -> None:
    target = out.resolve_target("a.json", None)
    out.write(target, "{}")
    result = out.write(target, "[]", allow_replace_server_file=False)
    assert result.path.name == "a-2.json"
    assert target.read_text() == "{}"


def test_foreign_file_gets_suffix(out: OutputDir, root: Path) -> None:
    root.mkdir()
    (root / "a.json").write_text("mine")
    (root / "a-2.json").write_text("also mine")
    target = out.resolve_target("a.json", None)
    result = out.write(target, "{}")
    assert result.path == root.resolve() / "a-3.json"
    assert result.replaced is False
    assert (root / "a.json").read_text() == "mine"
    assert (root / "a-2.json").read_text() == "also mine"
    assert out.is_server_created(result.path)
    assert not out.is_server_created(root / "a.json")


def test_foreign_file_overwritten_with_flag(out: OutputDir, root: Path) -> None:
    root.mkdir()
    (root / "a.json").write_text("mine")
    result = out.write(out.resolve_target("a.json", None), "{}", overwrite=True)
    assert result.path == root.resolve() / "a.json"
    assert result.replaced is True
    assert (root / "a.json").read_text() == "{}"


def test_builder_export_over_server_file_is_protected(out: OutputDir, root: Path) -> None:
    target = out.resolve_target("a.json", None)
    out.write(target, "{}")
    target.write_text('{"edited":"in builder"}')  # hash no longer matches
    assert not out.is_server_created(target)
    result = out.write(target, '{"new":1}')
    assert result.path.name == "a-2.json"
    assert target.read_text() == '{"edited":"in builder"}'


def test_interrupted_write_leaves_original(
    out: OutputDir, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root.mkdir()
    (root / "a.json").write_text("original")

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk unplugged")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(FileSafetyError, match="output directory"):
        out.write(out.resolve_target("a.json", None), "{}", overwrite=True)
    assert (root / "a.json").read_text() == "original"
    assert listing(root) == ["a.json"]


def test_missing_manifest_nothing_server_created(out: OutputDir, root: Path) -> None:
    target = out.resolve_target("a.json", None)
    out.write(target, "{}")
    (root / MANIFEST_NAME).unlink()
    assert not out.is_server_created(target)


@pytest.mark.parametrize("bad", ["not json", "[1,2]", '{"a.json": 5}', ""])
def test_corrupt_manifest(out: OutputDir, root: Path, bad: str) -> None:
    target = out.resolve_target("a.json", None)
    out.write(target, "{}")
    (root / MANIFEST_NAME).write_text(bad)
    assert not out.is_server_created(target)
    # A further write recovers: the foreign-looking file is kept and a fresh manifest written.
    result = out.write(target, "[]")
    assert result.path.name == "a-2.json"
    assert out.is_server_created(result.path)
    assert json.loads((root / MANIFEST_NAME).read_text()) == {
        "a-2.json": hashlib.sha256(b"[]").hexdigest()
    }


def test_unicode_content_written_utf8(out: OutputDir) -> None:
    result = out.write(out.resolve_target("u.json", None), '{"t":"日本語"}')
    assert result.path.read_bytes() == '{"t":"日本語"}'.encode()
    assert out.is_server_created(result.path)


# ---- read_workflow_file ----


def test_read_inside_allowed(tmp_path: Path) -> None:
    (tmp_path / "w.json").write_text('{"ok":true}', encoding="utf-8")
    assert read_workflow_file(tmp_path / "w.json", [tmp_path]) == '{"ok":true}'
    assert read_workflow_file(str(tmp_path / "w.json"), [tmp_path / "x", tmp_path]) != ""


def test_read_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "w.json").write_text("{}")
    assert read_workflow_file("~/w.json", [tmp_path]) == "{}"


def test_read_txt_refused(tmp_path: Path) -> None:
    (tmp_path / "w.txt").write_text("SECRET")
    with pytest.raises(FileSafetyError) as exc:
        read_workflow_file(tmp_path / "w.txt", [tmp_path])
    assert "SECRET" not in str(exc.value)


def test_read_outside_roots_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "w.json").write_text("SECRET")
    for p in [tmp_path / "w.json", allowed / ".." / "w.json"]:
        with pytest.raises(FileSafetyError) as exc:
            read_workflow_file(p, [allowed])
        assert str(allowed.resolve()) in str(exc.value)
        assert "SECRET" not in str(exc.value)


def test_read_symlink_pointing_out_refused(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (tmp_path / "w.json").write_text("SECRET")
    (allowed / "w.json").symlink_to(tmp_path / "w.json")
    with pytest.raises(FileSafetyError) as exc:
        read_workflow_file(allowed / "w.json", [allowed])
    assert "SECRET" not in str(exc.value)


def test_read_over_cap_refused(tmp_path: Path) -> None:
    p = tmp_path / "big.json"
    p.write_bytes(b"SECRET" + b" " * MAX_READ_BYTES)
    with pytest.raises(FileSafetyError) as exc:
        read_workflow_file(p, [tmp_path])
    assert "SECRET" not in str(exc.value)
    assert "4000000" in str(exc.value) or "4,000,000" in str(exc.value)


def test_read_at_cap_allowed(tmp_path: Path) -> None:
    p = tmp_path / "big.json"
    p.write_bytes(b" " * MAX_READ_BYTES)
    assert len(read_workflow_file(p, [tmp_path])) == MAX_READ_BYTES


def test_read_directory_and_missing_refused(tmp_path: Path) -> None:
    (tmp_path / "d.json").mkdir()
    with pytest.raises(FileSafetyError):
        read_workflow_file(tmp_path / "d.json", [tmp_path])
    with pytest.raises(FileSafetyError):
        read_workflow_file(tmp_path / "missing.json", [tmp_path])


def test_read_bad_utf8_refused(tmp_path: Path) -> None:
    (tmp_path / "w.json").write_bytes(b'{"SECRET":"\xff\xfe"}')
    with pytest.raises(FileSafetyError) as exc:
        read_workflow_file(tmp_path / "w.json", [tmp_path])
    assert "SECRET" not in str(exc.value)


def test_read_server_written_file(out: OutputDir, root: Path) -> None:
    result = out.write(out.resolve_target("a.json", None), "{}")
    assert read_workflow_file(result.path, [root]) == "{}"
    assert os.path.exists(root / MANIFEST_NAME)


# ---- review fixes ----


@pytest.mark.parametrize(
    "name", [".", "./", "sub/", "sub/.", ".json", "..json", "sub/.json", "...", ""]
)
def test_empty_or_dot_names_refused(out: OutputDir, root: Path, name: str) -> None:
    if not name:
        # An empty filename means "use the title", which is allowed.
        assert out.resolve_target(name, "T").name == "T.json"
        return
    with pytest.raises(FileSafetyError):
        out.resolve_target(name, None)
    assert not root.exists()


def test_read_bom_prefixed_file(tmp_path: Path) -> None:
    (tmp_path / "w.json").write_bytes(b'\xef\xbb\xbf{"version": 2}')
    assert read_workflow_file(tmp_path / "w.json", [tmp_path]) == '{"version": 2}'


def test_write_to_manifest_path_refused(out: OutputDir, root: Path) -> None:
    root.mkdir()
    with pytest.raises(FileSafetyError):
        out.write(root / MANIFEST_NAME, "{}")
    assert listing(root) == []


def test_oversized_manifest_treated_as_empty(out: OutputDir, root: Path) -> None:
    target = out.resolve_target("a.json", None)
    out.write(target, "{}")
    manifest = json.loads((root / MANIFEST_NAME).read_text())
    padding = " " * (MAX_READ_BYTES + 1)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest) + padding)
    assert not out.is_server_created(target)


def test_unwritable_output_dir_is_a_safety_error(tmp_path: Path) -> None:
    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        out = OutputDir(parent / "out")
        with pytest.raises(FileSafetyError) as exc:
            out.write(out.resolve_target("a.json", None), "{}")
        msg = str(exc.value)
        assert str((parent / "out").resolve()) in msg
        assert ".tmp-" not in msg and ".part" not in msg
    finally:
        parent.chmod(0o700)


def test_manifest_failure_keeps_file_and_warns(
    out: OutputDir, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mcp_solveit_workflows.files as files_mod

    real = files_mod._atomic_write_bytes

    def fail_manifest(target: Path, data: bytes) -> None:
        if target.name == MANIFEST_NAME:
            raise PermissionError(13, "Permission denied")
        real(target, data)

    monkeypatch.setattr(files_mod, "_atomic_write_bytes", fail_manifest)
    result = out.write(out.resolve_target("a.json", None), "{}")
    assert result.path.read_text() == "{}"
    assert result.warnings and "manifest" in result.warnings[0]
