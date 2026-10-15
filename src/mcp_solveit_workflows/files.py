"""Every filesystem touch the server makes.

Writes stay inside the output directory, checked after resolving symlinks,
and are atomic (temp file in the same directory, then rename). A manifest in
the output directory records the SHA-256 of each file the server wrote; a file
counts as server-created only while its bytes still match, so a builder export
saved over a server file is protected like any other file.

Reads are limited to allowed directories, ``.json`` files, and a size cap.
Error messages name the allowed directories and never quote file contents.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = ".solveit-workflows-manifest.json"
MAX_READ_BYTES = 4_000_000
MAX_SLUG_CHARS = 80
DEFAULT_STEM = "workflow"
JSON_SUFFIX = ".json"

_UNSAFE_SLUG_CHARS = re.compile(r"[^\w \-]")
_SPACES = re.compile(r" +")


class FileSafetyError(Exception):
    """A path or file was refused. Messages name directories, never file contents."""


@dataclass(frozen=True)
class WriteResult:
    path: Path
    replaced: bool
    # Problems that did not stop the write, such as a manifest that could not be updated.
    warnings: list[str] = field(default_factory=list)


def _slugify(title: str | None) -> str:
    text = (title or "").replace("\t", " ").replace("\n", " ")
    # \w is Unicode-aware, so accented letters survive; anything else goes.
    text = _UNSAFE_SLUG_CHARS.sub("", text).strip()
    text = _SPACES.sub("-", text)[:MAX_SLUG_CHARS].strip("-")
    return text or DEFAULT_STEM


def _is_inside(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)  # True for root itself


def _is_manifest_name(name: str) -> bool:
    return name.lower() == MANIFEST_NAME.lower()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write via a temp file in the target's directory and rename it into place.

    On any failure the temp file is removed and an existing target is untouched.
    """
    tmp = tempfile.NamedTemporaryFile(  # closed by the with block below
        delete=False, dir=target.parent, prefix=".tmp-", suffix=".part"
    )
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


class OutputDir:
    """The configured output directory. It is created on the first write."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def _resolved_root(self) -> Path:
        return self.root.resolve()

    def _refuse(self, why: str) -> FileSafetyError:
        return FileSafetyError(
            f"{why} Workflow files can only be written inside the output directory "
            f"{self._resolved_root}."
        )

    def _check_contained(self, path: Path) -> Path:
        root = self._resolved_root
        resolved = path.resolve()
        if not _is_inside(resolved, root) or resolved == root:
            raise self._refuse("That path is outside the output directory.")
        if _is_manifest_name(resolved.name):
            raise self._refuse("That name is reserved for the server's manifest.")
        return resolved

    def resolve_target(self, filename: str | None, title: str | None) -> Path:
        """The path a workflow should be written to, checked against the root."""
        if filename is None or not filename.strip():
            rel = Path(_slugify(title) + JSON_SUFFIX)
        else:
            raw = filename.strip()
            last = raw.replace("\\", "/").rsplit("/", 1)[-1]
            if last in ("", ".", ".."):
                raise self._refuse("Give a file name, not a directory.")
            rel = Path(raw)
            if rel.expanduser().is_absolute() or rel.is_absolute():
                raise self._refuse("Absolute paths are not allowed; give a file name.")
            if ".." in rel.parts:
                raise self._refuse("Paths may not contain '..'.")
            stem = (
                rel.name[: -len(JSON_SUFFIX)]
                if rel.name.lower().endswith(JSON_SUFFIX)
                else rel.name
            )
            if not stem.strip("."):
                raise self._refuse("The file name needs more than dots and an extension.")
            suffix = rel.suffix
            if not suffix:
                rel = rel.with_name(rel.name + JSON_SUFFIX)
            elif suffix.lower() != JSON_SUFFIX:
                raise self._refuse("Workflow files must have a .json extension.")
        if _is_manifest_name(rel.name):
            raise self._refuse("That name is reserved for the server's manifest.")
        return self._check_contained(self._resolved_root / rel)

    # ---- manifest ----

    @property
    def _manifest_path(self) -> Path:
        return self._resolved_root / MANIFEST_NAME

    def _manifest_key(self, path: Path) -> str:
        return path.resolve().relative_to(self._resolved_root).as_posix()

    def _load_manifest(self) -> dict[str, str]:
        """The recorded hashes, or an empty map if the manifest is missing or corrupt."""
        try:
            raw = self._manifest_path.read_bytes()
            if len(raw) > MAX_READ_BYTES:
                return {}
            data = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}

    def is_server_created(self, path: Path) -> bool:
        try:
            key = self._manifest_key(path)
            recorded = self._load_manifest().get(key)
            if recorded is None:
                return False
            return _sha256(path.read_bytes()) == recorded
        except (OSError, ValueError):
            return False

    # ---- writing ----

    def _free_name(self, target: Path) -> Path:
        n = 2
        while True:
            candidate = target.with_name(f"{target.stem}-{n}{target.suffix}")
            if not candidate.exists() and not candidate.is_symlink():
                return candidate
            n += 1

    def write(
        self,
        path: Path,
        content: str,
        *,
        overwrite: bool = False,
        allow_replace_server_file: bool = True,
    ) -> WriteResult:
        """Write ``content`` atomically at ``path`` (from resolve_target) or a suffixed name.

        An existing file is replaced only when ``overwrite`` is set, or when the
        server created it (and its bytes are unchanged) and that is allowed.
        Otherwise the smallest free ``<stem>-N.json`` is used.
        """
        target = self._check_contained(Path(path))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise self._cannot_write(exc) from None
        # mkdir may have followed a symlinked component; check again.
        target = self._check_contained(target)

        try:
            if target.exists() or target.is_symlink():
                if target.is_dir():
                    raise self._refuse("A directory already exists with that name.")
                keep = not (
                    overwrite or (allow_replace_server_file and self.is_server_created(target))
                )
                if keep:
                    target = self._check_contained(self._free_name(target))
            replaced = target.exists()
        except OSError as exc:
            # For example an output directory without search permission.
            raise self._cannot_write(exc) from None
        # The model layer has already checked the text encodes (lone surrogates are refused there).
        data = content.encode("utf-8")
        try:
            _atomic_write_bytes(target, data)
        except OSError as exc:
            raise self._cannot_write(exc) from None

        # Best effort: the workflow file is in place, so a manifest failure only
        # means the file will not count as server-created next time.
        warnings: list[str] = []
        manifest = self._load_manifest()
        manifest[self._manifest_key(target)] = _sha256(data)
        try:
            _atomic_write_bytes(
                self._manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            )
        except OSError as exc:
            warnings.append(
                f"The file was written, but the server's manifest in {self._resolved_root} "
                f"could not be updated ({exc.strerror or type(exc).__name__}), so a later "
                "write will not replace this file without overwrite."
            )
        return WriteResult(path=target, replaced=replaced, warnings=warnings)

    def _cannot_write(self, exc: OSError) -> FileSafetyError:
        reason = exc.strerror or type(exc).__name__
        return FileSafetyError(
            f"Cannot write to the output directory {self._resolved_root}: {reason}."
        )


def read_workflow_file(path: str | Path, allowed_dirs: Sequence[Path]) -> str:
    """Read a workflow file from an allowed directory, under the size cap."""
    roots = [Path(d).expanduser().resolve() for d in allowed_dirs]
    allowed_text = ", ".join(str(r) for r in roots) or "(none configured)"
    resolved = Path(path).expanduser().resolve()

    if not any(_is_inside(resolved, r) for r in roots):
        raise FileSafetyError(
            f"Workflow files can only be read from these directories: {allowed_text}."
        )
    if resolved.suffix.lower() != JSON_SUFFIX:
        raise FileSafetyError("Only .json workflow files can be read.")
    try:
        info = resolved.stat()
    except FileNotFoundError:
        raise FileSafetyError(f"No such file: {resolved}") from None
    except OSError as exc:
        raise FileSafetyError(f"Cannot read {resolved}: {exc.strerror}") from None
    if not stat.S_ISREG(info.st_mode):
        raise FileSafetyError(f"Not a regular file: {resolved}")
    if info.st_size > MAX_READ_BYTES:
        raise FileSafetyError(
            f"The file is {info.st_size:,} bytes, over the {MAX_READ_BYTES:,} byte limit."
        )
    try:
        with resolved.open("rb") as fh:
            data = fh.read(MAX_READ_BYTES + 1)
    except OSError as exc:
        raise FileSafetyError(f"Cannot read {resolved}: {exc.strerror}") from None
    if len(data) > MAX_READ_BYTES:
        raise FileSafetyError(f"The file is over the {MAX_READ_BYTES:,} byte limit.")
    try:
        # utf-8-sig drops a byte order mark, which some editors add on Windows.
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FileSafetyError(f"The file {resolved} is not valid UTF-8 text.") from None
