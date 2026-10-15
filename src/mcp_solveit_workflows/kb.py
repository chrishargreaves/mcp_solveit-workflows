"""Read-only index over a local SOLVE-IT knowledge base checkout.

The knowledge base is loaded by reading its per-item JSON files
(``data/techniques``, ``data/weaknesses``, ``data/mitigations``). No code from
the checkout is imported or run, and the revision is read from the ``.git``
files directly rather than by running ``git`` there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ASTM weakness categories, keyed by the strings used in a weakness's
# "categories" list. Values are the short labels the Workflow Builder shows in
# its compile panel. Copied from the builder's ASTM_CATEGORIES table:
# solveit-workflow-builder, origin/main:src/lib/constants.ts (which takes them
# from solve-it/reporting_scripts/generate_evaluation.py). Keep in step.
ASTM_LABELS: dict[str, str] = {
    "ASTM_INCOMP": "INCOMP",
    "ASTM_INAC_EX": "INAC-EX",
    "ASTM_INAC_AS": "INAC-AS",
    "ASTM_INAC_ALT": "INAC-ALT",
    "ASTM_INAC_COR": "INAC-COR",
    "ASTM_MISINT": "MISINT",
}

UNKNOWN_REVISION = "unknown"

_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class KnowledgeBaseError(Exception):
    """The knowledge base checkout cannot be used at all."""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _as_sha(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    return value if _SHA_RE.match(value) else None


def _git_dirs(root: Path) -> tuple[Path, Path] | None:
    """Return (git dir holding HEAD, common dir holding refs), or None."""
    dot_git = root / ".git"
    if dot_git.is_dir():
        git_dir = dot_git
    elif dot_git.is_file():
        content = _read_text(dot_git)
        if content is None or not content.startswith("gitdir:"):
            return None
        target = Path(content[len("gitdir:") :].strip())
        git_dir = target if target.is_absolute() else root / target
        if not git_dir.is_dir():
            return None
    else:
        return None

    common_dir = git_dir
    commondir = _read_text(git_dir / "commondir")
    if commondir:
        candidate = Path(commondir)
        common_dir = candidate if candidate.is_absolute() else git_dir / candidate
    return git_dir, common_dir


def _packed_ref(common_dir: Path, ref: str) -> str | None:
    content = _read_text(common_dir / "packed-refs")
    if content is None:
        return None
    for line in content.splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return _as_sha(parts[0])
    return None


def read_revision(root: Path) -> str:
    """Read the checked-out commit of ``root`` without running git."""
    dirs = _git_dirs(root)
    if dirs is None:
        return UNKNOWN_REVISION
    git_dir, common_dir = dirs
    head = _read_text(git_dir / "HEAD")
    if head is None:
        return UNKNOWN_REVISION
    if head.startswith("ref:"):
        ref = head[len("ref:") :].strip()
        if not ref.startswith("refs/") or ".." in ref.split("/"):
            return UNKNOWN_REVISION
        sha = None
        for base in dict.fromkeys([git_dir, common_dir]):
            sha = _as_sha(_read_text(base / ref))
            if sha:
                break
        sha = sha or _packed_ref(common_dir, ref)
        return sha or UNKNOWN_REVISION
    return _as_sha(head) or UNKNOWN_REVISION


def _load_items(directory: Path, warnings: list[str]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        warnings.append(f"{directory.name}: directory not found at {directory}")
        return items
    for path in sorted(directory.glob("*.json")):
        label = f"{directory.name}/{path.name}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            warnings.append(f"{label}: skipped, could not be read as JSON ({exc})")
            continue
        if not isinstance(data, dict):
            warnings.append(f"{label}: skipped, not a JSON object")
            continue
        item_id = data.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            warnings.append(f"{label}: skipped, no 'id' field")
            continue
        if item_id in items:
            warnings.append(f"{label}: skipped, duplicate id {item_id}")
            continue
        items[item_id] = data
    return items


class KnowledgeBase:
    """Techniques, weaknesses and mitigations from one SOLVE-IT checkout."""

    def __init__(
        self,
        techniques: dict[str, dict[str, Any]],
        weaknesses: dict[str, dict[str, Any]],
        mitigations: dict[str, dict[str, Any]],
        revision: str = UNKNOWN_REVISION,
        load_warnings: list[str] | None = None,
    ) -> None:
        self._techniques = techniques
        self._weaknesses = weaknesses
        self._mitigations = mitigations
        self.revision = revision
        self.load_warnings: list[str] = list(load_warnings or [])
        self._parents: dict[str, str] = {}
        for tid in sorted(techniques):
            subs = techniques[tid].get("subtechniques") or []
            if not isinstance(subs, list):
                continue
            for sub in subs:
                if isinstance(sub, str):
                    self._parents.setdefault(sub, tid)

    @classmethod
    def load(cls, root: Path) -> KnowledgeBase:
        root = Path(root)
        data = root / "data"
        if not (data / "techniques").is_dir():
            raise KnowledgeBaseError(
                f"No SOLVE-IT knowledge base at {root}: {data / 'techniques'} is missing"
            )
        warnings: list[str] = []
        techniques = _load_items(data / "techniques", warnings)
        weaknesses = _load_items(data / "weaknesses", warnings)
        mitigations = _load_items(data / "mitigations", warnings)
        return cls(techniques, weaknesses, mitigations, read_revision(root), warnings)

    def counts(self) -> dict[str, int]:
        return {
            "techniques": len(self._techniques),
            "weaknesses": len(self._weaknesses),
            "mitigations": len(self._mitigations),
        }

    def technique(self, tid: str) -> dict[str, Any] | None:
        return self._techniques.get(tid)

    def weakness(self, wid: str) -> dict[str, Any] | None:
        return self._weaknesses.get(wid)

    def mitigation(self, mid: str) -> dict[str, Any] | None:
        return self._mitigations.get(mid)

    def parent_of(self, tid: str) -> str | None:
        return self._parents.get(tid)

    def is_subtechnique_of(self, sub: str, parent: str) -> bool:
        return self._parents.get(sub) == parent

    def find_techniques(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Case-insensitive search on technique names and synonyms.

        Ranked: exact name, name prefix, name substring, synonym substring;
        ties broken by ID.
        """
        q = query.strip().casefold()
        if not q or limit <= 0:
            return []
        ranked: list[tuple[int, str]] = []
        for tid, item in self._techniques.items():
            name = str(item.get("name") or "").casefold()
            if name == q:
                rank = 0
            elif name.startswith(q):
                rank = 1
            elif q in name:
                rank = 2
            elif any(isinstance(s, str) and q in s.casefold() for s in item.get("synonyms") or []):
                rank = 3
            else:
                continue
            ranked.append((rank, tid))
        ranked.sort()
        return [
            {
                "id": tid,
                "name": self._techniques[tid].get("name", ""),
                "parent": self.parent_of(tid),
            }
            for _, tid in ranked[:limit]
        ]
