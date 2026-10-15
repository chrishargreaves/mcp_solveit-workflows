"""Read-only index over a local SOLVE-IT knowledge base checkout.

The knowledge base is loaded by reading its per-item JSON files
(``data/techniques``, ``data/weaknesses``, ``data/mitigations``). No code from
the checkout is imported or run, and the revision is read from the ``.git``
files directly rather than by running ``git`` there.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
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
        """Search techniques word by word, best matches first.

        Modelled on the SOLVE-IT library's search (solve_it_library `search`): the
        query is split into words, with "quoted phrases" kept whole and common
        words dropped. Each word is matched at the start of a word in the
        technique's text, after trimming common endings, so "imaging" finds
        "image" and "verification" finds "verify". A match in the name counts
        most, then synonyms, description, details and example tools.

        Techniques that match every word come first, then those matching most
        words; within that, by score, then ID. Each match carries a short
        description and the technique's CASE input and output classes, so it can
        be chosen without a separate lookup.
        """
        terms, phrases = _parse_query(query)
        if (not terms and not phrases) or limit <= 0:
            return []
        whole = " ".join(_WORD.findall(query.casefold()))
        wanted = len(terms) + len(phrases)
        ranked: list[tuple[int, float, str]] = []
        for tid, item in self._techniques.items():
            fields = _search_fields(item)
            matched = 0
            score = 0.0
            for term in terms:
                best = max((w * _term_fit(term, words) for w, words, _text in fields), default=0.0)
                if best:
                    matched += 1
                    score += best
            for phrase in phrases:
                best = max((w for w, _words, text in fields if phrase in text), default=0.0)
                if best:
                    matched += 1
                    score += 2 * best
            if not matched:
                continue
            # The whole query as the name or a synonym, or inside one.
            names = [
                fields[0][2],
                *(" ".join(_WORD.findall(s.casefold())) for s in _synonyms(item)),
            ]
            if whole and whole in names:
                score += 100
            elif whole and any(whole in n for n in names):
                score += 30
            ranked.append((-matched, -score, tid))
        ranked.sort()
        return [self._match(tid, wanted, -m) for m, _s, tid in ranked[:limit]]

    def _match(self, tid: str, wanted: int, matched: int) -> dict[str, Any]:
        item = self._techniques[tid]
        desc = " ".join(str(item.get("description") or "").split())
        if len(desc) > MATCH_DESCRIPTION_CHARS:
            desc = desc[: MATCH_DESCRIPTION_CHARS - 1].rstrip() + "\u2026"
        return {
            "id": tid,
            "name": item.get("name", ""),
            "parent": self.parent_of(tid),
            "description": desc,
            "inputs": _class_names(item.get("CASE_input_classes")),
            "outputs": _class_names(item.get("CASE_output_classes")),
            "matched_all_words": matched == wanted,
        }


# ---- technique search ----

# Shortened to this many characters in a search match.
MATCH_DESCRIPTION_CHARS = 200
# Field weights: a word matching in the name counts most.
_FIELD_WEIGHTS = (
    ("name", 10.0),
    ("synonyms", 8.0),
    ("description", 3.0),
    ("details", 1.0),
    ("examples", 1.0),
)
_WORD = re.compile(r"[0-9a-z]+")
_PHRASE = re.compile(r'"([^"]+)"')
# The library's stop words, plus a few common in requests for a workflow.
_STOP_WORDS = frozenset(
    "a an and are is not or the was were of to for from in on into with by via "
    "using use do does how what which".split()
)
# Common endings trimmed before matching; the longest that applies is used.
_ENDINGS = ("ication", "ation", "ition", "ings", "ing", "ion", "ure", "ers", "er", "ed", "es", "s")
# A query word that only starts a text word counts for this share of the field weight.
PREFIX_FIT = 0.8


def _stem(word: str) -> str:
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)]
    return word


def _parse_query(query: str) -> tuple[list[str], list[str]]:
    """Stemmed words and quoted phrases (each normalised to single spaces)."""
    q = query.casefold()
    phrases = [" ".join(_WORD.findall(p)) for p in _PHRASE.findall(q)]
    rest = _PHRASE.sub(" ", q)
    words = [w for w in _WORD.findall(rest) if w not in _STOP_WORDS]
    terms = list(dict.fromkeys(_stem(w) for w in words))
    return terms, [p for p in dict.fromkeys(phrases) if p]


def _text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value if isinstance(v, str))
    return str(value) if isinstance(value, str) else ""


def _search_fields(item: Mapping[str, Any]) -> list[tuple[float, list[str], str]]:
    """(weight, words, normalised text) for each searched field, name first."""
    out = []
    for key, weight in _FIELD_WEIGHTS:
        words = _WORD.findall(_text(item.get(key)).casefold())
        out.append((weight, words, " ".join(words)))
    return out


def _term_fit(term: str, words: Sequence[str]) -> float:
    """1 when a text word, or its stem, is the query word; PREFIX_FIT when one only
    starts with it; 0 for no match."""
    fit = 0.0
    for w in words:
        stem = _stem(w)
        if term in (w, stem):
            return 1.0
        if w.startswith(term) or stem.startswith(term):
            fit = PREFIX_FIT
    return fit


def _synonyms(item: Mapping[str, Any]) -> list[str]:
    value = item.get("synonyms")
    return [s for s in value if isinstance(s, str)] if isinstance(value, list) else []


def _class_names(value: object) -> list[str]:
    """CASE class IRIs, shortened to their last path segment."""
    if not isinstance(value, list):
        return []
    return [v.rstrip("/").rsplit("/", 1)[-1] for v in value if isinstance(v, str) and v]
