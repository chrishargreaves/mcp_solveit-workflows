"""Tests for the knowledge base index."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURE_KB

from mcp_solveit_workflows.kb import ASTM_LABELS, KnowledgeBase, KnowledgeBaseError

SHA_A = "a" * 40
SHA_B = "b" * 40


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    shutil.copytree(FIXTURE_KB, root)
    return root


# --- loading -----------------------------------------------------------------


def test_counts(kb: KnowledgeBase) -> None:
    assert kb.counts() == {"techniques": 8, "weaknesses": 6, "mitigations": 4}
    assert kb.load_warnings == []


def test_technique_lookup_returns_name_and_weaknesses(kb: KnowledgeBase) -> None:
    t = kb.technique("DFT-1001")
    assert t is not None
    assert t["name"] == "Acquire storage media"
    assert t["weaknesses"] == ["DFW-1001", "DFW-1002"]


def test_unknown_ids_return_none(kb: KnowledgeBase) -> None:
    assert kb.technique("DFT-9999") is None
    assert kb.weakness("DFW-1999") is None
    assert kb.mitigation("DFM-9999") is None


def test_weakness_and_mitigation_lookup(kb: KnowledgeBase) -> None:
    w = kb.weakness("DFW-1003")
    assert w is not None
    assert w["categories"] == ["ASTM_INAC_COR", "ASTM_INCOMP"]
    m = kb.mitigation("DFM-1002")
    assert m is not None
    assert m["technique"] == "DFT-1002"


def test_subtechnique_resolves_to_parent(kb: KnowledgeBase) -> None:
    assert kb.parent_of("DFT-1002") == "DFT-1001"
    assert kb.parent_of("DFT-1003") == "DFT-1001"
    assert kb.parent_of("DFT-1001") is None
    assert kb.parent_of("DFT-9999") is None
    assert kb.is_subtechnique_of("DFT-1002", "DFT-1001")
    assert not kb.is_subtechnique_of("DFT-1004", "DFT-1001")
    assert not kb.is_subtechnique_of("DFT-1001", "DFT-1002")
    assert kb.parent_of("DFT-1015") == "DFT-1004"


def test_missing_techniques_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseError):
        KnowledgeBase.load(tmp_path)


def test_missing_weaknesses_and_mitigations_dirs_are_not_fatal(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    shutil.rmtree(root / "data" / "weaknesses")
    shutil.rmtree(root / "data" / "mitigations")
    loaded = KnowledgeBase.load(root)
    assert loaded.counts() == {"techniques": 8, "weaknesses": 0, "mitigations": 0}


def test_malformed_item_file_is_skipped_and_reported(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / "data" / "techniques" / "DFT-1999.json").write_text("{ not json", encoding="utf-8")
    (root / "data" / "weaknesses" / "DFW-1998.json").write_text("[1, 2]", encoding="utf-8")
    (root / "data" / "mitigations" / "DFM-1998.json").write_text(
        json.dumps({"name": "no id"}), encoding="utf-8"
    )
    loaded = KnowledgeBase.load(root)
    assert loaded.counts() == {"techniques": 8, "weaknesses": 6, "mitigations": 4}
    assert len(loaded.load_warnings) == 3
    joined = "\n".join(loaded.load_warnings)
    assert "DFT-1999.json" in joined
    assert "DFW-1998.json" in joined
    assert "DFM-1998.json" in joined


def test_non_json_files_are_ignored(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    (root / "data" / "techniques" / "README.md").write_text("hello", encoding="utf-8")
    loaded = KnowledgeBase.load(root)
    assert loaded.counts()["techniques"] == 8
    assert loaded.load_warnings == []


# --- finder --------------------------------------------------------------------


def test_find_hash_is_ranked(kb: KnowledgeBase) -> None:
    results = kb.find_techniques("hash")
    ids = [r["id"] for r in results]
    # Name matches first (whole word, then word start), then synonym and description.
    assert ids[:3] == ["DFT-1004", "DFT-1006", "DFT-1005"]
    assert results[0] == {
        "id": "DFT-1004",
        "name": "Hashing",
        "parent": None,
        "description": "Fixture technique DFT-1004.",
        "inputs": [],
        "outputs": [],
        "matched_all_words": True,
    }


def test_find_exact_match_ranks_first(kb: KnowledgeBase) -> None:
    results = kb.find_techniques("HASHING")
    assert results[0]["id"] == "DFT-1004"


def test_find_reports_parent_and_matches_synonyms(kb: KnowledgeBase) -> None:
    results = kb.find_techniques("imaging")
    ids = [r["id"] for r in results]
    assert ids[:2] == ["DFT-1002", "DFT-1003"]
    assert all(r["parent"] == "DFT-1001" for r in results[:2])
    assert kb.find_techniques("disk imag")[0]["id"] == "DFT-1002"


def test_find_matches_each_word_and_word_forms(kb: KnowledgeBase) -> None:
    # Neither phrase appears whole; each word does, in some form.
    assert kb.find_techniques("verifying images")[0]["id"] == "DFT-1005"
    top = kb.find_techniques("hash verification")[0]
    assert top["id"] == "DFT-1005" and top["matched_all_words"] is True
    # Stop words are ignored.
    assert kb.find_techniques("the keyword search")[0]["id"] == "DFT-1007"


def test_find_whole_synonym_ranks_first(kb: KnowledgeBase) -> None:
    assert kb.find_techniques("disk imaging")[0]["id"] == "DFT-1002"


def test_find_quoted_phrase_must_match_whole(kb: KnowledgeBase) -> None:
    assert [r["id"] for r in kb.find_techniques('"image integrity"')] == ["DFT-1005"]
    assert kb.find_techniques('"integrity image"') == []


def test_find_partial_matches_come_after_full_ones(kb: KnowledgeBase) -> None:
    results = kb.find_techniques("hash lookup")
    # Only DFT-1006 has both words; the hash-only matches follow it.
    assert results[0]["id"] == "DFT-1006" and results[0]["matched_all_words"] is True
    assert len(results) > 1 and not any(r["matched_all_words"] for r in results[1:])


def test_find_shortens_descriptions_and_class_names() -> None:
    kb = KnowledgeBase(
        {
            "DFT-1": {
                "id": "DFT-1",
                "name": "Long one",
                "description": "word " * 100,
                "CASE_input_classes": ["https://example.org/ns/Disk"],
                "CASE_output_classes": ["https://example.org/ns/Bitstream/"],
            }
        },
        {},
        {},
    )
    (match,) = kb.find_techniques("long")
    assert len(match["description"]) <= 200 and match["description"].endswith("\u2026")
    assert (match["inputs"], match["outputs"]) == (["Disk"], ["Bitstream"])


def test_find_limit_and_empty_query(kb: KnowledgeBase) -> None:
    assert len(kb.find_techniques("hash", limit=2)) == 2
    assert kb.find_techniques("") == []
    assert kb.find_techniques("   ") == []
    assert kb.find_techniques("zzzz-no-match") == []


# --- ASTM labels ----------------------------------------------------------------


def test_astm_labels_cover_fixture_categories(kb: KnowledgeBase) -> None:
    for wid in ("DFW-1001", "DFW-1002", "DFW-1003", "DFW-1004", "DFW-1005", "DFW-1006"):
        w = kb.weakness(wid)
        assert w is not None
        for cat in w["categories"]:
            assert cat in ASTM_LABELS
    assert ASTM_LABELS["ASTM_INCOMP"] == "INCOMP"
    assert len(ASTM_LABELS) == 6


# --- revision -------------------------------------------------------------------


def test_revision_unknown_without_git(kb: KnowledgeBase) -> None:
    assert kb.revision == "unknown"


def _git_kb(tmp_path: Path) -> tuple[Path, Path]:
    root = _copy_fixture(tmp_path)
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    return root, git


def test_revision_from_loose_ref(tmp_path: Path) -> None:
    root, git = _git_kb(tmp_path)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(SHA_A + "\n")
    assert KnowledgeBase.load(root).revision == SHA_A


def test_revision_from_packed_ref(tmp_path: Path) -> None:
    root, git = _git_kb(tmp_path)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted \n"
        f"{SHA_B} refs/heads/other\n"
        f"{SHA_A} refs/heads/main\n"
        f"^{SHA_B}\n"
    )
    assert KnowledgeBase.load(root).revision == SHA_A


def test_revision_detached_head(tmp_path: Path) -> None:
    root, git = _git_kb(tmp_path)
    (git / "HEAD").write_text(SHA_B + "\n")
    assert KnowledgeBase.load(root).revision == SHA_B


def test_revision_unknown_when_ref_missing(tmp_path: Path) -> None:
    root, git = _git_kb(tmp_path)
    (git / "HEAD").write_text("ref: refs/heads/gone\n")
    assert KnowledgeBase.load(root).revision == "unknown"


def test_revision_unknown_when_head_garbage(tmp_path: Path) -> None:
    root, git = _git_kb(tmp_path)
    (git / "HEAD").write_text("not a sha\n")
    assert KnowledgeBase.load(root).revision == "unknown"


def test_revision_via_gitdir_file_worktree(tmp_path: Path) -> None:
    # A linked worktree: .git is a file pointing at <main>/.git/worktrees/<name>,
    # which holds HEAD and a commondir file pointing back at the shared refs.
    main_git = tmp_path / "main" / ".git"
    wt_git = main_git / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (main_git / "refs" / "heads").mkdir(parents=True)
    (main_git / "refs" / "heads" / "feature").write_text(SHA_A + "\n")
    (wt_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (wt_git / "commondir").write_text("../..\n")
    root = _copy_fixture(tmp_path)
    (root / ".git").write_text(f"gitdir: {wt_git}\n")
    assert KnowledgeBase.load(root).revision == SHA_A


def test_revision_via_relative_gitdir_file(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    real = root / "realgit"
    real.mkdir()
    (real / "HEAD").write_text(SHA_B + "\n")
    (root / ".git").write_text("gitdir: realgit\n")
    assert KnowledgeBase.load(root).revision == SHA_B


# --- real knowledge base ---------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MCP_APP_SOLVEIT_DATA_PATH"),
    reason="MCP_APP_SOLVEIT_DATA_PATH not set",
)
def test_real_knowledge_base_loads() -> None:
    root = Path(os.environ["MCP_APP_SOLVEIT_DATA_PATH"]).expanduser()
    real = KnowledgeBase.load(root)
    assert real.counts()["techniques"] >= 150
    categories: set[str] = set()
    for path in (root / "data" / "weaknesses").glob("*.json"):
        w = real.weakness(path.stem)
        if w is not None:
            categories.update(w.get("categories", []))
    assert categories <= set(ASTM_LABELS)
