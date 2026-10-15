"""Drift check: the server's output read by the deployed Workflow Builder's own code.

Runs the builder's `src/state/urlState.ts` and `scripts/lib/validate-workflow.mjs`
at `origin/main` through `vite-node` (see `probe.ts`), against a temporary git
worktree created outside the builder checkout. Opt in with `-m drift` and
`SOLVEIT_BUILDER_REPO=<builder checkout>`; otherwise the tests skip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mcp_solveit_workflows import sharelink
from mcp_solveit_workflows.kb import KnowledgeBase
from mcp_solveit_workflows.layout import layout_workflow
from mcp_solveit_workflows.model import (
    DataSourceInput,
    DecisionInput,
    EdgeInput,
    ForkInput,
    JoinInput,
    NoteInput,
    PlaceholderInput,
    SelectionInput,
    TechniqueInput,
    ViewportInput,
    Workflow,
    WorkflowInput,
    ZoneInput,
    build_workflow,
    read_workflow,
)

pytestmark = pytest.mark.drift

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
PROBE = HERE / "probe.ts"
FIXTURES = TESTS / "fixtures" / "workflows"
FIXTURE_KB = TESTS / "fixtures" / "kb"
BUILDER_REF = "origin/main"

# Fixtures the deployed builder (origin/main) can read. `synthetic_unreleased_fields`
# uses node types only on an unreleased branch, and `legacy_v1` is refused by
# the server, so neither is server output for main.
MAIN_FIXTURES = (
    "sample_workflow",
    "synthetic_small",
    "synthetic_21_nodes",
    "synthetic_63_nodes",
)

PROBE_TIMEOUT_SECONDS = 180

APP_ID = "https://ontology.unifiedcyberontology.org/uco/observable/applicationIdentifier"
ROUTE = (
    "<https://ontology.solveit-df.org/solveit/observable/sourceApplication>"
    "/<https://ontology.unifiedcyberontology.org/uco/core/hasFacet>"
    f"/<{APP_ID}>"
)
DISK_IMAGE = "https://ontology.solveit-df.org/solveit/observable/DiskImage"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def _builder_repo() -> Path:
    raw = os.environ.get("SOLVEIT_BUILDER_REPO")
    if not raw:
        pytest.skip("drift check skipped: SOLVEIT_BUILDER_REPO is not set")
    repo = Path(raw).expanduser()
    if not repo.is_dir():
        pytest.skip(f"drift check skipped: SOLVEIT_BUILDER_REPO ({repo}) is not a directory")
    if shutil.which("git") is None:
        pytest.skip("drift check skipped: git is not installed")
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        pytest.skip(f"drift check skipped: SOLVEIT_BUILDER_REPO ({repo}) is not a git repo")
    if not (repo / "node_modules").is_dir():
        pytest.skip("drift check skipped: the builder has no node_modules (run npm install there)")
    if not (repo / "node_modules" / ".bin" / "vite-node").exists():
        pytest.skip(
            "drift check skipped: node_modules/.bin/vite-node is missing (it comes in "
            "transitively with vitest; the builder's dependencies may have changed)"
        )
    if _git(repo, "rev-parse", "--verify", "--quiet", BUILDER_REF).returncode != 0:
        pytest.skip(f"drift check skipped: the builder has no {BUILDER_REF}")
    return repo


@pytest.fixture(scope="session")
def builder_worktree() -> Iterator[Path]:
    """A throwaway worktree of the builder's origin/main, outside the builder repo."""
    repo = _builder_repo()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    main_rev = _git(repo, "rev-parse", "--short", BUILDER_REF).stdout.strip()
    # node_modules follow whichever branch is checked out, not origin/main.
    print(
        f"\n[drift] builder checkout is on '{branch}'; its node_modules are reused for "
        f"{BUILDER_REF} ({main_rev})"
    )
    tmp = Path(tempfile.mkdtemp(prefix="solveit-drift-"))
    worktree = tmp / "builder"
    try:
        added = _git(repo, "worktree", "add", "--detach", str(worktree), BUILDER_REF)
        if added.returncode != 0:
            pytest.fail(f"could not create a worktree of {BUILDER_REF}: {added.stderr.strip()}")
        (worktree / "node_modules").symlink_to(repo / "node_modules", target_is_directory=True)
        yield worktree
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree))
        _git(repo, "worktree", "prune")
        shutil.rmtree(tmp, ignore_errors=True)


def _parent_of() -> Any:
    real = os.environ.get("MCP_APP_SOLVEIT_DATA_PATH")
    root = Path(real).expanduser() if real else FIXTURE_KB
    return KnowledgeBase.load(root).parent_of


# ---- server output ----


@dataclass
class Case:
    name: str
    workflow: Workflow


def _fixture_cases() -> list[Case]:
    return [
        Case(name, read_workflow((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))
        for name in MAIN_FIXTURES
    ]


def _every_kind_spec() -> WorkflowInput:
    """Every builder element: all eight node kinds, edge labels, selections."""
    return WorkflowInput(
        title="Drift: every element",
        description="Built by the MCP server for the drift check.",
        nodes=[
            DataSourceInput(
                kind="data_source", key="disk", label="Disk image", classes=[DISK_IMAGE, "Phone"]
            ),
            TechniqueInput(kind="technique", key="hash", technique_id="DFT-1002", label="Verify"),
            # A sub-technique: stored as its parent with the sub-technique as a method.
            TechniqueInput(kind="technique", key="sub", technique_id="DFT-1015"),
            TechniqueInput(
                kind="technique",
                key="carve",
                technique_id="DFT-1004",
                label="Carve",
                methods=["DFT-1015"],
                target={ROUTE: "com.example.app"},
            ),
            DecisionInput(kind="decision", key="enc", question="Is the volume encrypted?"),
            ForkInput(kind="fork", key="fork"),
            PlaceholderInput(
                kind="placeholder",
                key="new",
                name="Decrypt volume",
                temp_id="DFT-temp01",
                description="Not yet in SOLVE-IT.",
                details="Longer text\nover two lines.",
                input_classes=[DISK_IMAGE],
                output_classes=["Decrypted volume"],
            ),
            TechniqueInput(kind="technique", key="search", technique_id="DFT-1001"),
            JoinInput(kind="join", key="join"),
            NoteInput(kind="note", key="note", text="Check the hash first.", color="pink"),
            NoteInput(kind="note", key="plain", text="Default colour."),
            ZoneInput(
                kind="zone",
                key="acq",
                label="Acquisition",
                members=["disk", "hash"],
                color="green",
                label_corner="bl",
            ),
            ZoneInput(kind="zone", key="ana", label="Analysis", members=["fork", "join"]),
        ],
        edges=[
            EdgeInput(source="disk", target="hash"),
            EdgeInput(source="hash", target="enc"),
            EdgeInput(source="enc", target="new", label="yes"),
            EdgeInput(source="enc", target="fork", label="no"),
            EdgeInput(source="new", target="fork"),
            EdgeInput(source="fork", target="carve"),
            EdgeInput(source="fork", target="sub"),
            EdgeInput(source="fork", target="search", label="in parallel"),
            EdgeInput(source="carve", target="join"),
            EdgeInput(source="sub", target="join"),
            EdgeInput(source="search", target="join"),
        ],
        selections=[
            SelectionInput(
                step="hash", weakness_id="DFW-1004", mitigation_id="DFM-1003", value="Y"
            ),
            SelectionInput(
                step="hash", weakness_id="DFW-1004", mitigation_id="DFM-1004", value="N"
            ),
            SelectionInput(
                step="hash", weakness_id="DFW-1006", mitigation_id="DFM-1005", value="NA"
            ),
        ],
    )


def _built(name: str, spec: WorkflowInput, direction: Any, parent_of: Any) -> Case:
    result = build_workflow(spec, parent_of)
    layout_workflow(result.workflow, direction=direction, zone_members=result.zone_members)
    return Case(name, result.workflow)


def _built_cases() -> list[Case]:
    parent_of = _parent_of()
    small = WorkflowInput(
        title="Drift: minimal",
        nodes=[
            TechniqueInput(kind="technique", key="a", technique_id="DFT-1002"),
            TechniqueInput(kind="technique", key="b", technique_id="DFT-1001"),
        ],
        edges=[EdgeInput(source="a", target="b")],
        viewport=ViewportInput(x=12.6, y=-40.2, zoom=0.83333),
    )
    return [
        _built("built_every_kind_down", _every_kind_spec(), "down", parent_of),
        _built("built_every_kind_right", _every_kind_spec(), "right", parent_of),
        _built("built_minimal_right", small, "right", parent_of),
    ]


def _all_cases() -> list[Case]:
    return _fixture_cases() + _built_cases()


# ---- running the probe ----


@pytest.fixture(scope="session")
def cases() -> dict[str, Case]:
    _builder_repo()  # every drift test skips together when the builder is unavailable
    return {c.name: c for c in _all_cases()}


@pytest.fixture(scope="session")
def probe_results(builder_worktree: Path, cases: dict[str, Case]) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="solveit-drift-cases-"))
    try:
        listing = []
        for case in cases.values():
            path = work / f"{case.name}.json"
            path.write_text(case.workflow.to_json(), encoding="utf-8")
            link = sharelink.encode_hash(case.workflow.to_json(pretty=False))
            listing.append({"name": case.name, "json": str(path), "link": link})
        cases_path = work / "cases.json"
        cases_path.write_text(json.dumps(listing), encoding="utf-8")
        out_path = work / "results.json"
        kb_path = builder_worktree / "public" / "solve-it-fallback.json"
        argv = [
            str(builder_worktree / "node_modules" / ".bin" / "vite-node"),
            str(PROBE),
            str(builder_worktree),
            str(cases_path),
            str(out_path),
        ]
        if kb_path.is_file():
            argv.append(str(kb_path))
        run = subprocess.run(
            argv,
            cwd=builder_worktree,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if run.returncode != 0 or not out_path.is_file():
            pytest.fail(
                f"the builder probe failed (exit {run.returncode}):\n"
                f"{run.stderr.strip()}\n{run.stdout.strip()}"
            )
        results: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
        for name, result in results.items():
            for warning in result.get("validatorWarnings") or []:
                print(f"[drift] {name}: builder validator warning: {warning}")
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)


CASE_NAMES = [
    *MAIN_FIXTURES,
    "built_every_kind_down",
    "built_every_kind_right",
    "built_minimal_right",
]


def _result(probe_results: dict[str, Any], name: str) -> dict[str, Any]:
    result: dict[str, Any] = probe_results[name]
    assert result["error"] is None, f"{name}: the probe raised {result['error']}"
    return result


def test_cases_cover_every_case_name(cases: dict[str, Case]) -> None:
    assert sorted(cases) == sorted(CASE_NAMES)


@pytest.mark.parametrize("name", CASE_NAMES)
def test_builder_parses_with_same_counts(
    probe_results: dict[str, Any], cases: dict[str, Case], name: str
) -> None:
    result = _result(probe_results, name)
    wf = cases[name].workflow
    assert result["parsed"], f"{name}: the builder's isWorkflowState rejected the server's file"
    assert (result["nodeCount"], result["edgeCount"]) == (len(wf.nodes), len(wf.edges))


@pytest.mark.parametrize("name", CASE_NAMES)
def test_builder_deserialize_keeps_every_node_and_edge(
    probe_results: dict[str, Any], cases: dict[str, Case], name: str
) -> None:
    """The builder's real load path (urlState.ts deserialize) drops nothing."""
    result = _result(probe_results, name)
    wf = cases[name].workflow
    assert result["deserializedNodeCount"] is not None, "deserialize is not exported on main"
    assert (result["deserializedNodeCount"], result["deserializedEdgeCount"]) == (
        len(wf.nodes),
        len(wf.edges),
    )


@pytest.mark.parametrize("name", CASE_NAMES)
def test_builder_writes_the_same_json(probe_results: dict[str, Any], name: str) -> None:
    """The server's file is byte for byte the builder's JSON.stringify(state, null, 2)."""
    assert _result(probe_results, name)["stringifyEqual"], name


@pytest.mark.parametrize("name", CASE_NAMES)
def test_server_link_decodes_to_the_file(probe_results: dict[str, Any], name: str) -> None:
    result = _result(probe_results, name)
    assert result["roundTripEqual"], f"{name}: the builder decoded the server's link differently"
    assert result["builderLinkRoundTrip"], name


@pytest.mark.parametrize("name", CASE_NAMES)
def test_builder_validator_finds_no_errors(probe_results: dict[str, Any], name: str) -> None:
    result = _result(probe_results, name)
    assert result["validatorErrors"] == [], f"{name}: {result['validatorErrors']}"


def test_every_kind_case_holds_every_element(cases: dict[str, Case]) -> None:
    """Guard the drift input itself: the built cases really cover every builder element."""
    for name in ("built_every_kind_down", "built_every_kind_right"):
        wf = cases[name].workflow
        types = {n["type"] for n in wf.nodes}
        assert types == {
            "technique",
            "placeholder",
            "caseClass",
            "region",
            "note",
            "decision",
            "fork",
            "join",
        }
        assert any("methods" in n["data"] and "label" in n["data"] for n in wf.nodes)
        assert any(n["data"].get("inputClasses") for n in wf.nodes)
        assert any(n["type"] == "note" and n["data"].get("color") for n in wf.nodes)
        assert any(e.get("label") for e in wf.edges)
        assert wf.selections
        # The sub-technique case is real under the fixture knowledge base: DFT-1015
        # is a sub-technique of DFT-1004, so 'sub' is stored as its parent.
        assert not any(n["data"].get("techniqueId") == "DFT-1015" for n in wf.nodes)
        assert any(
            n["data"].get("techniqueId") == "DFT-1004"
            and n["data"].get("methods", [None])[0] == "DFT-1015"
            and "label" not in n["data"]
            for n in wf.nodes
        )


def test_sub_technique_node_is_stored_as_parent() -> None:
    """The 'sub' node by key: techniqueId DFT-1004 with DFT-1015 as its first method."""
    result = build_workflow(_every_kind_spec(), KnowledgeBase.load(FIXTURE_KB).parent_of)
    sub = result.workflow.node(result.key_to_id["sub"])
    assert sub is not None
    assert sub["data"]["techniqueId"] == "DFT-1004"
    assert sub["data"]["methods"][0] == "DFT-1015"
