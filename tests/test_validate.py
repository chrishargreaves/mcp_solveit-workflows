"""Tests for the validation rules."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from conftest import WORKFLOWS

from mcp_solveit_workflows.kb import KnowledgeBase
from mcp_solveit_workflows.model import Workflow, read_workflow
from mcp_solveit_workflows.validate import Finding, has_errors, validate_workflow


def _fixture(name: str) -> Workflow:
    return read_workflow((WORKFLOWS / f"{name}.json").read_text(encoding="utf-8"))


def _tech(
    node_id: str, tid: str = "DFT-1001", x: Any = 0, y: Any = 0, **data: Any
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "technique",
        "x": x,
        "y": y,
        "data": {"techniqueId": tid, **data},
    }


def _edge(edge_id: str, source: str, target: str, **extra: Any) -> dict[str, Any]:
    return {"id": edge_id, "source": source, "target": target, **extra}


def _wf(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]] | None = None, **top: Any
) -> Workflow:
    doc: dict[str, Any] = {"version": 2, "nodes": nodes, "edges": edges or [], "selections": {}}
    doc.update(top)
    return Workflow(doc)


def _base() -> list[dict[str, Any]]:
    """Two connected techniques: the clean starting point for broken documents."""
    return [_tech("n1"), _tech("n2", y=180)]


def _base_edges() -> list[dict[str, Any]]:
    return [_edge("e1", "n1", "n2")]


def _codes(findings: list[Finding], level: str | None = None) -> list[str]:
    return [f.code for f in findings if level is None or f.level == level]


def _only(findings: list[Finding], code: str) -> Finding:
    hits = [f for f in findings if f.code == code]
    assert len(hits) == 1, findings
    return hits[0]


# ---- the clean baseline -----------------------------------------------------


def test_minimal_clean_document_has_no_findings() -> None:
    assert validate_workflow(_wf(_base(), _base_edges()), None) == []


def test_finding_to_dict_omits_missing_ids() -> None:
    f = Finding("warning", "x", "m", node_id="n1")
    assert f.to_dict() == {"level": "warning", "code": "x", "message": "m", "node_id": "n1"}
    assert Finding("error", "y", "m").to_dict() == {"level": "error", "code": "y", "message": "m"}


def test_has_errors() -> None:
    assert has_errors([Finding("warning", "a", "m"), Finding("error", "b", "m")])
    assert not has_errors([Finding("warning", "a", "m")])
    assert not has_errors([])


def test_validate_does_not_change_the_document() -> None:
    doc = {"version": 2, "nodes": [_tech("n1")]}
    before = copy.deepcopy(doc)
    validate_workflow(Workflow(doc), None)
    assert doc == before


# ---- errors -----------------------------------------------------------------


def test_unknown_type_in_input_is_an_error() -> None:
    nodes = [*_base(), {"id": "n3", "type": "subworkflow", "x": 0, "y": 360, "data": {}}]
    edges = [*_base_edges(), _edge("e2", "n2", "n3")]
    findings = validate_workflow(_wf(nodes, edges), None, input_node_ids={"n3"})
    f = _only(findings, "unknown_type")
    assert f.level == "error" and f.node_id == "n3"
    assert has_errors(findings)


def test_unknown_type_read_from_file_is_a_pass_through_warning() -> None:
    nodes = [*_base(), {"id": "n3", "type": "subworkflow", "x": 0, "y": 360, "data": {}}]
    edges = [*_base_edges(), _edge("e2", "n2", "n3")]
    findings = validate_workflow(_wf(nodes, edges), None)
    f = _only(findings, "unreleased_or_unknown_type")
    assert f.level == "warning" and f.node_id == "n3"
    assert "carried through" in f.message
    assert not has_errors(findings)


@pytest.mark.parametrize(
    ("x", "y"),
    [(float("nan"), 0), (0, float("inf")), (None, 0), ("10", 0), (True, 0)],
)
def test_non_finite_or_missing_position_is_an_error(x: Any, y: Any) -> None:
    nodes = [_tech("n1", x=x, y=y), _tech("n2", y=180)]
    f = _only(validate_workflow(_wf(nodes, _base_edges()), None), "invalid_position")
    assert f.level == "error" and f.node_id == "n1"


def test_missing_x_key_is_an_error() -> None:
    node = _tech("n1")
    del node["x"]
    findings = validate_workflow(_wf([node, _tech("n2", y=180)], _base_edges()), None)
    assert _only(findings, "invalid_position").node_id == "n1"


def test_duplicate_node_id_is_an_error() -> None:
    nodes = [*_base(), _tech("n2", y=360)]
    f = _only(validate_workflow(_wf(nodes, _base_edges()), None), "duplicate_node_id")
    assert f.level == "error" and f.node_id == "n2"


def test_duplicate_edge_id_is_an_error() -> None:
    edges = [*_base_edges(), _edge("e1", "n2", "n1")]
    f = _only(validate_workflow(_wf(_base(), edges), None), "duplicate_edge_id")
    assert f.level == "error" and f.edge_id == "e1"


@pytest.mark.parametrize("end", ["source", "target"])
def test_dangling_edge_is_an_error(end: str) -> None:
    edge = _edge("e1", "n1", "n2")
    edge[end] = "n9"
    f = _only(validate_workflow(_wf(_base(), [edge]), None), "dangling_edge")
    assert f.level == "error" and f.edge_id == "e1" and "n9" in f.message


def test_dangling_parent_is_an_error() -> None:
    nodes = _base()
    nodes[1]["parent"] = "n9"
    f = _only(validate_workflow(_wf(nodes, _base_edges()), None), "dangling_parent")
    assert f.level == "error" and f.node_id == "n2"


def test_self_parent_is_an_error() -> None:
    nodes = _base()
    nodes[1]["parent"] = "n2"
    f = _only(validate_workflow(_wf(nodes, _base_edges()), None), "dangling_parent")
    assert f.node_id == "n2"


@pytest.mark.parametrize(
    ("node_type", "data"),
    [
        ("technique", {}),
        ("technique", {"techniqueId": ""}),
        ("placeholder", {}),
        ("placeholder", {"name": "   "}),
        ("note", {}),
        ("decision", {}),
        ("decision", {"question": 3}),
        ("region", {"w": 100, "h": 100}),
        ("caseClass", {}),
    ],
)
def test_missing_required_data_is_an_error(node_type: str, data: dict[str, Any]) -> None:
    nodes = [*_base(), {"id": "n3", "type": node_type, "x": 0, "y": 360, "data": data}]
    edges = [*_base_edges(), _edge("e2", "n2", "n3")]
    f = _only(validate_workflow(_wf(nodes, edges), None), "missing_data")
    assert f.level == "error" and f.node_id == "n3"


@pytest.mark.parametrize(
    "size", [{}, {"w": 0, "h": 100}, {"w": 100, "h": -1}, {"w": "100", "h": 100}]
)
def test_region_needs_positive_size(size: dict[str, Any]) -> None:
    region = {"id": "r1", "type": "region", "x": 0, "y": 0, "data": {"label": "Zone", **size}}
    findings = validate_workflow(_wf([*_base(), region], _base_edges()), None)
    f = _only(findings, "invalid_region_size")
    assert f.level == "error" and f.node_id == "r1"


def test_data_that_is_not_an_object_is_an_error() -> None:
    nodes = _base()
    nodes[0]["data"] = ["x"]
    f = _only(validate_workflow(_wf(nodes, _base_edges()), None), "invalid_data")
    assert f.level == "error" and f.node_id == "n1"


def test_methods_that_are_not_a_list_is_an_error() -> None:
    nodes = [_tech("n1", methods="DFT-1002"), _tech("n2", y=180)]
    f = _only(validate_workflow(_wf(nodes, _base_edges()), None), "invalid_data")
    assert f.node_id == "n1"


# ---- warnings: knowledge base -----------------------------------------------


def test_unknown_technique_is_a_warning_naming_node_and_id(kb: KnowledgeBase) -> None:
    nodes = [_tech("n1"), _tech("n2", tid="DFT-9999", y=180)]
    findings = validate_workflow(_wf(nodes, _base_edges()), kb)
    f = _only(findings, "unknown_technique")
    assert f.level == "warning" and f.node_id == "n2" and "DFT-9999" in f.message
    assert not has_errors(findings)


def test_kb_checks_skipped_without_kb() -> None:
    nodes = [_tech("n1"), _tech("n2", tid="DFT-9999", y=180)]
    assert validate_workflow(_wf(nodes, _base_edges()), None) == []


def test_unknown_method_is_a_warning(kb: KnowledgeBase) -> None:
    nodes = [_tech("n1", methods=["DFT-9998"]), _tech("n2", y=180)]
    f = _only(validate_workflow(_wf(nodes, _base_edges()), kb), "unknown_technique")
    assert f.node_id == "n1" and "DFT-9998" in f.message


def test_method_not_a_subtechnique_is_a_warning(kb: KnowledgeBase) -> None:
    nodes = [_tech("n1", methods=["DFT-1002", "DFT-1004"]), _tech("n2", y=180)]
    findings = validate_workflow(_wf(nodes, _base_edges()), kb)
    f = _only(findings, "method_not_subtechnique")
    assert f.level == "warning" and f.node_id == "n1"
    assert "DFT-1004" in f.message and "DFT-1001" in f.message


def test_selection_ids_checked_against_kb(kb: KnowledgeBase) -> None:
    sel = {"n1|DFW-1001|DFM-1001": "Y", "n1|DFW-1999|DFM-1999": "N"}
    findings = validate_workflow(_wf(_base(), _base_edges(), selections=sel), kb)
    assert _only(findings, "unknown_weakness").node_id == "n1"
    assert "DFW-1999" in _only(findings, "unknown_weakness").message
    assert "DFM-1999" in _only(findings, "unknown_mitigation").message
    assert not has_errors(findings)


# ---- warnings: structure ----------------------------------------------------


@pytest.mark.parametrize(("node_type", "mode"), [("fork", "join"), ("join", None)])
def test_fork_join_mode_mismatch_is_a_warning(node_type: str, mode: str | None) -> None:
    data = {} if mode is None else {"mode": mode}
    nodes = [*_base(), {"id": "g1", "type": node_type, "x": 0, "y": 360, "data": data}]
    edges = [*_base_edges(), _edge("e2", "n2", "g1")]
    f = _only(validate_workflow(_wf(nodes, edges), None), "fork_join_mode")
    assert f.level == "warning" and f.node_id == "g1"


def test_disconnected_flow_node_is_a_warning() -> None:
    nodes = [*_base(), _tech("n3", y=360)]
    findings = validate_workflow(_wf(nodes, _base_edges()), None)
    f = _only(findings, "disconnected")
    assert f.level == "warning" and f.node_id == "n3"


def test_single_flow_node_is_not_disconnected() -> None:
    note = {"id": "c1", "type": "note", "x": 300, "y": 0, "data": {"text": "hi"}}
    assert validate_workflow(_wf([_tech("n1"), note]), None) == []


def test_notes_and_regions_are_never_disconnected() -> None:
    note = {"id": "c1", "type": "note", "x": 300, "y": 0, "data": {"text": "hi"}}
    assert validate_workflow(_wf([*_base(), note], _base_edges()), None) == []


def test_several_branches_from_one_handle_is_a_warning() -> None:
    nodes = [*_base(), _tech("n3", x=300, y=180)]
    edges = [
        _edge("e1", "n1", "n2", sourceHandle="bottom-s"),
        _edge("e2", "n1", "n3", sourceHandle="bottom-s"),
    ]
    f = _only(validate_workflow(_wf(nodes, edges), None), "ambiguous_branch")
    assert f.level == "warning" and f.node_id == "n1"
    assert "decision" in f.message and "fork" in f.message


def test_branches_from_different_handles_are_fine() -> None:
    nodes = [*_base(), _tech("n3", x=300, y=180)]
    edges = [
        _edge("e1", "n1", "n2", sourceHandle="bottom-s"),
        _edge("e2", "n1", "n3", sourceHandle="right-s"),
    ]
    assert validate_workflow(_wf(nodes, edges), None) == []


def test_branches_from_a_decision_are_fine() -> None:
    nodes = [
        {"id": "d1", "type": "decision", "x": 0, "y": 0, "data": {"question": "Q?"}},
        _tech("n2", y=180),
        _tech("n3", x=300, y=180),
    ]
    edges = [_edge("e1", "d1", "n2"), _edge("e2", "d1", "n3")]
    assert validate_workflow(_wf(nodes, edges), None) == []


def test_selection_to_missing_node_is_a_warning() -> None:
    sel = {"n9|DFW-1001|DFM-1001": "Y"}
    findings = validate_workflow(_wf(_base(), _base_edges(), selections=sel), None)
    f = _only(findings, "selection_missing_node")
    assert f.level == "warning" and f.node_id == "n9"


def test_selection_on_non_technique_is_a_warning() -> None:
    note = {"id": "c1", "type": "note", "x": 300, "y": 0, "data": {"text": "hi"}}
    sel = {"c1|DFW-1001|DFM-1001": "Y"}
    findings = validate_workflow(_wf([*_base(), note], _base_edges(), selections=sel), None)
    assert _only(findings, "selection_not_technique").node_id == "c1"


def test_malformed_selection_key_and_value_are_warnings() -> None:
    sel = {"n1|DFW-1001": "Y", "n1|DFW-1001|DFM-1001": "maybe"}
    findings = validate_workflow(_wf(_base(), _base_edges(), selections=sel), None)
    assert set(_codes(findings)) == {"selection_bad_key", "selection_bad_value"}
    assert not has_errors(findings)


def test_long_description_is_a_warning() -> None:
    findings = validate_workflow(_wf(_base(), _base_edges(), description="x" * 281), None)
    f = _only(findings, "description_too_long")
    assert f.level == "warning" and "x" * 20 not in f.message


def test_unreleased_data_keys_warn_once_per_node() -> None:
    nodes = [_tech("n1", subWorkflow={"mode": "link"}, toolEvidence=[]), _tech("n2", y=180)]
    findings = validate_workflow(_wf(nodes, _base_edges()), None)
    f = _only(findings, "unreleased_content")
    assert f.level == "warning" and f.node_id == "n1"
    assert "subWorkflow" in f.message and "carried through" in f.message


def test_unreleased_top_level_fields_warn() -> None:
    findings = validate_workflow(_wf(_base(), _base_edges(), id="x", workflowVersion="3"), None)
    f = _only(findings, "unreleased_content")
    assert f.node_id is None and "workflowVersion" in f.message


def test_free_text_is_not_quoted_beyond_cap() -> None:
    long_label = "L" * 200
    nodes = [*_base(), _tech("n3", x=300, y=180, label=long_label)]
    edges = [*_base_edges(), _edge("e2", "n1", "n3", sourceHandle="h" * 200)]
    edges[0]["sourceHandle"] = "h" * 200
    for f in validate_workflow(_wf(nodes, edges), None):
        assert "L" * 61 not in f.message and "h" * 61 not in f.message


def test_every_node_finding_names_its_node() -> None:
    nodes = [
        _tech("n1", x=None),
        {"id": "n2", "type": "placeholder", "x": 0, "y": 0, "data": {}},
        {"id": "n3", "type": "fork", "x": 0, "y": 0, "data": {}},
        {"id": "n4", "type": "mystery", "x": 0, "y": 0, "data": {}},
    ]
    for f in validate_workflow(_wf(nodes), None):
        assert f.node_id or f.edge_id, f


# ---- fixtures ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["sample_workflow", "synthetic_small", "synthetic_21_nodes"])
def test_clean_fixtures_have_no_findings(name: str) -> None:
    assert validate_workflow(_fixture(name), None) == []


@pytest.mark.parametrize(
    "name",
    [
        "sample_workflow",
        "synthetic_small",
        "synthetic_21_nodes",
        "synthetic_63_nodes",
    ],
)
def test_fixtures_have_no_errors_against_kb(name: str, kb: KnowledgeBase) -> None:
    findings = validate_workflow(_fixture(name), kb)
    assert not has_errors(findings), [f.to_dict() for f in findings if f.level == "error"]
    # Every KB finding is a warning.
    assert all(f.level == "warning" for f in findings)


def test_big_fixture_has_no_errors() -> None:
    wf = _fixture("synthetic_63_nodes")
    assert len(wf.nodes) == 63
    assert not has_errors(validate_workflow(wf, None))


def test_unreleased_fixture_subworkflow_is_a_warning_not_an_error(kb: KnowledgeBase) -> None:
    wf = _fixture("synthetic_unreleased_fields")
    findings = validate_workflow(wf, kb)
    assert not has_errors(findings)
    node_hits = [f for f in findings if f.code == "unreleased_content" and f.node_id == "n5"]
    assert len(node_hits) == 1 and "subWorkflow" in node_hits[0].message
    top = [f for f in findings if f.code == "unreleased_content" and f.node_id is None]
    assert len(top) == 1
    # Findings serialise to JSON for tool results.
    json.dumps([f.to_dict() for f in findings])


def test_subworkflow_scoped_evaluation_key_is_pass_through(kb: KnowledgeBase) -> None:
    findings = validate_workflow(_fixture("synthetic_unreleased_fields"), kb)
    assert "evaluation_missing_node" not in _codes(findings)
    hits = [f for f in findings if f.code == "unreleased_content" and f.node_id == "n5/n1"]
    assert len(hits) == 1 and hits[0].level == "warning"


# ---- review fixes ----


@pytest.mark.parametrize("other_type", ["note", "region"])
@pytest.mark.parametrize("end", ["source", "target"])
def test_edge_to_note_or_region_is_an_error(other_type: str, end: str) -> None:
    data = {"text": "x"} if other_type == "note" else {"label": "Z", "w": 100, "h": 100}
    other = {"id": "o", "type": other_type, "x": 400, "y": 0, "data": data}
    edge = _edge("e9", "o", "n1") if end == "source" else _edge("e9", "n1", "o")
    findings = validate_workflow(_wf([*_base(), other], [*_base_edges(), edge]), None)
    f = _only(findings, "invalid_edge_endpoint")
    assert f.level == "error" and f.edge_id == "e9" and "'o'" in f.message


def test_edge_to_case_class_is_fine() -> None:
    cc = {"id": "c", "type": "caseClass", "x": 400, "y": 0, "data": {"label": "Disk"}}
    findings = validate_workflow(
        _wf([*_base(), cc], [*_base_edges(), _edge("e9", "c", "n1")]), None
    )
    assert "invalid_edge_endpoint" not in _codes(findings)


def test_evaluation_keys_checked(kb: KnowledgeBase) -> None:
    wf = _wf(
        _base(),
        _base_edges(),
        evaluations={"n1": {}, "n1|DFW-9999": {}, "n1|DFW-1001": {}, "n1|a|b": {}},
    )
    findings = validate_workflow(wf, kb)
    assert _codes(findings).count("evaluation_bad_key") == 2
    unknown = _only(findings, "unknown_weakness")
    assert unknown.level == "warning" and "DFW-9999" in unknown.message
    assert not has_errors(findings)


def test_non_text_ids_do_not_raise() -> None:
    nodes = [*_base(), {"id": ["x"], "type": "note", "x": 0, "y": 0, "data": {"text": "t"}}]
    edges = [*_base_edges(), {"id": {"a": 1}, "source": ["x"], "target": "n1"}]
    findings = validate_workflow(_wf(nodes, edges), None)
    assert has_errors(findings)
