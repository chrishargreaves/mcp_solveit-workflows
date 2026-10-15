"""Compile mirrors the deployed builder's useCompile / coverage / flowOrder."""

from __future__ import annotations

import json
from typing import Any

from mcp_solveit_workflows.compile import compile_workflow, flow_order
from mcp_solveit_workflows.kb import KnowledgeBase
from mcp_solveit_workflows.model import (
    EdgeInput,
    PlaceholderInput,
    SelectionInput,
    TechniqueInput,
    Workflow,
    WorkflowInput,
    build_workflow,
    read_workflow,
)


def tech(node_id: str, tid: str, **data: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "technique",
        "x": 0,
        "y": 0,
        "data": {"techniqueId": tid, **data},
    }


def edge(eid: str, s: str, t: str) -> dict[str, Any]:
    return {"id": eid, "source": s, "target": t}


def wf_of(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]] | None = None,
    selections: dict[str, str] | None = None,
    **extra: Any,
) -> Workflow:
    doc: dict[str, Any] = {
        "version": 2,
        "nodes": nodes,
        "edges": edges or [],
        "selections": selections or {},
        **extra,
    }
    return read_workflow(doc)


def step_of(result: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(s for s in result["steps"] if s["node_id"] == node_id)


def weakness_of(step: dict[str, Any], wid: str) -> dict[str, Any]:
    return next(w for w in step["weaknesses"] if w["id"] == wid)


def mitigation_of(weak: dict[str, Any], mid: str) -> dict[str, Any]:
    return next(m for m in weak["mitigations"] if m["id"] == mid)


# ---- flow order ----


def test_flow_order_follows_edges_not_array_order() -> None:
    wf = wf_of(
        [tech("b", "DFT-1004"), tech("a", "DFT-1001")],
        [edge("e1", "a", "b")],
    )
    assert flow_order(wf) == ["a", "b"]


def test_flow_order_kahn_ties_and_skips_non_flow_nodes() -> None:
    nodes = [
        tech("t1", "DFT-1001"),
        {"id": "note", "type": "note", "x": 0, "y": 0, "data": {"text": "x"}},
        {"id": "f", "type": "fork", "x": 0, "y": 0, "data": {"mode": "fork"}},
        tech("t2", "DFT-1004"),
        tech("t3", "DFT-1005"),
        {"id": "j", "type": "join", "x": 0, "y": 0, "data": {"mode": "join"}},
        {"id": "ds", "type": "caseClass", "x": 0, "y": 0, "data": {"label": "Disk"}},
    ]
    edges = [
        edge("e0", "ds", "t1"),
        edge("e1", "t1", "f"),
        edge("e2", "f", "t3"),
        edge("e3", "f", "t2"),
        edge("e4", "t2", "j"),
        edge("e5", "t3", "j"),
        edge("e6", "note", "t1"),
    ]
    # Successors are queued in edge order: t3 before t2.
    assert flow_order(wf_of(nodes, edges)) == ["t1", "f", "t3", "t2", "j"]


def test_cycle_leftovers_appended_in_file_order_and_deterministic(kb: KnowledgeBase) -> None:
    nodes = [tech("c", "DFT-1005"), tech("a", "DFT-1001"), tech("b", "DFT-1004")]
    edges = [edge("e1", "a", "b"), edge("e2", "b", "a"), edge("e3", "b", "c")]
    wf = wf_of(nodes, edges)
    # Every node has an incoming edge, so all three are leftovers.
    assert flow_order(wf) == ["c", "a", "b"]
    runs = [[s["node_id"] for s in compile_workflow(wf, kb)["steps"]] for _ in range(5)]
    assert all(r == flow_order(wf) for r in runs)


def test_partial_cycle_after_a_root() -> None:
    nodes = [tech("x", "DFT-1004"), tech("r", "DFT-1001"), tech("y", "DFT-1005")]
    edges = [edge("e1", "r", "x"), edge("e2", "x", "y"), edge("e3", "y", "x")]
    assert flow_order(wf_of(nodes, edges)) == ["r", "x", "y"]


# ---- compile ----


def test_two_techniques_in_flow_order_with_astm_labels(kb: KnowledgeBase) -> None:
    spec = WorkflowInput(
        nodes=[
            TechniqueInput(kind="technique", key="hash", technique_id="DFT-1004"),
            TechniqueInput(kind="technique", key="acq", technique_id="DFT-1001", label="Laptop"),
        ],
        edges=[EdgeInput(source="acq", target="hash")],
    )
    built = build_workflow(spec, kb.parent_of)
    result = compile_workflow(built.workflow, kb, detail=True)
    acq, hsh = built.key_to_id["acq"], built.key_to_id["hash"]

    assert result["knowledge_base_revision"] == kb.revision
    assert result["mode"] == "pattern"
    assert [s["node_id"] for s in result["steps"]] == [acq, hsh]
    first = result["steps"][0]
    assert first["technique_id"] == "DFT-1001"
    assert first["name"] == "Acquire storage media"
    assert first["label"] == "Laptop"
    assert "methods" not in first
    assert [w["id"] for w in first["weaknesses"]] == ["DFW-1001", "DFW-1002"]
    assert weakness_of(first, "DFW-1001")["astm"] == ["INCOMP"]
    assert weakness_of(result["steps"][1], "DFW-1005")["astm"] == ["INAC-AS", "MISINT"]
    assert "label" not in result["steps"][1]
    json.dumps(result)


def test_mitigation_in_place_when_implementing_technique_on_canvas(kb: KnowledgeBase) -> None:
    wf = wf_of([tech("a", "DFT-1001"), tech("h", "DFT-1004")], [edge("e1", "a", "h")])
    result = compile_workflow(wf, kb, detail=True)
    w = weakness_of(step_of(result, "a"), "DFW-1001")
    m = mitigation_of(w, "DFM-1001")
    assert m == {
        "id": "DFM-1001",
        "name": "Verify hashes with a second tool",
        "in_place": True,
        "applicable": True,
        "source": "canvas",
    }
    # DFM-1003 has no implementing technique: not in place, so partial.
    assert mitigation_of(w, "DFM-1003")["in_place"] is False
    assert w["status"] == "partial"
    # DFW-1005 on the Hashing step is mitigated by DFM-1001 (Hashing itself).
    assert weakness_of(step_of(result, "h"), "DFW-1005")["status"] == "mitigated"


def test_selection_n_overrides_canvas(kb: KnowledgeBase) -> None:
    wf = wf_of(
        [tech("a", "DFT-1001"), tech("h", "DFT-1004")],
        [edge("e1", "a", "h")],
        {"a|DFW-1001|DFM-1001": "N", "a|DFW-1001|DFM-1003": "Y"},
    )
    result = compile_workflow(wf, kb, detail=True)
    w = weakness_of(step_of(result, "a"), "DFW-1001")
    assert mitigation_of(w, "DFM-1001")["in_place"] is False
    assert mitigation_of(w, "DFM-1001")["source"] is None
    assert mitigation_of(w, "DFM-1003")["source"] == "selection"
    assert w["status"] == "partial"


def test_na_removes_mitigation_from_applicable_set(kb: KnowledgeBase) -> None:
    wf = wf_of(
        [tech("a", "DFT-1001")],
        selections={
            "a|DFW-1001|DFM-1001": "Y",
            "a|DFW-1001|DFM-1003": "NA",
            "a|DFW-1002|DFM-1002": "NA",
        },
    )
    result = compile_workflow(wf, kb, detail=True)
    step = step_of(result, "a")
    w1 = weakness_of(step, "DFW-1001")
    assert mitigation_of(w1, "DFM-1003")["applicable"] is False
    assert w1["status"] == "mitigated"
    assert weakness_of(step, "DFW-1002")["status"] == "na"
    assert (step["mitigated"], step["na"], step["worst"]) == (1, 1, "mitigated")


def test_method_only_subtechnique_puts_mitigation_in_place(kb: KnowledgeBase) -> None:
    wf = wf_of([tech("a", "DFT-1001", methods=["DFT-1002"])])
    result = compile_workflow(wf, kb, detail=True)
    step = step_of(result, "a")
    assert step["methods"] == ["DFT-1002"]
    # Parent weaknesses first, then the method's.
    assert [w["id"] for w in step["weaknesses"]] == ["DFW-1001", "DFW-1002", "DFW-1003"]
    m = mitigation_of(weakness_of(step, "DFW-1002"), "DFM-1002")
    assert (m["in_place"], m["source"]) == (True, "canvas")


def test_build_workflow_subtechnique_becomes_method(kb: KnowledgeBase) -> None:
    spec = WorkflowInput(
        nodes=[TechniqueInput(kind="technique", key="p", technique_id="DFT-1002")],
        selections=[
            SelectionInput(step="p", weakness_id="DFW-1003", mitigation_id="DFM-1004", value="Y")
        ],
    )
    built = build_workflow(spec, kb.parent_of)
    result = compile_workflow(built.workflow, kb, detail=True)
    step = result["steps"][0]
    assert step["technique_id"] == "DFT-1001"
    assert weakness_of(step, "DFW-1003")["status"] == "mitigated"


def test_invalid_methods_are_dropped(kb: KnowledgeBase) -> None:
    wf = wf_of([tech("a", "DFT-1001", methods=["DFT-1004"])])
    step = compile_workflow(wf, kb)["steps"][0]
    assert "methods" not in step
    assert step["weakness_count"] == 2


def test_assessment_mode_only_y_counts(kb: KnowledgeBase) -> None:
    wf = wf_of(
        [tech("a", "DFT-1001"), tech("h", "DFT-1004")],
        [edge("e1", "a", "h")],
        {"a|DFW-1001|DFM-1003": "Y"},
        mode="assessment",
    )
    result = compile_workflow(wf, kb, detail=True)
    assert result["mode"] == "assessment"
    w = weakness_of(step_of(result, "a"), "DFW-1001")
    assert mitigation_of(w, "DFM-1001")["in_place"] is False
    assert mitigation_of(w, "DFM-1003")["in_place"] is True
    assert w["status"] == "partial"
    assert weakness_of(step_of(result, "h"), "DFW-1005")["status"] == "unmitigated"


def test_placeholder_contributes_no_steps(kb: KnowledgeBase) -> None:
    spec = WorkflowInput(
        nodes=[
            TechniqueInput(kind="technique", key="a", technique_id="DFT-1004"),
            PlaceholderInput(kind="placeholder", key="p", name="Something new"),
        ],
        edges=[EdgeInput(source="p", target="a")],
    )
    built = build_workflow(spec, kb.parent_of)
    result = compile_workflow(built.workflow, kb)
    assert [s["node_id"] for s in result["steps"]] == [built.key_to_id["a"]]
    assert built.key_to_id["p"] in flow_order(built.workflow)


def test_missing_weakness_skipped_and_reported(kb: KnowledgeBase) -> None:
    wf = wf_of([tech("a", "DFT-1007"), tech("b", "DFT-1007")])
    result = compile_workflow(wf, kb, detail=True)
    assert result["skipped_ids"] == ["DFW-1999"]
    step = step_of(result, "a")
    assert [w["id"] for w in step["weaknesses"]] == ["DFW-1006"]
    assert step["weakness_count"] == 1


def test_unknown_technique(kb: KnowledgeBase) -> None:
    wf = wf_of([tech("a", "DFT-9999")])
    result = compile_workflow(wf, kb, detail=True)
    step = result["steps"][0]
    assert step["name"] is None
    assert step["weakness_count"] == 0
    assert step["worst"] == "none"
    assert step["weaknesses"] == []
    assert result["skipped_ids"] == ["DFT-9999"]


def test_weakness_without_mitigations_is_unmitigated(kb: KnowledgeBase) -> None:
    wf = wf_of([tech("a", "DFT-1005")])
    result = compile_workflow(wf, kb, detail=True)
    step = result["steps"][0]
    w = weakness_of(step, "DFW-1006")
    assert (w["status"], w["mitigations"]) == ("unmitigated", [])
    assert (step["unmitigated"], step["worst"]) == (1, "unmitigated")


def test_summary_has_no_weaknesses_and_totals(kb: KnowledgeBase) -> None:
    wf = wf_of(
        [tech("a", "DFT-1001"), tech("h", "DFT-1004"), tech("v", "DFT-1005")],
        [edge("e1", "a", "h"), edge("e2", "h", "v")],
    )
    result = compile_workflow(wf, kb)
    assert all("weaknesses" not in s for s in result["steps"])
    # a: DFW-1001 partial, DFW-1002 unmitigated; h: DFW-1005 mitigated; v: DFW-1006 unmitigated.
    assert result["totals"] == {
        "steps": 3,
        "weaknesses": 4,
        "unmitigated": 2,
        "partial": 1,
        "mitigated": 1,
        "na": 0,
    }
    assert [s["worst"] for s in result["steps"]] == ["unmitigated", "mitigated", "unmitigated"]
    assert result["skipped_ids"] == []
