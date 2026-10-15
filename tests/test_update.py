"""Update operations on an existing workflow."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from mcp_solveit_workflows.kb import KnowledgeBase
from mcp_solveit_workflows.model import (
    EdgeInput,
    NoteInput,
    SelectionInput,
    TechniqueInput,
    WorkflowError,
    ZoneInput,
    read_workflow,
)
from mcp_solveit_workflows.update import SetDataInput, apply_update


def _doc() -> dict[str, Any]:
    return {
        "version": 2,
        "title": "T",
        "nodes": [
            {"id": "a", "type": "technique", "x": 0, "y": 0, "data": {"techniqueId": "DFT-1001"}},
            {"id": "b", "type": "technique", "x": 0, "y": 200, "data": {"techniqueId": "DFT-1004"}},
            {"id": "c", "type": "note", "x": 400, "y": 0, "data": {"text": "hi"}},
            {"id": "z", "type": "subworkflow", "x": 0, "y": 400, "data": {"x": 1}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "z"},
        ],
        "selections": {"a|DFW-1001|DFM-1001": "Y", "b|DFW-1005|DFM-1001": "N"},
        "evaluations": {"b|DFW-1005": {"inherent": {}}, "b/n1|DFW-1": {}, "a|DFW-1001": {}},
        "mode": "assessment",
        "provenance": {"b": {"by": "x"}, "a": {"by": "y"}},
        "futureField": {"keep": True},
    }


def test_remove_node_drops_edges_selections_evaluations_provenance(kb: KnowledgeBase) -> None:
    wf = read_workflow(_doc())
    res = apply_update(wf, parent_of=kb.parent_of, remove_ids=["b"])
    assert [n["id"] for n in wf.nodes] == ["a", "c", "z"]
    assert wf.edges == []
    assert res.dropped == {
        "nodes": ["b"],
        "edges": ["e1", "e2"],
        "selections": ["b|DFW-1005|DFM-1001"],
        "evaluations": ["b|DFW-1005", "b/n1|DFW-1"],
        "provenance": ["b"],
    }
    assert wf.selections == {"a|DFW-1001|DFM-1001": "Y"}
    assert wf.doc["evaluations"] == {"a|DFW-1001": {}}
    assert wf.doc["provenance"] == {"a": {"by": "y"}}
    assert wf.doc["futureField"] == {"keep": True}


def test_remove_edge_only(kb: KnowledgeBase) -> None:
    wf = read_workflow(_doc())
    res = apply_update(wf, parent_of=kb.parent_of, remove_ids=["e1"])
    assert [e["id"] for e in wf.edges] == ["e2"]
    assert res.dropped["edges"] == ["e1"]
    assert res.dropped["nodes"] == []


def test_remove_unknown_id_is_an_error_and_changes_nothing(kb: KnowledgeBase) -> None:
    wf = read_workflow(_doc())
    before = copy.deepcopy(wf.doc)
    with pytest.raises(WorkflowError, match="nope"):
        apply_update(wf, parent_of=kb.parent_of, remove_ids=["b", "nope"])
    assert wf.doc == before


def test_set_data_sets_and_deletes(kb: KnowledgeBase) -> None:
    wf = read_workflow(_doc())
    apply_update(
        wf,
        parent_of=kb.parent_of,
        set_data=[
            SetDataInput(node="a", fields={"label": "Phone", "methods": ["DFT-1002"]}),
            SetDataInput(node="c", fields={"color": "blue"}),
        ],
    )
    assert wf.node("a")["data"] == {  # type: ignore[index]
        "techniqueId": "DFT-1001",
        "label": "Phone",
        "methods": ["DFT-1002"],
    }
    apply_update(
        wf, parent_of=kb.parent_of, set_data=[SetDataInput(node="a", fields={"label": None})]
    )
    assert "label" not in wf.node("a")["data"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("node", "fields", "match"),
    [
        ("a", {"type": "note"}, "type cannot be changed"),
        ("a", {"text": "x"}, "'text' is not a data field"),
        ("c", {"bogus": 1}, "'bogus'"),
        ("z", {"x": 2}, "cannot be set"),
        ("a", {"techniqueId": "DFW-1"}, "DFT-nnnn"),
        ("a", {"target": {"bad key": "v"}}, "route path key"),
        ("c", {"color": "purple"}, "color"),
        ("missing", {"label": "x"}, "no node"),
    ],
)
def test_set_data_refusals_leave_workflow_unchanged(
    kb: KnowledgeBase, node: str, fields: dict[str, Any], match: str
) -> None:
    wf = read_workflow(_doc())
    before = copy.deepcopy(wf.doc)
    with pytest.raises(WorkflowError, match=match):
        apply_update(
            wf,
            parent_of=kb.parent_of,
            remove_ids=["e1"],
            add_nodes=[NoteInput(kind="note", key="k", text="t")],
            set_data=[SetDataInput(node=node, fields=fields)],
        )
    assert wf.doc == before


def test_add_nodes_edges_selections_and_set_data_by_key(kb: KnowledgeBase) -> None:
    wf = read_workflow(_doc())
    res = apply_update(
        wf,
        parent_of=kb.parent_of,
        add_nodes=[
            TechniqueInput(kind="technique", key="new", technique_id="DFT-1005"),
            ZoneInput(kind="zone", key="zone", label="Z", members=["a", "new"]),
        ],
        add_edges=[EdgeInput(source="b", target="new")],
        selections=[
            SelectionInput(step="new", weakness_id="DFW-1006", mitigation_id="DFM-1001", value="NA")
        ],
        set_data=[SetDataInput(node="new", fields={"label": "check"})],
    )
    new_id = res.key_to_id["new"]
    assert new_id not in {"a", "b", "c", "z"}
    assert res.new_node_ids == [new_id, res.key_to_id["zone"]]
    assert res.zone_members[res.key_to_id["zone"]] == ["a", new_id]
    assert wf.node(new_id)["data"]["label"] == "check"  # type: ignore[index]
    assert wf.selections[f"{new_id}|DFW-1006|DFM-1001"] == "NA"
    assert wf.selections["a|DFW-1001|DFM-1001"] == "Y"


def test_removed_ids_are_never_reused(kb: KnowledgeBase) -> None:
    doc = {
        "version": 2,
        "nodes": [
            {"id": "n1", "type": "technique", "x": 0, "y": 0, "data": {"techniqueId": "DFT-1001"}}
        ],
        "edges": [],
        "selections": {},
    }
    wf = read_workflow(doc)
    res = apply_update(
        wf,
        parent_of=kb.parent_of,
        remove_ids=["n1"],
        add_nodes=[TechniqueInput(kind="technique", key="k", technique_id="DFT-1004")],
    )
    assert res.key_to_id["k"] != "n1"
    assert [n["id"] for n in wf.nodes] == [res.key_to_id["k"]]


def test_references_to_removed_nodes_are_refused(kb: KnowledgeBase) -> None:
    wf = read_workflow(_doc())
    before = copy.deepcopy(wf.doc)
    with pytest.raises(WorkflowError, match="removes"):
        apply_update(
            wf,
            parent_of=kb.parent_of,
            remove_ids=["b"],
            add_nodes=[NoteInput(kind="note", key="k", text="t")],
            add_edges=[EdgeInput(source="b", target="k")],
        )
    assert wf.doc == before


def test_removing_a_parent_without_its_child_is_refused(kb: KnowledgeBase) -> None:
    doc = _doc()
    doc["nodes"][2]["parent"] = "a"
    wf = read_workflow(doc)
    with pytest.raises(WorkflowError, match="parent"):
        apply_update(wf, parent_of=kb.parent_of, remove_ids=["a"])
    apply_update(wf, parent_of=kb.parent_of, remove_ids=["a", "c"])
    assert [n["id"] for n in wf.nodes] == ["b", "z"]
