"""Tests for the workflow model: v2 read/write with pass-through, and building."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from conftest import V2_FIXTURES, WORKFLOWS
from pydantic import TypeAdapter, ValidationError

from mcp_solveit_workflows.model import (
    BUILDER_FORMAT,
    FORMAT_VERSION,
    MAX_DESCRIPTION_LEN,
    MAX_NODES,
    DataSourceInput,
    DecisionInput,
    EdgeInput,
    ForkInput,
    IdAllocator,
    JoinInput,
    NodeInput,
    NoteInput,
    PlaceholderInput,
    SelectionInput,
    TechniqueInput,
    Workflow,
    WorkflowError,
    WorkflowInput,
    ZoneInput,
    add_to_workflow,
    build_workflow,
    read_workflow,
)

TOP_ORDER = [
    "version",
    "id",
    "workflowVersion",
    "title",
    "description",
    "nodes",
    "edges",
    "selections",
    "evaluations",
    "mode",
    "provenance",
    "runOverlay",
    "viewport",
]
NODE_ORDER = ["id", "type", "x", "y", "parent", "data"]
EDGE_ORDER = ["id", "source", "target", "sourceHandle", "targetHandle", "label"]

APP_ID = "https://ontology.unifiedcyberontology.org/uco/observable/applicationIdentifier"
ROUTE = (
    "<https://ontology.solveit-df.org/solveit/observable/sourceApplication>"
    "/<https://ontology.unifiedcyberontology.org/uco/core/hasFacet>"
    f"/<{APP_ID}>"
)

# A tiny stand-in for the knowledge base's sub-technique relation.
PARENTS = {"DFT-1137": "DFT-1069", "DFT-1138": "DFT-1069", "DFT-1139": "DFT-1069"}


def parent_of(tid: str) -> str | None:
    return PARENTS.get(tid)


def fixture_text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def in_order(keys: list[str], order: list[str]) -> bool:
    """The known keys appear in builder order, and before any unknown key."""
    known = [k for k in keys if k in order]
    if known != [k for k in order if k in keys]:
        return False
    first_unknown = next((i for i, k in enumerate(keys) if k not in order), len(keys))
    return all(i < first_unknown for i, k in enumerate(keys) if k in order)


def nodes_input(nodes: list[dict[str, Any]]) -> list[NodeInput]:
    adapter: TypeAdapter[list[NodeInput]] = TypeAdapter(list[NodeInput])
    return adapter.validate_python(nodes)


# ---- constants ----


def test_constants() -> None:
    assert FORMAT_VERSION == 2
    assert BUILDER_FORMAT == "0.13.0"
    assert MAX_NODES == 2000
    assert MAX_DESCRIPTION_LEN == 280


# ---- round trip ----


@pytest.mark.parametrize("name", V2_FIXTURES)
@pytest.mark.parametrize("pretty", [True, False])
def test_fixture_round_trips(name: str, pretty: bool) -> None:
    original = json.loads(fixture_text(name))
    out = read_workflow(fixture_text(name)).to_json(pretty=pretty)
    assert json.loads(out) == original
    written = json.loads(out)
    assert in_order(list(written), TOP_ORDER)
    for node in written["nodes"]:
        assert in_order(list(node), NODE_ORDER)
    for edge in written["edges"]:
        assert in_order(list(edge), EDGE_ORDER)
    # Writing is stable: a second pass produces the same text.
    assert read_workflow(out).to_json(pretty=pretty) == out


def test_pretty_and_compact_formats() -> None:
    wf = read_workflow(fixture_text("sample_workflow.json"))
    pretty = wf.to_json()
    assert pretty.startswith('{\n  "version": 2,\n  "title": ')
    compact = wf.to_json(pretty=False)
    assert compact.startswith('{"version":2,"title":')
    assert "\n" not in compact
    assert "—" in compact  # ensure_ascii=False keeps the em dash literal


def test_read_accepts_bytes_and_dict() -> None:
    text = fixture_text("sample_workflow.json")
    from_bytes = read_workflow(text.encode("utf-8"))
    as_dict = json.loads(text)
    from_dict = read_workflow(as_dict)
    assert from_bytes.to_json() == from_dict.to_json()
    # The caller's dict is not mutated by later edits.
    from_dict.nodes.append({"id": "zz", "type": "note", "x": 0, "y": 0, "data": {"text": "x"}})
    assert len(as_dict["nodes"]) == 3


def test_unreleased_fields_survive_unchanged() -> None:
    original = json.loads(fixture_text("synthetic_unreleased_fields.json"))
    wf = read_workflow(fixture_text("synthetic_unreleased_fields.json"))
    written = json.loads(wf.to_json())
    assert written["id"] == original["id"]
    assert written["nodes"][0]["data"]["subWorkflow"] == original["nodes"][0]["data"]["subWorkflow"]
    assert written["evaluations"] == original["evaluations"]


def test_unknown_node_types_and_keys_survive() -> None:
    doc = json.loads(fixture_text("synthetic_unreleased_fields.json"))
    doc["workflowVersion"] = "3"
    doc["futureTop"] = {"a": [1, 2]}
    doc["nodes"].append(
        {
            "id": "sw1",
            "type": "subworkflow",
            "x": 10,
            "y": 20,
            "data": {"ref": "lib/x.json", "pinned": True},
            "extra": [1],
        }
    )
    doc["edges"][0]["data"] = {"kind": "future"}
    # Put the known keys in a scrambled order to check the writer reorders them.
    doc = {
        k: doc[k]
        for k in [
            "futureTop",
            "nodes",
            "workflowVersion",
            *[k for k in doc if k not in ("futureTop", "nodes", "workflowVersion")],
        ]
    }
    written_text = read_workflow(doc).to_json()
    written = json.loads(written_text)
    assert written == doc
    keys = list(written)
    assert keys[:3] == ["version", "id", "workflowVersion"]
    assert keys[-1] == "futureTop"
    sw = written["nodes"][-1]
    assert list(sw) == ["id", "type", "x", "y", "data", "extra"]
    assert list(written["edges"][0])[-1] == "data"


def test_node_key_order_fixed_on_write() -> None:
    doc = {
        "version": 2,
        "selections": {},
        "edges": [{"label": "yes", "target": "b", "source": "a", "id": "e1"}],
        "nodes": [
            {"data": {"text": "x"}, "y": 2, "x": 1, "type": "note", "id": "a", "parent": "g"},
            {"id": "b", "type": "note", "x": 0, "y": 0, "data": {"text": "y"}},
        ],
        "title": "T",
    }
    written = json.loads(read_workflow(doc).to_json())
    assert list(written) == ["version", "title", "nodes", "edges", "selections"]
    assert list(written["nodes"][0]) == ["id", "type", "x", "y", "parent", "data"]
    assert list(written["edges"][0]) == ["id", "source", "target", "label"]


def test_empty_optionals_omitted_and_required_kept() -> None:
    doc = {
        "version": 2,
        "title": "",
        "description": "",
        "evaluations": {},
        "provenance": {},
        "id": None,
    }
    written = json.loads(read_workflow(doc).to_json())
    assert written == {"version": 2, "nodes": [], "edges": [], "selections": {}}


def test_integral_floats_written_as_ints() -> None:
    doc = {
        "version": 2.0,
        "nodes": [
            {
                "id": "a",
                "type": "note",
                "x": 120.0,
                "y": -44.0,
                "data": {"text": "t", "n": [1.0, 2.5]},
            }
        ],
        "edges": [],
        "selections": {},
        "viewport": {"x": 3.0, "y": 4.0, "zoom": 1.0},
    }
    text = read_workflow(doc).to_json(pretty=False)
    assert '"version":2,' in text
    assert '"x":120,"y":-44' in text
    assert '"n":[1,2.5]' in text
    assert '"viewport":{"x":3,"y":4,"zoom":1}' in text
    assert ".0" not in text


def test_workflow_accessors() -> None:
    wf = Workflow({"version": 2})
    assert wf.nodes == [] and wf.edges == [] and wf.selections == {}
    wf.nodes.append({"id": "n1", "type": "note", "x": 0, "y": 0, "data": {"text": "a"}})
    assert wf.doc["nodes"][0]["id"] == "n1"
    assert wf.node("n1") is wf.nodes[0]
    assert wf.node("nope") is None


# ---- read errors ----


def test_v1_refused_with_reexport_message() -> None:
    with pytest.raises(WorkflowError) as exc:
        read_workflow(fixture_text("legacy_v1.json"))
    msg = str(exc.value)
    assert "export" in msg.lower() and "builder" in msg.lower()
    # Never quotes file content.
    assert "Synthetic" not in msg


def test_v1_terse_keys_without_v_refused() -> None:
    with pytest.raises(WorkflowError, match="(?i)export"):
        read_workflow({"n": [], "e": [], "m": {}})


@pytest.mark.parametrize(
    "source, fragment",
    [
        ("not json at all {", "JSON"),
        (b"\xff\xfe\x00garbage", "JSON"),
        ("[1, 2]", "object"),
        ('{"version": 3, "nodes": []}', "version"),
        ('{"version": "2", "nodes": []}', "version"),
        ('{"nodes": []}', "version"),
        ('{"version": 2, "nodes": {}}', "nodes"),
        ('{"version": 2, "edges": "x"}', "edges"),
        ('{"version": 2, "selections": []}', "selections"),
        ('{"version": 2, "evaluations": []}', "evaluations"),
        ('{"version": 2, "viewport": 5}', "viewport"),
        ('{"version": 2, "title": 5}', "title"),
        ('{"version": 2, "description": []}', "description"),
        ('{"version": 2, "mode": "record"}', "mode"),
        ('{"version": 2, "provenance": []}', "provenance"),
        ('{"version": 2, "nodes": [], "x": NaN}', "JSON"),
    ],
)
def test_read_rejects_malformed(source: str | bytes, fragment: str) -> None:
    with pytest.raises(WorkflowError, match=fragment):
        read_workflow(source)


def test_read_error_never_quotes_content() -> None:
    secret = "TOP-SECRET-CASE-NAME"
    with pytest.raises(WorkflowError) as exc:
        read_workflow('{"title": "' + secret + '", "version": 9')
    assert secret not in str(exc.value)


def test_too_many_nodes_refused_on_read() -> None:
    nodes = [
        {"id": f"n{i}", "type": "note", "x": 0, "y": 0, "data": {"text": ""}}
        for i in range(MAX_NODES + 1)
    ]
    with pytest.raises(WorkflowError, match="2000"):
        read_workflow({"version": 2, "nodes": nodes})
    read_workflow({"version": 2, "nodes": nodes[:MAX_NODES]})


# ---- id allocation ----


def test_id_allocator_skips_existing() -> None:
    alloc = IdAllocator(["n1", "n3", "38fVjGwK", "e2", "n2x"])
    assert [alloc.node_id() for _ in range(3)] == ["n2", "n4", "n5"]
    assert [alloc.edge_id() for _ in range(2)] == ["e1", "e3"]


def test_id_allocator_shares_one_namespace() -> None:
    alloc = IdAllocator(["e1", "n1"])
    assert alloc.node_id() == "n2"
    assert alloc.edge_id() == "e2"


# ---- input models ----


def test_input_models_reject_unknown_fields_and_kinds() -> None:
    with pytest.raises(ValidationError):
        nodes_input([{"kind": "technique", "key": "a", "technique_id": "DFT-1001", "bogus": 1}])
    with pytest.raises(ValidationError):
        nodes_input([{"kind": "group", "key": "a"}])
    with pytest.raises(ValidationError):
        nodes_input([{"kind": "note", "key": "a", "text": "t", "color": "purple"}])
    with pytest.raises(ValidationError):
        SelectionInput(step="a", weakness_id="DFW-1", mitigation_id="DFM-1", value="maybe")  # type: ignore[arg-type]


def test_input_schema_has_descriptions() -> None:
    schema = WorkflowInput.model_json_schema()

    def walk(s: Any, path: str) -> None:
        if isinstance(s, dict):
            props = s.get("properties")
            if isinstance(props, dict):
                for name, prop in props.items():
                    if name != "kind":
                        assert "description" in prop, f"{path}.{name} has no description"
            for k, v in s.items():
                walk(v, f"{path}.{k}")
        elif isinstance(s, list):
            for i, v in enumerate(s):
                walk(v, f"{path}[{i}]")

    walk(schema, "WorkflowInput")


# ---- building ----


def full_spec() -> WorkflowInput:
    """Every builder element once."""
    return WorkflowInput(
        title="All elements",
        description="Covers every element the builder can hold.",
        nodes=[
            DataSourceInput(
                kind="data_source",
                key="media",
                label="Disk image",
                classes=[
                    "https://ontology.unifiedcyberontology.org/uco/observable/Image",
                    "free text class",
                ],
            ),
            TechniqueInput(
                kind="technique",
                key="hist",
                technique_id="DFT-1069",
                label="Chrome",
                methods=["DFT-1137", "DFT-1138"],
                target={ROUTE: "com.google.chrome"},
            ),
            PlaceholderInput(
                kind="placeholder",
                key="ph",
                name="Parse thing",
                temp_id="DFT-temp01",
                description="short",
                details="long",
                input_classes=["A"],
                output_classes=["B", "C"],
            ),
            DecisionInput(kind="decision", key="dec", question="Encrypted?"),
            ForkInput(kind="fork", key="fork"),
            JoinInput(kind="join", key="join"),
            NoteInput(kind="note", key="note", text="Remember", color="pink"),
            ZoneInput(
                kind="zone",
                key="zone",
                label="Core",
                members=["hist", "ph"],
                color="green",
                label_corner="bl",
            ),
        ],
        edges=[
            EdgeInput(source="media", target="hist"),
            EdgeInput(source="hist", target="dec", label="then"),
            EdgeInput(source="dec", target="fork", label="no"),
            EdgeInput(source="fork", target="ph"),
            EdgeInput(source="ph", target="join"),
        ],
        selections=[
            SelectionInput(
                step="hist", weakness_id="DFW-1170", mitigation_id="DFM-1001", value="Y"
            ),
            SelectionInput(
                step="hist", weakness_id="DFW-1170", mitigation_id="DFM-1002", value="NA"
            ),
        ],
        viewport={"x": 10.4, "y": -3.0, "zoom": 0.69414},
    )


def test_build_every_element_in_builder_shape() -> None:
    result = build_workflow(full_spec(), parent_of)
    wf = result.workflow
    ids = result.key_to_id
    assert list(ids) == ["media", "hist", "ph", "dec", "fork", "join", "note", "zone"]
    assert list(ids.values()) == [f"n{i}" for i in range(1, 9)]
    assert result.new_node_ids == list(ids.values())
    by_id = {n["id"]: n for n in wf.nodes}

    assert by_id[ids["media"]]["type"] == "caseClass"
    assert by_id[ids["media"]]["data"] == {
        "label": "Disk image",
        "classes": [
            "https://ontology.unifiedcyberontology.org/uco/observable/Image",
            "free text class",
        ],
    }
    assert by_id[ids["hist"]]["data"] == {
        "techniqueId": "DFT-1069",
        "label": "Chrome",
        "methods": ["DFT-1137", "DFT-1138"],
        "target": {ROUTE: "com.google.chrome"},
    }
    assert by_id[ids["ph"]]["data"] == {
        "tempId": "DFT-temp01",
        "name": "Parse thing",
        "description": "short",
        "details": "long",
        "inputClasses": ["A"],
        "outputClasses": ["B", "C"],
    }
    assert by_id[ids["dec"]] == {
        "id": "n4",
        "type": "decision",
        "x": 0,
        "y": 0,
        "data": {"question": "Encrypted?"},
    }
    assert by_id[ids["fork"]]["type"] == "fork"
    assert by_id[ids["fork"]]["data"] == {"mode": "fork"}
    assert by_id[ids["join"]]["data"] == {"mode": "join"}
    assert by_id[ids["note"]]["data"] == {"text": "Remember", "color": "pink"}
    zone = by_id[ids["zone"]]
    assert zone["type"] == "region"
    assert zone["data"] == {"label": "Core", "w": 0, "h": 0, "labelCorner": "bl", "color": "green"}
    assert result.zone_members == {ids["zone"]: [ids["hist"], ids["ph"]]}

    assert wf.edges[0] == {"id": "e1", "source": "n1", "target": "n2"}
    assert wf.edges[1] == {"id": "e2", "source": "n2", "target": "n4", "label": "then"}
    assert wf.selections == {"n2|DFW-1170|DFM-1001": "Y", "n2|DFW-1170|DFM-1002": "NA"}
    assert wf.doc["title"] == "All elements"
    assert wf.doc["viewport"] == {"x": 10, "y": -3, "zoom": 0.6941}

    text = wf.to_json()
    again = read_workflow(text)
    assert json.loads(again.to_json()) == json.loads(text)
    assert again.to_json() == text
    assert list(json.loads(text)) == [
        "version",
        "title",
        "description",
        "nodes",
        "edges",
        "selections",
        "viewport",
    ]


def test_build_minimal_omits_empty_fields() -> None:
    spec = WorkflowInput(
        nodes=nodes_input(
            [
                {"kind": "technique", "key": "a", "technique_id": "DFT-1001"},
                {"kind": "placeholder", "key": "b", "name": "P"},
                {"kind": "data_source", "key": "c", "label": "D"},
                {"kind": "note", "key": "d", "text": "N"},
                {"kind": "zone", "key": "z", "label": "Z", "members": []},
            ]
        )
    )
    result = build_workflow(spec, parent_of)
    data = [n["data"] for n in result.workflow.nodes]
    assert data == [
        {"techniqueId": "DFT-1001"},
        {"name": "P"},
        {"label": "D"},
        {"text": "N"},
        {"label": "Z", "w": 0, "h": 0},
    ]
    assert json.loads(result.workflow.to_json()) == {
        "version": 2,
        "nodes": result.workflow.nodes,
        "edges": [],
        "selections": {},
    }


def test_sub_technique_becomes_parent_with_method() -> None:
    spec = WorkflowInput(
        nodes=[
            TechniqueInput(kind="technique", key="a", technique_id="DFT-1137"),
            TechniqueInput(
                kind="technique", key="b", technique_id="DFT-1137", methods=["DFT-1139", "DFT-1137"]
            ),
        ]
    )
    nodes = build_workflow(spec, parent_of).workflow.nodes
    assert nodes[0]["data"] == {"techniqueId": "DFT-1069", "methods": ["DFT-1137"]}
    assert nodes[1]["data"] == {"techniqueId": "DFT-1069", "methods": ["DFT-1137", "DFT-1139"]}


def test_description_clamped_with_warning() -> None:
    spec = WorkflowInput(description="x" * 300, nodes=[])
    result = build_workflow(spec, parent_of)
    assert result.workflow.doc["description"] == "x" * MAX_DESCRIPTION_LEN
    assert any("280" in w for w in result.warnings)


def test_duplicate_key_rejected() -> None:
    spec = WorkflowInput(
        nodes=nodes_input(
            [
                {"kind": "note", "key": "dup", "text": "a"},
                {"kind": "note", "key": "dup", "text": "b"},
            ]
        )
    )
    with pytest.raises(WorkflowError, match="'dup'"):
        build_workflow(spec, parent_of)


@pytest.mark.parametrize(
    "edges, selections, zone_members, fragment",
    [
        ([{"source": "a", "target": "ghost"}], [], [], "'ghost'"),
        ([{"source": "ghost2", "target": "a"}], [], [], "'ghost2'"),
        (
            [],
            [{"step": "ghost3", "weakness_id": "DFW-1", "mitigation_id": "DFM-1", "value": "Y"}],
            [],
            "'ghost3'",
        ),
        (
            [],
            [{"step": "n", "weakness_id": "DFW-1", "mitigation_id": "DFM-1", "value": "Y"}],
            [],
            "'n'",
        ),
        ([], [], ["ghost4"], "'ghost4'"),
    ],
)
def test_unknown_references_rejected(
    edges: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    zone_members: list[str],
    fragment: str,
) -> None:
    spec = WorkflowInput.model_validate(
        {
            "nodes": [
                {"kind": "technique", "key": "a", "technique_id": "DFT-1001"},
                {"kind": "note", "key": "n", "text": "not a technique"},
                {"kind": "zone", "key": "z", "label": "Z", "members": zone_members},
            ],
            "edges": edges,
            "selections": selections,
        }
    )
    with pytest.raises(WorkflowError, match=fragment):
        build_workflow(spec, parent_of)


@pytest.mark.parametrize(
    "key",
    [
        APP_ID,  # bare IRI: a legacy key, never written
        "<>",
        "<a b>",
        "<a>/b",
        "<a><b>",
        "<a>/",
        "application",
        "<a<b>",
    ],
)
def test_invalid_target_key_rejected(key: str) -> None:
    spec = WorkflowInput(
        nodes=[
            TechniqueInput(
                kind="technique", key="chat", technique_id="DFT-1001", target={key: "x"}
            ),
        ]
    )
    with pytest.raises(WorkflowError, match="'chat'"):
        build_workflow(spec, parent_of)


def test_valid_target_keys_accepted() -> None:
    spec = WorkflowInput(
        nodes=[
            TechniqueInput(
                kind="technique",
                key="t",
                technique_id="DFT-1001",
                target={f"<{APP_ID}>": "com.whatsapp", ROUTE: "com.google.chrome"},
            ),
        ]
    )
    data = build_workflow(spec, parent_of).workflow.nodes[0]["data"]
    assert data["target"] == {f"<{APP_ID}>": "com.whatsapp", ROUTE: "com.google.chrome"}


def test_too_many_nodes_refused_on_build() -> None:
    nodes = [NoteInput(kind="note", key=f"k{i}", text="t") for i in range(MAX_NODES + 1)]
    with pytest.raises(WorkflowError, match="2000"):
        build_workflow(WorkflowInput(nodes=list(nodes)), parent_of)


# ---- adding to an existing workflow ----


def test_add_to_existing_keeps_existing_nodes() -> None:
    wf = read_workflow(fixture_text("synthetic_21_nodes.json"))
    before = copy.deepcopy(wf.doc)
    existing_ids = [n["id"] for n in wf.nodes]
    anchor = next(n["id"] for n in wf.nodes if n["type"] == "technique")
    result = add_to_workflow(
        wf,
        nodes=[
            TechniqueInput(kind="technique", key="new", technique_id="DFT-1138"),
            ZoneInput(kind="zone", key="z", label="Z", members=["new", anchor]),
        ],
        edges=[EdgeInput(source=anchor, target="new", label="next")],
        selections=[
            SelectionInput(step="new", weakness_id="DFW-1", mitigation_id="DFM-1", value="N"),
            SelectionInput(step=anchor, weakness_id="DFW-2", mitigation_id="DFM-2", value="Y"),
        ],
        parent_of=parent_of,
    )
    new_id = result.key_to_id["new"]
    assert new_id not in existing_ids
    assert result.new_node_ids == [new_id, result.key_to_id["z"]]
    assert wf.nodes[: len(existing_ids)] == before["nodes"]
    assert wf.edges[: len(before["edges"])] == before["edges"]
    assert wf.node(new_id) == {
        "id": new_id,
        "type": "technique",
        "x": 0,
        "y": 0,
        "data": {"techniqueId": "DFT-1069", "methods": ["DFT-1138"]},
    }
    new_edge = wf.edges[-1]
    assert new_edge["source"] == anchor and new_edge["target"] == new_id
    assert new_edge["id"] not in {e["id"] for e in before["edges"]}
    assert wf.selections[f"{new_id}|DFW-1|DFM-1"] == "N"
    assert wf.selections[f"{anchor}|DFW-2|DFM-2"] == "Y"
    assert result.zone_members == {result.key_to_id["z"]: [new_id, anchor]}


def test_add_rejects_key_shadowing_existing_id() -> None:
    wf = read_workflow(fixture_text("sample_workflow.json"))
    with pytest.raises(WorkflowError, match="'n1'"):
        add_to_workflow(wf, [NoteInput(kind="note", key="n1", text="x")], [], [], parent_of)


def test_add_selection_on_existing_non_technique_rejected() -> None:
    wf = read_workflow(fixture_text("sample_workflow.json"))  # n3 is a note
    with pytest.raises(WorkflowError, match="'n3'"):
        add_to_workflow(
            wf,
            [],
            [],
            [SelectionInput(step="n3", weakness_id="DFW-1", mitigation_id="DFM-1", value="Y")],
            parent_of,
        )


def test_add_failure_leaves_workflow_unchanged() -> None:
    wf = read_workflow(fixture_text("sample_workflow.json"))
    before = copy.deepcopy(wf.doc)
    with pytest.raises(WorkflowError):
        add_to_workflow(
            wf,
            [NoteInput(kind="note", key="ok", text="x")],
            [EdgeInput(source="ok", target="missing")],
            [],
            parent_of,
        )
    assert wf.doc == before


def test_add_respects_node_limit() -> None:
    nodes = [
        {"id": f"x{i}", "type": "note", "x": 0, "y": 0, "data": {"text": ""}}
        for i in range(MAX_NODES)
    ]
    wf = read_workflow({"version": 2, "nodes": nodes})
    with pytest.raises(WorkflowError, match="2000"):
        add_to_workflow(wf, [NoteInput(kind="note", key="k", text="t")], [], [], parent_of)


@pytest.mark.parametrize(
    "name",
    [
        "synthetic_small.json",
        "synthetic_21_nodes.json",
        "synthetic_63_nodes.json",
    ],
)
def test_builder_exports_round_trip_byte_for_byte(name: str) -> None:
    # These three were exported by the builder; the other fixtures are hand-formatted.
    text = fixture_text(name)
    assert read_workflow(text).to_json() == text.rstrip("\n")


# ---- review fixes ----


def test_lone_surrogate_is_a_workflow_error() -> None:
    text = (
        '{"version": 2, "nodes": [{"id": "n1", "type": "note", "x": 0, "y": 0, '
        '"data": {"text": "bad \\ud800 text"}}], "edges": [], "selections": {}}'
    )
    wf = read_workflow(text)
    with pytest.raises(WorkflowError, match="UTF-8"):
        wf.to_json_pair()
    with pytest.raises(WorkflowError, match="UTF-8"):
        wf.to_json()


@pytest.mark.parametrize("kind", ["note", "zone"])
@pytest.mark.parametrize("end", ["source", "target"])
def test_edge_to_note_or_zone_rejected(kind: str, end: str) -> None:
    other: dict[str, Any] = {"kind": kind, "key": "o"}
    other.update({"text": "x"} if kind == "note" else {"label": "Z", "members": ["t"]})
    edge = {"source": "o", "target": "t"} if end == "source" else {"source": "t", "target": "o"}
    spec = WorkflowInput.model_validate(
        {
            "nodes": [{"kind": "technique", "key": "t", "technique_id": "DFT-1001"}, other],
            "edges": [edge],
        }
    )
    with pytest.raises(WorkflowError) as exc:
        build_workflow(spec, parent_of)
    assert "'t'" in str(exc.value) and "'o'" in str(exc.value)


def test_edge_to_existing_note_rejected() -> None:
    wf = read_workflow(fixture_text("sample_workflow.json"))
    with pytest.raises(WorkflowError, match="n3"):
        add_to_workflow(wf, [], [EdgeInput(source="n1", target="n3")], [], parent_of)


@pytest.mark.parametrize("bad", [["n1"], {"a": 1}, 5])
def test_non_text_node_id_refused_on_read(bad: Any) -> None:
    doc = json.loads(fixture_text("sample_workflow.json"))
    doc["nodes"][0]["id"] = bad
    with pytest.raises(WorkflowError, match="id"):
        read_workflow(doc)


@pytest.mark.parametrize("field", ["id", "source", "target"])
def test_non_text_edge_ids_refused_on_read(field: str) -> None:
    doc = json.loads(fixture_text("sample_workflow.json"))
    doc["edges"][0][field] = ["x"]
    with pytest.raises(WorkflowError, match=field):
        read_workflow(doc)
