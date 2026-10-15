"""Layout, handles and zone boxes."""

from __future__ import annotations

import copy
import itertools
import json
import random
import time
from typing import Any

import pytest
from conftest import V2_FIXTURES, WORKFLOWS
from pydantic import TypeAdapter

from mcp_solveit_workflows import layout as lay
from mcp_solveit_workflows.layout import (
    CARD_W,
    LAYER_GAP,
    SIBLING_GAP,
    ZONE_PAD,
    estimate_size,
    infer_direction,
    layout_workflow,
    place_new_nodes,
    set_edge_handles,
)
from mcp_solveit_workflows.model import (
    EdgeInput,
    NodeInput,
    Workflow,
    WorkflowInput,
    add_to_workflow,
    build_workflow,
    read_workflow,
)

_NODES = TypeAdapter(list[NodeInput])


def _no_parent(_: str) -> str | None:
    return None


def tech(key: str, tid: str = "DFT-1001") -> dict[str, Any]:
    return {"kind": "technique", "key": key, "technique_id": tid}


def edge(s: str, t: str) -> dict[str, Any]:
    return {"source": s, "target": t}


def build(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[Any, dict[str, str]]:
    spec = WorkflowInput.model_validate({"nodes": nodes, "edges": edges})
    res = build_workflow(spec, _no_parent)
    return res, res.key_to_id


def rect(n: dict[str, Any]) -> tuple[float, float, float, float]:
    w, h = estimate_size(n)
    return n["x"], n["y"], w, h


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax, ay, aw, ah = rect(a)
    bx, by, bw, bh = rect(b)
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def cards(wf: Workflow) -> list[dict[str, Any]]:
    return [n for n in wf.nodes if isinstance(n, dict) and n.get("type") != "region"]


def assert_no_overlap(wf: Workflow) -> None:
    cs = cards(wf)
    for i, a in enumerate(cs):
        for b in cs[i + 1 :]:
            assert not overlaps(a, b), f"{a['id']} overlaps {b['id']}"


def edge_between(wf: Workflow, s: str, t: str) -> dict[str, Any]:
    return next(e for e in wf.edges if e["source"] == s and e["target"] == t)


def node(wf: Workflow, nid: str) -> dict[str, Any]:
    n = wf.node(nid)
    assert n is not None
    return n


def chain_spec(n: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = [f"s{i}" for i in range(n)]
    return [tech(k) for k in keys], [edge(a, b) for a, b in zip(keys, keys[1:], strict=False)]


# ---- create path, top to bottom ----


def test_chain_of_five_down() -> None:
    res, ids = build(*chain_spec())
    wf = res.workflow
    assert layout_workflow(wf) == []
    ns = [node(wf, ids[f"s{i}"]) for i in range(5)]
    assert len({n["x"] for n in ns}) == 1
    for a, b in zip(ns, ns[1:], strict=False):
        assert b["y"] == a["y"] + estimate_size(a)[1] + LAYER_GAP["down"]
    assert all(isinstance(n["x"], int) and isinstance(n["y"], int) for n in ns)
    for e in wf.edges:
        assert (e["sourceHandle"], e["targetHandle"]) == ("bottom-s", "top-t")


def test_fork_three_branches_and_join() -> None:
    nodes = [
        tech("a"),
        {"kind": "fork", "key": "f"},
        tech("b1"),
        tech("b2"),
        tech("b3"),
        {"kind": "join", "key": "j"},
        tech("z"),
    ]
    edges = [edge("a", "f")]
    edges += [edge("f", b) for b in ("b1", "b2", "b3")]
    edges += [edge(b, "j") for b in ("b1", "b2", "b3")]
    edges.append(edge("j", "z"))
    res, ids = build(nodes, edges)
    wf = res.workflow
    layout_workflow(wf)
    branches = sorted((node(wf, ids[b]) for b in ("b1", "b2", "b3")), key=lambda n: n["x"])
    assert len({n["y"] for n in branches}) == 1
    step = CARD_W + SIBLING_GAP["down"]
    assert [b["x"] - a["x"] for a, b in zip(branches, branches[1:], strict=False)] == [step] * 2
    # The fork and join sit centred over and under the middle branch.
    mid = branches[1]["x"] + CARD_W / 2
    for bar in (node(wf, ids["f"]), node(wf, ids["j"])):
        assert abs(bar["x"] + estimate_size(bar)[0] / 2 - mid) <= 1
    fork, join = node(wf, ids["f"]), node(wf, ids["j"])
    assert fork["y"] < branches[0]["y"] < join["y"] < node(wf, ids["z"])["y"]
    assert_no_overlap(wf)


CYCLE_NODES = [
    tech("a"),
    {"kind": "decision", "key": "d", "question": "Done?"},
    tech("b"),
    tech("c"),
    tech("z"),
]
CYCLE_EDGES = [edge("a", "d"), edge("d", "b"), edge("b", "c"), edge("c", "d"), edge("d", "z")]


def test_cycle_lays_out_and_back_edge_routes_round_the_side() -> None:
    res, ids = build(CYCLE_NODES, CYCLE_EDGES)
    wf = res.workflow
    layout_workflow(wf)
    assert_no_overlap(wf)
    back = edge_between(wf, ids["c"], ids["d"])
    assert node(wf, ids["c"])["y"] > node(wf, ids["d"])["y"]
    # Leaving and entering on the same side keeps the loop off the forward edges.
    assert back["sourceHandle"] == "right-s" and back["targetHandle"] == "right-t"
    assert all(e.get("sourceHandle") and e.get("targetHandle") for e in wf.edges)


def test_back_edge_left_to_right_routes_underneath() -> None:
    res, ids = build(CYCLE_NODES, CYCLE_EDGES)
    wf = res.workflow
    layout_workflow(wf, direction="right")
    back = edge_between(wf, ids["c"], ids["d"])
    assert node(wf, ids["c"])["x"] > node(wf, ids["d"])["x"]
    assert back["sourceHandle"] == "bottom-s" and back["targetHandle"] == "bottom-t"


def test_new_back_edge_in_update_routes_round_the_side() -> None:
    res, ids = build(CYCLE_NODES[:4], [edge("a", "d"), edge("d", "b"), edge("b", "c")])
    wf = res.workflow
    layout_workflow(wf)
    added = add_to_workflow(wf, [], [EdgeInput(source=ids["c"], target=ids["d"])], [], _no_parent)
    place_new_nodes(wf, added.new_node_ids)
    back = edge_between(wf, ids["c"], ids["d"])
    assert back["sourceHandle"] == "right-s" and back["targetHandle"] == "right-t"


def test_two_components_side_by_side() -> None:
    nodes = [tech("a"), tech("b"), tech("c"), tech("d"), tech("e")]
    res, ids = build(nodes, [edge("a", "b"), edge("c", "d"), edge("c", "e")])
    wf = res.workflow
    layout_workflow(wf)
    assert_no_overlap(wf)
    first_right = max(node(wf, ids[k])["x"] + 240 for k in ("a", "b"))
    assert all(node(wf, ids[k])["x"] >= first_right for k in ("c", "d", "e"))
    # Both components start at the top.
    assert node(wf, ids["a"])["y"] == node(wf, ids["c"])["y"]


def test_two_components_left_to_right_stack() -> None:
    nodes = [tech("a"), tech("b"), tech("c"), tech("d")]
    res, ids = build(nodes, [edge("a", "b"), edge("c", "d")])
    wf = res.workflow
    layout_workflow(wf, direction="right")
    assert_no_overlap(wf)
    first_bottom = max(node(wf, ids[k])["y"] + 96 for k in ("a", "b"))
    assert all(node(wf, ids[k])["y"] >= first_bottom for k in ("c", "d"))


def test_notes_beside_the_flow() -> None:
    nodes, edges = chain_spec(3)
    nodes += [
        {"kind": "note", "key": "n1", "text": "A note"},
        {"kind": "note", "key": "n2", "text": "Another, much longer note " * 5},
    ]
    res, ids = build(nodes, edges)
    wf = res.workflow
    layout_workflow(wf)
    assert_no_overlap(wf)
    flow_right = max(node(wf, ids[f"s{i}"])["x"] + 240 for i in range(3))
    for k in ("n1", "n2"):
        assert node(wf, ids[k])["x"] >= flow_right


def test_notes_below_the_flow_left_to_right() -> None:
    nodes, edges = chain_spec(3)
    nodes.append({"kind": "note", "key": "n1", "text": "A note"})
    res, ids = build(nodes, edges)
    wf = res.workflow
    layout_workflow(wf, direction="right")
    flow_bottom = max(node(wf, ids[f"s{i}"])["y"] + 100 for i in range(3))
    assert node(wf, ids["n1"])["y"] >= flow_bottom


# ---- handles ----


def _placed(nodes: list[tuple[str, int, int]], edges: list[tuple[str, str]]) -> Workflow:
    return Workflow(
        {
            "version": 2,
            "nodes": [
                {"id": i, "type": "technique", "x": x, "y": y, "data": {"techniqueId": "DFT-1"}}
                for i, x, y in nodes
            ],
            "edges": [{"id": f"e{k}", "source": s, "target": t} for k, (s, t) in enumerate(edges)],
            "selections": {},
        }
    )


def test_handles_follow_centre_direction_rule() -> None:
    wf = _placed(
        [
            ("o", 0, 0),
            ("down", 20, 300),
            ("right", 400, 30),
            ("left", -400, -30),
            ("up", -10, -300),
        ],
        [("o", "down"), ("o", "right"), ("o", "left"), ("o", "up")],
    )
    set_edge_handles(wf)
    got = {e["target"]: (e["sourceHandle"], e["targetHandle"]) for e in wf.edges}
    assert got == {
        "down": ("bottom-s", "top-t"),
        "right": ("right-s", "left-t"),
        "left": ("left-s", "right-t"),
        "up": ("top-s", "bottom-t"),
    }


@pytest.mark.parametrize(
    ("direction", "kind", "expected"),
    [
        ("down", "fork", ("bottom-s", "top-t")),
        ("down", "technique", ("bottom-s", "top-t")),
        ("down", "decision", None),
        ("right", "fork", ("right-s", "left-t")),
        ("right", "decision", None),
    ],
)
def test_wide_fan_out_follows_the_flow_except_from_decisions(
    direction: str, kind: str, expected: tuple[str, str] | None
) -> None:
    # Branches further across than along, as under a narrow fork bar.
    if direction == "down":
        wf = _placed([("o", 0, 0), ("a", -500, 200), ("b", 500, 200)], [("o", "a"), ("o", "b")])
    else:
        wf = _placed([("o", 0, 0), ("a", 200, -500), ("b", 200, 500)], [("o", "a"), ("o", "b")])
    wf.nodes[0]["type"] = kind
    set_edge_handles(wf, direction=direction)
    got = [(e["sourceHandle"], e["targetHandle"]) for e in wf.edges]
    if expected is None:
        # A decision keeps the centre rule: its branches leave from its sides.
        side = {"down": ["left-s", "right-s"], "right": ["top-s", "bottom-s"]}[direction]
        assert [g[0] for g in got] == side
    else:
        assert got == [expected, expected]


def test_set_edge_handles_only_named_edges() -> None:
    wf = _placed([("a", 0, 0), ("b", 0, 300), ("c", 400, 0)], [("a", "b"), ("a", "c")])
    set_edge_handles(wf, {"e1"})
    assert "sourceHandle" not in wf.edges[0]
    assert wf.edges[1]["sourceHandle"] == "right-s"


# ---- update path ----


def _load(name: str) -> Workflow:
    return read_workflow((WORKFLOWS / name).read_text())


def _add(wf: Workflow, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    res = add_to_workflow(
        wf,
        _NODES.validate_python(nodes),
        [EdgeInput.model_validate(e) for e in edges],
        [],
        _no_parent,
    )
    return res.new_node_ids


def test_update_fixture_keeps_positions_and_places_successor_below() -> None:
    wf = _load("synthetic_21_nodes.json")
    assert infer_direction(wf) == "down"
    before = {n["id"]: (n["x"], n["y"]) for n in wf.nodes}
    pred = node(wf, "MaiAJbMu")
    edges_before = copy.deepcopy(wf.edges)
    new = _add(wf, [tech("new")], [edge("MaiAJbMu", "new")])
    warnings = place_new_nodes(wf, new)
    assert warnings == []
    for n in wf.nodes:
        if n["id"] in before:
            assert (n["x"], n["y"]) == before[n["id"]]
    added = node(wf, new[0])
    assert added["y"] == pred["y"] + estimate_size(pred)[1] + LAYER_GAP["down"]
    for other in cards(wf):
        if other is not added:
            assert not overlaps(added, other), other["id"]
    # Existing edges are unchanged; the new one has handles.
    assert wf.edges[: len(edges_before)] == edges_before
    new_edge = wf.edges[-1]
    assert new_edge["sourceHandle"] and new_edge["targetHandle"]


def test_update_takes_first_free_slot() -> None:
    res, ids = build([tech("a"), tech("b")], [edge("a", "b")])
    wf = res.workflow
    layout_workflow(wf)
    a, b = node(wf, ids["a"]), node(wf, ids["b"])
    new = _add(wf, [tech("c")], [edge(ids["a"], "c")])
    place_new_nodes(wf, new)
    c = node(wf, new[0])
    assert c["y"] == b["y"]
    assert abs(c["x"] - a["x"]) == CARD_W + SIBLING_GAP["down"]
    assert (a["x"], b["x"]) == (node(wf, ids["a"])["x"], node(wf, ids["b"])["x"])
    assert_no_overlap(wf)


def test_update_new_chain_and_predecessor_before_successor() -> None:
    res, ids = build(*chain_spec(2))
    wf = res.workflow
    layout_workflow(wf)
    s0 = node(wf, ids["s0"])
    new = _add(
        wf,
        [tech("x"), tech("y"), {"kind": "data_source", "key": "ds", "label": "Disk image"}],
        [edge(ids["s1"], "x"), edge("x", "y"), edge("ds", ids["s0"])],
    )
    place_new_nodes(wf, new)
    x, y, ds = (node(wf, i) for i in new)
    assert y["y"] == x["y"] + estimate_size(x)[1] + LAYER_GAP["down"]
    assert ds["y"] + estimate_size(ds)[1] + LAYER_GAP["down"] == s0["y"]
    assert_no_overlap(wf)


def test_update_unconnected_note_goes_to_notes_area() -> None:
    res, ids = build(*chain_spec(3))
    wf = res.workflow
    layout_workflow(wf)
    new = _add(wf, [{"kind": "note", "key": "n", "text": "hi"}], [])
    place_new_nodes(wf, new)
    flow_right = max(node(wf, ids[f"s{i}"])["x"] + 240 for i in range(3))
    assert node(wf, new[0])["x"] >= flow_right


# ---- left to right ----


def test_chain_of_five_left_to_right() -> None:
    res, ids = build(*chain_spec())
    wf = res.workflow
    layout_workflow(wf, direction="right")
    ns = [node(wf, ids[f"s{i}"]) for i in range(5)]
    assert len({n["y"] for n in ns}) == 1
    for a, b in zip(ns, ns[1:], strict=False):
        assert b["x"] == a["x"] + estimate_size(a)[0] + LAYER_GAP["right"]
    for e in wf.edges:
        assert (e["sourceHandle"], e["targetHandle"]) == ("right-s", "left-t")
    assert infer_direction(wf) == "right"


def test_update_left_to_right_infers_direction() -> None:
    res, ids = build(*chain_spec(3))
    wf = res.workflow
    layout_workflow(wf, direction="right")
    before = {n["id"]: (n["x"], n["y"]) for n in wf.nodes}
    last = node(wf, ids["s2"])
    new = _add(wf, [tech("next")], [edge(ids["s2"], "next")])
    place_new_nodes(wf, new)
    added = node(wf, new[0])
    assert added["x"] == last["x"] + estimate_size(last)[0] + LAYER_GAP["right"]
    assert added["y"] == last["y"]
    assert wf.edges[-1]["sourceHandle"] == "right-s"
    for nid, pos in before.items():
        assert (node(wf, nid)["x"], node(wf, nid)["y"]) == pos


def test_infer_direction_ties_default_down() -> None:
    assert infer_direction(Workflow({"version": 2, "nodes": [], "edges": []})) == "down"


# ---- data sources ----


def test_data_source_one_layer_before_first_step() -> None:
    nodes = [{"kind": "data_source", "key": "ds", "label": "Disk image"}, *chain_spec(3)[0]]
    res, ids = build(nodes, [edge("ds", "s0"), *chain_spec(3)[1]])
    wf = res.workflow
    layout_workflow(wf)
    ds, s0 = node(wf, ids["ds"]), node(wf, ids["s0"])
    assert ds["y"] + estimate_size(ds)[1] + LAYER_GAP["down"] == s0["y"]
    e = edge_between(wf, ids["ds"], ids["s0"])
    assert (e["sourceHandle"], e["targetHandle"]) == ("bottom-s", "top-t")


def test_data_source_for_later_step_sits_just_before_it() -> None:
    nodes = [*chain_spec(3)[0], {"kind": "data_source", "key": "ds", "label": "Hash set"}]
    res, ids = build(nodes, [*chain_spec(3)[1], edge("ds", "s2")])
    wf = res.workflow
    layout_workflow(wf)
    ds, s1 = node(wf, ids["ds"]), node(wf, ids["s1"])
    assert ds["y"] == s1["y"]
    assert_no_overlap(wf)


# ---- zones ----


def test_zone_box_contains_members_and_warns_for_intruder() -> None:
    nodes, edges = chain_spec(4)
    nodes.append(
        {"kind": "zone", "key": "z", "label": "Acquisition", "members": ["s0", "s1", "s3"]}
    )
    res, ids = build(nodes, edges)
    wf = res.workflow
    warnings = layout_workflow(wf, zone_members=res.zone_members)
    zone = node(wf, ids["z"])
    zx, zy, zw, zh = zone["x"], zone["y"], zone["data"]["w"], zone["data"]["h"]
    assert all(isinstance(v, int) for v in (zx, zy, zw, zh))
    for k in ("s0", "s1", "s3"):
        x, y, w, h = rect(node(wf, ids[k]))
        assert zx + ZONE_PAD <= x and x + w + ZONE_PAD <= zx + zw
        assert zy + ZONE_PAD <= y and y + h + ZONE_PAD <= zy + zh
    # Drawn behind: the zone stays a region node, which the builder renders below cards.
    assert zone["type"] == "region"
    assert len(warnings) == 1
    assert "Acquisition" in warnings[0] and ids["s2"] in warnings[0]


def test_zone_without_members_warns() -> None:
    nodes, edges = chain_spec(2)
    nodes.append({"kind": "zone", "key": "z", "label": "Empty", "members": []})
    res, ids = build(nodes, edges)
    warnings = layout_workflow(res.workflow, zone_members=res.zone_members)
    assert any("Empty" in w for w in warnings)
    assert node(res.workflow, ids["z"])["data"]["w"] == 0


def test_new_zone_in_update() -> None:
    res, ids = build(*chain_spec(3))
    wf = res.workflow
    layout_workflow(wf)
    out = add_to_workflow(
        wf,
        _NODES.validate_python(
            [{"kind": "zone", "key": "z", "label": "Z", "members": [ids["s0"], ids["s1"]]}]
        ),
        [],
        [],
        _no_parent,
    )
    assert place_new_nodes(wf, out.new_node_ids, zone_members=out.zone_members) == []
    assert node(wf, out.new_node_ids[0])["data"]["w"] > 240


# ---- sizes ----


def test_estimate_size() -> None:
    assert estimate_size({"type": "mystery"}) == (240, 96)
    assert estimate_size({"type": "region", "data": {"w": 500, "h": 300}}) == (500, 300)
    short = estimate_size({"type": "technique", "data": {"techniqueId": "DFT-1"}})
    long = estimate_size({"type": "technique", "data": {"techniqueId": "DFT-1", "label": "x" * 80}})
    assert short[0] == 240 and long[1] > short[1]
    ph = estimate_size({"type": "placeholder", "data": {"name": "n", "description": "d" * 200}})
    assert ph[1] > 96


# ---- scale and fixtures ----


def test_300_nodes_under_one_second() -> None:
    rng = random.Random(7)
    keys = [f"k{i}" for i in range(300)]
    nodes = [tech(k) for k in keys]
    edges = [edge(keys[i], keys[i + 1]) for i in range(299) if rng.random() < 0.7]
    edges += [
        edge(keys[i], keys[j]) for i, j in (sorted(rng.sample(range(300), 2)) for _ in range(150))
    ]
    edges += [
        edge(keys[j], keys[i]) for i, j in (sorted(rng.sample(range(300), 2)) for _ in range(10))
    ]
    res, _ = build(nodes, edges)
    start = time.perf_counter()
    layout_workflow(res.workflow)
    assert time.perf_counter() - start < 1.0
    assert_no_overlap(res.workflow)


@pytest.mark.parametrize("name", V2_FIXTURES)
@pytest.mark.parametrize("direction", ["down", "right"])
def test_relaid_fixtures_have_no_overlaps(name: str, direction: str) -> None:
    wf = Workflow(json.loads((WORKFLOWS / name).read_text()))
    layout_workflow(wf, direction=direction)  # type: ignore[arg-type]
    assert_no_overlap(wf)
    for n in cards(wf):
        assert isinstance(n["x"], int) and isinstance(n["y"], int)
    for e in wf.edges:
        assert e["sourceHandle"].endswith("-s") and e["targetHandle"].endswith("-t")


def test_outer_zone_listed_first_contains_inner_zone() -> None:
    nodes, edges = chain_spec(4)
    nodes.append({"kind": "zone", "key": "outer", "label": "Outer", "members": ["inner", "s2"]})
    nodes.append({"kind": "zone", "key": "inner", "label": "Inner", "members": ["s0", "s1"]})
    res, ids = build(nodes, edges)
    wf = res.workflow
    warnings = layout_workflow(wf, zone_members=res.zone_members)
    outer, inner = node(wf, ids["outer"]), node(wf, ids["inner"])
    ix, iy, iw, ih = inner["x"], inner["y"], inner["data"]["w"], inner["data"]["h"]
    ox, oy, ow, oh = outer["x"], outer["y"], outer["data"]["w"], outer["data"]["h"]
    assert iw > 0 and ih > 0
    assert ox + ZONE_PAD <= ix and ix + iw + ZONE_PAD <= ox + ow
    assert oy + ZONE_PAD <= iy and iy + ih + ZONE_PAD <= oy + oh
    assert warnings == []


def test_zone_member_cycle_still_sizes() -> None:
    nodes, edges = chain_spec(2)
    nodes.append({"kind": "zone", "key": "a", "label": "A", "members": ["b", "s0"]})
    nodes.append({"kind": "zone", "key": "b", "label": "B", "members": ["a", "s1"]})
    res, ids = build(nodes, edges)
    layout_workflow(res.workflow, zone_members=res.zone_members)
    for k in ("a", "b"):
        assert node(res.workflow, ids[k])["data"]["w"] > 0


# ---- layout quality ----


def _behind_cards(wf: Workflow) -> list[tuple[str, str]]:
    """Edges whose drawn curve or label passes behind a card they do not touch."""
    by_id = {n["id"]: n for n in cards(wf)}
    obstacles = lay._Obstacles(cards(wf))
    bad = []
    for e in wf.edges:
        s, t = by_id[e["source"]], by_id[e["target"]]
        pts = lay._curve(s, t, (e["sourceHandle"], e["targetHandle"]), e.get("label"))
        if obstacles.hits(pts, {s["id"], t["id"]}):
            bad.append((e["source"], e["target"]))
    return bad


def _acquisition_spec() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """The shape of a real stress test: a fork, a labelled skip edge, a decision
    with ten branches, a nested decision, a join, and a zone at the far end."""
    nodes = [
        {"kind": "data_source", "key": "ds", "label": "Seized mobile device"},
        tech("faraday"),
        tech("isolate"),
        {"kind": "decision", "key": "locked", "question": "Is the device locked?"},
        {"kind": "fork", "key": "fork"},
        *[tech(f"pin{i}") for i in range(4)],
        {"kind": "join", "key": "pins"},
        {"kind": "decision", "key": "method", "question": "Which acquisition method?"},
        *[tech(f"m{i}") for i in range(10)],
        {"kind": "decision", "key": "logical", "question": "Which logical route?"},
        tech("l1"),
        tech("l2"),
        tech("decode"),
        tech("fs"),
        tech("emmc"),
        {"kind": "join", "key": "phys"},
        tech("hash"),
        tech("store"),
    ]
    edges = [
        edge("ds", "faraday"),
        edge("faraday", "isolate"),
        edge("isolate", "locked"),
        {"source": "locked", "target": "fork", "label": "Yes"},
        {"source": "locked", "target": "method", "label": "No / unlocked"},
        *[edge("fork", f"pin{i}") for i in range(4)],
        *[edge(f"pin{i}", "pins") for i in range(4)],
        edge("pins", "method"),
        *[{"source": "method", "target": f"m{i}", "label": f"Route {i}"} for i in range(10)],
        edge("m2", "logical"),
        {"source": "logical", "target": "l1", "label": "Exposed service"},
        {"source": "logical", "target": "l2", "label": "Device backup"},
        edge("l2", "decode"),
        edge("m4", "fs"),
        edge("m6", "emmc"),
        edge("m5", "phys"),
        edge("emmc", "phys"),
        edge("m7", "phys"),
        edge("phys", "hash"),
        edge("hash", "store"),
    ]
    zone = {"kind": "zone", "key": "cloud", "label": "Cloud", "members": ["m8", "m9"]}
    return nodes, edges, zone


@pytest.mark.parametrize("direction", ["down", "right"])
def test_stress_layout_keeps_connections_and_labels_clear_of_cards(direction: str) -> None:
    nodes, edges, zone = _acquisition_spec()
    res, ids = build([*nodes, zone], edges)
    wf = res.workflow
    zone_members = {ids["cloud"]: [ids["m8"], ids["m9"]]}
    assert layout_workflow(wf, direction=direction, zone_members=zone_members) == []  # type: ignore[arg-type]
    assert_no_overlap(wf)
    assert _behind_cards(wf) == []
    # The decision's branches enter their cards on the leading edge, never along a row.
    lead = "top-t" if direction == "down" else "left-t"
    method = ids["method"]
    for e in wf.edges:
        if e["source"] == method:
            assert e["targetHandle"] == lead


@pytest.mark.parametrize("direction", ["down", "right"])
def test_tree_declared_out_of_order_has_no_crossings(direction: str) -> None:
    rng = random.Random(7)
    keys = [f"t{i}" for i in range(30)]
    edges = [edge(keys[rng.randrange(i)], keys[i]) for i in range(1, 30)]
    shuffled = keys[:]
    rng.shuffle(shuffled)
    res, ids = build([tech(k) for k in shuffled], edges)
    wf = res.workflow
    layout_workflow(wf, direction=direction)  # type: ignore[arg-type]
    assert_no_overlap(wf)
    along, across = ("y", "x") if direction == "down" else ("x", "y")
    pairs = [(node(wf, ids[e["source"]]), node(wf, ids[e["target"]])) for e in edges]
    for (s1, t1), (s2, t2) in itertools.combinations(pairs, 2):
        if s1[along] != s2[along] or t1[along] != t2[along] or s1 is s2:
            continue
        assert (s1[across] - s2[across]) * (t1[across] - t2[across]) >= 0, "edges cross"


@pytest.mark.parametrize("direction", ["down", "right"])
def test_zone_members_stay_together_in_a_row(direction: str) -> None:
    nodes = [tech("root"), *[tech(f"b{i}") for i in range(5)]]
    edges = [edge("root", f"b{i}") for i in range(5)]
    zone = {"kind": "zone", "key": "z", "label": "Z", "members": ["b0", "b4"]}
    res, ids = build([*nodes, zone], edges)
    wf = res.workflow
    members = {ids["z"]: [ids["b0"], ids["b4"]]}
    warnings = layout_workflow(wf, direction=direction, zone_members=members)  # type: ignore[arg-type]
    assert warnings == [], "no stranger inside the zone box"


def test_pack_keeps_order_and_gaps_with_least_movement() -> None:
    # No conflict: everything stays where it wants to be.
    assert lay._pack([0, 500, 1000], [100, 100, 100], 50) == [0, 500, 1000]
    # Two cards wanting the same spot split the difference.
    assert lay._pack([0, 0], [100, 100], 50) == [-75, 75]
    # A heavy item barely moves; the light one makes the room.
    heavy, light = lay._pack([0, 0], [100, 100], 50, [8, 1])
    assert light - heavy == 150 and abs(heavy) < 20


@pytest.mark.parametrize("direction", ["down", "right"])
def test_zone_grouping_wins_even_when_it_costs_a_crossing(direction: str) -> None:
    # a and c have different parents; grouping them puts b outside, at the cost of
    # a crossing. Found by review: the ungrouped starting order used to win.
    nodes = [tech("p"), tech("q"), tech("a"), tech("b"), tech("c")]
    edges = [edge("p", "a"), edge("p", "b"), edge("q", "b"), edge("q", "c")]
    zone = {"kind": "zone", "key": "z", "label": "Z", "members": ["a", "c"]}
    res, ids = build([*nodes, zone], edges)
    wf = res.workflow
    members = {ids["z"]: [ids["a"], ids["c"]]}
    assert layout_workflow(wf, direction=direction, zone_members=members) == []  # type: ignore[arg-type]


def test_zone_is_widened_to_fit_a_long_label() -> None:
    label = "Route B: forensic boot environment, long label"
    res, ids = build(
        [tech("a"), {"kind": "zone", "key": "z", "label": label, "members": ["a"]}], []
    )
    wf = res.workflow
    layout_workflow(wf, zone_members={ids["z"]: [ids["a"]]})
    zone, a = node(wf, ids["z"]), node(wf, ids["a"])
    assert zone["data"]["w"] * lay.ZONE_LABEL_MAX_SHARE >= lay._zone_label_width(label)
    # Still centred on its member.
    assert abs((zone["x"] + zone["data"]["w"] / 2) - (a["x"] + CARD_W / 2)) <= 1


def _laptop_spec() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    nodes = [
        {"kind": "decision", "key": "d", "question": "Can the internal disk be removed?"},
        tech("a1"),
        tech("a2"),
        tech("b1"),
        tech("b2"),
        {"kind": "join", "key": "j"},
        {"kind": "zone", "key": "za", "label": "Route A: disk removal", "members": ["a1", "a2"]},
        {
            "kind": "zone",
            "key": "zb",
            "label": "Route B: forensic boot environment",
            "members": ["b1", "b2"],
        },
    ]
    edges = [
        {"source": "d", "target": "a1", "label": "Yes"},
        {"source": "d", "target": "b1", "label": "No (e.g. soldered SSD)"},
        edge("a1", "a2"),
        edge("b1", "b2"),
        edge("a2", "j"),
        edge("b2", "j"),
    ]
    return nodes, edges, {"za": ["a1", "a2"], "zb": ["b1", "b2"]}


def test_connections_do_not_cross_zone_labels() -> None:
    nodes, edges, members = _laptop_spec()
    res, ids = build(nodes, edges)
    wf = res.workflow
    zm = {ids[z]: [ids[m] for m in ms] for z, ms in members.items()}
    assert layout_workflow(wf, zone_members=zm) == []
    by_id = {n["id"]: n for n in cards(wf)}
    curves = [
        lay._curve(by_id[e["source"]], by_id[e["target"]], (e["sourceHandle"], e["targetHandle"]))
        for e in wf.edges
    ]
    for z in zm:
        zone = node(wf, z)
        d = zone["data"]
        chip_w = lay._zone_label_width(d["label"])
        assert chip_w <= d["w"] * lay.ZONE_LABEL_MAX_SHARE
        if d.get("labelCorner") == "tl":
            lo = zone["x"] + lay.ZONE_CHIP_INSET
        else:
            lo = zone["x"] + d["w"] - lay.ZONE_CHIP_INSET - chip_w
        top = zone["y"] + lay.ZONE_CHIP_TOP
        for pts in curves:
            for px, py in pts:
                assert not (lo <= px <= lo + chip_w and top <= py <= top + lay.ZONE_CHIP_H), z


def test_explicit_label_corner_is_kept() -> None:
    nodes, edges, members = _laptop_spec()
    nodes[-2] = {**nodes[-2], "label_corner": "tr"}
    res, ids = build(nodes, edges)
    wf = res.workflow
    zm = {ids[z]: [ids[m] for m in ms] for z, ms in members.items()}
    layout_workflow(wf, zone_members=zm)
    assert node(wf, ids["za"])["data"]["labelCorner"] == "tr"
