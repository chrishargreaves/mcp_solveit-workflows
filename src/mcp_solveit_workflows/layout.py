"""Node positions, edge handles and zone boxes.

A port of the deployed Workflow Builder's layered layout (`src/state/caseImport.ts`,
`buildWorkflowFromQuads`) with a direction setting:

- "down" (the builder's own): each layer is a row; siblings are spread across
  `X_STEP` apart, centred on `ORIGIN_X`; each row starts below the tallest
  estimated card of the row above, plus `ROW_GAP`.
- "right": the same layering transposed. Each layer is a column; columns are
  spaced by the widest estimated card plus `COL_GAP`; siblings stack vertically.

Back-edges found by depth-first search are ignored for layering, so cycles lay
out (the builder's import puts leftover nodes in overflow rows instead).

Positions are a node's top-left corner, as in the builder. Within a layer, cards
share the layer's leading edge (top in "down", left in "right") and are centred
on their slot across the flow, so a narrow fork bar sits over its branches.

Handles use the builder's centre-direction rule (`src/lib/edgeHandles.ts`
`defaultEdgeHandles`, same rule as `scripts/pin-handles.mjs` in auto mode), with
handle IDs from `src/components/Canvas/nodes/SideHandles.tsx`.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Literal

from .model import FLOW_TYPES, Workflow, is_number

Direction = Literal["down", "right"]

# ---- constants ported from the builder ----

# caseImport.ts: X_STEP, ORIGIN_X, ORIGIN_Y, ROW_GAP.
X_STEP = 320
ORIGIN_X = 120
ORIGIN_Y = 80
ROW_GAP = 44
# caseImport.ts: characters per line inside a 248px card at 13px, and line height.
CHARS_PER_LINE = 34
LINE_HEIGHT = 18
# edgeHandles.ts: fallback card size (NODE_W, NODE_H), as in FlowCanvas.
CARD_W = 240
CARD_H = 96

# ---- the server's own spacing ----

# Gap between columns in a left-to-right layout (X_STEP less a card width),
# used when stacking notes and placing nodes on update.
COL_GAP = X_STEP - CARD_W
# Gaps on create, wider than the builder's import so connection labels and the
# builder's data chips have room: between layers along the flow, and between
# neighbouring cards across it.
LAYER_GAP: dict[str, float] = {"down": 88, "right": 120}
SIBLING_GAP: dict[str, float] = {"down": 96, "right": 56}
# Room reserved in a layer for a connection passing through it, and how strongly
# the connection's ends are pulled into line with that room.
DUMMY_SIZE = 24
DUMMY_PULL = 8.0
# How strongly a zone's members are pulled towards the zone's mean position,
# so a zone spanning several layers lines up as one column.
ZONE_PULL = 2.0
# Barycentre sweeps when ordering layers, and placement sweeps across the flow.
ORDER_SWEEPS = 12
PLACE_SWEEPS = 8
# Gap between disconnected components, and between the flow and the notes area.
COMPONENT_GAP = 80
# Zone padding around its members' bounding box, plus room for the label on the
# side of its label corner (RegionNode.tsx; the default corner is "tr").
ZONE_PAD = 40
ZONE_LABEL_PAD = 28
# The label chip (.wf-region-label) is at most 70% of the zone's width, and is
# about 6.3px per character plus padding when measured; these allow some slack.
ZONE_LABEL_MAX_SHARE = 0.7
ZONE_LABEL_CHAR_W = 7.0
ZONE_LABEL_CHIP_PAD = 24
# The chip sits 6px from the top and 8px from the side, about 24px tall; a
# connection should pass at least ZONE_CHIP_MARGIN clear of it.
ZONE_CHIP_TOP = 6
ZONE_CHIP_INSET = 8
ZONE_CHIP_H = 24
ZONE_CHIP_MARGIN = 10
# Clearance kept between a newly placed card and any existing one.
PLACE_MARGIN = 16
# Gap between a zone member and a neighbour outside the zone, so the zone box,
# with its padding and label, does not take the neighbour in.
ZONE_CLEARANCE = ZONE_PAD + ZONE_LABEL_PAD + PLACE_MARGIN

# flowOrder.ts FLOW_TYPES, plus data sources, which the server layers with the
# steps they feed.
LAYERED_TYPES = FLOW_TYPES | {"caseClass"}

# Per-kind sizes from src/components/Canvas/canvas.css:
#   .wf-decision 124x124; .wf-forkjoin min-width 94px, height 24px;
#   .wf-caseclass min-width 188px, max-width 268px (taken at its maximum);
#   .wf-node (techniques, placeholders, notes) max-width 248px, which the
#   builder's own fallback rounds to CARD_W.
DECISION_SIZE = (124, 124)
FORKJOIN_SIZE = (94, 24)
CASECLASS_W = 268

_Rect = tuple[float, float, float, float]
# A node ID, or a dummy marking where an edge crosses a layer: ("dummy", source, target, depth).
_Vertex = str | tuple[str, str, str, int]


def _lines(text: object) -> int:
    return math.ceil(len(text) / CHARS_PER_LINE) if isinstance(text, str) else 0


def _data(node: Mapping[str, Any]) -> Mapping[str, Any]:
    d = node.get("data")
    return d if isinstance(d, Mapping) else {}


def _num(v: object) -> float:
    if is_number(v) and math.isfinite(v):
        return float(v)
    return 0.0


def estimate_size(node: Mapping[str, Any]) -> tuple[int, int]:
    """Rough rendered (width, height) of a node's card, in canvas units.

    Techniques and placeholders follow the builder's `estimateHeight`
    (caseImport.ts), which cannot see knowledge base titles and assumes a title
    wrapping to two lines. Other kinds use constants from the builder's CSS.
    """
    kind = node.get("type")
    d = _data(node)
    if kind == "technique":
        # estimateHeight: 100 + the instance label's lines. The server adds a row
        # for the target line and one for a method stack (TechniqueNode.tsx shows
        # both), which the import never produces.
        h = 100 + _lines(d.get("label")) * LINE_HEIGHT
        target = d.get("target")
        if isinstance(target, Mapping) and target:
            h += LINE_HEIGHT * max(1, _lines(" ".join(str(v) for v in target.values())))
        if d.get("methods"):
            h += LINE_HEIGHT
        return CARD_W, h
    if kind == "placeholder":
        # estimateHeight: 70 + (name lines + description lines) * LINE_HEIGHT.
        name = d.get("name") if isinstance(d.get("name"), str) and d.get("name") else None
        h = 70 + (_lines(name or "Undisclosed step") + _lines(d.get("description"))) * LINE_HEIGHT
        # The server's addition: a chip row each for input and output classes.
        h += LINE_HEIGHT * (bool(d.get("inputClasses")) + bool(d.get("outputClasses")))
        return CARD_W, h
    if kind == "decision":
        return DECISION_SIZE
    if kind in ("fork", "join"):
        return FORKJOIN_SIZE
    if kind == "caseClass":
        # "Data" kind row, the label, and one list row per class (CaseClassNode.tsx),
        # inside 10px/12px padding.
        classes = d.get("classes")
        n_classes = len(classes) if isinstance(classes, list) else 0
        return CASECLASS_W, 44 + max(1, _lines(d.get("label"))) * LINE_HEIGHT + (
            n_classes * LINE_HEIGHT
        )
    if kind == "note":
        # .wf-note-text at 13px, line-height 1.35, inside .wf-node's 9px padding.
        return CARD_W, 28 + max(1, _lines(d.get("text"))) * LINE_HEIGHT
    if kind == "region":
        return int(_num(d.get("w"))), int(_num(d.get("h")))
    return CARD_W, CARD_H


# ---- geometry helpers ----


def _dict_nodes(wf: Workflow) -> list[dict[str, Any]]:
    return [n for n in wf.nodes if isinstance(n, dict) and isinstance(n.get("id"), str)]


def _dict_edges(wf: Workflow) -> list[dict[str, Any]]:
    return [
        e
        for e in wf.edges
        if isinstance(e, dict)
        and isinstance(e.get("source"), str)
        and isinstance(e.get("target"), str)
    ]


def _rect(node: Mapping[str, Any]) -> _Rect:
    w, h = estimate_size(node)
    return _num(node.get("x")), _num(node.get("y")), float(w), float(h)


def _centre(node: Mapping[str, Any]) -> tuple[float, float]:
    x, y, w, h = _rect(node)
    return x + w / 2, y + h / 2


def _intersects(a: _Rect, b: _Rect, margin: float = 0.0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax < bx + bw + margin
        and bx < ax + aw + margin
        and ay < by + bh + margin
        and by < ay + ah + margin
    )


def _bbox(rects: Iterable[_Rect]) -> _Rect | None:
    rs = list(rects)
    if not rs:
        return None
    x0 = min(r[0] for r in rs)
    y0 = min(r[1] for r in rs)
    x1 = max(r[0] + r[2] for r in rs)
    y1 = max(r[1] + r[3] for r in rs)
    return x0, y0, x1 - x0, y1 - y0


# ---- handles ----


def _handles(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: Direction | None = None,
) -> tuple[str, str]:
    """The preferred handles for an edge, before obstacles are considered.

    Without a flow direction this is the builder's centre-direction rule
    (edgeHandles.ts defaultEdgeHandles). With one, the server's own rules apply:

    - An edge running with the flow leaves the flow side and enters the opposite
      one, even when its target is further across than along (branches fanning
      out from a fork).
    - A decision's branches leave from its sides when the target is off to one
      side, and still enter the target on its leading edge, so they do not run
      along a row of cards to reach a side handle.
    - An edge running against the flow (a loop back to an earlier step) leaves
      and enters on the same side, so it is drawn round the forward edges.
    """
    return _handle_candidates(source, target, direction)[0]


def _handle_candidates(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: Direction | None,
) -> list[tuple[str, str]]:
    """Handle pairs for an edge, most preferred first."""
    sx, sy = _centre(source)
    tx, ty = _centre(target)
    dx, dy = tx - sx, ty - sy
    if direction is None:
        if abs(dx) > abs(dy):
            return [("right-s", "left-t") if dx >= 0 else ("left-s", "right-t")]
        return [("bottom-s", "top-t") if dy >= 0 else ("top-s", "bottom-t")]

    # Work in flow terms: "along" runs with the flow, "across" at right angles.
    if direction == "down":
        along, across = dy, dx
        fwd, back, lo, hi = "bottom", "top", "left", "right"
        sw = _rect(source)[2]
        same_row = (_rect(source)[3] + _rect(target)[3]) / 2
    else:
        along, across = dx, dy
        fwd, back, lo, hi = "right", "left", "top", "bottom"
        sw = _rect(source)[3]
        same_row = (_rect(source)[2] + _rect(target)[2]) / 2
    near, far = (lo, hi) if across < 0 else (hi, lo)

    def pair(s: str, t: str) -> tuple[str, str]:
        return (f"{s}-s", f"{t}-t")

    if abs(along) < same_row:  # same layer
        return [pair(near, far), pair(fwd, fwd), pair(back, back)]
    if along < 0:  # loop back
        return [pair(hi, hi), pair(lo, lo)]
    forward = [
        pair(fwd, back),
        pair(near, back),
        pair(far, back),
        pair(near, near),
        pair(far, far),
    ]
    if source.get("type") == "decision" and abs(across) > sw / 2:
        forward.insert(0, forward.pop(1))  # leave from the side facing the target
    return forward


# Bezier curvature used by React Flow's getBezierPath, which the builder draws with.
BEZIER_CURVATURE = 0.25
# Points sampled along a drawn connection when testing it against cards.
CURVE_SAMPLES = 24
# Clearance a connection should keep from a card it does not touch.
CURVE_MARGIN = 6.0
# Estimated size of a connection label chip (.wf-edge-label, 11px text).
LABEL_CHAR_W = 6.5
LABEL_PAD_W = 14
LABEL_H = 20
# Cell size of the spatial index used for that test.
GRID_CELL = 400.0


def _anchor(node: Mapping[str, Any], side: str) -> tuple[float, float]:
    x, y, w, h = _rect(node)
    if side == "top":
        return x + w / 2, y
    if side == "bottom":
        return x + w / 2, y + h
    if side == "left":
        return x, y + h / 2
    return x + w, y + h / 2


def _control(side: str, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """React Flow getControlWithCurvature."""

    def offset(distance: float) -> float:
        if distance >= 0:
            return 0.5 * distance
        return BEZIER_CURVATURE * 25 * math.sqrt(-distance)

    if side == "left":
        return x1 - offset(x1 - x2), y1
    if side == "right":
        return x1 + offset(x2 - x1), y1
    if side == "top":
        return x1, y1 - offset(y1 - y2)
    return x1, y1 + offset(y2 - y1)


def _curve(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    handles: tuple[str, str],
    label: object = None,
) -> list[tuple[float, float]]:
    """Points along the builder's bezier for this edge and handle pair, plus points
    covering its label, which the builder centres on the curve's midpoint."""
    s_side = handles[0].rsplit("-", 1)[0]
    t_side = handles[1].rsplit("-", 1)[0]
    p0 = _anchor(source, s_side)
    p3 = _anchor(target, t_side)
    p1 = _control(s_side, *p0, *p3)
    p2 = _control(t_side, *p3, *p0)
    pts = []
    for k in range(1, CURVE_SAMPLES):
        t = k / CURVE_SAMPLES
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
        pts.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            )
        )
    if isinstance(label, str) and label.strip():
        mx = (p0[0] + 3 * p1[0] + 3 * p2[0] + p3[0]) / 8
        my = (p0[1] + 3 * p1[1] + 3 * p2[1] + p3[1]) / 8
        half_w = (len(label) * LABEL_CHAR_W + LABEL_PAD_W) / 2
        half_h = LABEL_H / 2
        for fx in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for fy in (-1.0, 0.0, 1.0):
                pts.append((mx + fx * half_w, my + fy * half_h))
    return pts


class _Obstacles:
    """Cards a connection should not pass behind, in a coarse grid."""

    def __init__(self, nodes: Iterable[Mapping[str, Any]]) -> None:
        self.rects: dict[str, _Rect] = {}
        self.grid: dict[tuple[int, int], list[str]] = {}
        for n in nodes:
            if n.get("type") == "region":
                continue
            r = _rect(n)
            self.rects[n["id"]] = r
            for cell in self._cells(r):
                self.grid.setdefault(cell, []).append(n["id"])

    def _cells(self, r: _Rect) -> Iterable[tuple[int, int]]:
        x0, y0 = int(r[0] // GRID_CELL), int(r[1] // GRID_CELL)
        x1, y1 = int((r[0] + r[2]) // GRID_CELL), int((r[1] + r[3]) // GRID_CELL)
        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                yield i, j

    def hits(self, points: Iterable[tuple[float, float]], skip: Collection[str]) -> int:
        """How many cards, other than those in `skip`, the points pass through."""
        hit: set[str] = set()
        m = CURVE_MARGIN
        for px, py in points:
            for nid in self.grid.get((int(px // GRID_CELL), int(py // GRID_CELL)), ()):
                if nid in hit or nid in skip:
                    continue
                x, y, w, h = self.rects[nid]
                if x - m <= px <= x + w + m and y - m <= py <= y + h + m:
                    hit.add(nid)
        return len(hit)


def _choose_handles(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: Direction | None,
    obstacles: _Obstacles | None,
    label: object = None,
) -> tuple[str, str]:
    """The most preferred handle pair whose drawn curve and label pass behind the
    fewest cards."""
    candidates = _handle_candidates(source, target, direction)
    if obstacles is None or len(candidates) == 1:
        return candidates[0]
    skip = {source["id"], target["id"]}
    best, best_hits = candidates[0], math.inf
    for cand in candidates:
        hits = obstacles.hits(_curve(source, target, cand, label), skip)
        if hits < best_hits:
            best, best_hits = cand, hits
        if hits == 0:
            break
    return best


def set_edge_handles(
    wf: Workflow,
    edge_ids: Collection[str] | None = None,
    *,
    direction: Direction | None = None,
) -> None:
    """Set both handles on the named edges (all edges when None).

    Edges whose endpoints cannot be found are left unchanged, as in the builder.
    With a `direction`, each edge takes the most preferred handles (see _handles)
    whose curve does not pass behind another card, where such a pair exists.
    """
    nodes = _dict_nodes(wf)
    by_id = {n["id"]: n for n in nodes}
    obstacles = _Obstacles(nodes) if direction is not None else None
    for e in _dict_edges(wf):
        if edge_ids is not None and e.get("id") not in edge_ids:
            continue
        s, t = by_id.get(e["source"]), by_id.get(e["target"])
        if s is None or t is None:
            continue
        e["sourceHandle"], e["targetHandle"] = _choose_handles(
            s, t, direction, obstacles, e.get("label")
        )


def infer_direction(wf: Workflow) -> Direction:
    """The dominant direction of the existing edges between positioned nodes.

    Signed centre deltas are summed, so branches fanning out to both sides of a
    top-to-bottom flow cancel out instead of reading as sideways. Ties and
    workflows with no usable edges are "down", the builder's own direction.
    """
    by_id = {n["id"]: n for n in _dict_nodes(wf) if n.get("type") != "region"}
    sum_dx = sum_dy = 0.0
    for e in _dict_edges(wf):
        s, t = by_id.get(e["source"]), by_id.get(e["target"])
        if s is None or t is None or s is t:
            continue
        sx, sy = _centre(s)
        tx, ty = _centre(t)
        sum_dx += tx - sx
        sum_dy += ty - sy
    return "right" if sum_dx > sum_dy else "down"


# ---- layering ----


def _graph(
    ids: Sequence[str], edges: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Successor and predecessor lists among `ids`, with back-edges removed.

    Back-edges are found by an iterative depth-first search that starts from
    each node in node order.
    """
    idset = set(ids)
    out: dict[str, list[str]] = {i: [] for i in ids}
    for e in edges:
        s, t = e["source"], e["target"]
        if s in idset and t in idset and s != t and t not in out[s]:
            out[s].append(t)

    state: dict[str, int] = {}  # 1 = on the stack, 2 = done
    back: set[tuple[str, str]] = set()
    for root in ids:
        if root in state:
            continue
        state[root] = 1
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            v, i = stack[-1]
            if i < len(out[v]):
                stack[-1] = (v, i + 1)
                w = out[v][i]
                st = state.get(w)
                if st == 1:
                    back.add((v, w))
                elif st is None:
                    state[w] = 1
                    stack.append((w, 0))
            else:
                state[v] = 2
                stack.pop()

    succ = {v: [w for w in out[v] if (v, w) not in back] for v in ids}
    pred: dict[str, list[str]] = {i: [] for i in ids}
    for v in ids:
        for w in succ[v]:
            pred[w].append(v)
    return succ, pred


def _topo(
    ids: Sequence[str], succ: Mapping[str, list[str]], pred: Mapping[str, list[str]]
) -> list[str]:
    """Kahn's algorithm, ties in node order (the graph is acyclic here)."""
    order_of = {v: k for k, v in enumerate(ids)}
    indeg = {v: len(pred[v]) for v in ids}
    ready = [(order_of[v], v) for v in ids if indeg[v] == 0]
    heapq.heapify(ready)
    out: list[str] = []
    while ready:
        _, v = heapq.heappop(ready)
        out.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, (order_of[w], w))
    return out


def _depths(
    ids: Sequence[str], succ: Mapping[str, list[str]], pred: Mapping[str, list[str]]
) -> dict[str, int]:
    """Longest path from a root; then roots are pulled down to just before their
    earliest successor, so a data source sits one layer before its consumer."""
    order = _topo(ids, succ, pred)
    depth: dict[str, int] = {}
    for v in order:
        depth[v] = max((depth[p] + 1 for p in pred[v]), default=0)
    for v in reversed(order):
        if not pred[v] and succ[v]:
            depth[v] = min(depth[w] for w in succ[v]) - 1
    return depth


def _components(ids: Sequence[str], succ: Mapping[str, list[str]]) -> list[list[str]]:
    adj: dict[str, set[str]] = {v: set() for v in ids}
    for v in ids:
        for w in succ[v]:
            adj[v].add(w)
            adj[w].add(v)
    comp_of: dict[str, int] = {}
    n_comps = 0
    for root in ids:
        if root in comp_of:
            continue
        c = comp_of[root] = n_comps
        n_comps += 1
        stack = [root]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if w not in comp_of:
                    comp_of[w] = c
                    stack.append(w)
    comps: list[list[str]] = [[] for _ in range(n_comps)]
    for v in ids:  # keep node order
        comps[comp_of[v]].append(v)
    return comps


def _layout_component(
    comp: Sequence[str],
    by_id: Mapping[str, dict[str, Any]],
    succ: Mapping[str, list[str]],
    pred: Mapping[str, list[str]],
    depth: Mapping[str, int],
    direction: Direction,
    zone_of: Mapping[str, str],
    edge_labels: Mapping[tuple[str, str], str],
    zone_extra: Mapping[str, float] | None = None,
) -> None:
    """Sugiyama-style placement of one connected component.

    1. Layers come from `depth`. An edge spanning several layers gets a dummy in
       each layer it passes, so it takes part in ordering like a real branch.
    2. Each layer is ordered by repeated barycentre sweeps, down then up, keeping
       the order with the fewest crossings. Members of the same zone are kept
       together, so a zone box never encloses a stranger.
    3. Across the flow, each card moves towards the mean of its neighbours,
       solved exactly per layer with the cards' order and gaps as constraints.
       Dummies pull hard, so a long edge's ends line up with the gap reserved
       for it and the builder's curve runs down that gap, not through a card.
    """
    order_of = {v: k for k, v in enumerate(comp)}
    base = min(depth[v] for v in comp)
    n_layers = max(depth[v] for v in comp) - base + 1

    # ---- layered graph with dummies ----
    layers: list[list[_Vertex]] = [[] for _ in range(n_layers)]
    up: dict[_Vertex, list[_Vertex]] = {}  # neighbours in the layer above
    down: dict[_Vertex, list[_Vertex]] = {}  # neighbours in the layer below
    for v in comp:
        layers[depth[v] - base].append(v)
        up.setdefault(v, [])
        down.setdefault(v, [])
    for v in comp:
        for w in succ[v]:
            prev: _Vertex = v
            for d in range(depth[v] + 1, depth[w]):
                dummy: _Vertex = ("dummy", v, w, d)
                layers[d - base].append(dummy)
                up[dummy] = [prev]
                down[dummy] = []
                down[prev].append(dummy)
                prev = dummy
            down[prev].append(w)
            up[w].append(prev)

    def tie(x: _Vertex) -> float:
        return float(order_of[x]) if isinstance(x, str) else float(order_of[x[1]])

    # Initial order: one top-down barycentre pass, ties in node order.
    pos: dict[_Vertex, float] = {}
    for layer in layers:
        # Nodes with nothing above (data sources pulled down to their consumer) go last.
        layer.sort(
            key=lambda x: (0, _mean([pos[u] for u in up[x]]), tie(x)) if up[x] else (1, 0.0, tie(x))
        )
        for k, x in enumerate(layer):
            pos[x] = float(k)

    def zone_key(x: _Vertex) -> str | None:
        return zone_of.get(x) if isinstance(x, str) else None

    def reorder(layer: list[_Vertex], nbrs: Mapping[_Vertex, list[_Vertex]]) -> None:
        bary = {x: _mean([pos[u] for u in nbrs[x]], default=pos[x]) for x in layer}
        # Zone members share their zone's mean, so they stay adjacent.
        groups: dict[str, list[float]] = {}
        for x in layer:
            z = zone_key(x)
            if z is not None:
                groups.setdefault(z, []).append(bary[x])

        def key(x: _Vertex) -> tuple[float, str, float, float]:
            z = zone_key(x)
            if z is None:
                return (bary[x], "", bary[x], pos[x])
            return (_mean(groups[z]), z, bary[x], pos[x])

        layer.sort(key=key)
        for k, x in enumerate(layer):
            pos[x] = float(k)

    best = [list(layer) for layer in layers]
    # The initial order ignores zones, so it never wins: the first sweep is always taken.
    best_cross = math.inf
    for sweep in range(ORDER_SWEEPS):
        if sweep % 2 == 0:
            for layer in layers[1:]:
                reorder(layer, up)
        else:
            for layer in reversed(layers[:-1]):
                reorder(layer, down)
        crossed = _crossings(layers, down, pos)
        # Ties go to the later order, which has had zone members grouped.
        if crossed <= best_cross:
            best, best_cross = [list(layer) for layer in layers], crossed
        if best_cross == 0:
            break
    layers = best

    # ---- coordinates across the flow ----
    across = 0 if direction == "down" else 1  # index into (w, h)
    gap = SIBLING_GAP[direction]
    size: dict[_Vertex, tuple[int, int]] = {}
    for layer in layers:
        for x in layer:
            if isinstance(x, str):
                size[x] = estimate_size(by_id[x])
            else:
                # Room for the edge's label too, wherever along it the label falls.
                label = edge_labels.get((x[1], x[2]), "")
                room = DUMMY_SIZE + (round(len(label) * LABEL_CHAR_W + LABEL_PAD_W) if label else 0)
                size[x] = (room, room) if direction == "down" else (room, DUMMY_SIZE + LABEL_H)

    centre: dict[_Vertex, float] = {}
    for layer in layers:
        extents = [size[x][across] for x in layer]
        total = sum(extents) + max(gap, ZONE_CLEARANCE) * (len(layer) - 1)
        at = -total / 2
        for x, ext in zip(layer, extents, strict=True):
            centre[x] = at + ext / 2
            at += ext + max(gap, ZONE_CLEARANCE)

    # Zone members across all layers, so a zone spanning several layers can be
    # pulled into one column.
    zone_items: dict[str, list[_Vertex]] = {}
    for layer in layers:
        for x in layer:
            z = zone_key(x)
            if z is not None:
                zone_items.setdefault(z, []).append(x)

    def settle(layer: list[_Vertex], nbrs_of: Sequence[Mapping[_Vertex, list[_Vertex]]]) -> None:
        desired = []
        for x in layer:
            total = weight = 0.0
            for nbrs in nbrs_of:
                for u in nbrs[x]:
                    pull = DUMMY_PULL if not (isinstance(x, str) and isinstance(u, str)) else 1.0
                    total += pull * centre[u]
                    weight += pull
            z = zone_key(x)
            if z is not None and len(zone_items[z]) > 1:
                total += ZONE_PULL * _mean([centre[m] for m in zone_items[z]])
                weight += ZONE_PULL
            desired.append(total / weight if weight else centre[x])
        extents = [size[x][across] for x in layer]
        weights = [1.0 if isinstance(x, str) else DUMMY_PULL for x in layer]
        extra = zone_extra or {}

        def boundary(a: _Vertex, b: _Vertex) -> float:
            za, zb = zone_key(a), zone_key(b)
            if za == zb:
                return gap
            # Room for each zone's padding, label, and any widening for a long label.
            return max(gap, ZONE_CLEARANCE) + extra.get(za or "", 0.0) + extra.get(zb or "", 0.0)

        gaps = [boundary(a, b) for a, b in itertools.pairwise(layer)]
        for x, packed in zip(layer, _pack(desired, extents, gaps, weights), strict=True):
            centre[x] = packed

    for sweep in range(PLACE_SWEEPS):
        if sweep % 2 == 0:
            for layer in layers[1:]:
                settle(layer, (up, down) if sweep else (up,))
        else:
            for layer in reversed(layers[:-1]):
                settle(layer, (up, down))

    _clear_zone_bands(layers, centre, size, across, zone_items, zone_key, zone_extra or {})

    # ---- write positions ----
    origin = min(centre[x] - size[x][across] / 2 for layer in layers for x in layer)
    start = (ORIGIN_X if direction == "down" else ORIGIN_Y) - origin
    lead = float(ORIGIN_Y if direction == "down" else ORIGIN_X)
    for layer in layers:
        real = [r for r in layer if isinstance(r, str)]
        if not real:
            along_room = max(size[x][1 if direction == "down" else 0] for x in layer)
            lead += along_room + LAYER_GAP[direction]
            continue
        for x in real:
            card_w, card_h = size[x]
            node = by_id[x]
            if direction == "down":
                node["x"] = round(start + centre[x] - card_w / 2)
                node["y"] = round(lead)
            else:
                node["x"] = round(lead)
                node["y"] = round(start + centre[x] - card_h / 2)
        along = 1 if direction == "down" else 0
        lead += max(size[x][along] for x in real) + LAYER_GAP[direction]


def _clear_zone_bands(
    layers: Sequence[Sequence[_Vertex]],
    centre: dict[_Vertex, float],
    size: Mapping[_Vertex, tuple[int, int]],
    across: int,
    zone_items: Mapping[str, Sequence[_Vertex]],
    zone_key: Callable[[_Vertex], str | None],
    zone_extra: Mapping[str, float],
) -> None:
    """Move non-members out of each zone's box.

    A zone's box spans its members' layers and their extent across the flow. In
    every layer of that span, what lies before the zone (in layer order, or by
    position in a layer the zone has no member in) moves out to one side, and
    what lies after it to the other. Run twice so one zone's moves settle
    against another's.
    """
    layer_of = {x: i for i, layer in enumerate(layers) for x in layer}
    for _ in range(2):
        for z, items in zone_items.items():
            if not items:
                continue
            half = {x: size[x][across] / 2 for x in items}
            pad = ZONE_PAD + zone_extra.get(z, 0.0) + PLACE_MARGIN
            lo = min(centre[x] - half[x] for x in items) - pad
            hi = max(centre[x] + half[x] for x in items) + pad
            mid = (lo + hi) / 2
            first = min(layer_of[x] for x in items)
            last = max(layer_of[x] for x in items)
            for i in range(first, last + 1):
                layer = layers[i]
                inside = [k for k, x in enumerate(layer) if zone_key(x) == z]
                if inside:
                    before, after = layer[: inside[0]], layer[inside[-1] + 1 :]
                else:
                    before = [x for x in layer if centre[x] < mid]
                    after = [x for x in layer if centre[x] >= mid]
                cards = [x for x in before if isinstance(x, str)]
                if cards:
                    shift = min(0.0, lo - max(centre[x] + size[x][across] / 2 for x in cards))
                    for x in before:
                        centre[x] += shift
                cards = [x for x in after if isinstance(x, str)]
                if cards:
                    shift = max(0.0, hi - min(centre[x] - size[x][across] / 2 for x in cards))
                    for x in after:
                        centre[x] += shift


def _mean(values: Sequence[float], default: float = -1.0) -> float:
    return sum(values) / len(values) if values else default


def _pack(
    desired: Sequence[float],
    extents: Sequence[float],
    gap: float | Sequence[float],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """Centres closest to `desired` (weighted least squares) that keep the given
    order with at least `gap` between neighbouring cards (one gap for all, or one
    per neighbouring pair).

    Subtracting each card's minimum offset from the first turns the spacing
    constraints into "non-decreasing", which pool-adjacent-violators solves exactly.
    """
    gaps = [gap] * max(0, len(extents) - 1) if isinstance(gap, int | float) else list(gap)
    offs = [0.0]
    for (a, b), g in zip(itertools.pairwise(extents), gaps, strict=True):
        offs.append(offs[-1] + (a + b) / 2 + g)
    ws = list(weights) if weights is not None else [1.0] * len(desired)
    blocks: list[list[float]] = []  # [weighted sum, total weight, count]
    for d, o, wt in zip(desired, offs, ws, strict=True):
        blocks.append([wt * (d - o), wt, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, wsum, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += wsum
            blocks[-1][2] += n
    fitted: list[float] = []
    for s, wsum, n in blocks:
        fitted.extend([s / wsum] * int(n))
    return [f + o for f, o in zip(fitted, offs, strict=True)]


def _crossings(
    layers: Sequence[Sequence[_Vertex]],
    down: Mapping[_Vertex, list[_Vertex]],
    pos: Mapping[_Vertex, float],
) -> int:
    """Edge crossings between adjacent layers, counted as inversions."""
    total = 0
    for layer in layers:
        ends = sorted((pos[v], pos[w]) for v in layer for w in down[v])
        lower = [b for _a, b in ends]
        total += _inversions(lower)
    return total


def _inversions(seq: list[float]) -> int:
    if len(seq) < 2:
        return 0
    mid = len(seq) // 2
    left, right = seq[:mid], seq[mid:]
    count = _inversions(left) + _inversions(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            seq[k] = left[i]
            i += 1
        else:
            seq[k] = right[j]
            j += 1
            count += len(left) - i
        k += 1
    seq[k:] = left[i:] + right[j:]
    return count


def _shift(nodes: Iterable[dict[str, Any]], dx: float, dy: float) -> None:
    for n in nodes:
        n["x"] = round(_num(n.get("x")) + dx)
        n["y"] = round(_num(n.get("y")) + dy)


def _stack_aside(
    nodes: Sequence[dict[str, Any]], placed: Sequence[_Rect], direction: Direction
) -> None:
    """Notes and other unconnected nodes: a column to the right of everything
    ("down") or a row below everything ("right")."""
    box = _bbox(placed)
    if direction == "down":
        x = (box[0] + box[2] + COMPONENT_GAP) if box else ORIGIN_X
        y = box[1] if box else ORIGIN_Y
        for n in nodes:
            n["x"], n["y"] = round(x), round(y)
            y += estimate_size(n)[1] + ROW_GAP
    else:
        x = box[0] if box else ORIGIN_X
        y = (box[1] + box[3] + COMPONENT_GAP) if box else ORIGIN_Y
        for n in nodes:
            n["x"], n["y"] = round(x), round(y)
            x += estimate_size(n)[0] + COL_GAP


# ---- zones ----


def _nested_members(zid: str, members_of: Mapping[str, Sequence[str]]) -> set[str]:
    """A zone's members, plus the members of any zone among them, transitively."""
    seen: set[str] = set()
    stack = list(members_of.get(zid, ()))
    while stack:
        m = stack.pop()
        if m in seen or m == zid:
            continue
        seen.add(m)
        stack.extend(members_of.get(m, ()))
    return seen


def _zone_label_width(label: str) -> float:
    """Estimated width of a zone's label chip (.wf-region-label: uppercase 12.5px
    condensed heading font, 10px padding each side)."""
    return len(label) * ZONE_LABEL_CHAR_W + ZONE_LABEL_CHIP_PAD


def _size_zones(
    wf: Workflow, zone_ids: Iterable[str], zone_members: Mapping[str, Sequence[str]] | None
) -> list[str]:
    warnings: list[str] = []
    by_id = {n["id"]: n for n in _dict_nodes(wf)}
    members_of = zone_members or {}
    # A zone named as a member of another is sized first, so the outer box is
    # drawn round the inner one's final box. On a cycle, node order decides.
    zones = list(dict.fromkeys(zone_ids))
    nesting = [{"source": m, "target": z} for z in zones for m in members_of.get(z, ()) if m != z]
    succ, pred = _graph(zones, nesting)
    for zid in _topo(zones, succ, pred):
        zone = by_id.get(zid)
        if zone is None or zone.get("type") != "region":
            continue
        data = zone.get("data")
        if not isinstance(data, dict):
            data = {}
            zone["data"] = data
        label = data.get("label") or zid
        members = [by_id[m] for m in members_of.get(zid, ()) if m in by_id and m != zid]
        box = _bbox(_rect(m) for m in members)
        if box is None:
            warnings.append(
                f"Zone '{label}' ({zid}) has no positioned member nodes, so it was not sized."
            )
            continue
        corner = data.get("labelCorner") or "tr"
        top = ZONE_PAD + (ZONE_LABEL_PAD if str(corner).startswith("t") else 0)
        bottom = ZONE_PAD + (ZONE_LABEL_PAD if str(corner).startswith("b") else 0)
        x0 = math.floor(box[0] - ZONE_PAD)
        y0 = math.floor(box[1] - top)
        x1 = math.ceil(box[0] + box[2] + ZONE_PAD)
        # The label chip is capped at a share of the zone's width; widen the zone,
        # evenly on both sides, until the whole label fits.
        need = math.ceil(_zone_label_width(str(label)) / ZONE_LABEL_MAX_SHARE)
        if x1 - x0 < need:
            extra = need - (x1 - x0)
            x0 -= extra // 2
            x1 += extra - extra // 2
        y1 = math.ceil(box[1] + box[3] + bottom)
        zone["x"], zone["y"] = x0, y0
        data["w"], data["h"] = x1 - x0, y1 - y0
        zrect = (float(x0), float(y0), float(x1 - x0), float(y1 - y0))
        member_ids = _nested_members(zid, members_of)
        for n in by_id.values():
            if n["id"] in member_ids or n["id"] == zid or n.get("type") == "region":
                continue
            if _intersects(_rect(n), zrect):
                warnings.append(
                    f"Node {n['id']} is not a member of zone '{label}' ({zid}) "
                    "but falls inside its box."
                )
    return warnings


# ---- create path ----


def layout_workflow(
    wf: Workflow,
    *,
    direction: Direction = "down",
    zone_members: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Position every node, size every zone and set every edge's handles."""
    nodes = _dict_nodes(wf)
    by_id = {n["id"]: n for n in nodes}
    layered = [n["id"] for n in nodes if n.get("type") in LAYERED_TYPES]
    zones = [n["id"] for n in nodes if n.get("type") == "region"]
    aside = [n for n in nodes if n.get("type") not in LAYERED_TYPES and n.get("type") != "region"]

    succ, pred = _graph(layered, _dict_edges(wf))
    depth = _depths(layered, succ, pred)
    edge_labels = {
        (e["source"], e["target"]): e["label"]
        for e in _dict_edges(wf)
        if isinstance(e.get("label"), str) and e["label"].strip()
    }
    zone_of: dict[str, str] = {}
    for zid, members in (zone_members or {}).items():
        for m in members:
            zone_of.setdefault(m, zid)
    # Across a top-to-bottom flow, a zone with a long label is widened on both sides
    # (see _size_zones). Estimate that from its widest member, so the layout leaves
    # room for it at the zone's edges.
    zone_extra: dict[str, float] = {}
    if direction == "down":
        for zid, members in (zone_members or {}).items():
            zone = by_id.get(zid)
            widths = [estimate_size(by_id[m])[0] for m in members if m in by_id]
            if zone is None or not widths:
                continue
            label = str((zone.get("data") or {}).get("label") or zid)
            need = _zone_label_width(label) / ZONE_LABEL_MAX_SHARE
            zone_extra[zid] = max(0.0, (need - (max(widths) + 2 * ZONE_PAD)) / 2)

    placed: list[_Rect] = []
    # Right and bottom edges of _bbox(placed), folded component by component.
    right = bottom = -math.inf
    for comp in _components(layered, succ):
        _layout_component(
            comp, by_id, succ, pred, depth, direction, zone_of, edge_labels, zone_extra
        )
        comp_nodes = [by_id[v] for v in comp]
        box = _bbox(_rect(n) for n in comp_nodes)
        if box is not None and placed:
            if direction == "down":
                _shift(comp_nodes, right + COMPONENT_GAP - box[0], 0)
            else:
                _shift(comp_nodes, 0, bottom + COMPONENT_GAP - box[1])
        rects = [_rect(n) for n in comp_nodes]
        placed.extend(rects)
        right = max([right, *(r[0] + r[2] for r in rects)])
        bottom = max([bottom, *(r[1] + r[3] for r in rects)])

    _stack_aside(aside, placed, direction)
    warnings = _size_zones(wf, zones, zone_members)
    set_edge_handles(wf, direction=direction)
    _clear_zone_labels(wf, zones, zone_members)
    return warnings


def _strip_clear(xs: Sequence[float], lo: float, hi: float) -> bool:
    """True when no crossing in `xs` comes within ZONE_CHIP_MARGIN of [lo, hi]."""
    m = ZONE_CHIP_MARGIN
    return not any(lo - m <= x <= hi + m for x in xs)


def _clear_zone_labels(
    wf: Workflow, zone_ids: Iterable[str], zone_members: Mapping[str, Sequence[str]] | None
) -> None:
    """Keep connections off zone labels.

    A connection into or out of a zone's first card crosses the zone's top strip,
    where the label chip sits. For a zone with no corner set, use the top-right
    corner (the builder's default) or the top-left, whichever no connection
    crosses; if both are crossed, widen the zone on the side that needs less. A
    zone whose corner is already top-left or top-right keeps it, and is widened on
    that side if needed. Widening never takes in a card that is not a member, and
    bottom corners are left alone.
    """
    nodes = _dict_nodes(wf)
    by_id = {n["id"]: n for n in nodes}
    members_of = zone_members or {}
    curves = []
    for e in _dict_edges(wf):
        s, t = by_id.get(e["source"]), by_id.get(e["target"])
        if s is None or t is None or not e.get("sourceHandle") or not e.get("targetHandle"):
            continue
        curves.append(_curve(s, t, (e["sourceHandle"], e["targetHandle"])))
    for zid in dict.fromkeys(zone_ids):
        zone = by_id.get(zid)
        data = zone.get("data") if zone else None
        if zone is None or not isinstance(data, dict):
            continue
        preset = data.get("labelCorner")
        if preset not in (None, "tl", "tr"):
            continue
        x0, y0 = _num(zone.get("x")), _num(zone.get("y"))
        w, h = _num(data.get("w")), _num(data.get("h"))
        if w <= 0 or h <= 0:
            continue
        chip_w = min(_zone_label_width(str(data.get("label") or zid)), w * ZONE_LABEL_MAX_SHARE)
        top, bottom = y0 + ZONE_CHIP_TOP, y0 + ZONE_CHIP_TOP + ZONE_CHIP_H
        # Where connections cross the label strip.
        xs = [
            px
            for pts in curves
            for px, py in pts
            if top - ZONE_CHIP_MARGIN <= py <= bottom + ZONE_CHIP_MARGIN and x0 <= px <= x0 + w
        ]
        right_chip = (x0 + w - ZONE_CHIP_INSET - chip_w, x0 + w - ZONE_CHIP_INSET)
        left_chip = (x0 + ZONE_CHIP_INSET, x0 + ZONE_CHIP_INSET + chip_w)

        if preset == "tl":
            if _strip_clear(xs, *left_chip):
                continue
        elif _strip_clear(xs, *right_chip):
            continue
        elif preset is None and _strip_clear(xs, *left_chip):
            data["labelCorner"] = "tl"
            continue
        # Widen: to the right so the chip clears the rightmost crossing, or to the
        # left so it clears the leftmost one.
        grow_right = max(xs) + ZONE_CHIP_MARGIN - right_chip[0]
        grow_left = left_chip[1] - (min(xs) - ZONE_CHIP_MARGIN)
        members = _nested_members(zid, members_of)
        others = [
            _rect(n)
            for n in nodes
            if n["id"] not in members and n["id"] != zid and n.get("type") != "region"
        ]
        if preset == "tl":
            options = [(grow_left, "left")]
        elif preset == "tr":
            options = [(grow_right, "right")]
        else:
            options = sorted([(grow_right, "right"), (grow_left, "left")], key=lambda o: o[0])
        for grow, side in options:
            grow = math.ceil(grow)
            box = (x0 - grow, y0, w + grow, h) if side == "left" else (x0, y0, w + grow, h)
            if any(_intersects(box, r) for r in others):
                continue
            if side == "left":
                zone["x"] = round(x0 - grow)
                data["labelCorner"] = "tl"
            data["w"] = round(w + grow)
            break


def zone_members_by_box(wf: Workflow) -> dict[str, list[str]]:
    """Each sized zone's members, read from where things are: every node lying
    wholly inside the zone's box, including zones nested in it.

    The builder keeps no membership, so this is how a zone already in a workflow
    keeps its contents when the workflow is laid out again.
    """
    nodes = _dict_nodes(wf)
    out: dict[str, list[str]] = {}
    for zone in nodes:
        data = zone.get("data")
        if zone.get("type") != "region" or not isinstance(data, dict):
            continue
        zx, zy = _num(zone.get("x")), _num(zone.get("y"))
        zw, zh = _num(data.get("w")), _num(data.get("h"))
        if zw <= 0 or zh <= 0:
            continue
        inside = []
        for n in nodes:
            if n is zone:
                continue
            nd = n.get("data")
            if n.get("type") == "region" and isinstance(nd, dict):
                x, y, w, h = (
                    _num(n.get("x")),
                    _num(n.get("y")),
                    _num(nd.get("w")),
                    _num(nd.get("h")),
                )
            else:
                x, y, w, h = _rect(n)
            if zx <= x and zy <= y and x + w <= zx + zw and y + h <= zy + zh:
                inside.append(n["id"])
        out[zone["id"]] = inside
    return out


# ---- update path ----


def place_new_nodes(
    wf: Workflow,
    new_node_ids: Sequence[str],
    *,
    direction: Direction | None = None,
    zone_members: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Position the nodes in `new_node_ids` without moving any existing node."""
    nodes = _dict_nodes(wf)
    by_id = {n["id"]: n for n in nodes}
    new_set = set(new_node_ids)
    if direction is None:
        # Inferred from the existing workflow only.
        existing = Workflow(
            {
                "nodes": [n for n in nodes if n["id"] not in new_set],
                "edges": [
                    e
                    for e in _dict_edges(wf)
                    if e["source"] not in new_set and e["target"] not in new_set
                ],
            }
        )
        direction = infer_direction(existing)

    occupied: list[_Rect] = [
        _rect(n) for n in nodes if n["id"] not in new_set and n.get("type") != "region"
    ]
    positioned = {n["id"] for n in nodes if n["id"] not in new_set}
    edges = _dict_edges(wf)
    preds: dict[str, list[str]] = {}
    succs: dict[str, list[str]] = {}
    for e in edges:
        if e["source"] != e["target"]:
            succs.setdefault(e["source"], []).append(e["target"])
            preds.setdefault(e["target"], []).append(e["source"])

    todo = [i for i in new_node_ids if i in by_id and by_id[i].get("type") != "region"]
    # New layered nodes in topological order among themselves, so a new chain is
    # placed head first.
    layered_new = [i for i in todo if by_id[i].get("type") in LAYERED_TYPES]
    s_new, p_new = _graph(layered_new, edges)
    order = _topo(layered_new, s_new, p_new)
    in_order = set(order)
    order += [i for i in todo if i not in in_order]

    def free_slot(node: dict[str, Any], x: float, y: float) -> None:
        w, h = estimate_size(node)
        step = (
            (CARD_W + SIBLING_GAP["down"], 0.0)
            if direction == "down"
            else (0.0, h + SIBLING_GAP["right"])
        )
        for k in range(len(occupied) + 2):
            # 0, +1, -1, +2, -2, ... steps across the flow.
            m = (k + 1) // 2 * (1 if k % 2 else -1)
            cx, cy = x + m * step[0], y + m * step[1]
            r = (float(round(cx)), float(round(cy)), float(w), float(h))
            if not any(_intersects(r, o, PLACE_MARGIN) for o in occupied):
                break
        node["x"], node["y"] = int(r[0]), int(r[1])
        occupied.append(r)
        positioned.add(node["id"])

    def after(node: dict[str, Any], placed_preds: list[str]) -> None:
        w, h = estimate_size(node)
        rects = {p: _rect(by_id[p]) for p in placed_preds}
        if direction == "down":
            far = max(placed_preds, key=lambda p: rects[p][1] + rects[p][3])
            r = rects[far]
            free_slot(node, r[0] + r[2] / 2 - w / 2, r[1] + r[3] + LAYER_GAP["down"])
        else:
            far = max(placed_preds, key=lambda p: rects[p][0] + rects[p][2])
            r = rects[far]
            free_slot(node, r[0] + r[2] + LAYER_GAP["right"], r[1] + r[3] / 2 - h / 2)

    def before(node: dict[str, Any], placed_succs: list[str]) -> None:
        w, h = estimate_size(node)
        rects = {s: _rect(by_id[s]) for s in placed_succs}
        if direction == "down":
            first = min(placed_succs, key=lambda s: rects[s][1])
            r = rects[first]
            free_slot(node, r[0] + r[2] / 2 - w / 2, r[1] - h - LAYER_GAP["down"])
        else:
            first = min(placed_succs, key=lambda s: rects[s][0])
            r = rects[first]
            free_slot(node, r[0] - w - LAYER_GAP["right"], r[1] + r[3] / 2 - h / 2)

    def aside(node: dict[str, Any]) -> None:
        box = _bbox(occupied)
        if direction == "down":
            x = (box[0] + box[2] + COMPONENT_GAP) if box else ORIGIN_X
            y = box[1] if box else ORIGIN_Y
            # Stack down the notes column.
            w, h = estimate_size(node)
            while any(_intersects((x, y, w, h), o, PLACE_MARGIN) for o in occupied):
                y += h + ROW_GAP
        else:
            x = box[0] if box else ORIGIN_X
            y = (box[1] + box[3] + COMPONENT_GAP) if box else ORIGIN_Y
            w, h = estimate_size(node)
            while any(_intersects((x, y, w, h), o, PLACE_MARGIN) for o in occupied):
                x += w + COL_GAP
        node["x"], node["y"] = round(x), round(y)
        occupied.append(_rect(node))
        positioned.add(node["id"])

    remaining = list(order)
    while remaining:
        for i in remaining:
            node = by_id[i]
            if node.get("type") not in LAYERED_TYPES:
                continue
            if any(p not in positioned for p in p_new.get(i, [])):
                continue  # a new predecessor goes first
            placed_preds = [p for p in preds.get(i, []) if p in positioned and p in by_id]
            if placed_preds:
                after(node, placed_preds)
                break
        else:
            for i in remaining:
                node = by_id[i]
                if node.get("type") not in LAYERED_TYPES:
                    continue
                if any(p in by_id and p not in positioned for p in preds.get(i, [])):
                    continue  # wait for its predecessor
                placed_succs = [s for s in succs.get(i, []) if s in positioned and s in by_id]
                if placed_succs:
                    before(node, placed_succs)
                    break
            else:
                i = remaining[0]
                aside(by_id[i])
        remaining.remove(i)

    warnings = _size_zones(
        wf,
        [i for i in new_node_ids if i in by_id and by_id[i].get("type") == "region"],
        zone_members,
    )

    # Handles: every edge touching a new node, and missing handles elsewhere.
    obstacles = _Obstacles(nodes)
    for e in edges:
        s, t = by_id.get(e["source"]), by_id.get(e["target"])
        if s is None or t is None:
            continue
        sh, th = _choose_handles(s, t, direction, obstacles, e.get("label"))
        if e["source"] in new_set or e["target"] in new_set:
            e["sourceHandle"], e["targetHandle"] = sh, th
        else:
            if not e.get("sourceHandle"):
                e["sourceHandle"] = sh
            if not e.get("targetHandle"):
                e["targetHandle"] = th
    _clear_zone_labels(
        wf,
        [i for i in new_node_ids if i in by_id and by_id[i].get("type") == "region"],
        zone_members,
    )
    return warnings
