"""Validation rules shared by create, update and validate.

Errors mirror what the deployed builder's strict validator
(`scripts/lib/validate-workflow.mjs`) rejects: the builder would mis-render
or drop something. Warnings mean the workflow opens but something looks
wrong. Knowledge base findings are always warnings, as in the builder: the
local checkout may lag or lead the live knowledge base.

Nodes of a type this server does not know, and data keys or top-level fields
from unreleased builder versions, are carried through untouched. They
are only an error when the caller supplied them as create or update input.

Messages name keys, IDs and short labels only, never free text from the file
beyond a capped quote.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .kb import KnowledgeBase
from .model import (
    DFT_ID,
    FLOW_TYPES,
    MAX_DESCRIPTION_LEN,
    NODE_DATA_KEYS,
    UNCONNECTABLE_TYPES,
    Workflow,
    is_number,
)

Level = Literal["error", "warning"]

# Nodes whose same-handle branching the builder flags (branches.ts).
_BRANCH_WARNED = frozenset({"technique", "placeholder"})

# Data keys of each node type in the deployed builder; anything else is
# unreleased content, carried through but not checked.
_KNOWN_DATA = NODE_DATA_KEYS
# validate-workflow.mjs NODE_KINDS: the node types above.
NODE_KINDS = frozenset(_KNOWN_DATA)
_KNOWN_TOP = frozenset(
    {
        "version",
        "title",
        "description",
        "nodes",
        "edges",
        "selections",
        "evaluations",
        "mode",
        "provenance",
        "viewport",
    }
)

_SELECTION_VALUES = ("Y", "N", "NA")
_QUOTE_CAP = 60


@dataclass(frozen=True)
class Finding:
    level: Level
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"level": self.level, "code": self.code, "message": self.message}
        if self.node_id is not None:
            out["node_id"] = self.node_id
        if self.edge_id is not None:
            out["edge_id"] = self.edge_id
        return out


def has_errors(findings: Iterable[Finding]) -> bool:
    return any(f.level == "error" for f in findings)


def _q(value: object) -> str:
    """A short, quoted rendering of a value from the file."""
    text = value if isinstance(value, str) else repr(value)
    if len(text) > _QUOTE_CAP:
        text = text[: _QUOTE_CAP - 1] + "…"
    return f"'{text}'"


def _is_finite(v: object) -> bool:
    return is_number(v) and math.isfinite(v)


def _nonempty_str(v: object) -> bool:
    return isinstance(v, str) and bool(v)


class _Collector:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def error(
        self, code: str, message: str, *, node: str | None = None, edge: str | None = None
    ) -> None:
        self.findings.append(Finding("error", code, message, node, edge))

    def warn(
        self, code: str, message: str, *, node: str | None = None, edge: str | None = None
    ) -> None:
        self.findings.append(Finding("warning", code, message, node, edge))


def _node_ref(index: int, node_id: str | None) -> str:
    return f"Node {_q(node_id)}" if node_id else f"Node {index}"


def _check_technique(
    out: _Collector, ref: str, nid: str | None, d: dict[str, Any], kb: KnowledgeBase | None
) -> None:
    tid = d.get("techniqueId")
    if not _nonempty_str(tid):
        out.error("missing_data", f"{ref}: a technique node needs 'data.techniqueId'.", node=nid)
        return
    assert isinstance(tid, str)
    if not DFT_ID.match(tid):
        out.warn(
            "bad_technique_id",
            f"{ref}: technique ID {_q(tid)} is not of the form DFT-nnnn.",
            node=nid,
        )
    elif kb is not None and kb.technique(tid) is None:
        out.warn(
            "unknown_technique",
            f"{ref}: technique {_q(tid)} is not in the knowledge base; "
            "the builder will show it as an unknown box.",
            node=nid,
        )
    methods = d.get("methods")
    if methods is None:
        return
    if not isinstance(methods, list):
        out.error("invalid_data", f"{ref}: 'data.methods' is not a list.", node=nid)
        return
    for m in methods:
        if not isinstance(m, str):
            out.error("invalid_data", f"{ref}: 'data.methods' holds a non-text entry.", node=nid)
            continue
        if kb is None:
            continue
        if kb.technique(m) is None:
            out.warn(
                "unknown_technique",
                f"{ref}: method {_q(m)} is not in the knowledge base.",
                node=nid,
            )
        elif kb.technique(tid) is not None and not kb.is_subtechnique_of(m, tid):
            out.warn(
                "method_not_subtechnique",
                f"{ref}: method {_q(m)} is not a sub-technique of {_q(tid)}.",
                node=nid,
            )


def _check_data(
    out: _Collector,
    ref: str,
    nid: str | None,
    ntype: str,
    d: dict[str, Any],
    kb: KnowledgeBase | None,
) -> None:
    if ntype == "technique":
        _check_technique(out, ref, nid, d, kb)
    elif ntype == "placeholder":
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            out.error(
                "missing_data",
                f"{ref}: a placeholder node needs a non-empty 'data.name'.",
                node=nid,
            )
    elif ntype == "note":
        if not isinstance(d.get("text"), str):
            out.error("missing_data", f"{ref}: a note node needs 'data.text'.", node=nid)
    elif ntype == "decision":
        if not isinstance(d.get("question"), str):
            out.error("missing_data", f"{ref}: a decision node needs 'data.question'.", node=nid)
    elif ntype in ("fork", "join"):
        if d.get("mode") != ntype:
            out.warn(
                "fork_join_mode",
                f"{ref}: a {ntype} node's 'data.mode' should be '{ntype}' "
                f"(got {_q(d.get('mode'))}).",
                node=nid,
            )
    elif ntype == "region":
        if not isinstance(d.get("label"), str):
            out.error(
                "missing_data", f"{ref}: a zone (region) node needs a text 'data.label'.", node=nid
            )
        w, h = d.get("w"), d.get("h")
        if not (_is_finite(w) and _is_finite(h) and w > 0 and h > 0):  # type: ignore[operator]
            out.error(
                "invalid_region_size",
                f"{ref}: a zone (region) needs positive numeric 'data.w' and 'data.h' "
                f"(got {_q(w)} x {_q(h)}).",
                node=nid,
            )
    elif ntype == "caseClass" and not isinstance(d.get("label"), str):
        out.error(
            "missing_data",
            f"{ref}: a data source (caseClass) node needs a text 'data.label'.",
            node=nid,
        )

    unknown = sorted(k for k in d if k not in _KNOWN_DATA[ntype])
    if unknown:
        out.warn(
            "unreleased_content",
            f"{ref}: data field(s) {_q(', '.join(unknown))} are from an unreleased or "
            "unknown builder version; carried through but not checked.",
            node=nid,
        )


def validate_workflow(
    wf: Workflow, kb: KnowledgeBase | None, *, input_node_ids: Collection[str] = ()
) -> list[Finding]:
    """Check `wf` against the builder's rules.

    `input_node_ids` are the IDs of nodes the caller supplied in this create or
    update call; an unknown node type is an error only for those.
    """
    out = _Collector()
    doc = wf.doc
    raw_nodes = doc.get("nodes")
    raw_edges = doc.get("edges")
    nodes: list[Any] = raw_nodes if isinstance(raw_nodes, list) else []
    edges: list[Any] = raw_edges if isinstance(raw_edges, list) else []
    inputs = set(input_node_ids)

    # ---- top level ----
    description = doc.get("description")
    if isinstance(description, str) and len(description) > MAX_DESCRIPTION_LEN:
        out.warn(
            "description_too_long",
            f"The description is {len(description)} characters; the builder cuts it at "
            f"{MAX_DESCRIPTION_LEN}.",
        )
    unknown_top = sorted(k for k in doc if k not in _KNOWN_TOP)
    if unknown_top:
        out.warn(
            "unreleased_content",
            f"Top-level field(s) {_q(', '.join(unknown_top))} are from an unreleased or "
            "unknown builder version; carried through but not checked.",
        )
    if not nodes:
        out.warn("no_nodes", "The workflow has no nodes; there is nothing to draw.")

    # ---- nodes ----
    ids: set[str] = set()
    type_of: dict[str, str] = {}
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            out.error("invalid_node", f"Node {i} is not an object.")
            continue
        raw_id = n.get("id")
        nid = raw_id if _nonempty_str(raw_id) else None
        ref = _node_ref(i, nid)
        if nid is None:
            out.error("missing_id", f"{ref} has no text 'id'.")
        elif nid in ids:
            out.error(
                "duplicate_node_id", f"{ref}: duplicate node ID; IDs must be unique.", node=nid
            )
        else:
            ids.add(nid)

        ntype = n.get("type")
        known_type = isinstance(ntype, str) and ntype in NODE_KINDS
        if nid is not None and isinstance(ntype, str) and nid not in type_of:
            type_of[nid] = ntype

        if not _is_finite(n.get("x")) or not _is_finite(n.get("y")):
            out.error("invalid_position", f"{ref}: 'x' and 'y' must be finite numbers.", node=nid)

        if not isinstance(ntype, str):
            out.error("missing_type", f"{ref} has no text 'type'.", node=nid)
            continue
        if not known_type:
            if nid is not None and nid in inputs:
                out.error(
                    "unknown_type",
                    f"{ref}: unknown node type {_q(ntype)}; expected one of "
                    f"{', '.join(sorted(NODE_KINDS))}.",
                    node=nid,
                )
            else:
                out.warn(
                    "unreleased_or_unknown_type",
                    f"{ref}: node type {_q(ntype)} is from an unreleased or unknown builder "
                    "version; carried through but not checked.",
                    node=nid,
                )
            continue

        data = n.get("data", {})
        if not isinstance(data, dict):
            out.error("invalid_data", f"{ref}: 'data' is not an object.", node=nid)
            continue
        _check_data(out, ref, nid, ntype, data, kb)

    for i, n in enumerate(nodes):
        if not isinstance(n, dict) or "parent" not in n:
            continue
        raw_id = n.get("id")
        nid = raw_id if _nonempty_str(raw_id) else None
        ref = _node_ref(i, nid)
        parent = n["parent"]
        if not isinstance(parent, str):
            out.error("dangling_parent", f"{ref}: 'parent' is not text.", node=nid)
        elif parent == nid:
            out.error("dangling_parent", f"{ref}: 'parent' points at the node itself.", node=nid)
        elif parent not in ids:
            out.error(
                "dangling_parent",
                f"{ref}: 'parent' refers to node {_q(parent)}, which does not exist.",
                node=nid,
            )

    # ---- edges ----
    edge_ids: set[str] = set()
    touched: set[str] = set()
    per_handle: dict[tuple[str, str], list[str]] = {}
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            out.error("invalid_edge", f"Edge {i} is not an object.")
            continue
        raw_id = e.get("id")
        eid = raw_id if _nonempty_str(raw_id) else None
        ref = f"Edge {_q(eid)}" if eid else f"Edge {i}"
        if eid is None:
            out.error("missing_id", f"{ref} has no text 'id'.")
        elif eid in edge_ids:
            out.error("duplicate_edge_id", f"{ref}: duplicate edge ID.", edge=eid)
        else:
            edge_ids.add(eid)

        ends_ok = True
        for end in ("source", "target"):
            v = e.get(end)
            if not _nonempty_str(v):
                out.error("invalid_edge", f"{ref} has no text '{end}'.", edge=eid)
                ends_ok = False
            elif v not in ids:
                out.error(
                    "dangling_edge",
                    f"{ref}: '{end}' refers to node {_q(v)}, which does not exist; "
                    "the builder would drop the edge.",
                    edge=eid,
                )
                ends_ok = False
        if "label" in e and not isinstance(e["label"], str):
            out.error("invalid_edge", f"{ref}: 'label' is not text.", edge=eid)
        if not ends_ok:
            continue
        source, target = e["source"], e["target"]
        for end_id in dict.fromkeys((source, target)):
            if type_of.get(end_id) in UNCONNECTABLE_TYPES:
                out.error(
                    "invalid_edge_endpoint",
                    f"{ref}: node {_q(end_id)} is a {type_of[end_id]}, which has no "
                    "connection points in the builder; the edge would not be drawn.",
                    edge=eid,
                )
        touched.update((source, target))
        if source == target:
            out.warn("self_loop", f"{ref}: source and target are the same node.", edge=eid)
        handle = e.get("sourceHandle")
        key = (source, handle if isinstance(handle, str) else "")
        per_handle.setdefault(key, []).append(eid or f"#{i}")

    # Several edges out of one technique or placeholder handle (branches.ts).
    for (source, handle), eids in per_handle.items():
        if len(eids) >= 2 and type_of.get(source) in _BRANCH_WARNED:
            where = f"handle {_q(handle)}" if handle else "the same handle"
            out.warn(
                "ambiguous_branch",
                f"Node {_q(source)}: {len(eids)} connections leave {where}; it is unclear "
                "whether this is a choice or parallel work. Add a decision or fork node.",
                node=source,
            )

    # Flow nodes with no connections at all.
    flow_ids = [nid for nid in ids if type_of.get(nid) in FLOW_TYPES]
    if len(flow_ids) > 1:
        for nid in sorted(flow_ids):
            if nid not in touched:
                out.warn(
                    "disconnected",
                    f"Node {_q(nid)} is not connected to the rest of the workflow.",
                    node=nid,
                )

    # ---- selections and evaluations ----
    selections = doc.get("selections")
    if isinstance(selections, dict):
        for key, value in selections.items():
            parts = key.split("|") if isinstance(key, str) else []
            if len(parts) != 3:
                out.warn(
                    "selection_bad_key",
                    f"Selection key {_q(key)} is not '<nodeId>|<DFW-id>|<DFM-id>'.",
                )
                continue
            nid, wid, mid = parts
            if nid not in ids:
                out.warn(
                    "selection_missing_node",
                    f"Selection {_q(key)} refers to node {_q(nid)}, which does not exist.",
                    node=nid,
                )
            elif type_of.get(nid) != "technique":
                out.warn(
                    "selection_not_technique",
                    f"Selection {_q(key)} is on node {_q(nid)}, which is not a technique.",
                    node=nid,
                )
            if value not in _SELECTION_VALUES:
                out.warn(
                    "selection_bad_value",
                    f"Selection {_q(key)} has value {_q(value)}; expected Y, N or NA.",
                    node=nid,
                )
            if kb is not None:
                if kb.weakness(wid) is None:
                    out.warn(
                        "unknown_weakness",
                        f"Selection {_q(key)}: weakness {_q(wid)} is not in the knowledge base.",
                        node=nid,
                    )
                if kb.mitigation(mid) is None:
                    out.warn(
                        "unknown_mitigation",
                        f"Selection {_q(key)}: mitigation {_q(mid)} is not in the knowledge base.",
                        node=nid,
                    )

    evaluations = doc.get("evaluations")
    if isinstance(evaluations, dict):
        for key in evaluations:
            parts = key.split("|") if isinstance(key, str) else []
            if len(parts) != 2:
                out.warn(
                    "evaluation_bad_key",
                    f"Evaluation key {_q(key)} is not '<nodeId>|<DFW-id>'.",
                )
                continue
            nid, wid = parts
            if "/" in nid and nid not in ids:
                # A sub-workflow path key from an unreleased builder version.
                out.warn(
                    "unreleased_content",
                    f"Evaluation {_q(key)} is scoped to a sub-workflow node path, from an "
                    "unreleased builder version; carried through but not checked.",
                    node=nid,
                )
                continue
            if nid not in ids:
                out.warn(
                    "evaluation_missing_node",
                    f"Evaluation {_q(key)} refers to node {_q(nid)}, which does not exist.",
                    node=nid,
                )
            if kb is not None and kb.weakness(wid) is None:
                out.warn(
                    "unknown_weakness",
                    f"Evaluation {_q(key)}: weakness {_q(wid)} is not in the knowledge base.",
                    node=nid,
                )

    return out.findings
