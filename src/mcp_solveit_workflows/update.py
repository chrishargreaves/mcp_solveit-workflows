"""Operations on an existing workflow.

Update is operation-based: remove nodes or edges, add nodes, edges and
selections, and set data fields on a node. Everything else in the document is
left exactly as it was read, including fields from unreleased builder versions.
Node IDs are never reused, because selections are keyed by node ID.

All operations run on a copy; the workflow is only changed when every
operation succeeded.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field

from .model import (
    DFT_ID,
    NODE_DATA_KEYS,
    EdgeInput,
    LabelCorner,
    NodeInput,
    NoteColor,
    ParentOf,
    SelectionInput,
    Workflow,
    WorkflowError,
    ZoneInput,
    add_to_workflow,
    check_route_keys,
    is_number,
)

# Data keys a set-data operation may change, per builder node type. Fork and
# join `mode` follows the node type, so it is not settable; unknown node types
# are never edited.
SETTABLE_DATA: dict[str, frozenset[str]] = {
    t: keys for t, keys in NODE_DATA_KEYS.items() if t not in ("fork", "join")
}

_TEXT_KEYS = frozenset(
    {"label", "name", "tempId", "description", "details", "text", "question", "techniqueId"}
)
_LIST_KEYS = frozenset({"methods", "inputClasses", "outputClasses", "classes"})
_NUMBER_KEYS = frozenset({"w", "h"})
_COLORS = frozenset(get_args(NoteColor))
_CORNERS = frozenset(get_args(LabelCorner))


class SetDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node: str = Field(
        description="ID of an existing node, or the key of a node added in this call."
    )
    fields: dict[str, Any] = Field(
        description=(
            "Builder data keys to set on the node, with their new values; null deletes "
            "the key. Allowed keys: technique: techniqueId, label, methods, target; "
            "placeholder: name, tempId, description, details, inputClasses, outputClasses; "
            "note: text, color; decision: question; data source: label, classes; zone: "
            "label, w, h, labelCorner, color. A node's type cannot be changed."
        ),
    )


@dataclass
class UpdateResult:
    workflow: Workflow
    new_node_ids: list[str] = field(default_factory=list)
    key_to_id: dict[str, str] = field(default_factory=dict)
    zone_members: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _dict_of(doc: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = doc.get(key)
    return value if isinstance(value, dict) else None


def _remove(wf: Workflow, remove_ids: Sequence[str]) -> dict[str, list[str]]:
    # read_workflow refuses non-text IDs; only text ones are ever looked up here.
    node_ids = {n["id"] for n in wf.nodes if isinstance(n, dict) and isinstance(n.get("id"), str)}
    edge_ids = {e["id"] for e in wf.edges if isinstance(e, dict) and isinstance(e.get("id"), str)}
    nodes_gone: list[str] = []
    edges_gone: list[str] = []
    for rid in dict.fromkeys(remove_ids):
        if rid in node_ids:
            nodes_gone.append(rid)
        elif rid in edge_ids:
            edges_gone.append(rid)
        else:
            raise WorkflowError(f"Cannot remove '{rid}': no node or edge has that ID.")

    gone = set(nodes_gone)
    for n in wf.nodes:
        if isinstance(n, dict) and n.get("parent") in gone and n.get("id") not in gone:
            raise WorkflowError(
                f"Cannot remove node '{n['parent']}': node '{n.get('id')}' has it as its "
                "parent. Remove that node too."
            )

    dropped: dict[str, list[str]] = {
        "nodes": nodes_gone,
        "edges": [],
        "selections": [],
        "evaluations": [],
        "provenance": [],
    }
    if not nodes_gone and not edges_gone:
        return dropped

    wf.doc["nodes"] = [n for n in wf.nodes if not (isinstance(n, dict) and n.get("id") in gone)]
    explicit = set(edges_gone)
    kept_edges: list[Any] = []
    for e in wf.edges:
        if isinstance(e, dict) and (
            e.get("id") in explicit or e.get("source") in gone or e.get("target") in gone
        ):
            dropped["edges"].append(str(e.get("id")))
        else:
            kept_edges.append(e)
    wf.doc["edges"] = kept_edges

    def keyed_to_removed(key: object) -> bool:
        if not isinstance(key, str):
            return False
        # "<id>|..." keys, plus "<id>/..." sub-workflow path keys from unreleased builders:
        # some prefix ending just before a separator is a removed node ID.
        return any(key[:i] in gone for i, ch in enumerate(key) if ch in "|/")

    for name in ("selections", "evaluations"):
        table = _dict_of(wf.doc, name)
        if table is None:
            continue
        for key in [k for k in table if keyed_to_removed(k)]:
            del table[key]
            dropped[name].append(key)

    provenance = _dict_of(wf.doc, "provenance")
    if provenance is not None:
        for nid in nodes_gone:
            if nid in provenance:
                del provenance[nid]
                dropped["provenance"].append(nid)
    return dropped


def _check_value(node_ref: str, key: str, value: Any) -> None:
    def bad(wanted: str) -> WorkflowError:
        return WorkflowError(f"Node '{node_ref}': data field '{key}' must be {wanted}.")

    if key in _TEXT_KEYS and not isinstance(value, str):
        raise bad("text")
    if key == "techniqueId" and not DFT_ID.match(value):
        raise bad("a technique ID of the form DFT-nnnn")
    if key in _LIST_KEYS and not (
        isinstance(value, list) and all(isinstance(v, str) for v in value)
    ):
        raise bad("a list of text")
    if key == "methods" and not all(DFT_ID.match(v) for v in value):
        raise bad("a list of technique IDs of the form DFT-nnnn")
    if key in _NUMBER_KEYS and not (is_number(value) and value > 0):
        raise bad("a positive number")
    if key == "color" and value not in _COLORS:
        raise bad(f"one of {', '.join(sorted(_COLORS))}")
    if key == "labelCorner" and value not in _CORNERS:
        raise bad(f"one of {', '.join(sorted(_CORNERS))}")
    if key == "target":
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise bad("an object of route path keys to text values")
        check_route_keys(node_ref, value)


def _set_data(wf: Workflow, ops: Sequence[SetDataInput], key_to_id: dict[str, str]) -> None:
    for op in ops:
        node_id = key_to_id.get(op.node, op.node)
        node = wf.node(node_id)
        if node is None:
            raise WorkflowError(f"Cannot set data on '{op.node}': no node has that ID or key.")
        ntype = node.get("type")
        if "type" in op.fields:
            raise WorkflowError(
                f"Node '{op.node}': a node's type cannot be changed. Remove it and add a new node."
            )
        allowed = SETTABLE_DATA.get(ntype) if isinstance(ntype, str) else None
        if allowed is None:
            raise WorkflowError(
                f"Node '{op.node}': data on a node of this type cannot be set by this server."
            )
        for key, value in op.fields.items():
            if key not in allowed:
                raise WorkflowError(
                    f"Node '{op.node}': '{key}' is not a data field of a {ntype} node. "
                    f"Allowed: {', '.join(sorted(allowed))}."
                )
            if value is not None:
                _check_value(op.node, key, value)
        data = node.get("data")
        if not isinstance(data, dict):
            data = {}
            node["data"] = data
        for key, value in op.fields.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = copy.deepcopy(value)


def apply_update(
    wf: Workflow,
    *,
    parent_of: ParentOf,
    add_nodes: Sequence[NodeInput] = (),
    add_edges: Sequence[EdgeInput] = (),
    remove_ids: Sequence[str] = (),
    set_data: Sequence[SetDataInput] = (),
    selections: Sequence[SelectionInput] = (),
) -> UpdateResult:
    """Apply the operations to ``wf``: add, then remove, then set data.

    Edges, selections, zone members and set-data operations may refer to
    existing node IDs or to the keys of added nodes, but not to removed nodes.
    New nodes are added before anything is removed, so a removed node's ID is
    never handed out again. On any error ``wf`` is left unchanged.
    """
    removing = set(remove_ids)
    refs: list[tuple[str, str]] = [(e.source, "An edge") for e in add_edges]
    refs += [(e.target, "An edge") for e in add_edges]
    refs += [(s.step, "A selection") for s in selections]
    refs += [(op.node, "A set-data operation") for op in set_data]
    for node in add_nodes:
        if isinstance(node, ZoneInput):
            refs += [(m, f"Zone '{node.key}'") for m in node.members]
    new_keys = {n.key for n in add_nodes}
    for ref, what in refs:
        if ref in removing and ref not in new_keys:
            raise WorkflowError(f"{what} refers to '{ref}', which this update removes.")

    work = Workflow(copy.deepcopy(wf.doc))
    built = add_to_workflow(work, list(add_nodes), list(add_edges), list(selections), parent_of)
    dropped = _remove(work, remove_ids)
    _set_data(work, set_data, built.key_to_id)

    wf.doc = work.doc
    return UpdateResult(
        workflow=wf,
        new_node_ids=list(built.new_node_ids),
        key_to_id=dict(built.key_to_id),
        zone_members=dict(built.zone_members),
        dropped=dropped,
        warnings=list(built.warnings),
    )
