"""The Workflow Builder's v2 document: reading, writing and building.

The format authority is the deployed builder (`BUILDER_FORMAT`), in particular
`serialize`, `isWorkflowStateV2` and `deserialize` in its `src/state/urlState.ts`.

A read document is kept as the raw dict it came in as, so that every field the
server does not manage (unknown top-level fields, node types, data keys, fields
from unreleased builder versions) is written back unchanged. Writing only
reorders known keys into the builder's order and drops the empty optional
fields the builder itself would never write.

Nodes hold knowledge base IDs and user text only, never knowledge base names,
as in the builder.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field

from .sharelink import BUILDER_ORIGIN

FORMAT_VERSION = 2
BUILDER_FORMAT = "0.13.0"
MAX_NODES = 2000
MAX_DESCRIPTION_LEN = 280

BUILDER_URL = BUILDER_ORIGIN.rstrip("/")

# flowOrder.ts FLOW_TYPES.
FLOW_TYPES = frozenset({"technique", "placeholder", "decision", "fork", "join"})

# Data keys of each node type in the deployed builder (src/types/workflow.ts).
# Validation treats any other key as unreleased content; update may set every
# key except fork and join `mode`, which follows the node type.
NODE_DATA_KEYS: dict[str, frozenset[str]] = {
    "technique": frozenset({"techniqueId", "label", "methods", "target"}),
    "placeholder": frozenset(
        {"tempId", "name", "description", "details", "inputClasses", "outputClasses"}
    ),
    "note": frozenset({"text", "color"}),
    "decision": frozenset({"question"}),
    "fork": frozenset({"mode"}),
    "join": frozenset({"mode"}),
    "region": frozenset({"label", "w", "h", "labelCorner", "color"}),
    "caseClass": frozenset({"label", "classes"}),
}

# Node types the builder renders without handles (NoteNode.tsx, RegionNode.tsx),
# so an edge to or from one is never drawn.
UNCONNECTABLE_TYPES = frozenset({"note", "region"})

# A SOLVE-IT technique ID, matched from the start (re.match).
DFT_ID_PATTERN = r"^DFT-\d+$"
DFT_ID = re.compile(DFT_ID_PATTERN)

# Builder `serialize` order, plus the unreleased fields (`id`, `workflowVersion`,
# `runOverlay`) in the places the unreleased builder puts them.
_TOP_ORDER = (
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
)
# Fields always written, each with the default used when it is missing.
_ALWAYS_WRITTEN: dict[str, Callable[[], Any]] = {
    "version": lambda: FORMAT_VERSION,
    "nodes": list,
    "edges": list,
    "selections": dict,
}
_NODE_ORDER = ("id", "type", "x", "y", "parent", "data")
_NODE_OPTIONAL = ("parent",)
_EDGE_ORDER = ("id", "source", "target", "sourceHandle", "targetHandle", "label")
_EDGE_OPTIONAL = ("sourceHandle", "targetHandle", "label")

# Terse keys of the legacy v1 schema (types/workflow.ts WorkflowStateV1).
_V1_KEYS = frozenset({"v", "tt", "ds", "n", "e", "m", "r", "vp"})

# One `<iri>` segment of a route path key (lib/target.ts pathKey). An IRI may
# never contain `<`, `>` or whitespace, which is what keeps the encoding
# injective.
ROUTE_KEY = re.compile(r"<[^<>\s]+>(?:/<[^<>\s]+>)*")


class WorkflowError(Exception):
    """A problem with a workflow, worded for the user.

    Messages name keys, IDs and limits, never raw file content.
    """


def check_route_keys(ref: str, target: Iterable[object]) -> None:
    """Raise unless every key of a technique's `target` is a route path key."""
    for k in target:
        if not isinstance(k, str) or not ROUTE_KEY.fullmatch(k):
            raise WorkflowError(
                f"Node '{ref}': target key is not a route path key. Write one or more "
                "full IRIs, each in angle brackets, joined by '/'."
            )


# ---- reading and writing ----


def is_number(v: object) -> TypeGuard[int | float]:
    """A JSON number: int or float, but not bool."""
    return isinstance(v, int | float) and not isinstance(v, bool)


def _ints_for_integral_floats(v: Any) -> Any:
    """Write 1.0 as 1, as JavaScript's JSON.stringify does."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _ints_for_integral_floats(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_ints_for_integral_floats(x) for x in v]
    return v


def _is_empty(v: object) -> bool:
    return v is None or v == "" or v == {} or v == []


def _ordered(
    item: dict[str, Any], order: tuple[str, ...], optional: tuple[str, ...]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in order:
        if k in item and not (k in optional and _is_empty(item[k])):
            out[k] = item[k]
    for k, v in item.items():
        if k not in order:
            out[k] = v
    return out


def _ordered_list(items: list[Any], order: tuple[str, ...], opt: tuple[str, ...]) -> list[Any]:
    return [_ordered(x, order, opt) if isinstance(x, dict) else x for x in items]


class Workflow:
    """A v2 workflow document, held as its raw dict for full pass-through."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self.doc = doc

    @property
    def nodes(self) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = self.doc.setdefault("nodes", [])
        return nodes

    @property
    def edges(self) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = self.doc.setdefault("edges", [])
        return edges

    @property
    def selections(self) -> dict[str, str]:
        selections: dict[str, str] = self.doc.setdefault("selections", {})
        return selections

    def node(self, node_id: str) -> dict[str, Any] | None:
        for n in self.nodes:
            if isinstance(n, dict) and n.get("id") == node_id:
                return n
        return None

    def _ordered_doc(self) -> dict[str, Any]:
        doc = self.doc
        out: dict[str, Any] = {}
        for k in _TOP_ORDER:
            if k in _ALWAYS_WRITTEN:
                v = doc.get(k)
                out[k] = _ALWAYS_WRITTEN[k]() if v is None else v
            elif k in doc and not _is_empty(doc[k]):
                out[k] = doc[k]
        for k, v in doc.items():
            if k not in _TOP_ORDER:
                out[k] = v
        out["nodes"] = _ordered_list(out["nodes"], _NODE_ORDER, _NODE_OPTIONAL)
        out["edges"] = _ordered_list(out["edges"], _EDGE_ORDER, _EDGE_OPTIONAL)
        result: dict[str, Any] = _ints_for_integral_floats(out)
        return result

    def to_json(self, pretty: bool = True) -> str:
        """The document as the builder writes it (`JSON.stringify(state, null, 2)`)."""
        return _dump(self._ordered_doc(), pretty)

    def to_json_pair(self) -> tuple[str, str]:
        """`to_json(pretty=True)` and `to_json(pretty=False)`, ordering the doc once."""
        doc = self._ordered_doc()
        return _dump(doc, True), _dump(doc, False)


def _dump(doc: dict[str, Any], pretty: bool) -> str:
    """The JSON text, checked once here to encode as UTF-8.

    A lone surrogate (from a ``\\ud800`` escape in a file or link) survives
    ``json.loads`` and ``json.dumps`` but cannot be encoded, so it is refused
    here; callers can then encode the text without that failure.
    """
    try:
        if pretty:
            text = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False)
        else:
            text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WorkflowError("The workflow holds a value that cannot be written as JSON.") from exc
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise WorkflowError("The workflow holds text that cannot be written as UTF-8.") from None
    return text


def _reject_constant(name: str) -> Any:
    # JSON.parse rejects NaN and Infinity, so the builder could never open them.
    raise ValueError(name)


def _structure_error(doc: dict[str, Any]) -> str | None:
    """Mirror of the builder's isWorkflowStateV2, for a doc already known to be v2."""
    checks: tuple[tuple[str, type | tuple[type, ...], str], ...] = (
        ("nodes", list, "a list"),
        ("edges", list, "a list"),
        ("selections", dict, "an object"),
        ("evaluations", dict, "an object"),
        ("viewport", dict, "an object"),
        ("title", str, "text"),
        ("description", str, "text"),
        ("provenance", dict, "an object"),
    )
    for key, kind, wanted in checks:
        if key in doc and not isinstance(doc[key], kind):
            return f"The workflow's '{key}' field must be {wanted}."
    if "mode" in doc and doc["mode"] != "assessment":
        return "The workflow's 'mode' field must be 'assessment' when present."
    # IDs are used as set members and dict keys everywhere after this, so a
    # list or object where an ID belongs is refused here, by position.
    for i, n in enumerate(doc.get("nodes") or []):
        if isinstance(n, dict) and "id" in n and not isinstance(n["id"], str):
            return f"Node {i}: its 'id' must be text."
    for i, e in enumerate(doc.get("edges") or []):
        if not isinstance(e, dict):
            continue
        for key in ("id", "source", "target"):
            if key in e and not isinstance(e[key], str):
                return f"Edge {i}: its '{key}' must be text."
    return None


def read_workflow(source: str | bytes | dict[str, Any]) -> Workflow:
    """Parse a builder v2 document from JSON text or an already parsed dict."""
    if isinstance(source, dict):
        parsed: Any = copy.deepcopy(source)
    else:
        try:
            text = source.decode("utf-8-sig") if isinstance(source, bytes) else source
            parsed = json.loads(text, parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise WorkflowError("The workflow is not valid JSON.") from None

    if not isinstance(parsed, dict):
        raise WorkflowError("The workflow must be a JSON object.")

    if parsed.get("v") == 1 or ("version" not in parsed and _V1_KEYS & parsed.keys()):
        raise WorkflowError(
            "This is a version 1 Workflow Builder file, which this server does not read. "
            f"Open it in the Workflow Builder ({BUILDER_URL}), export it again as JSON, "
            "and use the new file."
        )
    version = parsed.get("version")
    if version is None:
        raise WorkflowError("This is not a Workflow Builder file: it has no 'version' field.")
    if not (is_number(version) and version == FORMAT_VERSION):
        raise WorkflowError(
            f"Unsupported workflow version; this server reads version {FORMAT_VERSION} "
            f"files (Workflow Builder {BUILDER_FORMAT})."
        )

    problem = _structure_error(parsed)
    if problem:
        raise WorkflowError(problem)
    nodes = parsed.get("nodes")
    if isinstance(nodes, list) and len(nodes) > MAX_NODES:
        raise WorkflowError(f"The workflow has {len(nodes)} nodes; the limit is {MAX_NODES}.")
    return Workflow(parsed)


# ---- id allocation ----


class IdAllocator:
    """Hands out `n<N>` and `e<N>` IDs that collide with no existing ID.

    Existing IDs may be the server's own, the builder's random ones or
    hand-written; node and edge IDs share one namespace here.
    """

    def __init__(self, existing: Iterable[str]) -> None:
        self._used: set[str] = set(existing)
        self._next = {"n": 1, "e": 1}

    def _take(self, prefix: str) -> str:
        n = self._next[prefix]
        while f"{prefix}{n}" in self._used:
            n += 1
        new_id = f"{prefix}{n}"
        self._used.add(new_id)
        self._next[prefix] = n + 1
        return new_id

    def node_id(self) -> str:
        return self._take("n")

    def edge_id(self) -> str:
        return self._take("e")


# ---- input models ----

NoteColor = Literal["yellow", "blue", "green", "pink", "grey"]
LabelCorner = Literal["tl", "tr", "bl", "br"]

_KEY_DESC = (
    "Your name for this node, unique within the call. Edges, selections and zone "
    "members refer to nodes by this key. The server assigns the node's real ID."
)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TechniqueInput(_Input):
    kind: Literal["technique"]
    key: str = Field(min_length=1, description=_KEY_DESC)
    technique_id: str = Field(
        pattern=DFT_ID_PATTERN,
        description=(
            "SOLVE-IT technique ID, e.g. DFT-1001. A sub-technique ID is stored as its "
            "parent technique with the sub-technique as a method."
        ),
    )
    label: str | None = Field(
        default=None, description="Optional instance label, to tell duplicates apart."
    )
    methods: list[Annotated[str, Field(pattern=DFT_ID_PATTERN)]] = Field(
        default_factory=list,
        description="Sub-technique IDs chosen as this technique's methods.",
    )
    target: dict[str, str] | None = Field(
        default=None,
        description=(
            "What this placement examines: route path key to stated value. A key is one "
            "or more full IRIs, each in angle brackets, joined by '/', e.g. "
            "'<https://...#sourceApplication>/<https://...#applicationIdentifier>'."
        ),
    )


class PlaceholderInput(_Input):
    kind: Literal["placeholder"]
    key: str = Field(min_length=1, description=_KEY_DESC)
    name: str = Field(description="Name of the technique not yet in SOLVE-IT.")
    temp_id: str | None = Field(default=None, description="Provisional ID, e.g. DFT-temp01.")
    description: str | None = Field(default=None, description="Short description.")
    details: str | None = Field(default=None, description="Longer description.")
    input_classes: list[str] = Field(
        default_factory=list,
        description="Ontology classes consumed: full IRIs or free-text class names.",
    )
    output_classes: list[str] = Field(
        default_factory=list,
        description="Ontology classes produced: full IRIs or free-text class names.",
    )


class DataSourceInput(_Input):
    kind: Literal["data_source"]
    key: str = Field(min_length=1, description=_KEY_DESC)
    label: str = Field(description="What the data is, e.g. 'Disk image'.")
    classes: list[str] = Field(
        default_factory=list,
        description="Ontology classes the data is declared as: full IRIs or free text.",
    )


class ZoneInput(_Input):
    kind: Literal["zone"]
    key: str = Field(min_length=1, description=_KEY_DESC)
    label: str = Field(description="Zone label.")
    members: list[str] = Field(
        description=(
            "Keys (or existing node IDs) of the nodes the zone is drawn around. Used only "
            "to size the zone; the builder keeps no membership."
        ),
    )
    color: NoteColor | None = Field(default=None, description="Tint; the builder default is blue.")
    label_corner: LabelCorner | None = Field(
        default=None, description="Corner carrying the label; the builder default is tr."
    )


class NoteInput(_Input):
    kind: Literal["note"]
    key: str = Field(min_length=1, description=_KEY_DESC)
    text: str = Field(description="Note text.")
    color: NoteColor | None = Field(
        default=None, description="Tint; the builder default is yellow."
    )


class DecisionInput(_Input):
    kind: Literal["decision"]
    key: str = Field(min_length=1, description=_KEY_DESC)
    question: str = Field(description="The question the branch depends on.")


class ForkInput(_Input):
    kind: Literal["fork"]
    key: str = Field(min_length=1, description=_KEY_DESC)


class JoinInput(_Input):
    kind: Literal["join"]
    key: str = Field(min_length=1, description=_KEY_DESC)


NodeInput = Annotated[
    TechniqueInput
    | PlaceholderInput
    | DataSourceInput
    | ZoneInput
    | NoteInput
    | DecisionInput
    | ForkInput
    | JoinInput,
    Field(discriminator="kind"),
]


class EdgeInput(_Input):
    source: str = Field(description="Key of a node in this call, or an existing node ID.")
    target: str = Field(description="Key of a node in this call, or an existing node ID.")
    label: str | None = Field(default=None, description="Optional connection label.")


class SelectionInput(_Input):
    step: str = Field(description="Key or existing ID of a technique node.")
    weakness_id: str = Field(pattern=r"^DFW-\d+$", description="Weakness ID, e.g. DFW-1001.")
    mitigation_id: str = Field(pattern=r"^DFM-\d+$", description="Mitigation ID, e.g. DFM-1001.")
    value: Literal["Y", "N", "NA"] = Field(
        description="Y = applied, N = not applied, NA = not applicable."
    )


class ViewportInput(_Input):
    x: float = Field(description="Canvas pan x.")
    y: float = Field(description="Canvas pan y.")
    zoom: float = Field(gt=0, description="Canvas zoom factor.")


class WorkflowInput(_Input):
    title: str | None = Field(default=None, description="Workflow title.")
    description: str | None = Field(
        default=None,
        description=f"Short summary; clamped to {MAX_DESCRIPTION_LEN} characters.",
    )
    nodes: list[NodeInput] = Field(description="The workflow's nodes.")
    edges: list[EdgeInput] = Field(default_factory=list, description="Connections between nodes.")
    selections: list[SelectionInput] = Field(
        default_factory=list,
        description="Mitigation selections per technique step and weakness.",
    )
    viewport: ViewportInput | None = Field(
        default=None,
        description="Saved pan and zoom. Leave out to let the builder centre the workflow.",
    )


# ---- building ----


@dataclass
class BuildResult:
    workflow: Workflow
    key_to_id: dict[str, str] = field(default_factory=dict)
    zone_members: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    new_node_ids: list[str] = field(default_factory=list)


ParentOf = Callable[[str], str | None]
# Maps a class name to its full IRI where it can (KnowledgeBase.resolve_class).
ClassResolver = Callable[[str], str]


def _keep(name: str) -> str:
    return name


_BUILDER_TYPE = {
    "technique": "technique",
    "placeholder": "placeholder",
    "data_source": "caseClass",
    "zone": "region",
    "note": "note",
    "decision": "decision",
    "fork": "fork",
    "join": "join",
}


def _put(data: dict[str, Any], key: str, value: Any) -> None:
    if not _is_empty(value):
        data[key] = value


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _node_data(
    node: Any, parent_of: ParentOf, resolve_class: ClassResolver = _keep
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if isinstance(node, TechniqueInput):
        parent = parent_of(node.technique_id)
        methods = node.methods
        tid = node.technique_id
        if parent:
            methods = [tid, *methods]
            tid = parent
        data["techniqueId"] = tid
        _put(data, "label", node.label)
        _put(data, "methods", _dedupe(methods))
        if node.target:
            check_route_keys(node.key, node.target)
            data["target"] = dict(node.target)
    elif isinstance(node, PlaceholderInput):
        _put(data, "tempId", node.temp_id)
        data["name"] = node.name
        _put(data, "description", node.description)
        _put(data, "details", node.details)
        _put(data, "inputClasses", _dedupe(resolve_class(c) for c in node.input_classes))
        _put(data, "outputClasses", _dedupe(resolve_class(c) for c in node.output_classes))
    elif isinstance(node, DataSourceInput):
        data["label"] = node.label
        _put(data, "classes", _dedupe(resolve_class(c) for c in node.classes))
    elif isinstance(node, ZoneInput):
        # Size is computed by layout from the members' bounding box.
        data.update(label=node.label, w=0, h=0)
        _put(data, "labelCorner", node.label_corner)
        _put(data, "color", node.color)
    elif isinstance(node, NoteInput):
        data["text"] = node.text
        _put(data, "color", node.color)
    elif isinstance(node, DecisionInput):
        data["question"] = node.question
    elif isinstance(node, ForkInput | JoinInput):
        data["mode"] = node.kind
    return data


def _apply(
    wf: Workflow,
    nodes: list[Any],
    edges: list[EdgeInput],
    selections: list[SelectionInput],
    parent_of: ParentOf,
    resolve_class: ClassResolver = _keep,
) -> BuildResult:
    """Add nodes, edges and selections to `wf`, all or nothing."""
    existing_nodes = [n for n in wf.nodes if isinstance(n, dict)]
    existing_ids = {n["id"] for n in existing_nodes if isinstance(n.get("id"), str)}
    edge_ids = {e["id"] for e in wf.edges if isinstance(e, dict) and isinstance(e.get("id"), str)}

    if len(wf.nodes) + len(nodes) > MAX_NODES:
        raise WorkflowError(
            f"The workflow would have {len(wf.nodes) + len(nodes)} nodes; the limit is {MAX_NODES}."
        )

    alloc = IdAllocator(existing_ids | edge_ids)
    key_to_id: dict[str, str] = {}
    types: dict[str, Any] = {
        n["id"]: n.get("type") for n in existing_nodes if isinstance(n.get("id"), str)
    }
    for node in nodes:
        if node.key in key_to_id:
            raise WorkflowError(f"Node key '{node.key}' is used more than once.")
        if node.key in existing_ids:
            raise WorkflowError(
                f"Node key '{node.key}' is already a node ID in the workflow; use a different key."
            )
        key_to_id[node.key] = alloc.node_id()

    def resolve(ref: str, what: str) -> str:
        if ref in key_to_id:
            return key_to_id[ref]
        if ref in existing_ids:
            return ref
        raise WorkflowError(f"{what} refers to '{ref}', which is not a node key or node ID.")

    new_nodes: list[dict[str, Any]] = []
    zone_members: dict[str, list[str]] = {}
    for node in nodes:
        node_id = key_to_id[node.key]
        btype = _BUILDER_TYPE[node.kind]
        types[node_id] = btype
        new_nodes.append(
            {
                "id": node_id,
                "type": btype,
                "x": 0,
                "y": 0,
                "data": _node_data(node, parent_of, resolve_class),
            }
        )
        if isinstance(node, ZoneInput):
            members = [resolve(m, f"Zone '{node.key}'") for m in node.members]
            if node_id in members:
                raise WorkflowError(f"Zone '{node.key}' lists itself as a member.")
            zone_members[node_id] = _dedupe(members)

    new_edges: list[dict[str, Any]] = []
    for edge in edges:
        source = resolve(edge.source, "An edge")
        target = resolve(edge.target, "An edge")
        for ref, end in ((edge.source, source), (edge.target, target)):
            if types.get(end) in UNCONNECTABLE_TYPES:
                raise WorkflowError(
                    f"The edge from '{edge.source}' to '{edge.target}' cannot be drawn: "
                    f"'{ref}' is a note or zone, which have no connection points in the "
                    "builder. Connect steps, decisions, forks, joins and data sources only."
                )
        out = {"id": alloc.edge_id(), "source": source, "target": target}
        _put(out, "label", edge.label)
        new_edges.append(out)

    new_selections: dict[str, str] = {}
    for sel in selections:
        node_id = resolve(sel.step, "A selection")
        if types.get(node_id) != "technique":
            raise WorkflowError(f"A selection's step '{sel.step}' is not a technique node.")
        new_selections[f"{node_id}|{sel.weakness_id}|{sel.mitigation_id}"] = sel.value

    # Nothing above touched wf, so a failure leaves it as it was.
    wf.nodes.extend(new_nodes)
    wf.edges.extend(new_edges)
    wf.selections.update(new_selections)
    return BuildResult(
        workflow=wf,
        key_to_id=key_to_id,
        zone_members=zone_members,
        new_node_ids=[n["id"] for n in new_nodes],
    )


def build_workflow(
    spec: WorkflowInput, parent_of: ParentOf, resolve_class: ClassResolver = _keep
) -> BuildResult:
    """A new workflow from `spec`. Positions are left at 0,0 for layout to fill."""
    doc: dict[str, Any] = {"version": FORMAT_VERSION}
    cut = False
    if spec.title:
        doc["title"] = spec.title
    if spec.description:
        description = spec.description
        if len(description) > MAX_DESCRIPTION_LEN:
            description = description[:MAX_DESCRIPTION_LEN]
            cut = True
        doc["description"] = description
    doc.update(nodes=[], edges=[], selections={})
    if spec.viewport:
        # Rounded as the builder's serialize does.
        doc["viewport"] = {
            "x": round(spec.viewport.x),
            "y": round(spec.viewport.y),
            "zoom": round(spec.viewport.zoom * 10000) / 10000,
        }
    result = _apply(
        Workflow(doc), spec.nodes, spec.edges, spec.selections, parent_of, resolve_class
    )
    if cut:
        result.warnings.append(
            f"The description was cut to {MAX_DESCRIPTION_LEN} characters, the builder's limit."
        )
    return result


def add_to_workflow(
    wf: Workflow,
    nodes: list[NodeInput],
    edges: list[EdgeInput],
    selections: list[SelectionInput],
    parent_of: ParentOf,
    resolve_class: ClassResolver = _keep,
) -> BuildResult:
    """Append nodes, edges and selections to `wf`. Existing nodes are never changed.

    Edges, selections and zone members may refer to new node keys or to
    existing node IDs. On error, `wf` is left unchanged.
    """
    return _apply(wf, nodes, edges, selections, parent_of, resolve_class)
