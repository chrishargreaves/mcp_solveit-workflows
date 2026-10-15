"""FastMCP server: tool registration and startup.

stdio only. Do not add HTTP or SSE transports without a security design for
them: the tools read and write local files.

Every tool returns a dict. Failures are ``{"ok": False, "error": ...}`` with a
message that names keys, IDs, paths and limits, never raw file content. Text
taken from workflow files only ever appears inside data fields.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from . import __version__, sharelink
from .compile import compile_workflow
from .config import KB_PATH_VAR, Config
from .files import MAX_READ_BYTES, FileSafetyError, OutputDir, read_workflow_file
from .kb import KnowledgeBase, KnowledgeBaseError
from .layout import infer_direction, layout_workflow, place_new_nodes, zone_members_by_box
from .model import (
    BUILDER_FORMAT,
    BUILDER_URL,
    MAX_DESCRIPTION_LEN,
    MAX_NODES,
    EdgeInput,
    NodeInput,
    SelectionInput,
    Workflow,
    WorkflowError,
    WorkflowInput,
    build_workflow,
    read_workflow,
)
from .update import SetDataInput, apply_update
from .validate import Finding, validate_workflow

SERVER_NAME = "solveit-workflows"
MAX_STATUS_WARNINGS = 20
MAX_FIND_LIMIT = 100

mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        f"Build workflows for the SOLVE-IT Workflow Builder ({BUILDER_URL}). "
        "Call workflow_status first if a tool reports a configuration error. Give the user "
        "share links exactly as returned: changing a single character breaks them. The "
        "written file, or open_in_browser, is the dependable way to open a workflow."
    ),
)

Opener = Callable[[str], tuple[bool, str | None]]


@dataclass
class ServerState:
    config: Config
    kb: KnowledgeBase | None = None
    kb_error: str | None = None
    # Replaced in tests so no real browser is ever launched.
    opener: Opener = field(default=sharelink.open_in_browser)


_state: ServerState | None = None


def init_state(config: Config) -> ServerState:
    """Build the server state from config. Called at startup, and by tests."""
    global _state
    kb: KnowledgeBase | None = None
    kb_error: str | None = None
    if config.kb_path is None:
        kb_error = (
            f"No SOLVE-IT knowledge base configured. Set {KB_PATH_VAR} to the root of a "
            "SOLVE-IT checkout (the folder containing data/techniques)."
        )
    else:
        try:
            kb = KnowledgeBase.load(config.kb_path)
        except KnowledgeBaseError as exc:
            kb_error = f"{exc}. Set {KB_PATH_VAR} to the root of a SOLVE-IT checkout."
    _state = ServerState(config=config, kb=kb, kb_error=kb_error)
    return _state


def get_state() -> ServerState:
    if _state is None:
        return init_state(Config.from_env())
    return _state


# ---- helpers ----


class _ToolError(Exception):
    """A failure to report as {"ok": False, "error": message}."""


def _fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _require_kb(state: ServerState) -> KnowledgeBase:
    if state.kb is None:
        raise _ToolError(state.kb_error or "The knowledge base is not loaded.")
    return state.kb


def _guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except (_ToolError, WorkflowError, FileSafetyError, sharelink.ShareLinkError) as exc:
        return _fail(str(exc))
    except ValidationError as exc:
        return _fail(f"Invalid input: {exc.error_count()} problem(s): {_pydantic_summary(exc)}")
    except UnicodeError:
        # For example a lone surrogate in a path, which the filesystem cannot encode.
        return _fail("The request holds text that cannot be encoded, such as a lone surrogate.")


def _pydantic_summary(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', '')}" if loc else str(err.get("msg", "")))
    return "; ".join(parts)


def _split(findings: list[Finding]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors = [f.to_dict() for f in findings if f.level == "error"]
    warnings = [f.to_dict() for f in findings if f.level == "warning"]
    return errors, warnings


def _note(code: str, message: str) -> dict[str, Any]:
    return {"level": "warning", "code": code, "message": message}


def _counts(wf: Workflow) -> dict[str, int]:
    return {"nodes": len(wf.nodes), "edges": len(wf.edges)}


_BARE_HASH = re.compile(r"~[A-Za-z0-9_-]*")


def _is_link(source: str) -> bool:
    """A URL, a "#..." hash, or a bare "~<base64url>" hash; "~/x.json" is a path."""
    s = source.strip()
    return s.startswith(("http://", "https://", "#")) or _BARE_HASH.fullmatch(s) is not None


def _load_source(state: ServerState, source: str) -> tuple[Workflow, Path | None]:
    """Read a workflow from a share link or a file in an allowed directory.

    Returns the workflow and, for a file, its resolved path.
    """
    if _is_link(source):
        return read_workflow(sharelink.decode_link(source)), None
    text = read_workflow_file(source, state.config.read_dirs)
    return read_workflow(text), Path(source).expanduser().resolve()


def _load_inline_or_file(
    state: ServerState, file_path: str | None, workflow: dict[str, Any] | None
) -> Workflow:
    if (file_path is None) == (workflow is None):
        raise _ToolError("Give exactly one of 'file_path' or 'workflow'.")
    if file_path is not None:
        return read_workflow(read_workflow_file(file_path, state.config.read_dirs))
    assert workflow is not None
    return read_workflow(workflow)


def _write_and_link(
    state: ServerState,
    out: OutputDir,
    wf: Workflow,
    target: Path,
    *,
    overwrite: bool,
    open_in_browser: bool,
) -> dict[str, Any]:
    pretty, compact = wf.to_json_pair()
    written = out.write(target, pretty, overwrite=overwrite)
    url = sharelink.build_url(compact)
    link = sharelink.link_for_url(url)
    opened: bool | None = None
    open_note: str | None = None
    if open_in_browser:
        # The full URL, whatever its length; the opener enforces the builder's cap.
        opened, open_note = state.opener(url)
    return {
        "ok": True,
        "file": str(written.path),
        "replaced": written.replaced,
        "link": link.url,
        "link_length": link.length,
        "link_note": link.reason,
        "opened": opened,
        "open_note": open_note,
        "counts": _counts(wf),
        "warnings": [_note("manifest", w) for w in written.warnings],
    }


TargetPicker = Callable[[], tuple[Path, bool]]


def _validate_and_write(
    state: ServerState,
    kb: KnowledgeBase,
    out: OutputDir,
    wf: Workflow,
    *,
    build_warnings: list[str],
    layout_warnings: list[str],
    new_node_ids: list[str],
    pick_target: TargetPicker,
    open_in_browser: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """The shared create/update tail: validate, then write and link.

    `pick_target` gives the path and overwrite flag; it runs only once the
    workflow has no errors. `extra` goes into the result before the warnings.
    """
    notes = [_note("build", w) for w in build_warnings]
    notes += [_note("layout", w) for w in layout_warnings]
    errors, warnings = _split(validate_workflow(wf, kb, input_node_ids=new_node_ids))
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings + notes}
    target, overwrite = pick_target()
    result = _write_and_link(
        state, out, wf, target, overwrite=overwrite, open_in_browser=open_in_browser
    )
    write_notes = result.pop("warnings")
    result.update(extra)
    result["warnings"] = warnings + notes + write_notes
    result["knowledge_base_revision"] = kb.revision
    return result


# ---- tools ----

_LINK_ADVICE = (
    "If a link is returned, give it to the user exactly as returned, in full: changing "
    "even one character breaks it. The file path, or open_in_browser=true, is the "
    "dependable way to open the workflow; offer those, especially when there is no link "
    "(large workflows exceed the 8000-character link limit)."
)

_INPUT_ADVICE = (
    "Nodes are given with your own keys; edges, selections and zone members refer to "
    "those keys, and the result maps each key to the node ID the server assigned. A "
    "sub-technique ID can be given directly as a technique: it is stored as its parent "
    "technique with the sub-technique as a method. Zones take the keys of their member "
    "nodes and are drawn around them. Use decision or fork nodes for branches out of a "
    "step. Use workflow_find_techniques to look up DFT IDs."
)


@mcp.tool()
def workflow_status() -> dict[str, Any]:
    """Report whether the server is ready: knowledge base, output and read directories, limits."""
    state = get_state()
    cfg = state.config
    kb = state.kb
    status: dict[str, Any] = {
        "server": SERVER_NAME,
        "version": __version__,
        "ready": state.kb_error is None and kb is not None,
        "error": state.kb_error,
        "knowledge_base_path": str(cfg.kb_path) if cfg.kb_path else None,
        "knowledge_base_revision": kb.revision if kb else None,
        "counts": kb.counts() if kb else None,
        "load_warnings": kb.load_warnings[:MAX_STATUS_WARNINGS] if kb else [],
        "load_warning_count": len(kb.load_warnings) if kb else 0,
        "output_dir": str(cfg.output_dir),
        "read_dirs": [str(p) for p in cfg.read_dirs],
        "builder_format": BUILDER_FORMAT,
        "limits": {
            "max_nodes": MAX_NODES,
            "share_link_max_chars": sharelink.SHARE_LINK_MAX_CHARS,
            "max_link_hash_chars": sharelink.MAX_HASH_LENGTH,
            "max_file_bytes": MAX_READ_BYTES,
            "max_inflated_bytes": sharelink.MAX_INFLATED_BYTES,
            "max_description_chars": MAX_DESCRIPTION_LEN,
        },
    }
    return status


@mcp.tool()
def workflow_find_techniques(
    query: Annotated[
        str,
        Field(
            description=(
                "Words to look for, e.g. 'hash verification' or 'remove disk'. Each word is "
                "matched separately (imaging also finds image); put a phrase in double "
                "quotes to match it whole."
            )
        ),
    ],
    limit: Annotated[
        int, Field(ge=1, le=MAX_FIND_LIMIT, description="Maximum number of matches to return.")
    ] = 10,
) -> dict[str, Any]:
    """Find SOLVE-IT technique IDs (DFT-...), best matches first.

    Searches names, synonyms, descriptions, details and example tools. Techniques
    matching every word come first (matched_all_words), then those matching most.
    Each match gives the ID, name, parent technique (for a sub-technique), a short
    description, and its CASE input and output classes, which is usually enough
    to choose techniques and order them without looking each one up.
    """

    def run() -> dict[str, Any]:
        kb = _require_kb(get_state())
        capped = max(1, min(int(limit), MAX_FIND_LIMIT))
        return {
            "ok": True,
            "query": query,
            "matches": kb.find_techniques(query, capped),
            "knowledge_base_revision": kb.revision,
        }

    return _guarded(run)


@mcp.tool(
    description=f"""Create a new SOLVE-IT workflow, write it to a JSON file and
return the file path and, when it fits, a share link for the Workflow Builder
({BUILDER_URL}).

{_INPUT_ADVICE}

Layout is computed: direction "down" (the default, as the builder draws) or "right"
(left to right, for example for slides). The workflow is checked against the SOLVE-IT
knowledge base first; if there are errors nothing is written and the errors are
returned. Warnings are returned with the result.

{_LINK_ADVICE}"""
)
def workflow_create(
    workflow: Annotated[WorkflowInput, Field(description="The workflow to build.")],
    direction: Annotated[
        Literal["down", "right"],
        Field(description='Layout direction: "down" (default) or "right" for slides.'),
    ] = "down",
    filename: Annotated[
        str | None,
        Field(
            description=(
                "File name inside the output directory (.json added if missing). "
                "Defaults to a name made from the title."
            )
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Field(
            description=(
                "Replace an existing file of that name even if this server did not write "
                "it. Otherwise a numbered name is used."
            )
        ),
    ] = False,
    open_in_browser: Annotated[
        bool,
        Field(description="Also open the workflow in the default browser on this machine."),
    ] = False,
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        state = get_state()
        kb = _require_kb(state)
        spec = (
            workflow
            if isinstance(workflow, WorkflowInput)
            else WorkflowInput.model_validate(workflow)
        )
        built = build_workflow(spec, kb.parent_of, kb.resolve_class)
        wf = built.workflow
        out = OutputDir(state.config.output_dir)
        return _validate_and_write(
            state,
            kb,
            out,
            wf,
            build_warnings=built.warnings,
            layout_warnings=layout_workflow(
                wf, direction=direction, zone_members=built.zone_members
            ),
            new_node_ids=built.new_node_ids,
            pick_target=lambda: (out.resolve_target(filename, spec.title), overwrite),
            open_in_browser=open_in_browser,
            extra={"node_ids": built.key_to_id},
        )

    return _guarded(run)


@mcp.tool(
    description=f"""Change an existing SOLVE-IT workflow and write the result.

The source is a workflow .json file in an allowed directory (see workflow_status), or
a Workflow Builder share link ({BUILDER_URL}/#~...). If the user has
opened or edited the workflow in the builder since it was written, ask them for a
fresh export or a share link from the builder's Share button and use that as source,
so their edits are not lost.

Operations: add nodes and edges, remove nodes or edges by ID (a removed node's
edges, selections, evaluations and provenance go with it, and are reported), set
data fields on a node, and set mitigation selections. Existing nodes keep their IDs,
positions and data except for fields you set; everything else in the file is kept.
New nodes are placed after their predecessors. {_INPUT_ADVICE} New edges, selections
and set_data may also refer to existing node IDs; call workflow_inspect first to get
node IDs when the workflow came from the builder or an earlier session. Edges cannot
connect notes or zones: the builder draws no connection points on them.

Where it is written: a source file inside the output directory is updated in place;
a source elsewhere, or a link, gives a new file in the output directory (named by
'filename', else the source file's name, else the title). The result gives the path
written, which later updates should use.

{_LINK_ADVICE}"""
)
def workflow_update(
    source: Annotated[
        str,
        Field(
            description=(
                "Path to a workflow .json file in an allowed directory, or a Workflow "
                f"Builder share link {BUILDER_URL}/#~..."
            )
        ),
    ],
    add_nodes: Annotated[
        Sequence[NodeInput], Field(description="Nodes to add, each with your own key.")
    ] = (),
    add_edges: Annotated[
        Sequence[EdgeInput],
        Field(description="Connections to add, between new node keys or existing node IDs."),
    ] = (),
    remove_ids: Annotated[
        Sequence[str], Field(description="IDs of existing nodes or edges to remove.")
    ] = (),
    set_data: Annotated[
        Sequence[SetDataInput],
        Field(description="Data fields to set on existing or added nodes."),
    ] = (),
    selections: Annotated[
        Sequence[SelectionInput],
        Field(description="Mitigation selections to set (Y, N or NA) per step and weakness."),
    ] = (),
    direction: Annotated[
        Literal["down", "right"] | None,
        Field(
            description=(
                "Direction for placing new nodes. Leave out to keep the workflow's own direction."
            )
        ),
    ] = None,
    filename: Annotated[
        str | None,
        Field(
            description=(
                "File name in the output directory for the result. Leave out to update a "
                "file in the output directory in place."
            )
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Field(
            description=(
                "Replace an existing file of the chosen name even if this server did not "
                "write it. Otherwise a numbered name is used."
            )
        ),
    ] = False,
    open_in_browser: Annotated[
        bool,
        Field(description="Also open the workflow in the default browser on this machine."),
    ] = False,
    relayout: Annotated[
        bool,
        Field(
            description=(
                "Lay the whole workflow out again instead of only placing new nodes. Use it "
                "when adding a zone round existing nodes that are not next to each other, or "
                "after large changes. It moves every node, so arrangements made by hand in "
                "the builder are lost. Existing zones keep the nodes that lie inside them."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        state = get_state()
        kb = _require_kb(state)
        out = OutputDir(state.config.output_dir)

        wf, source_path = _load_source(state, source)
        # Read before the update: new nodes start at 0,0 and could land in a box.
        boxed = zone_members_by_box(wf) if relayout else {}

        updated = apply_update(
            wf,
            parent_of=kb.parent_of,
            resolve_class=kb.resolve_class,
            add_nodes=list(add_nodes),
            add_edges=list(add_edges),
            remove_ids=list(remove_ids),
            set_data=list(set_data),
            selections=list(selections),
        )

        def pick_target() -> tuple[Path, bool]:
            root = out.root.resolve()
            if filename is None and source_path is not None and source_path.is_relative_to(root):
                # Naming a file in the output directory is the explicit request to replace it.
                return source_path, True
            name = filename if filename is not None else (source_path.name if source_path else None)
            title = wf.doc.get("title")
            return out.resolve_target(name, title if isinstance(title, str) else None), overwrite

        return _validate_and_write(
            state,
            kb,
            out,
            wf,
            build_warnings=updated.warnings,
            layout_warnings=(
                layout_workflow(
                    wf,
                    direction=direction or infer_direction(wf),
                    zone_members={**boxed, **updated.zone_members},
                )
                if relayout
                else place_new_nodes(
                    wf,
                    updated.new_node_ids,
                    direction=direction,
                    zone_members=updated.zone_members,
                )
            ),
            new_node_ids=updated.new_node_ids,
            pick_target=pick_target,
            open_in_browser=open_in_browser,
            extra={"node_ids": updated.key_to_id, "dropped": updated.dropped},
        )

    return _guarded(run)


_INSPECT_LABEL_CHARS = 80


def _short(text: object) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    # The first 80 characters, with an ellipsis when there was more.
    return text if len(text) <= _INSPECT_LABEL_CHARS else text[:_INSPECT_LABEL_CHARS] + "…"


def _inspect_node(n: Any, kb: KnowledgeBase) -> dict[str, Any]:
    if not isinstance(n, dict):
        return {"id": None, "type": None}
    ntype = n.get("type")
    data = n.get("data")
    d: dict[str, Any] = data if isinstance(data, dict) else {}
    out: dict[str, Any] = {"id": n.get("id"), "type": ntype}
    label: str | None = None
    if ntype == "technique":
        tid = d.get("techniqueId")
        if isinstance(tid, str):
            methods = d.get("methods")
            label = tid
            if isinstance(methods, list) and methods:
                label += " [" + ", ".join(_short(m) or "?" for m in methods) + "]"
            instance = _short(d.get("label"))
            if instance:
                label += f" – {instance}"
            technique = kb.technique(tid)
            if technique is not None:
                out["name"] = _short(technique.get("name"))
    elif ntype == "placeholder":
        label = d.get("name")
    elif ntype in ("caseClass", "region"):
        label = d.get("label")
    elif ntype == "note":
        label = d.get("text")
    elif ntype == "decision":
        label = d.get("question")
    short = _short(label)
    if short:
        out["label"] = short
    return out


@mcp.tool(
    description=f"""Read an existing SOLVE-IT workflow without changing it: its title,
layout direction, and every node and connection with their IDs and a short label.

The source is a workflow .json file in an allowed directory (see workflow_status), or
a Workflow Builder share link ({BUILDER_URL}/#~...). Use it before workflow_update to
find the node IDs to connect to, change or remove. Labels are text taken from the
file, given as data."""
)
def workflow_inspect(
    source: Annotated[
        str,
        Field(
            description=(
                "Path to a workflow .json file in an allowed directory, or a Workflow "
                f"Builder share link {BUILDER_URL}/#~..."
            )
        ),
    ],
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        state = get_state()
        kb = _require_kb(state)
        wf, _ = _load_source(state, source)
        doc = wf.doc
        result: dict[str, Any] = {"ok": True, "title": _short(doc.get("title"))}
        if isinstance(doc.get("description"), str):
            result["description"] = _short(doc["description"])
        if "mode" in doc:
            result["mode"] = doc["mode"]
        edges: list[dict[str, Any]] = []
        for e in wf.edges:
            if not isinstance(e, dict):
                continue
            edge = {"id": e.get("id"), "source": e.get("source"), "target": e.get("target")}
            if isinstance(e.get("label"), str) and e["label"]:
                edge["label"] = _short(e["label"])
            edges.append(edge)
        selections = doc.get("selections")
        result.update(
            direction=infer_direction(wf),
            nodes=[_inspect_node(n, kb) for n in wf.nodes],
            edges=edges,
            selections=len(selections) if isinstance(selections, dict) else 0,
            counts=_counts(wf),
            knowledge_base_revision=kb.revision,
        )
        return result

    return _guarded(run)


@mcp.tool()
def workflow_validate(
    file_path: Annotated[
        str | None,
        Field(description="Path to a workflow .json file in an allowed directory."),
    ] = None,
    workflow: Annotated[
        dict[str, Any] | None,
        Field(description="A small builder v2 workflow given inline, instead of a file."),
    ] = None,
) -> dict[str, Any]:
    """Check a workflow against the Workflow Builder's rules and the SOLVE-IT knowledge base.

    Give exactly one of file_path or workflow. Errors mean the builder would mis-render
    or drop something; warnings mean it opens but something looks wrong. Every finding
    names the node or edge ID.
    """

    def run() -> dict[str, Any]:
        state = get_state()
        kb = _require_kb(state)
        wf = _load_inline_or_file(state, file_path, workflow)
        errors, warnings = _split(validate_workflow(wf, kb))
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "counts": _counts(wf),
            "knowledge_base_revision": kb.revision,
        }

    return _guarded(run)


@mcp.tool()
def workflow_compile(
    file_path: Annotated[
        str | None,
        Field(description="Path to a workflow .json file in an allowed directory."),
    ] = None,
    workflow: Annotated[
        dict[str, Any] | None,
        Field(description="A small builder v2 workflow given inline, instead of a file."),
    ] = None,
    detail: Annotated[
        bool,
        Field(description="Include every weakness (with ASTM categories) and mitigation row."),
    ] = False,
) -> dict[str, Any]:
    """List the weaknesses and mitigations for every technique step, in flow order.

    Mirrors the builder's Compile step. Give exactly one of file_path or workflow. By
    default each step is summarised (weakness count, counts per status, worst status);
    detail=true adds full weakness and mitigation rows. A mitigation is in place when
    its selection is Y, or when it has no selection and its implementing technique is on
    the canvas (not in assessment mode). If the workflow has validation errors, only the
    errors are returned.
    """

    def run() -> dict[str, Any]:
        state = get_state()
        kb = _require_kb(state)
        wf = _load_inline_or_file(state, file_path, workflow)
        errors, warnings = _split(validate_workflow(wf, kb))
        if errors:
            return {"ok": False, "errors": errors, "warnings": warnings}
        compiled = compile_workflow(wf, kb, detail=detail)
        return {"ok": True, **compiled, "warnings": warnings}

    return _guarded(run)


def main() -> None:
    init_state(Config.from_env())
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
