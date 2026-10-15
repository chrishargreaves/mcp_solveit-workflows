"""The MCP tools, called through FastMCP so arguments are parsed as a client's would be."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import WORKFLOWS, init_fixture_state
from mcp.server.fastmcp.exceptions import ToolError

from mcp_solveit_workflows import server, sharelink
from mcp_solveit_workflows.config import KB_PATH_VAR, OUTPUT_DIR_VAR, READ_DIRS_VAR, Config
from mcp_solveit_workflows.files import OutputDir
from mcp_solveit_workflows.layout import estimate_size


class FakeRunner:
    """Stands in for the opener's launcher (subprocess.Popen): records argv, exits 0."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> Any:
        self.calls.append(argv)

        class Done:
            def wait(self, timeout: float | None = None) -> int:
                return 0

        return Done()


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    out = tmp_path / "out"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    return {"out": out, "downloads": downloads}


@pytest.fixture
def runner(dirs: dict[str, Path]) -> FakeRunner:
    state = init_fixture_state(dirs["out"], **{READ_DIRS_VAR: str(dirs["downloads"])})
    fake = FakeRunner()
    # Never a real browser: the opener always goes through the fake runner.
    state.opener = functools.partial(sharelink.open_in_browser, launcher=fake, platform="darwin")
    return fake


def call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    result = asyncio.run(server.mcp.call_tool(name, args or {}))
    assert isinstance(result, tuple)
    structured = result[1]
    assert isinstance(structured, dict)
    return structured


def fixture_doc(name: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))
    return doc


def compact(doc: dict[str, Any]) -> str:
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False)


def to_input(doc: dict[str, Any]) -> dict[str, Any]:
    """Turn a builder file into create input, keyed by the file's node IDs."""
    nodes: list[dict[str, Any]] = []
    plain = [n for n in doc["nodes"] if n["type"] != "region"]
    for n in doc["nodes"]:
        d = n.get("data", {})
        t = n["type"]
        node: dict[str, Any] = {"key": n["id"]}
        if t == "technique":
            node.update(kind="technique", technique_id=d["techniqueId"])
            for src, dst in (("label", "label"), ("methods", "methods"), ("target", "target")):
                if src in d:
                    node[dst] = d[src]
        elif t == "placeholder":
            node.update(kind="placeholder", name=d["name"])
            for src, dst in (
                ("tempId", "temp_id"),
                ("description", "description"),
                ("details", "details"),
                ("inputClasses", "input_classes"),
                ("outputClasses", "output_classes"),
            ):
                if src in d:
                    node[dst] = d[src]
        elif t == "caseClass":
            node.update(kind="data_source", label=d["label"], classes=d.get("classes", []))
        elif t == "region":
            x0, y0, x1, y1 = n["x"], n["y"], n["x"] + d["w"], n["y"] + d["h"]
            members = []
            for m in plain:
                w, h = estimate_size(m)
                cx, cy = m["x"] + w / 2, m["y"] + h / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    members.append(m["id"])
            node.update(kind="zone", label=d["label"], members=members)
            if "color" in d:
                node["color"] = d["color"]
            if "labelCorner" in d:
                node["label_corner"] = d["labelCorner"]
        elif t == "note":
            node.update(kind="note", text=d["text"])
            if "color" in d:
                node["color"] = d["color"]
        elif t == "decision":
            node.update(kind="decision", question=d["question"])
        elif t in ("fork", "join"):
            node["kind"] = t
        else:
            raise AssertionError(t)
        nodes.append(node)
    edges = []
    for e in doc["edges"]:
        edge = {"source": e["source"], "target": e["target"]}
        if e.get("label"):
            edge["label"] = e["label"]
        edges.append(edge)
    spec: dict[str, Any] = {"nodes": nodes, "edges": edges}
    for k in ("title", "description"):
        if k in doc:
            spec[k] = doc[k]
    return spec


def written(result: dict[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(Path(result["file"]).read_text(encoding="utf-8"))
    return doc


def codes(findings: list[dict[str, Any]]) -> set[str]:
    return {f["code"] for f in findings}


# ---- registration and status ----


def test_registered_tools_are_exactly_the_seven() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == {
        "workflow_status",
        "workflow_find_techniques",
        "workflow_create",
        "workflow_update",
        "workflow_inspect",
        "workflow_validate",
        "workflow_compile",
    }
    by_name = {t.name: t for t in tools}
    for name in ("workflow_create", "workflow_update"):
        desc = by_name[name].description or ""
        assert "exactly as returned" in desc
        assert "open_in_browser" in desc
    assert "fresh export" in (by_name["workflow_update"].description or "")
    assert "workflow_inspect" in (by_name["workflow_update"].description or "")
    # Every parameter is described.
    for tool in tools:
        for pname, prop in tool.inputSchema.get("properties", {}).items():
            assert prop.get("description"), (tool.name, pname)


def test_status_reports_kb(runner: FakeRunner) -> None:
    status = call("workflow_status")
    assert status["ready"] is True
    assert status["counts"] == {"techniques": 8, "weaknesses": 6, "mitigations": 4}
    assert status["builder_format"] == "0.13.0"
    assert status["limits"]["share_link_max_chars"] == 8000
    assert status["knowledge_base_revision"] == "unknown"


def test_degraded_mode_every_tool_but_status_errors(tmp_path: Path) -> None:
    server.init_state(Config.from_env({OUTPUT_DIR_VAR: str(tmp_path)}))
    calls: list[tuple[str, dict[str, Any]]] = [
        ("workflow_find_techniques", {"query": "hash"}),
        ("workflow_create", {"workflow": {"nodes": [{"kind": "fork", "key": "f"}]}}),
        ("workflow_update", {"source": str(tmp_path / "x.json")}),
        ("workflow_validate", {"workflow": {"version": 2, "nodes": []}}),
        ("workflow_compile", {"workflow": {"version": 2, "nodes": []}}),
    ]
    for name, args in calls:
        res = call(name, args)
        assert res["ok"] is False, name
        assert KB_PATH_VAR in res["error"], name
    assert list(tmp_path.iterdir()) == []
    assert call("workflow_status")["ready"] is False


def test_degraded_mode_when_kb_path_is_not_a_checkout(tmp_path: Path) -> None:
    server.init_state(Config.from_env({KB_PATH_VAR: str(tmp_path), OUTPUT_DIR_VAR: str(tmp_path)}))
    status = call("workflow_status")
    assert status["ready"] is False
    assert "techniques" in status["error"]
    assert call("workflow_find_techniques", {"query": "hash"})["ok"] is False


def test_find_techniques(runner: FakeRunner) -> None:
    res = call("workflow_find_techniques", {"query": "imaging", "limit": 5})
    assert res["ok"] is True
    assert [(m["id"], m["name"], m["parent"]) for m in res["matches"][:2]] == [
        ("DFT-1002", "Physical imaging", "DFT-1001"),
        ("DFT-1003", "Logical imaging", "DFT-1001"),
    ]
    assert {"description", "inputs", "outputs", "matched_all_words"} <= set(res["matches"][0])
    assert {m["parent"] for m in res["matches"] if m["id"] == "DFT-1002"} == {"DFT-1001"}


# ---- create ----


def test_ae1_create_21_node_fixture_gives_file_and_link(runner: FakeRunner) -> None:
    doc = fixture_doc("synthetic_21_nodes.json")
    res = call("workflow_create", {"workflow": to_input(doc)})
    assert res["ok"] is True, res
    path = Path(res["file"])
    assert path.is_file() and path.parent == Path(server.get_state().config.output_dir)
    assert res["link"] is not None and res["link"].startswith(sharelink.SHARE_PREFIX)
    assert res["link_length"] <= 8000 and res["link_note"] is None
    assert res["counts"] == {"nodes": 21, "edges": 22}
    assert set(res["node_ids"]) == {n["id"] for n in doc["nodes"]}
    assert "errors" not in res
    # The link decodes to exactly what was written.
    assert sharelink.decode_link(res["link"]) == written(res)
    assert res["opened"] is None and runner.calls == []


def test_create_recreates_63_node_fixture_left_to_right(runner: FakeRunner) -> None:
    doc = fixture_doc("synthetic_63_nodes.json")
    spec = to_input(doc)
    res = call("workflow_create", {"workflow": spec, "direction": "right"})
    assert res["ok"] is True, res
    out = written(res)
    ids = res["node_ids"]
    by_id = {n["id"]: n for n in out["nodes"]}
    assert len(out["nodes"]) == len(doc["nodes"]) == 63
    assert len(out["edges"]) == len(doc["edges"])
    for n in doc["nodes"]:
        new = by_id[ids[n["id"]]]
        assert new["type"] == n["type"]
        if n["type"] == "region":
            for k in ("label", "color", "labelCorner"):
                assert new["data"].get(k) == n["data"].get(k)
            assert new["data"]["w"] > 0 and new["data"]["h"] > 0
        else:
            assert new["data"] == n["data"], n["id"]
    old_edges = {(ids[e["source"]], ids[e["target"]]) for e in doc["edges"]}
    assert {(e["source"], e["target"]) for e in out["edges"]} == old_edges
    # Left to right: a technique step's outgoing edges leave on the right.
    assert any(e["sourceHandle"].startswith("right") for e in out["edges"])
    assert {"technique", "region", "fork", "placeholder", "note", "caseClass"} <= {
        n["type"] for n in out["nodes"]
    }


def test_create_validation_error_writes_nothing(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    spec = {
        "nodes": [{"kind": "zone", "key": "z", "label": "Empty", "members": []}],
    }
    res = call("workflow_create", {"workflow": spec})
    assert res["ok"] is False
    assert "invalid_region_size" in codes(res["errors"])
    assert not dirs["out"].exists() or list(dirs["out"].iterdir()) == []


def test_create_model_error_is_reported(runner: FakeRunner) -> None:
    spec = {
        "nodes": [
            {"kind": "fork", "key": "a"},
            {"kind": "join", "key": "a"},
        ]
    }
    res = call("workflow_create", {"workflow": spec})
    assert res == {"ok": False, "error": "Node key 'a' is used more than once."}


def test_create_bad_input_rejected_by_schema(runner: FakeRunner) -> None:
    with pytest.raises(ToolError):
        call("workflow_create", {"workflow": {"nodes": [{"kind": "subworkflow", "key": "s"}]}})


@pytest.mark.parametrize("filename", ["~/.zshrc", "../../x.json", "/etc/x.json", "a.txt"])
def test_ae4_bad_filenames_refused_and_nothing_written(
    runner: FakeRunner, dirs: dict[str, Path], filename: str
) -> None:
    spec = {"title": "T", "nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
    res = call("workflow_create", {"workflow": spec, "filename": filename})
    assert res["ok"] is False
    assert str(dirs["out"].resolve()) in res["error"]
    assert not dirs["out"].exists() or list(dirs["out"].rglob("*.json")) == []


def test_create_collision_rule_and_overwrite(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    dirs["out"].mkdir()
    foreign = dirs["out"] / "mine.json"
    foreign.write_text("{}", encoding="utf-8")
    spec = {"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
    res = call("workflow_create", {"workflow": spec, "filename": "mine"})
    assert Path(res["file"]).name == "mine-2.json" and res["replaced"] is False
    assert foreign.read_text(encoding="utf-8") == "{}"
    res = call("workflow_create", {"workflow": spec, "filename": "mine", "overwrite": True})
    assert Path(res["file"]).name == "mine.json" and res["replaced"] is True


def _big_spec(n: int) -> dict[str, Any]:
    nodes = [
        {
            "kind": "technique",
            "key": f"s{i}",
            "technique_id": "DFT-1004",
            "label": hashlib.sha256(str(i).encode()).hexdigest(),
        }
        for i in range(n)
    ]
    edges = [{"source": f"s{i}", "target": f"s{i + 1}"} for i in range(n - 1)]
    return {"title": "Big", "nodes": nodes, "edges": edges}


def test_ae2_large_workflow_has_no_link_but_opens(runner: FakeRunner) -> None:
    res = call("workflow_create", {"workflow": _big_spec(150), "open_in_browser": True})
    assert res["ok"] is True
    assert res["link"] is None
    assert res["link_length"] > 8000
    assert "8000" in res["link_note"]
    assert Path(res["file"]).is_file()
    assert res["opened"] is True and res["open_note"] is None
    assert len(runner.calls) == 1
    argv = runner.calls[0]
    assert argv[0] == "open"
    assert argv[1].startswith(sharelink.SHARE_PREFIX)
    assert len(argv[1]) == res["link_length"]
    assert sharelink.decode_link(argv[1]) == written(res)


def test_open_failure_still_returns_file(runner: FakeRunner) -> None:
    state = server.get_state()
    state.opener = functools.partial(sharelink.open_in_browser, platform="win32")
    spec = {"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
    res = call("workflow_create", {"workflow": spec, "open_in_browser": True})
    assert res["ok"] is True and res["opened"] is False
    assert "not opened" in res["open_note"]
    assert Path(res["file"]).is_file()


# ---- update ----


def _exported_demo() -> dict[str, Any]:
    """synthetic_small as if a user dragged nodes and ticked mitigations."""
    doc = fixture_doc("synthetic_small.json")
    for i, n in enumerate(doc["nodes"]):
        n["x"] += 7 * (i + 1)
        n["y"] += 3 * (i + 1)
    tech = [n["id"] for n in doc["nodes"] if n["type"] == "technique"]
    doc["selections"] = {
        f"{tech[0]}|DFW-1001|DFM-1001": "Y",
        f"{tech[1]}|DFW-1002|DFM-1002": "N",
    }
    return doc


def _last_technique(doc: dict[str, Any]) -> str:
    sources = {e["source"] for e in doc["edges"]}
    return str(next(n["id"] for n in doc["nodes"] if n["id"] not in sources))


def _check_ae3(res: dict[str, Any], before: dict[str, Any]) -> None:
    assert res["ok"] is True, res
    after = written(res)
    new_id = res["node_ids"]["hash"]
    old_by_id = {n["id"]: n for n in before["nodes"]}
    for n in after["nodes"]:
        if n["id"] == new_id:
            continue
        assert (n["x"], n["y"], n["data"]) == (
            old_by_id[n["id"]]["x"],
            old_by_id[n["id"]]["y"],
            old_by_id[n["id"]]["data"],
        )
    assert len(after["nodes"]) == len(before["nodes"]) + 1
    assert after["selections"] == before["selections"]
    assert after["viewport"] == before["viewport"]
    new = next(n for n in after["nodes"] if n["id"] == new_id)
    last = old_by_id[_last_technique(before)]
    assert new["y"] > last["y"]  # one row further down
    assert new_id not in old_by_id


def test_ae3_update_file_source_keeps_positions_and_selections(
    runner: FakeRunner, dirs: dict[str, Path]
) -> None:
    before = _exported_demo()
    src = dirs["downloads"] / "demo export.json"
    src.write_text(json.dumps(before, indent=2), encoding="utf-8")
    src_bytes = src.read_bytes()
    res = call(
        "workflow_update",
        {
            "source": str(src),
            "add_nodes": [{"kind": "technique", "key": "hash", "technique_id": "DFT-1004"}],
            "add_edges": [{"source": _last_technique(before), "target": "hash"}],
        },
    )
    _check_ae3(res, before)
    # Outside the output dir: a new file there, named from the source; source untouched.
    assert Path(res["file"]).parent == dirs["out"].resolve()
    assert Path(res["file"]).name == "demo export.json"
    assert src.read_bytes() == src_bytes


def test_ae3_update_link_source(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    before = _exported_demo()
    link = sharelink.build_url(compact(before))
    res = call(
        "workflow_update",
        {
            "source": link,
            "add_nodes": [{"kind": "technique", "key": "hash", "technique_id": "DFT-1004"}],
            "add_edges": [{"source": _last_technique(before), "target": "hash"}],
        },
    )
    _check_ae3(res, before)
    path = Path(res["file"])
    assert path.parent == dirs["out"].resolve()
    assert path.name == "Synthetic-small-workflow.json"
    assert res["link"] is not None


def test_update_link_with_filename(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    before = _exported_demo()
    res = call(
        "workflow_update",
        {"source": sharelink.build_url(compact(before)), "filename": "named"},
    )
    assert Path(res["file"]).name == "named.json"


def test_update_in_output_dir_writes_back(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    spec = {
        "title": "Back",
        "nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1001"}],
    }
    created = call("workflow_create", {"workflow": spec})
    path = created["file"]
    res = call(
        "workflow_update",
        {
            "source": path,
            "add_nodes": [{"kind": "technique", "key": "b", "technique_id": "DFT-1004"}],
            "add_edges": [{"source": created["node_ids"]["a"], "target": "b"}],
        },
    )
    assert res["ok"] is True
    assert res["file"] == path and res["replaced"] is True
    assert len(written(res)["nodes"]) == 2
    # Even after the builder replaced the file (no longer server-created), naming it wins.
    doc = written(res)
    Path(path).write_text(json.dumps(doc, indent=4), encoding="utf-8")
    assert not OutputDir(dirs["out"]).is_server_created(Path(path))
    res2 = call("workflow_update", {"source": path, "remove_ids": [res["node_ids"]["b"]]})
    assert res2["file"] == path and len(written(res2)["nodes"]) == 1
    assert [p.name for p in dirs["out"].glob("[!.]*.json")] == ["Back.json"]


def test_update_unreleased_fixture_keeps_subworkflow(
    runner: FakeRunner, dirs: dict[str, Path]
) -> None:
    before = fixture_doc("synthetic_unreleased_fields.json")
    src = dirs["downloads"] / "unreleased.json"
    shutil.copy(WORKFLOWS / "synthetic_unreleased_fields.json", src)
    res = call(
        "workflow_update",
        {
            "source": str(src),
            "add_nodes": [{"kind": "technique", "key": "k", "technique_id": "DFT-1007"}],
            "add_edges": [{"source": "n6", "target": "k"}],
        },
    )
    assert res["ok"] is True, res
    after = written(res)
    assert after["nodes"][0] == before["nodes"][0]
    assert after["nodes"][0]["data"]["subWorkflow"] == before["nodes"][0]["data"]["subWorkflow"]
    assert after["id"] == before["id"]
    assert after["evaluations"] == before["evaluations"]
    assert "unreleased_content" in codes(res["warnings"])


def test_update_remove_reports_drops(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    doc = {
        "version": 2,
        "title": "Rm",
        "nodes": [
            {"id": "a", "type": "technique", "x": 0, "y": 0, "data": {"techniqueId": "DFT-1001"}},
            {"id": "b", "type": "technique", "x": 0, "y": 200, "data": {"techniqueId": "DFT-1004"}},
            {"id": "c", "type": "technique", "x": 0, "y": 400, "data": {"techniqueId": "DFT-1005"}},
        ],
        "edges": [
            {"id": "e1", "source": "a", "target": "b"},
            {"id": "e2", "source": "b", "target": "c"},
        ],
        "selections": {"b|DFW-1005|DFM-1001": "Y", "a|DFW-1001|DFM-1001": "N"},
        "evaluations": {"b|DFW-1005": {"inherent": {}}},
        "provenance": {"b": {"source": "x"}},
    }
    src = dirs["downloads"] / "rm.json"
    src.write_text(json.dumps(doc), encoding="utf-8")
    res = call(
        "workflow_update",
        {"source": str(src), "remove_ids": ["b"], "add_edges": [{"source": "a", "target": "c"}]},
    )
    assert res["ok"] is True, res
    assert res["dropped"] == {
        "nodes": ["b"],
        "edges": ["e1", "e2"],
        "selections": ["b|DFW-1005|DFM-1001"],
        "evaluations": ["b|DFW-1005"],
        "provenance": ["b"],
    }
    after = written(res)
    assert [n["id"] for n in after["nodes"]] == ["a", "c"]
    assert after["selections"] == {"a|DFW-1001|DFM-1001": "N"}
    assert "evaluations" not in after and "provenance" not in after
    assert [(e["source"], e["target"]) for e in after["edges"]] == [("a", "c")]


def test_update_ids_never_collide(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    # A builder export with random IDs.
    shutil.copy(WORKFLOWS / "synthetic_21_nodes.json", dirs["downloads"] / "hash.json")
    before = fixture_doc("synthetic_21_nodes.json")
    first = before["nodes"][0]["id"]
    res = call(
        "workflow_update",
        {
            "source": str(dirs["downloads"] / "hash.json"),
            "add_nodes": [
                {"kind": "note", "key": "x", "text": "x"},
                {"kind": "technique", "key": "y", "technique_id": "DFT-1004"},
            ],
            "add_edges": [{"source": first, "target": "y"}],
        },
    )
    assert res["ok"] is True
    after = written(res)
    node_ids = [n["id"] for n in after["nodes"]]
    edge_ids = [e["id"] for e in after["edges"]]
    assert len(set(node_ids)) == len(node_ids) and len(set(edge_ids)) == len(edge_ids)
    assert not set(node_ids) & set(edge_ids)

    # A hand-written file already holding n3 and e1.
    src = dirs["downloads"] / "sample.json"
    shutil.copy(WORKFLOWS / "sample_workflow.json", src)
    res = call(
        "workflow_update",
        {
            "source": str(src),
            "add_nodes": [
                {"kind": "technique", "key": "p", "technique_id": "DFT-1004"},
                {"kind": "technique", "key": "q", "technique_id": "DFT-1006"},
            ],
            "add_edges": [{"source": "n2", "target": "p"}, {"source": "p", "target": "q"}],
        },
    )
    assert res["ok"] is True
    assert set(res["node_ids"].values()).isdisjoint({"n1", "n2", "n3"})
    after = written(res)
    assert len({e["id"] for e in after["edges"]}) == 3
    assert "e1" in {e["id"] for e in after["edges"]}


def test_update_set_data_and_refusals(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    src = dirs["downloads"] / "sample.json"
    shutil.copy(WORKFLOWS / "sample_workflow.json", src)
    res = call(
        "workflow_update",
        {"source": str(src), "set_data": [{"node": "n1", "fields": {"label": "Phone"}}]},
    )
    assert res["ok"] is True
    assert written(res)["nodes"][0]["data"] == {"techniqueId": "DFT-1005", "label": "Phone"}
    out_files = sorted(p.name for p in dirs["out"].glob("*.json"))

    res = call(
        "workflow_update",
        {"source": str(src), "set_data": [{"node": "n1", "fields": {"type": "note"}}]},
    )
    assert res["ok"] is False and "type" in res["error"]
    res = call(
        "workflow_update",
        {"source": str(src), "set_data": [{"node": "n3", "fields": {"question": "?"}}]},
    )
    assert res["ok"] is False and "'question'" in res["error"]
    # Deleting a required field is a validation error, and nothing is written.
    res = call(
        "workflow_update",
        {"source": str(src), "set_data": [{"node": "n1", "fields": {"techniqueId": None}}]},
    )
    assert res["ok"] is False and "missing_data" in codes(res["errors"])
    assert sorted(p.name for p in dirs["out"].glob("*.json")) == out_files


def test_update_source_errors(runner: FakeRunner, dirs: dict[str, Path], tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.json"
    outside.write_text('{"version": 2, "secret": "do-not-echo"}', encoding="utf-8")
    res = call("workflow_update", {"source": str(outside)})
    assert res["ok"] is False and "do-not-echo" not in res["error"]

    bad = dirs["downloads"] / "bad.json"
    bad.write_text("not json do-not-echo", encoding="utf-8")
    res = call("workflow_update", {"source": str(bad)})
    assert res["ok"] is False and "do-not-echo" not in res["error"]

    res = call("workflow_update", {"source": "https://workflows.hargs.co.uk/#N4Ig"})
    assert res["ok"] is False and "older-format" in res["error"]

    v1 = dirs["downloads"] / "v1.json"
    shutil.copy(WORKFLOWS / "legacy_v1.json", v1)
    res = call("workflow_update", {"source": str(v1)})
    assert res["ok"] is False and "version 1" in res["error"]


# ---- validate ----


def test_validate_file_and_inline(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    src = dirs["downloads"] / "sample.json"
    shutil.copy(WORKFLOWS / "sample_workflow.json", src)
    res = call("workflow_validate", {"file_path": str(src)})
    assert res["ok"] is True and res["errors"] == []
    assert res["counts"] == {"nodes": 3, "edges": 1}
    assert res["knowledge_base_revision"] == "unknown"

    doc = fixture_doc("sample_workflow.json")
    doc["edges"].append({"id": "e9", "source": "n1", "target": "gone"})
    res = call("workflow_validate", {"workflow": doc})
    assert res["ok"] is False
    assert "dangling_edge" in codes(res["errors"])


def test_validate_needs_exactly_one_source(runner: FakeRunner) -> None:
    assert call("workflow_validate", {})["ok"] is False
    res = call("workflow_validate", {"file_path": "x.json", "workflow": {"version": 2}})
    assert "exactly one" in res["error"]


# ---- compile ----


def _create(spec: dict[str, Any]) -> dict[str, Any]:
    res = call("workflow_create", {"workflow": spec})
    assert res["ok"] is True, res
    return res


def _weakness(step: dict[str, Any], wid: str) -> dict[str, Any]:
    return next(w for w in step["weaknesses"] if w["id"] == wid)


def _mitigation(weakness: dict[str, Any], mid: str) -> dict[str, Any]:
    return next(m for m in weakness["mitigations"] if m["id"] == mid)


def test_compile_method_implemented_mitigation_in_place(runner: FakeRunner) -> None:
    # DFT-1002 is a sub-technique of DFT-1001 and implements DFM-1002.
    res = _create({"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1002"}]})
    out = written(res)
    assert out["nodes"][0]["data"] == {"techniqueId": "DFT-1001", "methods": ["DFT-1002"]}
    comp = call("workflow_compile", {"file_path": res["file"], "detail": True})
    assert comp["ok"] is True
    step = comp["steps"][0]
    m = _mitigation(_weakness(step, "DFW-1002"), "DFM-1002")
    assert m["in_place"] is True and m["source"] == "canvas"
    assert _weakness(step, "DFW-1001")["astm"] == ["INCOMP"]


def test_compile_two_techniques_in_flow_order_and_n_overrides(runner: FakeRunner) -> None:
    spec = {
        "nodes": [
            {"kind": "technique", "key": "hash", "technique_id": "DFT-1004"},
            {"kind": "technique", "key": "acq", "technique_id": "DFT-1001"},
        ],
        "edges": [{"source": "acq", "target": "hash"}],
        "selections": [
            {"step": "hash", "weakness_id": "DFW-1005", "mitigation_id": "DFM-1001", "value": "N"}
        ],
    }
    res = _create(spec)
    comp = call("workflow_compile", {"file_path": res["file"], "detail": True})
    ids = res["node_ids"]
    assert [s["node_id"] for s in comp["steps"]] == [ids["acq"], ids["hash"]]
    # DFM-1001 is implemented by DFT-1004 on the canvas: in place on acq, overridden by N on hash.
    acq, hsh = comp["steps"]
    assert _mitigation(_weakness(acq, "DFW-1001"), "DFM-1001")["in_place"] is True
    assert _mitigation(_weakness(hsh, "DFW-1005"), "DFM-1001")["in_place"] is False
    summary = call("workflow_compile", {"file_path": res["file"]})
    assert "weaknesses" not in summary["steps"][0]
    assert summary["steps"][0]["weakness_count"] == 2


def test_compile_assessment_mode_only_y_counts(runner: FakeRunner) -> None:
    doc = {
        "version": 2,
        "nodes": [
            {"id": "a", "type": "technique", "x": 0, "y": 0, "data": {"techniqueId": "DFT-1001"}},
            {"id": "b", "type": "technique", "x": 0, "y": 200, "data": {"techniqueId": "DFT-1004"}},
        ],
        "edges": [{"id": "e1", "source": "a", "target": "b"}],
        "selections": {},
        "mode": "assessment",
    }
    comp = call("workflow_compile", {"workflow": doc, "detail": True})
    assert comp["mode"] == "assessment"
    m = _mitigation(_weakness(comp["steps"][0], "DFW-1001"), "DFM-1001")
    assert m["in_place"] is False


def test_compile_na_via_update(runner: FakeRunner) -> None:
    res = _create({"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1002"}]})
    nid = res["node_ids"]["a"]
    upd = call(
        "workflow_update",
        {
            "source": res["file"],
            "selections": [
                {"step": nid, "weakness_id": "DFW-1002", "mitigation_id": "DFM-1002", "value": "NA"}
            ],
        },
    )
    assert upd["ok"] is True and upd["file"] == res["file"]
    comp = call("workflow_compile", {"file_path": upd["file"], "detail": True})
    w = _weakness(comp["steps"][0], "DFW-1002")
    assert w["status"] == "na"
    assert _mitigation(w, "DFM-1002")["applicable"] is False


def test_compile_placeholder_has_no_rows_and_cycle_is_deterministic(runner: FakeRunner) -> None:
    spec = {
        "nodes": [
            {"kind": "technique", "key": "a", "technique_id": "DFT-1001"},
            {"kind": "decision", "key": "d", "question": "Again?"},
            {"kind": "technique", "key": "b", "technique_id": "DFT-1004"},
            {"kind": "placeholder", "key": "p", "name": "New thing"},
        ],
        "edges": [
            {"source": "a", "target": "d"},
            {"source": "d", "target": "b"},
            {"source": "b", "target": "a"},
            {"source": "b", "target": "p"},
        ],
    }
    res = _create(spec)
    first = call("workflow_compile", {"file_path": res["file"]})
    again = call("workflow_compile", {"file_path": res["file"]})
    assert first["steps"] == again["steps"]
    assert [s["node_id"] for s in first["steps"]] == [res["node_ids"]["a"], res["node_ids"]["b"]]


def test_compile_with_dangling_edge_returns_errors_only(runner: FakeRunner) -> None:
    doc = fixture_doc("sample_workflow.json")
    doc["edges"].append({"id": "e9", "source": "n1", "target": "gone"})
    comp = call("workflow_compile", {"workflow": doc})
    assert comp["ok"] is False
    assert "dangling_edge" in codes(comp["errors"])
    assert "steps" not in comp and "totals" not in comp


# ---- review fixes ----


def test_tilde_path_source_is_a_file_not_a_link(
    runner: FakeRunner, dirs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    shutil.copy(WORKFLOWS / "sample_workflow.json", dirs["downloads"] / "home.json")
    res = call("workflow_update", {"source": "~/downloads/home.json"})
    assert res["ok"] is True, res
    assert Path(res["file"]).name == "home.json"
    assert written(res)["title"] == fixture_doc("sample_workflow.json")["title"]


def test_bare_hash_and_tilde_hash_are_links(runner: FakeRunner) -> None:
    doc = fixture_doc("sample_workflow.json")
    hash_ = sharelink.encode_hash(compact(doc))
    for source in (hash_, "#" + hash_):
        res = call("workflow_inspect", {"source": source})
        assert res["ok"] is True, (source, res)


def test_bom_prefixed_source_reads(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    src = dirs["downloads"] / "bom.json"
    src.write_bytes(b"\xef\xbb\xbf" + (WORKFLOWS / "sample_workflow.json").read_bytes())
    res = call("workflow_validate", {"file_path": str(src)})
    assert res["ok"] is True, res


def test_unwritable_output_dir_reports_error(dirs: dict[str, Path], tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    init_fixture_state(locked / "out", **{READ_DIRS_VAR: str(dirs["downloads"])})
    locked.chmod(0o500)
    try:
        spec = {"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
        res = call("workflow_create", {"workflow": spec})
    finally:
        locked.chmod(0o700)
    assert res["ok"] is False
    assert str((locked / "out").resolve()) in res["error"]
    assert ".part" not in res["error"]


def test_manifest_failure_still_writes_with_warning(
    runner: FakeRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mcp_solveit_workflows.files as files_mod

    real = files_mod._atomic_write_bytes

    def fail_manifest(target: Path, data: bytes) -> None:
        if target.name == files_mod.MANIFEST_NAME:
            raise OSError(28, "No space left on device")
        real(target, data)

    monkeypatch.setattr(files_mod, "_atomic_write_bytes", fail_manifest)
    spec = {"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
    res = call("workflow_create", {"workflow": spec})
    assert res["ok"] is True, res
    assert Path(res["file"]).is_file()
    assert any(w["code"] == "manifest" for w in res["warnings"])


def test_edge_to_note_or_zone_refused(runner: FakeRunner) -> None:
    for kind in (
        {"kind": "note", "key": "x", "text": "hi"},
        {"kind": "zone", "key": "x", "label": "Z", "members": ["a"]},
    ):
        spec = {
            "nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}, kind],
            "edges": [{"source": "a", "target": "x"}],
        }
        res = call("workflow_create", {"workflow": spec})
        assert res["ok"] is False
        assert "'a'" in res["error"] and "'x'" in res["error"]


def test_data_source_edges_allowed(runner: FakeRunner) -> None:
    spec = {
        "nodes": [
            {"kind": "data_source", "key": "d", "label": "Disk"},
            {"kind": "technique", "key": "a", "technique_id": "DFT-1004"},
        ],
        "edges": [{"source": "d", "target": "a"}],
    }
    assert call("workflow_create", {"workflow": spec})["ok"] is True


def test_lone_surrogate_reported_not_raised(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    src = dirs["downloads"] / "surrogate.json"
    text = (WORKFLOWS / "sample_workflow.json").read_text(encoding="utf-8")
    src.write_text(text.replace("Loaded from a URL", "\\ud800 Loaded"), encoding="utf-8")
    res = call("workflow_update", {"source": str(src)})
    assert res["ok"] is False
    assert "UTF-8" in res["error"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["nodes"][0].__setitem__("id", ["n1"]),
        lambda d: d["nodes"][0].__setitem__("id", {"a": 1}),
        lambda d: d["edges"][0].__setitem__("id", ["e1"]),
        lambda d: d["edges"][0].__setitem__("source", {"a": 1}),
    ],
)
def test_malformed_ids_are_reported_not_raised(
    runner: FakeRunner, dirs: dict[str, Path], mutate: Any
) -> None:
    doc = fixture_doc("sample_workflow.json")
    mutate(doc)
    src = dirs["downloads"] / "ids.json"
    src.write_text(json.dumps(doc), encoding="utf-8")
    for name, args in (
        ("workflow_update", {"source": str(src)}),
        ("workflow_update", {"source": str(src), "remove_ids": ["n2"]}),
        ("workflow_validate", {"file_path": str(src)}),
        ("workflow_validate", {"workflow": doc}),
        ("workflow_compile", {"file_path": str(src)}),
        ("workflow_inspect", {"source": str(src)}),
    ):
        res = call(name, args)
        assert res["ok"] is False, (name, res)


@pytest.mark.parametrize(
    ("node", "fields"),
    [
        ("t", {"label": 5}),
        ("t", {"methods": [1]}),
        ("t", {"methods": ["x"]}),
        ("t", {"target": {"a": 1}}),
        ("z", {"w": 0}),
        ("z", {"labelCorner": "nope"}),
    ],
)
def test_set_data_invalid_values(
    runner: FakeRunner, dirs: dict[str, Path], node: str, fields: dict[str, Any]
) -> None:
    doc = {
        "version": 2,
        "nodes": [
            {"id": "t", "type": "technique", "x": 0, "y": 0, "data": {"techniqueId": "DFT-1004"}},
            {
                "id": "z",
                "type": "region",
                "x": -40,
                "y": -40,
                "data": {"label": "Z", "w": 400, "h": 300},
            },
        ],
        "edges": [],
        "selections": {},
    }
    src = dirs["downloads"] / "setdata.json"
    src.write_text(json.dumps(doc), encoding="utf-8")
    res = call(
        "workflow_update", {"source": str(src), "set_data": [{"node": node, "fields": fields}]}
    )
    assert res["ok"] is False
    assert "must be" in res["error"]


# ---- inspect ----


def test_inspect_fixture_file(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    src = dirs["downloads"] / "demo.json"
    shutil.copy(WORKFLOWS / "synthetic_small.json", src)
    doc = fixture_doc("synthetic_small.json")
    res = call("workflow_inspect", {"source": str(src)})
    assert res["ok"] is True, res
    assert res["title"] == doc["title"]
    assert [n["id"] for n in res["nodes"]] == [n["id"] for n in doc["nodes"]]
    assert [n["type"] for n in res["nodes"]] == [n["type"] for n in doc["nodes"]]
    assert [e["id"] for e in res["edges"]] == [e["id"] for e in doc["edges"]]
    assert [(e["source"], e["target"]) for e in res["edges"]] == [
        (e["source"], e["target"]) for e in doc["edges"]
    ]
    assert res["counts"] == {"nodes": len(doc["nodes"]), "edges": len(doc["edges"])}
    assert res["direction"] in ("down", "right")
    assert res["selections"] == len(doc.get("selections", {}))
    assert res["knowledge_base_revision"] == "unknown"
    for n, raw in zip(res["nodes"], doc["nodes"], strict=True):
        if raw["type"] == "technique":
            assert n["label"].startswith(raw["data"]["techniqueId"])
            # Only DFT-1002 of this fixture's techniques is in the fixture knowledge base.
            if raw["data"]["techniqueId"] == "DFT-1002":
                assert n["name"] == "Physical imaging"
            else:
                assert "name" not in n
        assert len(n.get("label", "")) <= 120


def test_inspect_share_link(runner: FakeRunner) -> None:
    doc = fixture_doc("sample_workflow.json")
    res = call("workflow_inspect", {"source": sharelink.build_url(compact(doc))})
    assert res["ok"] is True, res
    assert [n["id"] for n in res["nodes"]] == ["n1", "n2", "n3"]
    assert res["edges"] == [{"id": "e1", "source": "n1", "target": "n2"}]
    by_id = {n["id"]: n for n in res["nodes"]}
    assert by_id["n1"]["label"] == "DFT-1005"
    assert by_id["n1"]["name"] == "Verify image integrity"
    note = doc["nodes"][2]["data"]["text"]
    assert len(note) > 80 and by_id["n3"]["label"] == note[:80] + "…"
    assert "description" in res and "mode" not in res


def test_inspect_refuses_outside_file(runner: FakeRunner, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.json"
    outside.write_text('{"version": 2}', encoding="utf-8")
    assert call("workflow_inspect", {"source": str(outside)})["ok"] is False


# ---- second review round ----


def test_lone_surrogate_in_filename_is_a_tool_error(runner: FakeRunner) -> None:
    spec = {"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
    res = call("workflow_create", {"workflow": spec, "filename": "ab\ud800.json"})
    assert res["ok"] is False
    assert "\ud800" not in res["error"]


def test_lone_surrogate_in_inspect_source_is_a_tool_error(
    runner: FakeRunner, dirs: dict[str, Path]
) -> None:
    res = call("workflow_inspect", {"source": str(dirs["downloads"] / "x\ud800.json")})
    assert res["ok"] is False


def test_output_dir_without_search_permission_is_a_tool_error(
    runner: FakeRunner, dirs: dict[str, Path]
) -> None:
    dirs["out"].mkdir()
    dirs["out"].chmod(0)
    try:
        spec = {"nodes": [{"kind": "technique", "key": "a", "technique_id": "DFT-1004"}]}
        res = call("workflow_create", {"workflow": spec, "filename": "t.json"})
    finally:
        dirs["out"].chmod(0o700)
    assert res["ok"] is False
    assert str(dirs["out"]) in res["error"]


def test_inspect_caps_technique_labels_and_title(runner: FakeRunner, dirs: dict[str, Path]) -> None:
    doc = {
        "version": 2,
        "title": "T" * 5000,
        "description": "D" * 5000,
        "nodes": [
            {
                "id": "n1",
                "type": "technique",
                "x": 0,
                "y": 0,
                "data": {"techniqueId": "DFT-1004", "methods": ["M" * 5000, {"x": 1}]},
            }
        ],
        "edges": [],
        "selections": {},
    }
    path = dirs["downloads"] / "long.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    res = call("workflow_inspect", {"source": str(path)})
    assert res["ok"] is True
    assert len(res["title"]) <= 81
    assert len(res["description"]) <= 81
    label = res["nodes"][0]["label"]
    assert len(label) <= 81
    assert "{" not in label
