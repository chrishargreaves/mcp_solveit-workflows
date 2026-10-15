"""Drive the real server over stdio with the MCP SDK's client: create, then compile."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
from conftest import FIXTURE_KB
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = Path(__file__).parent
SRC = HERE.parent / "src"


def _structured(result: Any) -> dict[str, Any]:
    assert not result.isError, result
    if result.structuredContent is not None:
        data: dict[str, Any] = result.structuredContent
        return data
    parsed: dict[str, Any] = json.loads(result.content[0].text)
    return parsed


async def _session(out_dir: Path) -> dict[str, Any]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "MCP_APP_SOLVEIT_DATA_PATH": str(FIXTURE_KB),
        "SOLVEIT_WORKFLOWS_OUTPUT_DIR": str(out_dir),
        "SOLVEIT_WORKFLOWS_READ_DIRS": "",
        "PYTHONPATH": str(SRC),
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_solveit_workflows.server"], env=env
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        created = _structured(
            await session.call_tool(
                "workflow_create",
                {
                    "workflow": {
                        "title": "E2E",
                        "nodes": [
                            {"kind": "technique", "key": "acq", "technique_id": "DFT-1002"},
                            {"kind": "technique", "key": "hash", "technique_id": "DFT-1004"},
                        ],
                        "edges": [{"source": "acq", "target": "hash"}],
                    }
                },
            )
        )
        compiled = _structured(
            await session.call_tool("workflow_compile", {"file_path": created["file"]})
        )
        return {"tools": [t.name for t in tools.tools], "created": created, "compiled": compiled}


def test_stdio_create_then_compile(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = anyio.run(_session, out)
    assert len(result["tools"]) == 7
    created = result["created"]
    assert created["ok"] is True
    path = Path(created["file"])
    assert path.parent == out.resolve() and path.is_file()
    assert created["link"].startswith("https://workflows.hargs.co.uk/#~")
    compiled = result["compiled"]
    assert compiled["ok"] is True
    assert [s["node_id"] for s in compiled["steps"]] == [
        created["node_ids"]["acq"],
        created["node_ids"]["hash"],
    ]
    assert compiled["steps"][0]["technique_id"] == "DFT-1001"


def test_readme_example_is_valid_and_its_layout_is_stable() -> None:
    """examples/laptop-to-e01.json is shown in the README; keep it valid and keep the
    picture true to what the server produces."""
    from mcp_solveit_workflows.layout import layout_workflow
    from mcp_solveit_workflows.model import read_workflow
    from mcp_solveit_workflows.validate import has_errors, validate_workflow

    path = Path(__file__).parent.parent / "examples" / "laptop-to-e01.json"
    text = path.read_text(encoding="utf-8")
    wf = read_workflow(text)
    assert not has_errors(validate_workflow(wf, None))
    from mcp_solveit_workflows.layout import zone_members_by_box

    members = zone_members_by_box(wf)
    assert members == {"n17": ["n3", "n4", "n5", "n6", "n7"]}
    assert layout_workflow(wf, direction="down", zone_members=members) == []
    assert wf.to_json_pair()[0] == text
