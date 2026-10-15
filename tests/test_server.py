import asyncio
from pathlib import Path

from conftest import init_fixture_state

from mcp_solveit_workflows import server
from mcp_solveit_workflows.config import KB_PATH_VAR, OUTPUT_DIR_VAR, Config

EXPECTED_TOOLS = {
    "workflow_status",
    "workflow_find_techniques",
    "workflow_create",
    "workflow_update",
    "workflow_inspect",
    "workflow_validate",
    "workflow_compile",
}


def test_status_degraded_without_kb(tmp_path: Path) -> None:
    server.init_state(Config.from_env({OUTPUT_DIR_VAR: str(tmp_path)}))
    status = server.workflow_status()
    assert status["ready"] is False
    assert KB_PATH_VAR in status["error"]
    assert status["output_dir"] == str(tmp_path)
    assert status["counts"] is None


def test_registered_tools_are_exactly_the_seven() -> None:
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_status_ready_with_fixture_kb(tmp_path: Path) -> None:
    init_fixture_state(tmp_path)
    status = server.workflow_status()
    assert status["ready"] is True
    assert status["error"] is None
    assert status["counts"] == {"techniques": 8, "weaknesses": 6, "mitigations": 4}
    assert status["knowledge_base_revision"] == "unknown"
    assert status["builder_format"] == "0.13.0"
    assert status["limits"]["max_nodes"] == 2000


def test_status_degraded_when_kb_path_has_no_data(tmp_path: Path) -> None:
    server.init_state(Config.from_env({KB_PATH_VAR: str(tmp_path), OUTPUT_DIR_VAR: str(tmp_path)}))
    status = server.workflow_status()
    assert status["ready"] is False
    assert "data/techniques" in status["error"] or "techniques" in status["error"]
    assert KB_PATH_VAR in status["error"]


def test_status_caps_load_warnings(tmp_path: Path) -> None:
    techniques = tmp_path / "data" / "techniques"
    techniques.mkdir(parents=True)
    for i in range(25):
        (techniques / f"bad{i}.json").write_text("not json", encoding="utf-8")
    server.init_state(Config.from_env({KB_PATH_VAR: str(tmp_path), OUTPUT_DIR_VAR: str(tmp_path)}))
    status = server.workflow_status()
    assert status["ready"] is True
    assert len(status["load_warnings"]) == 20
    # 25 bad files plus the missing weaknesses and mitigations directories.
    assert status["load_warning_count"] == 27
