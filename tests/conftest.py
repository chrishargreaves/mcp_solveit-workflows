"""Shared test fixtures: the fixture knowledge base and workflow files."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_solveit_workflows import server
from mcp_solveit_workflows.config import KB_PATH_VAR, OUTPUT_DIR_VAR, Config
from mcp_solveit_workflows.kb import KnowledgeBase

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_KB = FIXTURES / "kb"
WORKFLOWS = FIXTURES / "workflows"

# The builder v2 workflow fixtures (legacy_v1.json is the v1 one).
V2_FIXTURES = [
    "sample_workflow.json",
    "synthetic_small.json",
    "synthetic_21_nodes.json",
    "synthetic_63_nodes.json",
    "synthetic_unreleased_fields.json",
]


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.load(FIXTURE_KB)


def init_fixture_state(output_dir: Path, **env: str) -> server.ServerState:
    """Server state with the fixture knowledge base, writing to `output_dir`."""
    return server.init_state(
        Config.from_env({KB_PATH_VAR: str(FIXTURE_KB), OUTPUT_DIR_VAR: str(output_dir), **env})
    )
