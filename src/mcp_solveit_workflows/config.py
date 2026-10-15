"""Configuration from environment variables.

Every setting has a documented default so the server runs without a config
file. The knowledge base path has no usable default: without it the server
starts in degraded mode and only the status tool works.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

KB_PATH_VAR = "MCP_APP_SOLVEIT_DATA_PATH"
OUTPUT_DIR_VAR = "SOLVEIT_WORKFLOWS_OUTPUT_DIR"
READ_DIRS_VAR = "SOLVEIT_WORKFLOWS_READ_DIRS"

DEFAULT_OUTPUT_DIR = "~/Documents/SOLVE-IT Workflows"
DEFAULT_EXTRA_READ_DIRS = ("~/Downloads",)


def _expand(value: str) -> Path:
    return Path(value).expanduser().absolute()


def _is_placeholder(value: str) -> bool:
    # mcp_solve-it ships "<full path to your>/solve-it-main" as its toml default.
    return not value.strip() or "<" in value or ">" in value


@dataclass(frozen=True)
class Config:
    kb_path: Path | None
    output_dir: Path
    read_dirs: tuple[Path, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env

        raw_kb = env.get(KB_PATH_VAR, "")
        kb_path = None if _is_placeholder(raw_kb) else _expand(raw_kb)

        output_dir = _expand(env.get(OUTPUT_DIR_VAR) or DEFAULT_OUTPUT_DIR)

        raw_read = env.get(READ_DIRS_VAR)
        if raw_read is None:
            extra = [_expand(p) for p in DEFAULT_EXTRA_READ_DIRS]
        else:
            extra = [_expand(p) for p in raw_read.split(os.pathsep) if p.strip()]
        # The output directory is always readable, so a file the server wrote can be updated.
        read_dirs = tuple(dict.fromkeys([output_dir, *extra]))

        return cls(kb_path=kb_path, output_dir=output_dir, read_dirs=read_dirs)
