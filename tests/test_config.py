from pathlib import Path

from mcp_solveit_workflows.config import (
    KB_PATH_VAR,
    OUTPUT_DIR_VAR,
    READ_DIRS_VAR,
    Config,
)


def test_defaults_expand_home_and_are_absolute() -> None:
    cfg = Config.from_env({})
    assert cfg.kb_path is None
    assert cfg.output_dir.is_absolute()
    assert "~" not in str(cfg.output_dir)
    assert cfg.output_dir == Path.home() / "Documents" / "SOLVE-IT Workflows"
    assert cfg.read_dirs == (cfg.output_dir, Path.home() / "Downloads")


def test_placeholder_kb_path_counts_as_unset() -> None:
    cfg = Config.from_env({KB_PATH_VAR: "<full path to your>/solve-it-main"})
    assert cfg.kb_path is None


def test_explicit_settings(tmp_path: Path) -> None:
    out = tmp_path / "out"
    extra = tmp_path / "extra"
    cfg = Config.from_env(
        {
            KB_PATH_VAR: str(tmp_path / "kb"),
            OUTPUT_DIR_VAR: str(out),
            READ_DIRS_VAR: f"{extra}:{out}",
        }
    )
    assert cfg.kb_path == tmp_path / "kb"
    assert cfg.output_dir == out
    # Output dir comes first and is not duplicated.
    assert cfg.read_dirs == (out, extra)


def test_empty_read_dirs_leaves_only_output_dir(tmp_path: Path) -> None:
    cfg = Config.from_env({OUTPUT_DIR_VAR: str(tmp_path), READ_DIRS_VAR: ""})
    assert cfg.read_dirs == (tmp_path,)
