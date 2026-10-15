#!/bin/bash
# Sets up the SOLVE-IT workflow MCP server for Claude Desktop (macOS and Linux).
#
#   ./setup_claude_desktop.sh /path/to/solve-it [output-dir]
#
# /path/to/solve-it is a local clone of https://github.com/SOLVE-IT-DF/solve-it
# (the folder containing data/techniques). The output directory defaults to
# ~/Documents/SOLVE-IT Workflows. On Windows, add the config entry by hand (see README).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/solve-it [output-dir]" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
KB="$(cd "$1" && pwd)"
OUT="${2:-$HOME/Documents/SOLVE-IT Workflows}"

if [ ! -d "$KB/data/techniques" ]; then
  echo "Not a SOLVE-IT checkout: $KB/data/techniques does not exist." >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin) CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
  Linux) CFG="${XDG_CONFIG_HOME:-$HOME/.config}/Claude/claude_desktop_config.json" ;;
  *) echo "Unsupported platform; add the config entry by hand (see README)." >&2; exit 1 ;;
esac

# 1. uv manages its own Python, so the system Python version doesn't matter.
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
fi

# 2. Environment with locked dependencies.
cd "$REPO"
uv sync --frozen --no-dev --inexact
PY="$REPO/.venv/bin/python"

# 3. Smoke test before touching the Desktop config: the knowledge base must load.
MCP_APP_SOLVEIT_DATA_PATH="$KB" SOLVEIT_WORKFLOWS_OUTPUT_DIR="$OUT" "$PY" - <<'PY'
from mcp_solveit_workflows import server
from mcp_solveit_workflows.config import Config

server.init_state(Config.from_env())
status = server.workflow_status()
if not status["ready"]:
    raise SystemExit(f"Server not ready: {status['error']}")
print(f"Knowledge base loaded: {status['counts']} at revision {status['knowledge_base_revision'][:12]}")
PY

# 4. Merge the entry into the Claude Desktop config, keeping a backup.
mkdir -p "$(dirname "$CFG")"
if [ -f "$CFG" ]; then
  cp "$CFG" "$CFG.bak-$(date +%Y%m%d%H%M%S)"
fi
"$PY" - "$CFG" "$PY" "$KB" "$OUT" <<'PY'
import json
import os
import sys

cfg, py, kb, out = sys.argv[1:5]
data = {}
if os.path.exists(cfg) and os.path.getsize(cfg):
    with open(cfg, encoding="utf-8") as fh:
        data = json.load(fh)
data.setdefault("mcpServers", {})["solveit-workflows"] = {
    "command": py,
    "args": ["-m", "mcp_solveit_workflows.server"],
    "env": {
        "MCP_APP_SOLVEIT_DATA_PATH": kb,
        "SOLVEIT_WORKFLOWS_OUTPUT_DIR": out,
    },
}
tmp = cfg + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
os.replace(tmp, cfg)
print("Updated", cfg)
PY

echo "Done. Quit Claude Desktop fully and reopen it; the tools appear under 'solveit-workflows'."
