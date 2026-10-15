# Changelog

## [Unreleased]

First release.

### Added
- `workflow_create`, `workflow_update`, `workflow_inspect`, `workflow_validate`, `workflow_compile`, `workflow_find_techniques` and `workflow_status` tools.
- Workflow model for the deployed builder's v2 format (0.13.0), covering every core element, with pass-through of unknown and unreleased fields.
- Knowledge base index read from the SOLVE-IT JSON files, with a technique finder.
- Layered layout, top to bottom or left to right, with crossing reduction, zone members kept together, room reserved for connections that skip layers, and connection points chosen so each drawn connection and its label stay clear of other cards. Zones are sized around their members.
- Validation, compile (weaknesses, ASTM categories and mitigations per step), and update operations that keep builder edits. Connections to or from notes and zones are refused, because the builder draws no connection points on them.
- Confined file reading and writing, share links with the builder's encoding, and an optional browser opener that starts `open` or `xdg-open` detached, without a shell.
- Drift check against the deployed builder, and a Claude Desktop setup script.
