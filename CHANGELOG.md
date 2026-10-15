# Changelog

## [Unreleased]

First release.

### Added
- `workflow_create`, `workflow_update`, `workflow_inspect`, `workflow_validate`, `workflow_compile`, `workflow_find_techniques` and `workflow_status` tools.
- Workflow model for the deployed builder's v2 format (0.13.0), covering every core element, with pass-through of unknown and unreleased fields.
- Knowledge base index read from the SOLVE-IT JSON files, with a technique search modelled on the SOLVE-IT library's: word by word with simple stemming and quoted phrases, across names, synonyms, descriptions, details and example tools, returning each match's description and input and output classes.
- Layered layout, top to bottom or left to right, with crossing reduction, zone members kept together, room reserved for connections that skip layers, and connection points chosen so each drawn connection and its label stay clear of other cards. Zones are sized around their members, widened to fit their labels, and labelled in a corner no connection crosses.
- Validation, compile (weaknesses, ASTM categories and mitigations per step), and update operations that keep builder edits. Connections to or from notes and zones are refused, because the builder draws no connection points on them.
- Confined file reading and writing, share links with the builder's encoding, and an optional browser opener that starts `open` or `xdg-open` detached, without a shell.
- Drift check against the deployed builder.
- Ontology class names given as short names (Computer), paths (observable/Computer) or prefixed names (uco-observable:Computer) are written as full IRIs when the knowledge base declares exactly one match; validation warns about any class that is not a full IRI.
- `relayout` option on `workflow_update`, which lays the whole workflow out again, keeping existing zones' contents, so that a zone added round scattered steps holds only its members.
