"""Compile a workflow against the knowledge base, as the builder's Compile step does.

Mirrors the deployed builder (solveit-workflow-builder, origin/main):

- `src/lib/flowOrder.ts` `computeFlowOrder`: Kahn's algorithm over flow nodes,
  leftovers (cycles, or nodes stuck behind one) appended in file order.
- `src/components/CompilePanel/useCompile.ts`: technique instances sorted by
  flow order; "on the canvas" is every technique node's `techniqueId` plus its
  `methods`; nothing is auto-satisfied in an assessment (`mode: 'assessment'`).
- `src/lib/methods.ts` `validMethods` / `effectiveWeaknessIds`.
- `src/lib/coverage.ts` `weaknessStatus`: mitigated / partial / unmitigated / na.

Selection keys are `<nodeId>|<weaknessId>|<mitigationId>`. The layered build's
external tool-evidence matches are not mirrored.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import Any

from .kb import ASTM_LABELS, KnowledgeBase
from .model import FLOW_TYPES, Workflow

# Worst first.
_STATUSES = ("unmitigated", "partial", "mitigated", "na")


def _dicts(items: list[Any]) -> list[dict[str, Any]]:
    return [x for x in items if isinstance(x, dict)]


def _order_map(wf: Workflow) -> dict[Any, int]:
    """Port of computeFlowOrder: 1-based position of each flow node ID."""
    flow_nodes = [
        n
        for n in _dicts(wf.nodes)
        if n.get("type") in FLOW_TYPES and isinstance(n.get("id"), Hashable)
    ]
    flow_ids = {n["id"] for n in flow_nodes}

    indeg: dict[Any, int] = {}
    outgoing: dict[Any, list[Any]] = {}
    for n in flow_nodes:
        indeg[n["id"]] = 0
        outgoing[n["id"]] = []
    for e in _dicts(wf.edges):
        s, t = e.get("source"), e.get("target")
        if isinstance(s, Hashable) and isinstance(t, Hashable) and s in flow_ids and t in flow_ids:
            outgoing[s].append(t)
            indeg[t] = indeg.get(t, 0) + 1

    queue = deque(n["id"] for n in flow_nodes if indeg.get(n["id"], 0) == 0)
    seen = set(queue)
    order: dict[Any, int] = {}
    i = 1
    while queue:
        node_id = queue.popleft()
        order[node_id] = i
        i += 1
        for m in outgoing.get(node_id, []):
            indeg[m] = indeg.get(m, 0) - 1
            if indeg[m] == 0 and m not in seen:
                seen.add(m)
                queue.append(m)
    for n in flow_nodes:
        if n["id"] not in order:
            order[n["id"]] = i
            i += 1
    return order


def flow_order(wf: Workflow) -> list[str]:
    """IDs of the flow nodes in the builder's flow order."""
    order = _order_map(wf)
    return [node_id for node_id, _ in sorted(order.items(), key=lambda kv: kv[1])]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


def _valid_methods(kb: KnowledgeBase, tid: str, methods: list[str]) -> list[str]:
    item = kb.technique(tid)
    subs = set(_str_list(item.get("subtechniques"))) if item else set()
    return [m for m in methods if m in subs]


def _effective_weakness_ids(kb: KnowledgeBase, tid: str, valid: list[str]) -> list[str]:
    items = (kb.technique(source) for source in (tid, *valid))
    return list(
        dict.fromkeys(wid for item in items if item for wid in _str_list(item.get("weaknesses")))
    )


def compile_workflow(wf: Workflow, kb: KnowledgeBase, *, detail: bool = False) -> dict[str, Any]:
    """Per technique step, in flow order: weaknesses, mitigations and coverage."""
    assessment = wf.doc.get("mode") == "assessment"
    selections: dict[str, Any] = wf.doc.get("selections") or {}
    if not isinstance(selections, dict):
        selections = {}
    order = _order_map(wf)
    # IDs not in the knowledge base, in first-seen order (an ordered set).
    skipped: dict[str, None] = {}

    instances = [
        n for n in _dicts(wf.nodes) if n.get("type") == "technique" and isinstance(n.get("id"), str)
    ]
    # Stable, as Array.prototype.sort is; nodes missing from the order sort last.
    instances.sort(key=lambda n: order.get(n["id"], float("inf")))

    def data_of(n: dict[str, Any]) -> dict[str, Any]:
        d = n.get("data")
        return d if isinstance(d, dict) else {}

    canvas: set[str] = set()
    for n in instances:
        d = data_of(n)
        if isinstance(d.get("techniqueId"), str):
            canvas.add(d["techniqueId"])
        canvas.update(_str_list(d.get("methods")))

    def in_place_source(key: str, m: dict[str, Any] | None) -> str | None:
        status = selections.get(key)
        if status == "Y":
            return "selection"
        if status:  # N or NA (or anything else set): a manual decision wins.
            return None
        if assessment:
            return None
        technique = m.get("technique") if m else None
        return "canvas" if isinstance(technique, str) and technique in canvas else None

    steps: list[dict[str, Any]] = []
    totals = {"steps": 0, "weaknesses": 0, **{s: 0 for s in _STATUSES}}

    for n in instances:
        d = data_of(n)
        node_id: str = n["id"]
        raw_tid = d.get("techniqueId")
        tid: str | None = raw_tid if isinstance(raw_tid, str) else None
        technique = kb.technique(tid) if tid else None
        if tid and technique is None:
            skipped[tid] = None
        methods = _str_list(d.get("methods"))
        valid = _valid_methods(kb, tid, methods) if technique and tid else []
        for m in valid:
            if kb.technique(m) is None:
                skipped[m] = None

        step: dict[str, Any] = {
            "node_id": node_id,
            "technique_id": tid,
            "name": technique.get("name") if technique else None,
        }
        label = d.get("label")
        if isinstance(label, str) and label:
            step["label"] = label
        if valid:
            step["methods"] = valid

        counts = {s: 0 for s in _STATUSES}
        rows: list[dict[str, Any]] = []
        for wid in _effective_weakness_ids(kb, tid, valid) if technique and tid else []:
            weakness = kb.weakness(wid)
            if weakness is None:
                skipped[wid] = None
                continue
            mids = _str_list(weakness.get("mitigations"))
            mit_rows: list[dict[str, Any]] = []
            in_place = na = 0
            for mid in mids:
                key = f"{node_id}|{wid}|{mid}"
                mitigation = kb.mitigation(mid)
                applicable = selections.get(key) != "NA"
                source = in_place_source(key, mitigation)
                if not applicable:
                    na += 1
                elif source:
                    in_place += 1
                if mitigation is None:
                    # Still counts towards the status, as in coverage.ts; no row.
                    skipped[mid] = None
                elif detail:
                    mit_rows.append(
                        {
                            "id": mid,
                            "name": mitigation.get("name"),
                            "in_place": source is not None,
                            "applicable": applicable,
                            "source": source,
                        }
                    )
            status = _status(len(mids), in_place, na)
            counts[status] += 1
            if detail:
                rows.append(
                    {
                        "id": wid,
                        "name": weakness.get("name"),
                        "astm": [
                            ASTM_LABELS.get(c, c) for c in _str_list(weakness.get("categories"))
                        ],
                        "status": status,
                        "mitigations": mit_rows,
                    }
                )

        total = sum(counts.values())
        step["weakness_count"] = total
        step.update(counts)
        step["worst"] = next((s for s in _STATUSES if counts[s]), "none")
        if detail:
            step["weaknesses"] = rows
        steps.append(step)
        totals["steps"] += 1
        totals["weaknesses"] += total
        for s in _STATUSES:
            totals[s] += counts[s]

    return {
        "knowledge_base_revision": kb.revision,
        "mode": "assessment" if assessment else "pattern",
        "steps": steps,
        "skipped_ids": list(skipped),
        "totals": totals,
    }


def _status(total: int, in_place: int, na: int) -> str:
    """coverage.ts weaknessStatusBy."""
    if total == 0:
        return "unmitigated"
    applicable = total - na
    if applicable == 0:
        return "na"
    if in_place == 0:
        return "unmitigated"
    if in_place == applicable:
        return "mitigated"
    return "partial"
