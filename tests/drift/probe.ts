// Drift probe: runs the Workflow Builder's own parser, link codec and strict
// validator over workflows written by the MCP server.
//
// Run with vite-node from a worktree of the builder:
//   cd <worktree> && npx vite-node <this file> <worktree> <cases.json> <out.json> [<kb.json>]
//
// cases.json is [{name, json: <path to a server-written file>, link: <hash or null>}].
// Results go to out.json (not stdout, which vite-node may also write to), as
// {name: {parsed, nodeCount, edgeCount, deserializedNodeCount, deserializedEdgeCount,
//         roundTripEqual, builderLinkRoundTrip, stringifyEqual, validatorErrors,
//         validatorWarnings, error}}.
// The deserialized counts come from urlState.ts `deserialize`, the builder's real
// load path onto the canvas, which drops malformed nodes and edges; they are null
// when that function is not exported.
//
// importWorkflow.ts is deliberately not imported: it pulls in React-context
// modules that may need a DOM.
import { readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { isDeepStrictEqual } from 'node:util'

interface Case {
  name: string
  json: string
  link: string | null
}

interface Result {
  parsed: boolean
  nodeCount: number | null
  edgeCount: number | null
  deserializedNodeCount: number | null
  deserializedEdgeCount: number | null
  roundTripEqual: boolean | null
  builderLinkRoundTrip: boolean | null
  stringifyEqual: boolean | null
  validatorErrors: string[]
  validatorWarnings: string[]
  error: string | null
}

async function main(): Promise<void> {
  const [root, casesPath, outPath, kbPath] = process.argv.slice(2)
  if (!root || !casesPath || !outPath) {
    throw new Error('usage: probe.ts <worktree> <cases.json> <out.json> [<kb.json>]')
  }
  const url = await import(resolve(join(root, 'src/state/urlState.ts')))
  const val = await import(resolve(join(root, 'scripts/lib/validate-workflow.mjs')))
  const kb = kbPath ? JSON.parse(readFileSync(kbPath, 'utf8')) : undefined
  const cases: Case[] = JSON.parse(readFileSync(casesPath, 'utf8'))

  const out: Record<string, Result> = {}
  for (const c of cases) {
    const r: Result = {
      parsed: false,
      nodeCount: null,
      edgeCount: null,
      deserializedNodeCount: null,
      deserializedEdgeCount: null,
      roundTripEqual: null,
      builderLinkRoundTrip: null,
      stringifyEqual: null,
      validatorErrors: [],
      validatorWarnings: [],
      error: null,
    }
    out[c.name] = r
    try {
      const text = readFileSync(c.json, 'utf8')
      const raw: unknown = JSON.parse(text)
      // What the builder's parseWorkflowFile does, minus the React-side import.
      if (url.isWorkflowState(raw)) {
        const state = url.toCurrentState(raw)
        r.parsed = true
        r.nodeCount = Array.isArray(state.nodes) ? state.nodes.length : 0
        r.edgeCount = Array.isArray(state.edges) ? state.edges.length : 0
        if (typeof url.deserialize === 'function') {
          const loaded = url.deserialize(raw)
          r.deserializedNodeCount = loaded.nodes.length
          r.deserializedEdgeCount = loaded.edges.length
        }
        // The server writes the builder's JSON.stringify(state, null, 2) form.
        r.stringifyEqual = JSON.stringify(state, null, 2) === text.replace(/\n$/, '')
        if (c.link !== null) {
          const decoded = await url.decodeFromHash(c.link)
          r.roundTripEqual = decoded !== null && isDeepStrictEqual(decoded, state)
        }
        const again = await url.decodeFromHash(await url.encodeToHash(state))
        r.builderLinkRoundTrip = again !== null && isDeepStrictEqual(again, state)
      }
      const report = val.validateWorkflow(raw, kb)
      r.validatorErrors = report.errors
      r.validatorWarnings = report.warnings
    } catch (e) {
      r.error = e instanceof Error ? `${e.name}: ${e.message}` : String(e)
    }
  }
  writeFileSync(outPath, JSON.stringify(out, null, 2))
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
