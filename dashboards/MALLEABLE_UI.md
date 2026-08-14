# Malleable dashboard UI: views, layouts, and the substrate

The dashboard's run pages are not hardcoded. They are **layouts** — ordered
trees of **views** — and both are documents stored on the metadata volume, so
they can be rearranged, edited, forked, and authored from the browser at runtime
without rebuilding the frontend or redeploying the Modal app.

The shipped UI uses this system with no privileged access: every section of the
Training Run page is a view, registered the same way a view you write at 2am is.

Design background: `MALLEABLE_PANELS_PLAN.md`-era discussion, Ink & Switch's
[Malleable software](https://www.inkandswitch.com/essay/malleable-software/)
(gentle slope, tools-not-apps, bootstrapping), and the existing
`Sample.metadata` convention in this repo, which is the same idea at the scale of
one sample: producers write data, the UI progressively enhances known shapes,
unknown shapes degrade instead of erroring.

---

## 1. Substrate: addressable documents

Everything the dashboard can render is reachable as a document.

```
GET /api/docs/{source}/{path}          -> JSON
GET /api/docs/{source}/{path}?ptr=/a/0/b   -> subdocument (RFC 6901 JSON pointer)
GET /api/docs                          -> [{name, kind, description, paths}]
```

`source` is a server-registered name; a view can only address registered
sources, so a view document never widens data access. Sources at launch:

| source | kind | backing |
| --- | --- | --- |
| `gym` | volume | `training-gym-metadata` — `runs/{id}`, `runs/{id}/rollouts`, `runs/{id}/rollouts/{rollout_id}`, `runs/{id}/advantages/{rollout_id}`, `train-results/{id}`, `evals/{id}`, `deployments/{id}`, and the list documents `runs`, `evals`, `deployments` |
| `ui` | volume | the view/layout store described in §4 |

Reads are cached per store key with the same TTL discipline the existing summary
cache uses. The existing typed routes (`/api/runs`, `/api/evals`, …) stay exactly
as they are — they are convenience views over the same data, and the frontend
keeps using them for the builtin views. `/api/docs` exists so a user-authored
view can reach data no typed route exposes.

Adding a producer (e.g. the learning-agent observatory volume) means registering
a source, not patching the frontend.

## 2. ViewDoc

```jsonc
{
  "id": "rollout-reward-chart",       // unique within scope, [a-z0-9-]+
  "scope": "builtin",                 // builtin | org | user | run  (see §5)
  "title": "Reward over rollouts",
  "accepts": "run.rollouts",          // free-form shape tag; advisory only
  "source": {                         // optional; omitted when the view takes props only
    "doc": "gym/runs/{run_id}/rollouts",  // {…} placeholders filled from layout context
    "ptr": null,
    "poll_ms": 5000
  },
  "config": { "y": "mean", "x": "rollout_id" },   // no-code knobs, view-defined
  "lens": null,                       // optional lens id (§6)
  "code": null,                       // Svelte source; null for builtins (resolved from the registry)
  "component": "RolloutRewardChart",  // builtin registry key; null for authored views
  "updated_at": 1765000000,
  "author": "joy",
  "forked_from": null                 // "builtin:rollout-reward-chart" after a fork
}
```

`accepts` is documentation, not validation: a view that gets a shape it does not
understand must render its own empty state, never throw. Unknown/missing
`component` and uncompilable `code` both fall back to the `json` view.

## 3. LayoutDoc

```jsonc
{
  "id": "training-run.default",
  "scope": "builtin",
  "title": "Training run",
  "route": "/training/:run_id",
  "context": ["run_id"],              // placeholders available to view sources
  "tabs": [
    {
      "id": "summary", "label": "Summary",
      "slots": {
        "main": [ {"view": "run-header", "h": null},
                  {"view": "rollout-reward-chart"},
                  {"view": "custom-tag-charts"} ],
        "rail": [ {"view": "run-summary-card"}, {"view": "step-timings"} ]
      }
    },
    { "id": "rollouts", "label": "Rollouts", "slots": {"main": [{"view": "rollout-explorer"}]} },
    { "id": "logs",     "label": "Logs",     "slots": {"main": [{"view": "run-logs"}]} }
  ]
}
```

A layout instance entry is `{view, config?, lens?, hidden?, w?, h?}`; `config`
here overrides the ViewDoc's `config` for this placement only. Slot names are
part of the layout chrome contract (`main`, `rail`), not free-form: the chrome
owns the grid, views never position themselves.

The layout serializes into the URL for sharing (`?layout=<base64url of a diff
against the resolved layout>`), so a rearrangement can be linked without saving.

**Constraint on the first cut: `training-run.default` must render pixel-identical
to today's page.** The builtin views are the existing markup moved, not rewritten.

## 4. Persistence

Two new `MetadataStore` entries on the existing `training-gym-metadata` volume,
so views and layouts survive redeploys and are shared by everyone hitting the
deployment:

```
MetadataStore.UI_VIEWS   = "ui-views"      # key: "{scope}__{id}"
MetadataStore.UI_LAYOUTS  = "ui-layouts"   # key: "{scope}__{id}"
```

```
GET    /api/ui/views                  -> [ViewDoc]     (builtins merged in, see §5)
PUT    /api/ui/views/{scope}/{id}     <- ViewDoc        (write)
DELETE /api/ui/views/{scope}/{id}                       (revert to shadowed doc)
GET    /api/ui/layouts, PUT, DELETE   -> same shape
GET    /api/ui/schema                 -> JSON Schema for ViewDoc + LayoutDoc
```

Writes require the dashboard password when one is set (they are *not* added to
`PASSWORD_EXEMPT_PATHS`); on an open deployment they are open, exactly like every
other route there. Caps: 256 KB per document, 512 documents per store.

`GET /api/ui/schema` exists so an agent — Claude Code, or the learning agent
inside a run — can author a view without a browser.

## 5. Scopes and fork-on-edit

Resolution order, later shadowing earlier: `builtin < org < user < run`.

- `builtin` views live in the frontend bundle (`src/views/builtin/*.svelte` +
  `src/views/registry.js`) and are listed by the server as synthetic ViewDocs.
- Editing a builtin writes a copy at the current scope with
  `forked_from: "builtin:<id>"`. The builtin is untouched; `DELETE` restores it.
- `run` scope is keyed by `training_run_id` and lets a producer ship UI with its
  data (a run whose agent wrote its own log format can ship the view that reads
  it).
- `org` is the shared scope: "publish" copies a user view to it.

Top of the slope is Git: a view that earns it gets exported to
`src/views/builtin/` in a PR, and existing docs that `forked_from` it keep
working because a fork holds its own code.

## 6. Lenses

A lens adapts a producer's shape into what a view expects, so two producers never
have to agree on a schema:

```jsonc
{ "id": "lab-log-to-timeline", "scope": "org",
  "map": {"ts": "ts", "kind": "kind", "title": "what", "body": "why", "chips": "artifacts"},
  "where": "kind != 'note'",          // optional filter expression
  "derive": {"score": "dev_score * 100"}  // optional derived fields
}
```

`map` values are JSON pointers or dotted paths; `where`/`derive` are a small
expression language over the row (no function calls, no property access beyond
the row). Lenses are how rungs 2–3 of the slope stay no-code, and they are data,
so an LLM can write one.

## 7. Authoring runtime (rung 4)

Authored views are Svelte 5 components compiled **in the browser** and run **in
the app origin**, with access to the host API. In-origin is deliberate: a view
must be able to start from `ConversationView` and share hover/selection with its
neighbours, which an iframe cannot do. The dashboard is deployed behind proxy
auth / a password, Modal credentials are server-side only, and anyone who can
write a view can already `modal deploy` this app.

Compile pipeline, in `src/lib/views/compile.js`:

1. `const { compile } = await import("svelte/compiler")` — lazy chunk, only
   fetched when the editor opens or an authored view is first rendered.
2. `compile(source, {generate: "client", runes: true, name: "AuthoredView"})`.
3. Rewrite the emitted module's static imports into lookups against a host
   registry (`svelte/internal/client`, `svelte`, `$host/*`), and its
   `export default X` into `return X`; instantiate with
   `new Function("__require", body)`.
4. Cache the compiled module keyed by `sha256(source)` in memory, and store the
   hash on the ViewDoc so a reload skips recompilation of an unchanged view.

Host registry (`$host/*`, the only imports an authored view may use besides
`svelte`):

| import | contents |
| --- | --- |
| `$host/components` | `LineChart`, `ComparativeBarChart`, `MinimalTable`, `ResizableTable`, `ConversationView`, `StatusPill`, `TimeAgo`, `CollapsibleSection`, `Loading`, `SkeletonPulse`, `Drawer`, `FilterBar` |
| `$host/format` | everything in `src/lib/format.js` (`fmtDuration`, `truncateId`, …) |
| `$host/data` | `doc(address, {ptr, poll_ms})` → reactive substrate reader; `api` (the typed client) |
| `$host/icons` | the `lucide-svelte` icons already bundled |

A view component receives `{data, config, context, navigate}` as props: `data` is
the resolved (and lensed) document, `config` the merged config, `context` the
layout context (`{run_id, …}`), `navigate(path)` the router hook.

**Styling contract:** Tailwind utility classes are scanned at build time, so a
class that appears only in a runtime-authored view has no CSS. Authored views use
a `<style>` block plus the theme CSS variables (`--text-bright`, `--muted`,
`--green`, `--yellow`, `--color-c-gray-10`, …) — or, better, reuse
`$host/components`, which already carry their styling. This is documented in the
editor's starter template.

## 8. Failure containment

- Every view instance renders inside an error boundary: a compile error, a throw
  during render, or a failed source fetch collapses that card to an error state
  with the message and a "open in editor" / "reset to builtin" action. It never
  takes down the page.
- `?safe=1` on any route ignores all non-builtin views and layouts.
- `json` is a builtin view that renders any document as collapsible JSON, and is
  the fallback for every unresolvable view.
- A layout referencing a missing view id drops that instance silently (same
  gating philosophy as the rest of the dashboard: absent data is absent, not an
  error).

## 9. The slope, summarized

| rung | action | surface |
| --- | --- | --- |
| 1 | rearrange, hide, resize, "open document as…" | layout editing in the page, URL-shareable |
| 2 | configure a view; pick or edit a lens | config drawer, no code |
| 3 | derived fields and filters | lens `where` / `derive` expressions |
| 4 | write or AI-generate a whole view | source editor, in-browser compile, live preview, saved to the volume |
| 5 | publish to org, or export to `src/views/builtin/` as a PR | Git |
