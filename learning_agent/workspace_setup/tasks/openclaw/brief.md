> DRAFT — operator review pending

`tasks/openclaw/corpus/` is a snapshot of the **openclaw** project's own upstream
repository — the primary source for this task. `package.json` declares
`"name": "openclaw"`, and its `repository.url` field and the checkout's git remote
(`tasks/openclaw/corpus/.git/config`) both point at
`https://github.com/openclaw/openclaw`. Fetch/download that repository and place
its contents at `tasks/openclaw/corpus/` — the application source (`src/`), the
plugin/extension packages (`extensions/`, `packages/`), and the docs (`docs/`) are
all part of the primary source and should all be present.

## Manifest

- Project: **openclaw** — `"name": "openclaw"` in `package.json`; its
  `"description"` reads "Multi-channel AI gateway with extensible messaging
  integrations".
- Version: **2026.4.18** — `"version": "2026.4.18"` in `package.json`, matching the
  `## 2026.4.18` release section in `CHANGELOG.md`.
- Source: `https://github.com/openclaw/openclaw`
- Exact commit this snapshot is checked out at: `da228660306b55a9cce3b973946f3aacfc515848`
  (read from `tasks/openclaw/corpus/.git`) — the same commit (by its 8-character
  prefix) pinned at `tasks.openclaw.corpus_commit` in `bench/config.yaml`. The
  upstream repository carries no version tags (an empty tag list at this commit),
  so fetch by commit rather than by tag: `da228660306b55a9cce3b973946f3aacfc515848`.
  If that exact commit is unavailable, any commit whose `package.json` declares
  `"version": "2026.4.18"` is the intended target.

## Layout on disk today (`tasks/openclaw/corpus/`, 14,360 tracked files)

```
.agents/, .codex, .pi/, .vscode/, git-hooks/          (editor/agent/tooling config)
.github/                                              CI workflows, issue/PR templates
.detect-secrets.cfg, .secrets.baseline                 secret-scan config/baseline
.dockerignore, .gitattributes, .gitignore
.jscpd.json, .markdownlint-cli2.jsonc, .npmignore, .npmrc,
.oxfmtrc.jsonc, .oxlintrc.json, .prettierignore, .shellcheckrc,
.swiftformat, .swiftlint.yml                           lint/format config
.env.example, .mailmap
AGENTS.md, CLAUDE.md, CONTRIBUTING.md, INCIDENT_RESPONSE.md, LICENSE, README.md,
SECURITY.md, VISION.md
CHANGELOG.md
Dockerfile (+ 3 additional Dockerfile.* build-target variants), docker-compose.yml,
docker-setup.sh, fly.private.toml, fly.toml, openclaw.podman.env, render.yaml,
setup-podman.sh                                        (deploy/container config)
Makefile, knip.config.ts, tsconfig*.json (11 variants), tsdown.config.ts,
vitest.config.ts, zizmor.yml                            (build/lint/test config)
appcast.xml, docs.acp.md, fix2.py, openclaw.mjs (CLI entrypoint), package.json,
pnpm-lock.yaml, pnpm-workspace.yaml, pyproject.toml
dream-diary-preview-v2.html, dream-diary-preview-v3.html
apps/                         android/, ios/, macos/, shared/
assets/
docs/                         AGENTS.md, CLAUDE.md, docs.json, index.md, and
                               topic pages/dirs: .generated/, .i18n/, assets/,
                               automation/, auth-credential-semantics.md,
                               brave-search.md, channels/, ci.md, cli/, concepts/,
                               date-time.md, debug/, diagnostics/, gateway/,
                               help/, images/, install/, logging.md,
                               nav-tabs-underline.js, network.md, nodes/,
                               perplexity.md, pi-dev.md, pi.md, platforms/,
                               plugins/, prose.md, providers/, refactor/,
                               reference/, security/, snippets/, start/,
                               style.css, tools/, tts.md, vps.md, web/, and two
                               image assets
extensions/                   107 plugin/provider directories (one per channel or
                               model provider — e.g. anthropic/, discord/, google/,
                               memory-core/, openai/, slack/, telegram/,
                               whatsapp/, ...; see the directory itself for the
                               full, current list)
packages/                     memory-host-sdk/, plugin-package-contract/,
                               plugin-sdk/
patches/, qa/, scripts/, skills/, Swabble/, test/, test-fixtures/, ui/, vendor/
src/                           (application source)
  entry.ts (+ entry.*.test.ts), library.ts, logger.ts, logging.ts, runtime.ts,
  utils.ts, version.ts, param-key.ts, globals.ts, global-state.ts,
  extensionAPI.ts, channel-web.ts, and their *.test.ts siblings
  acp/, agents/, auto-reply/, bindings/, bootstrap/, browser-lifecycle-cleanup*,
  canvas-host/, channels/, chat/, cli/, commands/, compat/, config/,
  context-engine/, cron/, daemon/, docker-build-cache*, docker-image-digests*,
  docker-setup.e2e*, dockerfile*, docs/, flows/, gateway/, hooks/, i18n/,
  image-generation/, infra/, install-sh-version*, interactive/,
  link-understanding/, logging/, markdown/, mcp/, media/, media-generation/,
  media-understanding/, memory-host-sdk/, music-generation/, node-host/,
  pairing/, plugin-activation-boundary*, plugin-sdk/, plugins/, poll-params*,
  polls*, process/, proxy-capture/, realtime-transcription/, realtime-voice/,
  routing/, scripts/, secrets/, security/, sessions/, shared/, status/, tasks/,
  terminal/, test-helpers/, test-utils/, tts/, tui/, types/,
  ui-app-settings.agents-files-refresh*, utils/, video-generation/, web/,
  web-fetch/, web-search/, wizard/
```

## Expected normalized layout

Place the fetched repository's contents at `tasks/openclaw/corpus/`, mirroring the
tree above exactly — `src/`, `extensions/`, `packages/`, `docs/`, and the other
top-level files/directories. Version-control metadata (`.git/`) is excluded from
the integrity pin (see `CORPUS_EXCLUDE_NAMES` in `harness/integrity.py`) and does
not need to be reproduced; everything else in the tree above should be present as
shown. `git status` against this snapshot's own commit is clean (nothing untracked
or ignored-but-present), so there are no local build byproducts to exclude here as
there are for the dspy corpus.
