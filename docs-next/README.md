# Training Gym docs site

The [gym.modal.dev](https://gym.modal.dev) docs, built with
[Starlight](https://starlight.astro.build/) (Astro) and served from a Modal
ASGI app.

## Dev

```bash
cd docs-next
npm install
npm run dev        # http://localhost:4321
```

## Build

```bash
npm run build      # emits dist/
npm run preview    # serves dist/ on http://localhost:4321
```

## Deploy

```bash
uv run modal deploy docs-next/docs_next_app.py
```

This serves the site at `gym.modal.dev`.

## Layout

```
docs-next/
├── astro.config.mjs        # Starlight config: sidebar, theme hooks
├── docs_next_app.py        # Modal ASGI app that serves dist/
├── src/
│   ├── styles/custom.css   # Modal-green accent, Inter font
│   └── content/
│       └── docs/           # page tree (frontmatter-led)
│           ├── index.md
│           ├── support.md
│           └── tutorials/
│               └── index.md
└── package.json
```

The top-level `index.md` and `tutorials/index.md` are **regenerated** from
the repo's `README.md` and `tutorials/README.md` by
`scripts/generate_docs_pages.py`. Edit those source files, not the generated
pages. API reference pages under `reference/` come from
`scripts/generate_api_reference.py`. Use `scripts/generate_all.py` to
regenerate everything at once.
