# Training Gym docs — Starlight preview

Side-by-side preview of the docs site, migrating from `mkdocs-material` to
[Starlight](https://starlight.astro.build/) (Astro).

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

## Deploy (preview URL)

Once `docs_next_app.py` is added in this directory:

```bash
uv run modal deploy docs-next/docs_next_app.py
```

This will attach to `training-gym-next.modal.dev` (must be registered in
workspace Domains first). The existing `docs/docs_app.py` continues to serve
the current mkdocs-material site at `training-gym.modal.dev` until cutover.

## Layout

```
docs-next/
├── astro.config.mjs        # Starlight config: sidebar, theme hooks
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

Content under `src/content/docs/` is largely regenerated from the repo's
`README.md` and `tutorials/README.md` via `scripts/generate_docs_pages.py`.
Edit those source files, not the generated pages.
