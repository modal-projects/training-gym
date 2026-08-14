#!/usr/bin/env python3
"""Validate the toolbox tool contract (TOOLS.md rules); optionally regenerate
the generated surfaces (dashboard catalog JSON + the TOOLS.md catalog).

    python3 observatory/validate_tools.py           # exit 0 iff all valid
    python3 observatory/validate_tools.py --emit    # also write the catalogs

Operator/dashboard tooling — the agent never needs to run this; its run
timeline picks up every executed toolbox path automatically.

A tool is a .py file in a `toolbox/*_tool/` category directory (or a folder
with run.py when it genuinely needs several files). Everything
machine-readable is DERIVED, never hand-maintained: name from the filename,
category from the path, kind from CATEGORY_KINDS below, the description from
the module docstring's first line, flags from `--help`. Cloned reference
packages (toolbox/repos.yaml dests) and files starting with `_` are not
tools. Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The agent surface lives under learning_agent_workspace/; catalog paths are
# emitted relative to WS_ROOT (i.e. "toolbox/...") so they match the paths
# agents actually type in their workspaces.
WS_ROOT = REPO / "learning_agent_workspace"
TOOLBOX = WS_ROOT / "toolbox"
# Selectable modules live OUTSIDE the workspace and are copied in at seeding
# (workspace_setup/prepare_workspace.sh); their catalog paths still read toolbox/…
# because that is where they land in a seeded workspace.
TOOLBOX_BANK = REPO / "workspace_setup" / "toolbox_bank"
CATALOG_PATH = REPO / "observatory" / "static" / "assets" / "tools.json"
TOOLS_MD = TOOLBOX_BANK / "TOOLS.md"
MARKER = "<!-- CATALOG BELOW IS GENERATED at seeding from the tools in THIS workspace — do not edit -->"

SKIP_DIRS = {"__pycache__", ".git", "recipes", "runs", "tests", "papers",
             "prompts", "entries", "apply_history"}
# Library modules that live inside category trees but are imported, not run.
LIB_FILES = {"corpus_sampling.py", "__init__.py"}


# Known category dirs -> learning-timeline kind (observatory/schema.py).
# Any other toolbox/*_tool/ directory is scanned too, with kind "tool" —
# same catch-all default the trace classifier uses.
CATEGORY_KINDS = {
    "data_tool": "data",
    "training_tool": "train",
    "eval_tool": "eval",
    "harness_tool": "harness",
    "inference_tool": "infra",
}


def load_categories() -> dict[str, str]:
    """Every toolbox/*_tool/ directory -> timeline kind (bank included)."""
    dirs = [d for d in sorted(TOOLBOX.iterdir()) if d.is_dir()]
    if TOOLBOX_BANK.is_dir():
        dirs += [d for d in sorted(TOOLBOX_BANK.iterdir()) if d.is_dir()]
    cats = {d.name: CATEGORY_KINDS.get(d.name, "tool")
            for d in dirs if d.name.endswith("_tool")}
    if not cats:
        raise SystemExit("no toolbox/*_tool/ category directories found")
    return cats


def read_repos(manifest: Path | None = None) -> dict[str, dict]:
    """{name: {dest, notes}} from repos.yaml (never scanned as tools)."""
    repos: dict[str, dict] = {}
    if manifest is None:
        manifest = TOOLBOX_BANK / "repos.yaml"
        if not manifest.exists():
            manifest = TOOLBOX / "repos.yaml"   # a seeded workspace
    if not manifest.is_file():
        return repos
    current = None
    for raw in manifest.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            current = line.rstrip(":").strip()
            repos[current] = {}
        elif current and ":" in line:
            k, _, v = line.strip().partition(":")
            v = v.strip()
            if v.startswith('"') and v.endswith('"'):
                try:                      # YAML double-quoted scalar ~= JSON
                    v = json.loads(v)
                except ValueError:
                    pass
            repos[current][k.strip()] = v
    return repos


def repo_dests() -> set[Path]:
    """Both places a manifest dest can exist: a seeded-workspace layout under
    WS_ROOT and the bank's own clone cache (dest minus the toolbox/ prefix)."""
    dests: set[Path] = set()
    for r in read_repos().values():
        d = r.get("dest")
        if not d:
            continue
        dests.add(WS_ROOT / d)
        dests.add(TOOLBOX_BANK / Path(d).relative_to("toolbox"))
    return dests


def find_tools(categories: dict[str, str]) -> list[tuple[Path, Path]]:
    """[(tool_dir_or_file, entry_py)] — folder tools (run.py) and file tools."""
    dests = repo_dests()
    found: list[tuple[Path, Path]] = []
    for cat in categories:
        # scan every existing root for this category: the workspace core AND
        # the bank (bank modules land at the same toolbox/<cat>/ path when a
        # workspace is seeded)
        for base in (TOOLBOX, TOOLBOX_BANK):
            root = base / cat
            if not root.is_dir():
                continue
            for py in sorted(root.rglob("*.py")):
                if any(part in SKIP_DIRS or part.startswith("_") for part in
                       py.relative_to(base).parts[:-1]):
                    continue
                if any(dest in py.parents for dest in dests):
                    continue
                if py.name == "run.py":
                    found.append((py.parent, py))
                elif (py.parent / "run.py").exists():
                    continue  # internal file of a folder tool
                elif py.name.startswith("_") or py.name in LIB_FILES:
                    continue
                else:
                    found.append((py, py))
    return found


def docstring_summary(entry: Path) -> tuple[str | None, str, str]:
    """(error, summary, full docstring) from the module docstring."""
    try:
        doc = ast.get_docstring(ast.parse(entry.read_text(encoding="utf-8"))) or ""
    except SyntaxError as e:
        return f"unparseable ({e.msg}, line {e.lineno})", "", ""
    if not doc.strip():
        return "missing module docstring (first line = what the tool does)", "", ""
    first = doc.strip().splitlines()[0].strip()
    name = entry.parent.name if entry.name == "run.py" else entry.stem
    # accept "<name> — summary", "<name> - summary", or a bare summary line
    m = re.match(rf"^{re.escape(name)}(?:\.py)?\s*(?:—|--|-)\s*(.+)$", first)
    summary = m.group(1).strip() if m else first
    if len(summary) > 100:
        return f"docstring first line is {len(summary)} chars (max 100)", summary, doc
    return None, summary, doc


def harvest_help(entry: Path) -> tuple[str | None, str]:
    # bank tools import sibling bank libraries (api_clients etc.) — in a
    # seeded workspace they sit side by side, so give them that sys.path here
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TOOLBOX_BANK) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        cp = subprocess.run([sys.executable, str(entry), "--help"],
                            capture_output=True, text=True, timeout=30,
                            cwd=str(WS_ROOT), env=env)
    except subprocess.TimeoutExpired:
        return "--help timed out (30s)", ""
    if cp.returncode != 0:
        return f"--help exited {cp.returncode}: {cp.stderr.strip()[:200]}", ""
    return None, cp.stdout


def validate(unit: Path, entry: Path, categories: dict[str, str],
             run_help: bool) -> tuple[list[str], dict]:
    # bank modules report the path they occupy in a SEEDED workspace
    base = TOOLBOX_BANK if TOOLBOX_BANK in unit.parents else TOOLBOX
    rel = unit.relative_to(base)
    errs: list[str] = []
    for legacy in ("tool.yaml", "tool.md", "README.md"):
        if unit.is_dir() and (unit / legacy).is_file():
            errs.append(f"{rel}: stale {legacy} — the module docstring is the doc")
    doc_err, summary, doc = docstring_summary(entry)
    if doc_err:
        errs.append(f"{rel}: {doc_err}")
    name = unit.name if unit.is_dir() else unit.stem
    tool = {
        "name": name,
        "category": str(rel.parent),
        "kind": categories[rel.parts[0]],
        "summary": summary,
        "path": ("toolbox/" + str(entry.relative_to(TOOLBOX_BANK))
                 if TOOLBOX_BANK in entry.parents
                 else str(entry.relative_to(WS_ROOT))),
        "doc": doc,
    }
    if run_help:
        help_err, help_text = harvest_help(entry)
        if help_err:
            errs.append(f"{rel}: {help_err}")
        tool["help"] = help_text
    return errs, tool


def catalog_json(tools: list[dict]) -> str:
    cats: dict[str, list] = {}
    for t in sorted(tools, key=lambda t: (t["category"], t["name"])):
        cats.setdefault(t["category"], []).append(t)
    return json.dumps({"generated_from": "observatory/validate_tools.py --emit",
                       "categories": cats,
                       "packages": [{"name": n, **r} for n, r in read_repos().items()]},
                      indent=1) + "\n"


def catalog_md(tools: list[dict], repos: dict[str, dict] | None = None) -> str:
    """The generated catalog section of a workspace TOOLS.md (below MARKER)."""
    cats: dict[str, list] = {}
    for t in sorted(tools, key=lambda t: (t["category"], t["name"])):
        cats.setdefault(t["category"], []).append(t)
    lines = [""]
    for cat, ts in cats.items():
        lines.append(f"## {cat}")
        for t in ts:
            lines.append(f"- `{t['path']}` — {t['summary']}")
        lines.append("")
    if repos is None:
        repos = read_repos()
    if repos:
        lines.append("## cloned packages (drive them directly; pins: repos.yaml)")
        for name, r in repos.items():
            lines.append(f"- `{r.get('dest', '?')}` — {r.get('notes', '')}")
        lines.append("")
    return "\n".join(lines)


def catalog_for(root: Path) -> str:
    """Catalog md for ONE composed toolbox directory (a seeded workspace's
    toolbox/, or the bank itself): rows generated from the files actually
    present, module-docstring first lines as summaries, packages from the
    repos.yaml sitting in that directory. This is what workspace seeding
    appends below the MARKER — the catalog never lists a tool that did not
    ship, because it is derived from the shipped tree, not filtered from a
    superset."""
    root = Path(root).resolve()
    repos = read_repos(root / "repos.yaml")
    dests = {root / Path(r["dest"]).relative_to("toolbox")
             for r in repos.values() if r.get("dest")}
    tools: list[dict] = []
    for catdir in sorted(p for p in root.iterdir()
                         if p.is_dir() and p.name.endswith("_tool")):
        for py in sorted(catdir.rglob("*.py")):
            rel = py.relative_to(root)
            if any(part in SKIP_DIRS or part.startswith("_")
                   for part in rel.parts[:-1]):
                continue
            if any(dest in py.parents for dest in dests):
                continue
            if py.name == "run.py":
                unit = py.parent
            elif (py.parent / "run.py").exists():
                continue  # internal file of a folder tool
            elif py.name.startswith("_") or py.name in LIB_FILES:
                continue
            else:
                unit = py
            _, summary, _ = docstring_summary(py)
            tools.append({"name": unit.stem if unit.is_file() else unit.name,
                          "category": str(rel.parent),
                          "path": "toolbox/" + rel.as_posix(),
                          "summary": summary})
    return catalog_md(tools, repos)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", action="store_true",
                    help="regenerate observatory tools.json + the TOOLS.md catalog")
    ap.add_argument("--emit-json", default="",
                    help="write the catalog JSON to this path only")
    ap.add_argument("--catalog-for", default="",
                    help="print the catalog md for ONE composed toolbox dir "
                         "(what seeding appends to a workspace TOOLS.md)")
    args = ap.parse_args()

    if args.catalog_for:
        print(catalog_for(Path(args.catalog_for)), end="")
        return 0

    categories = load_categories()
    run_help = bool(args.emit or args.emit_json)
    tools, errors = [], []
    for unit, entry in find_tools(categories):
        errs, tool = validate(unit, entry, categories, run_help)
        errors.extend(errs)
        if not errs:
            tools.append(tool)

    if errors:
        print(f"[tools] {len(errors)} spec violation(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"[tools] {len(tools)} tool(s) valid")
    if args.emit:
        CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CATALOG_PATH.write_text(catalog_json(tools))
        print(f"[tools] catalog -> {CATALOG_PATH.relative_to(REPO)}")
    if args.emit_json:
        out = Path(args.emit_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(catalog_json(tools))
        if not args.emit:
            print(f"[tools] catalog -> {args.emit_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
