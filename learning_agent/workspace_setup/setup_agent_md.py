#!/usr/bin/env python3
"""Stitch the AGENTS.md handed to a learning agent from instructions/ blocks.

The template is instructions/AGENTS.md (pinned, never edited by this script).
It holds slots that this script fills for a concrete run:

  <OBJECTIVE>    instructions/objective/<archetype>.md (qa or agentic),
                 or the path in the task's `instructions.objective:`
  <DATA_ACCESS>  instructions/data_access/<track>.md (easy/medium/hard),
                 or the path in the task's `instructions.data_access:`
  <SETUP>        the path in `global.setup_instructions` (bench/config.yaml),
                 or the task's `instructions.setup:`
  <METHODS>      instructions/methods/<m>.md concatenated for every selected
                 training method (--methods), in the order given
  <HARNESS>      instructions/harness/<archetype>.md, or the task's
                 `instructions.harness:`
  <TRAINING_TIPS> instructions/tips/<archetype>.md, or the task's
                 `instructions.tips:`
  <RULES>        instructions/rules/default.md, or the task's
                 `instructions.rules:`
  <TASK>         the assigned task name (resolved last: the blocks above
  <TASK_MODEL>   may themselves contain these two placeholders)

Method markers (<!-- if:sft --> ... <!-- endif:sft -->) are kept or stripped
per the same selection; the template itself has none left, but --strip-only
still resolves them in composed files like toolbox TOOLS.md.

Usage:
  python3 workspace_setup/setup_agent_md.py --task fav2              # -> stdout
  python3 workspace_setup/setup_agent_md.py --task fav2 --track medium \\
      --archetype qa --out /tmp/AGENTS.filled.md

make_prompt.py uses the same fill logic when assembling the launch prompt;
workspace_setup/prepare_workspace.sh passes the block paths the task config
resolved (harness/config.py --emit-seed-env).
"""
from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = "instructions/AGENTS.md"
TRACKS = ("easy", "medium", "hard")
ARCHETYPES = ("qa", "agentic")

MARK_IF = "<!-- if:{tag} -->"
MARK_END = "<!-- endif:{tag} -->"
METHOD_TAGS = ("sft", "dpo", "rl", "opd", "sdft")


def default_task_model(root: Path) -> str:
    """The operative task model pin: global.base_model from bench/config.yaml."""
    import yaml
    cfg = yaml.safe_load((root / "bench" / "config.yaml").read_text())
    model = (cfg.get("global") or {}).get("base_model")
    if not model:
        raise SystemExit("bench/config.yaml has no global.base_model — pass --task-model")
    return str(model)


def default_setup_path(root: Path) -> str:
    """global.setup_instructions from bench/config.yaml (the deployment's
    infrastructure block — Modal here; other deployments point it at their own)."""
    import yaml
    cfg = yaml.safe_load((root / "bench" / "config.yaml").read_text())
    return str((cfg.get("global") or {}).get(
        "setup_instructions", "instructions/setup/modal.md"))


def load_block(root: Path, rel: str, what: str) -> str:
    p = root / rel
    if not p.exists():
        raise SystemExit(f"no {what} block at {p}")
    return p.read_text()


def strip_method_markers(text: str, keep: set[str]) -> str:
    """Resolve <!-- if:METHOD --> ... <!-- endif:METHOD --> blocks.

    A kept method's markers vanish (content stays); an unselected method's
    whole block vanishes. The tag may be a comma list (<!-- if:rl,opd -->):
    the block is kept when ANY listed method is selected — for content
    shared by several methods, like the Training Gym guide. Line-based so
    the markers can sit on their own lines inside lists and paragraphs."""
    import re
    mark_re = re.compile(r"^<!-- (if|endif):([a-z0-9_,]+) -->$")
    out = []
    skipping: str | None = None   # the tag string of the block being dropped
    for line in text.splitlines(keepends=True):
        m = mark_re.match(line.strip())
        if m:
            kind, tag_s = m.group(1), m.group(2)
            tags = set(tag_s.split(","))
            if not tags <= set(METHOD_TAGS):
                if skipping is None:
                    out.append(line)   # not a method marker; leave it alone
                continue
            if skipping is None:
                if kind == "if" and not (tags & keep):
                    skipping = tag_s
                continue               # marker lines never survive
            if kind == "endif" and tag_s == skipping:
                skipping = None
            continue
        if skipping is None:
            out.append(line)
    return "".join(out)


def fill_spec(spec: str, task: str, task_model: str, blocks: dict[str, str]) -> str:
    """Resolve the template slots for one run. Blocks substitute BEFORE
    <TASK>/<TASK_MODEL>: the blocks themselves contain those placeholders."""
    for slot, text in blocks.items():
        spec = spec.replace(f"<{slot}>", text.strip())
    return spec.replace("<TASK>", task).replace("<TASK_MODEL>", task_model)


def resolve_blocks(root: Path, track: str, archetype: str,
                   objective: str = "", data_access: str = "",
                   setup: str = "", methods: str = "sft dpo rl",
                   rules: str = "", harness: str = "", tips: str = "") -> dict[str, str]:
    """Load the prose blocks, explicit paths beating the defaults."""
    if track not in TRACKS:
        raise SystemExit(f"unknown track {track!r} — must be one of {TRACKS}")
    if archetype not in ARCHETYPES:
        raise SystemExit(f"unknown archetype {archetype!r} — must be one of {ARCHETYPES}")
    selected = [m for m in methods.replace(",", " ").split()]
    unknown = [m for m in selected if m not in METHOD_TAGS]
    if unknown:
        raise SystemExit(f"unknown training methods {unknown} — known: {METHOD_TAGS}")
    return {
        "OBJECTIVE": load_block(
            root, objective or f"instructions/objective/{archetype}.md", "objective"),
        "DATA_ACCESS": load_block(
            root, data_access or f"instructions/data_access/{track}.md", "data_access"),
        "SETUP": load_block(root, setup or default_setup_path(root), "setup"),
        "METHODS": "".join(
            load_block(root, f"instructions/methods/{m}.md", f"method {m}")
            for m in selected).rstrip(),
        "HARNESS": load_block(
            root, harness or f"instructions/harness/{archetype}.md", "harness"),
        "TRAINING_TIPS": load_block(
            root, tips or f"instructions/tips/{archetype}.md", "tips"),
        "RULES": load_block(
            root, rules or "instructions/rules/default.md", "rules"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch AGENTS.md for a run from instructions/ blocks.")
    ap.add_argument("--task", required=True, help="assigned task name (fills <TASK>)")
    ap.add_argument("--track", default="easy", choices=TRACKS,
                    help="which data_access block to inject (default easy)")
    ap.add_argument("--archetype", default="qa", choices=ARCHETYPES,
                    help="which objective block to inject (default qa)")
    ap.add_argument("--objective", default="",
                    help="path to an objective block md; overrides --archetype")
    ap.add_argument("--instructions", default="",
                    help="path to a data_access block md; overrides --track")
    ap.add_argument("--setup", default="",
                    help="path to a setup block md; overrides global.setup_instructions")
    ap.add_argument("--rules", default="",
                    help="path to a rules block md; overrides instructions/rules/default.md")
    ap.add_argument("--harness", default="",
                    help="path to a harness block md; overrides --archetype")
    ap.add_argument("--tips", default="",
                    help="path to a training-tips block md; overrides --archetype")
    ap.add_argument("--methods", default="sft dpo rl",
                    help="training methods to keep (if:/endif: markers)")
    ap.add_argument("--strip-only", default="",
                    help="strip method markers in THIS file in place and exit")
    ap.add_argument("--task-model", default="",
                    help="task model id (fills <TASK_MODEL>); "
                         "default: global.base_model from bench/config.yaml")
    ap.add_argument("--root", default=".",
                    help="root holding instructions/ + bench/config.yaml")
    ap.add_argument("--out", default="", help="write here instead of stdout")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    keep = set(args.methods.replace(",", " ").split())
    if args.strip_only:
        f = Path(args.strip_only)
        f.write_text(strip_method_markers(f.read_text(), keep))
        print(f"stripped {f} (methods kept: {sorted(keep)})")
        return
    spec_path = root / TEMPLATE
    if not spec_path.exists():
        raise SystemExit(f"no template at {spec_path}")
    model = args.task_model or default_task_model(root)
    blocks = resolve_blocks(root, args.track, args.archetype,
                            args.objective, args.instructions, args.setup,
                            args.methods, args.rules, args.harness, args.tips)
    filled = fill_spec(spec_path.read_text(), args.task, model, blocks)
    filled = strip_method_markers(filled, keep)
    if args.out:
        Path(args.out).write_text(filled)
        print(f"wrote {args.out}  (task={args.task}, track={args.track}, "
              f"archetype={args.archetype}, task_model={model})")
    else:
        print(filled, end="")


if __name__ == "__main__":
    main()
