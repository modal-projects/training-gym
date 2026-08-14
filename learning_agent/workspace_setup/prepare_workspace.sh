#!/bin/bash
# The ONE workspace-seeding routine, shared by every runner so a sandbox is
# identical no matter where the agent executes:
#
#   agents/run_sandbox_modal.sh   container under the Modal app `lab-agent`
#   agents/run_sandbox_docker.sh  local Docker container
#
# Source this file, then:
#
#   prepare_workspace <seed_root> <run_parent> <track> <scaffold> <task> <hours>
#
# Produces <run_parent>/workspace (the agent's own copy — the submission)
# plus seed_manifest.txt and run_meta.json beside it. Steps:
#   1. export the committed learning_agent_workspace/ tree (the agent surface:
#      AGENTS.md, submission/, task/ placeholder, runs/ ledgers, and a toolbox/
#      PLACEHOLDER) to the workspace root, then inject the run machinery
#      (agents/run.sh, agents/lib/, the ONE chosen scaffold) and
#      bench/config.yaml (the global pins the tools read)
#   2. stitch AGENTS.md from the instructions/ blocks: objective by
#      archetype, data_access by track or task override, setup from
#      global config — the workspace ships with a concrete spec
#   3. copy the ONE task into workspace task/: committed files, then the
#      gitignored corpus and dev gold per track; delete test.json always and
#      dev.json on medium/hard; empty the run ledgers
#   4. compose toolbox/ from the tool bank (toolbox_bank/): the shared core,
#      then this task's harness starters, data cards, and pinned packages
#      (clone_repos.py); write .env (API keys STRIPPED; judge-service
#      pointers kept)
#   5. write .learning_agent_sandbox, seed_manifest.txt (workspace-relative, task files
#      under task/), run_meta.json
#   6. git init a fresh repo so the agent can version its own work

# task-config readers (yaml_top). Sourced here rather than assumed from the
# caller: the runners already source it, but the tests source THIS file alone.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../agents/lib/session_config.sh"

prepare_workspace() {
    local SEED_ROOT="$1" RUN_PARENT="$2" TRACK="$3" SCAFFOLD="$4" TASK="$5" HOURS="$6"
    local WS="$RUN_PARENT/workspace"
    mkdir -p "$WS"

    if ! git -C "$SEED_ROOT" diff --quiet HEAD -- 2>/dev/null; then
        echo "note: uncommitted changes in $SEED_ROOT are NOT in the workspace (it is HEAD)" >&2
    fi

    # HEAD:<path> forms resolve from the GIT ROOT, which is no longer the
    # seed root when learning_agent/ lives as a subtree inside another repo
    # (the training gym): show-prefix is "" at a repo root, "<subdir>/" in a
    # subtree — prepend it everywhere a tree path is named. AND those
    # commands must run FROM the toplevel: git archive silently filters the
    # output to the current directory when run inside a subdir, producing a
    # valid empty tar (observed 2026-08-15). Pathspec-form commands
    # (archive HEAD -- <paths>) stay on $SEED_ROOT: their paths are
    # cwd-relative in and out.
    local GITPFX GITROOT
    GITPFX="$(git -C "$SEED_ROOT" rev-parse --show-prefix 2>/dev/null)"
    GITROOT="$(git -C "$SEED_ROOT" rev-parse --show-toplevel 2>/dev/null || echo "$SEED_ROOT")"

    # 0) the seeding plan: task_configs/<task>.yaml (with extends + toolbox
    #    resolution) says which harnesses, training methods, and packages this
    #    workspace gets, which task dir holds the gitignored assets, and which
    #    instructions blocks stitch AGENTS.md. The config itself never enters
    #    the workspace.
    eval "$(python3 "$SEED_ROOT/harness/config.py" --root "$SEED_ROOT" --task "$TASK" --emit-seed-env)"
    echo "  toolbox : harnesses=[$LEARNING_AGENT_TB_HARNESSES] training=[$LEARNING_AGENT_TB_TRAINING] assets=$LEARNING_AGENT_TB_ASSETS_TASK" >&2

    # 1) committed tree only, no .git. The workspace root IS the
    #    learning_agent_workspace/ subtree; the run machinery, the prompt
    #    material, and the ONE task ride in beside it (their repo-relative
    #    paths equal their workspace-relative paths).
    git -C "$GITROOT" archive "HEAD:${GITPFX}learning_agent_workspace" | tar -x -C "$WS"
    git -C "$SEED_ROOT" archive HEAD -- \
        agents/run.sh agents/lib "agents/$SCAFFOLD" \
        bench/config.yaml | tar -x -C "$WS"

    # 2) stitch AGENTS.md from the instructions/ blocks the task config
    #    resolved: objective by archetype, data_access by track (or the
    #    task's own override), setup from global config. The workspace
    #    ships concrete, not as a template.
    local INSTR_ARGS=(--track "$TRACK" --archetype "$LEARNING_AGENT_TB_ARCHETYPE"
                      --objective "$SEED_ROOT/$LEARNING_AGENT_TB_INSTR_OBJECTIVE")
    if [ -n "$LEARNING_AGENT_TB_INSTR_DATA_ACCESS" ]; then
        INSTR_ARGS+=(--instructions "$SEED_ROOT/$LEARNING_AGENT_TB_INSTR_DATA_ACCESS")
    fi
    if [ -n "$LEARNING_AGENT_TB_INSTR_SETUP" ]; then
        INSTR_ARGS+=(--setup "$SEED_ROOT/$LEARNING_AGENT_TB_INSTR_SETUP")
    fi
    if [ -n "$LEARNING_AGENT_TB_INSTR_RULES" ]; then
        INSTR_ARGS+=(--rules "$SEED_ROOT/$LEARNING_AGENT_TB_INSTR_RULES")
    fi
    if [ -n "$LEARNING_AGENT_TB_INSTR_HARNESS" ]; then
        INSTR_ARGS+=(--harness "$SEED_ROOT/$LEARNING_AGENT_TB_INSTR_HARNESS")
    fi
    if [ -n "$LEARNING_AGENT_TB_INSTR_TIPS" ]; then
        INSTR_ARGS+=(--tips "$SEED_ROOT/$LEARNING_AGENT_TB_INSTR_TIPS")
    fi
    python3 "$SEED_ROOT/workspace_setup/setup_agent_md.py" --task "$TASK" "${INSTR_ARGS[@]}" \
        --methods "$LEARNING_AGENT_TB_TRAINING" \
        --root "$SEED_ROOT" --out "$WS/AGENTS.md"

    # 3) copy the ONE task into task/ (committed files first). The held-out
    #    sets never enter the workspace: test.json always deleted, dev.json
    #    deleted too on medium/hard.
    if [ "$LEARNING_AGENT_TB_ASSETS_TASK" != "$TASK" ]; then
        git -C "$GITROOT" archive "HEAD:${GITPFX}workspace_setup/tasks/$LEARNING_AGENT_TB_ASSETS_TASK" | tar -x -C "$WS/task"
    fi
    # a variant may have no asset folder of its own (its config lives in
    # task_configs/, which never enters a workspace)
    if git -C "$GITROOT" cat-file -e "HEAD:${GITPFX}workspace_setup/tasks/$TASK" 2>/dev/null; then
        git -C "$GITROOT" archive "HEAD:${GITPFX}workspace_setup/tasks/$TASK" | tar -x -C "$WS/task"
    fi
    rm -f "$WS/task/test.json"
    if [ "$TRACK" != "easy" ] || [ "$LEARNING_AGENT_TB_SEED_DEV" != 1 ]; then
        rm -f "$WS/task/dev.json"
    fi
    #    and the workspace starts with ZERO prior-run product: committed learning-log
    #    rows would leak earlier generations' methods and scores into a run that is
    #    supposed to discover its own. The operator's curated ledgers stay at the
    #    repo root, outside the seeded surface.
    : > "$WS/runs/LEARNING_LOG.jsonl"

    # 2b) task ARCHETYPE (qa vs agentic) is recorded for the run metadata —
    #     resolved by the config loader in step 0.
    local ARCHETYPE="$LEARNING_AGENT_TB_ARCHETYPE"

    # 3) gitignored inputs, per track: corpus on easy/medium, dev.json on easy only.
    #    copy() uses APFS copy-on-write when available (instant for multi-hundred-MB
    #    corpora), plain copy elsewhere.
    _pw_copy() { cp -Rc "$1" "$2" 2>/dev/null || cp -R "$1" "$2"; }
    local ASSETS="$SEED_ROOT/workspace_setup/tasks/$LEARNING_AGENT_TB_ASSETS_TASK"
    if [ "$TRACK" = "hard" ] || [ "$LEARNING_AGENT_TB_SEED_CORPUS" != 1 ]; then
        echo "corpus not seeded (task/track config: agent acquires or works without it)" >&2
    elif [ -z "$(yaml_top "$SEED_ROOT/workspace_setup/task_configs/$LEARNING_AGENT_TB_ASSETS_TASK.yaml" corpus)" ]; then
        # No `corpus:` key: an env task (alfworld) ships no study material — the
        # environment IS the material. Absence is the design, not a missing file.
        echo "task $TASK declares no corpus: nothing to seed" >&2
    elif [ ! -d "$WS/task/corpus" ]; then
        if [ -d "$ASSETS/corpus" ]; then
            _pw_copy "$ASSETS/corpus" "$WS/task/corpus"
        else
            echo "warn: no corpus at workspace_setup/tasks/$LEARNING_AGENT_TB_ASSETS_TASK/corpus — seed the workspace manually" >&2
        fi
    fi
    if [ "$TRACK" = "easy" ] && [ "$LEARNING_AGENT_TB_SEED_DEV" = 1 ] && [ -f "$ASSETS/dev.json" ] && [ ! -f "$WS/task/dev.json" ]; then
        _pw_copy "$ASSETS/dev.json" "$WS/task/dev.json"
    fi
    # .env is seeded WITHOUT any raw provider keys — those are the operator's
    # identity and must never ride into a workspace.
    #
    # LEARNING_AGENT_USER_SIM_* and LEARNING_AGENT_JUDGE_* are NOT such keys and are deliberately
    # kept: they point at operator-run SERVICES (harness/user_sim.py,
    # harness/judge_service.py) that pin their model server-side, hold the
    # provider key themselves, and budget each session. The judge service is
    # how the agent's dev-set and intermediate-checkpoint evals come from the
    # SAME pinned judge that scores the submission — without it the judge
    # tools fall back to the local `claude` CLI (canonical:false), and the
    # numbers stop being comparable.
    if [ -f "$SEED_ROOT/.env" ]; then
        grep -vE '^[[:space:]]*(ANTHROPIC_API_KEY|OPENAI_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)=' "$SEED_ROOT/.env" > "$WS/.env" || : > "$WS/.env"
    fi
    # per-session id: the user-sim service budgets and logs against it
    printf 'LEARNING_AGENT_SESSION=%s\n' "$(basename "$RUN_PARENT")" >> "$WS/.env"

    # 3b) the workspace toolbox/ is a PLACEHOLDER in the committed tree; its
    #     content lives in the tool bank (toolbox_bank/, outside the seeded
    #     surface) and is composed in here. First the shared core: everything
    #     in the bank except the selectable pieces — harness starters (copied
    #     by name below), data-card families (copied by training method below),
    #     and the full repos.yaml registry (a filtered one is generated below).
    local BANK_TMP; BANK_TMP="$(mktemp -d)"
    git -C "$GITROOT" archive "HEAD:${GITPFX}workspace_setup/toolbox_bank" | tar -x -C "$BANK_TMP"
    rm -f "$BANK_TMP/repos.yaml"
    rm -rf "$BANK_TMP/harness_tool"
    local fam
    for fam in $LEARNING_AGENT_TB_ALL_CARDS; do rm -rf "$BANK_TMP/data_tool/$fam"; done
    cp -R "$BANK_TMP"/. "$WS/toolbox/"
    rm -rf "$BANK_TMP"

    # 3c) the pinned training packages (gitignored, pinned in the bank registry)
    #     are MATERIALIZED at setup: copied from the seed repo's bank where
    #     present (APFS copy-on-write = instant, offline), cloned from upstream
    #     at the exact pin otherwise.
    #     the workspace's repos.yaml is GENERATED from the bank registry with
    #     only this task's packages; the agent never sees pins it cannot use.
    python3 - "$SEED_ROOT" "$WS" "$LEARNING_AGENT_TB_PACKAGES" <<'PYREPOS'
import sys, yaml
root, ws, wanted = sys.argv[1], sys.argv[2], set(sys.argv[3].split())
full = yaml.safe_load(open(f"{root}/workspace_setup/toolbox_bank/repos.yaml"))
kept = {k: v for k, v in full.items() if k in wanted}
header = ("# repos.yaml: the cloned packages pinned for THIS task's training methods\n"
          "# (generated at seeding from the tool bank registry).\n")
open(f"{ws}/toolbox/repos.yaml", "w").write(
    header + yaml.safe_dump(kept, sort_keys=False, allow_unicode=True, width=10000))
PYREPOS
    python3 "$WS/toolbox/clone_repos.py" --copy-from "$SEED_ROOT/workspace_setup/toolbox_bank" >&2 \
        || echo "warn: package materialization incomplete (see toolbox/repos.yaml)" >&2

    # 3d) the selected modules: harness starters by name, data cards by
    #     training method.
    mkdir -p "$WS/toolbox/harness_tool"
    git -C "$GITROOT" archive "HEAD:${GITPFX}workspace_setup/toolbox_bank/harness_tool" -- README.md 2>/dev/null \
        | tar -x -C "$WS/toolbox/harness_tool" 2>/dev/null || true
    local h
    for h in $LEARNING_AGENT_TB_HARNESSES; do
        if git -C "$GITROOT" cat-file -e "HEAD:${GITPFX}workspace_setup/toolbox_bank/harness_tool/$h.py" 2>/dev/null; then
            git -C "$GITROOT" show "HEAD:${GITPFX}workspace_setup/toolbox_bank/harness_tool/$h.py" \
                > "$WS/toolbox/harness_tool/$h.py"
        else
            echo "warn: no harness starter '$h' in toolbox_bank/harness_tool" >&2
        fi
    done
    _pw_bank_cards() {  # copy one data-card family from the bank
        git -C "$GITROOT" archive "HEAD:${GITPFX}workspace_setup/toolbox_bank/data_tool/$1" \
            | { mkdir -p "$WS/toolbox/data_tool/$1"; tar -x -C "$WS/toolbox/data_tool/$1"; }
    }
    local card
    for card in $LEARNING_AGENT_TB_CARDS; do _pw_bank_cards "$card"; done
    # the shelf docs arrived with the bank core: resolve their method-marker
    # blocks for this task's training selection
    local md
    for md in TOOLS.md training_tool/README.md data_tool/README.md; do
        python3 "$SEED_ROOT/workspace_setup/setup_agent_md.py" --task "$TASK" \
            --strip-only "$WS/toolbox/$md" --methods "$LEARNING_AGENT_TB_TRAINING" >&2
    done
    # the TOOLS.md catalog is GENERATED from the tools that actually shipped
    # into THIS workspace — it never lists anything the agent does not have
    python3 "$SEED_ROOT/observatory/validate_tools.py" \
        --catalog-for "$WS/toolbox" >> "$WS/toolbox/TOOLS.md" \
        || echo "warn: TOOLS.md catalog generation failed" >&2


    # 4) mark this tree as a sandbox — agents/run.sh refuses to launch without it,
    #    so an agent can never be started against the seed repo by accident
    : > "$WS/.learning_agent_sandbox"

    # 5) seed manifest (exactly what was seeded, WORKSPACE-relative paths —
    #    the observatory compares these against workspace snapshot paths to
    #    tell seed tools from invented ones) + run metadata.
    #    Written into RUN_PARENT (not the workspace — not part of the agent's own repo).
    {
        git -C "$GITROOT" ls-tree -r "HEAD:${GITPFX}learning_agent_workspace"
        git -C "$SEED_ROOT" ls-tree -r HEAD -- \
            agents/run.sh agents/lib "agents/$SCAFFOLD" \
            bench/config.yaml
        git -C "$SEED_ROOT" ls-tree -r HEAD -- workspace_setup/toolbox_bank | sed $'s#\tworkspace_setup/toolbox_bank/#\ttoolbox/#'
        if [ "$LEARNING_AGENT_TB_ASSETS_TASK" != "$TASK" ]; then
            git -C "$GITROOT" ls-tree -r "HEAD:${GITPFX}workspace_setup/tasks/$LEARNING_AGENT_TB_ASSETS_TASK" | sed $'s#\t#\ttask/#'
        fi
        git -C "$GITROOT" ls-tree -r "HEAD:${GITPFX}workspace_setup/tasks/$TASK" 2>/dev/null | sed $'s#\t#\ttask/#' || true
    } | while IFS=$'\t' read -r meta path; do
        # the manifest must list what was ACTUALLY seeded: toolbox composition
        # prunes harnesses/packages/cards, so drop entries whose file is absent
        [ -e "$WS/$path" ] && printf '%s\t%s\n' "$meta" "$path"
    done > "$RUN_PARENT/seed_manifest.txt"
    python3 -c '
import json, sys
from datetime import datetime, timezone

track, scaffold, task, hours, archetype, out = sys.argv[1:7]
meta = {
    "track": track,
    "scaffold": scaffold,
    "task": task,
    "hours": hours,
    "archetype": archetype,
    "prepared_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(out, "w") as f:
    json.dump(meta, f)
' "$TRACK" "$SCAFFOLD" "$TASK" "$HOURS" "$ARCHETYPE" "$RUN_PARENT/run_meta.json"

    # 6) the agent's history starts here (.gitignore keeps corpus/dev/.env out of it)
    # gc.auto=0 + autodetach=false: the commit of a ~50k-file tree otherwise
    # triggers a DETACHED background repack that deletes loose objects while
    # the Modal runner is still uploading them (observed 2026-08-12).
    git -C "$WS" init -q
    git -C "$WS" -c gc.auto=0 add -A
    git -C "$WS" -c user.name="learning-agent-operator" -c user.email="lab@localhost" \
        -c gc.auto=0 -c gc.autodetach=false \
        commit -q -m "Learning Agent workspace: ${SCAFFOLD} ${TASK} $(basename "$RUN_PARENT")"
}
