# instructions/ — the agent-facing prose, one knob per file

Everything a learning agent reads at launch is assembled from this
folder when its workspace is seeded (`workspace_setup/prepare_workspace.sh`
calls `workspace_setup/setup_agent_md.py`). The workspace source
(`learning_agent_workspace/`) carries no instruction text; the filled
AGENTS.md is written into each workspace at seeding, the way the toolbox
is composed from `toolbox_bank/`.

## The pieces

| file | fills | picked by |
|---|---|---|
| `AGENTS.md` | the template holding the slots below | always |
| `objective/<archetype>.md` | `<OBJECTIVE>` | the task's `archetype:` (qa or agentic), or `instructions.objective:` in task.yaml |
| `data_access/<track>.md` | `<DATA_ACCESS>` | the launch track (easy/medium/hard), or `instructions.data_access:` in task.yaml |
| `setup/modal.md` | `<SETUP>` | `global.setup_instructions` in bench/config.yaml, or `instructions.setup:` in task.yaml |
| `methods/<method>.md` | `<METHODS>` | one bullet per selected training method, concatenated per the task's `toolbox.training:` |
| `harness/<archetype>.md` | `<HARNESS>` | the harness-improvement bullet, picked by archetype, or `instructions.harness:` in task.yaml |
| `tips/<archetype>.md` | `<TRAINING_TIPS>` | data-scale and data-source tips, picked by archetype, or `instructions.tips:` in task.yaml |
| `rules/default.md` | `<RULES>` | the run rules, or `instructions.rules:` in task.yaml |

Method markers (`<!-- if:sft --> ... <!-- endif:sft -->`) remain only in
`toolbox_bank/TOOLS.md`, kept or stripped per the same selection when the
file is composed into a workspace. `<TASK>` and `<TASK_MODEL>` resolve
last.

A task.yaml stitches its own combination:

    instructions:
      objective: instructions/objective/qa.md        # optional override
      data_access: instructions/data_access/medium.md
      setup: instructions/setup/modal.md
    # a bare string is shorthand for the data_access override:
    # instructions: instructions/data_access/medium.md

## Guidelines

1. To change what agents are told, edit the block here; every later seed
   picks it up. These files are pinned (bench/pins.json), so run
   `python bench.py freeze` after a deliberate edit or scoring refuses
   to run.
2. Not on Modal? Write your own setup block (where GPUs run, where the
   base weights live, how the judge is reached) and point
   `global.setup_instructions` in bench/config.yaml at it.
3. A task variant with its own data story (fav2_no_dev, for example)
   overrides just the block it needs in its task.yaml; everything else
   keeps the defaults.
4. Preview the exact filled spec without seeding a workspace:

       python3 workspace_setup/setup_agent_md.py --task fav2 --track easy --root .
