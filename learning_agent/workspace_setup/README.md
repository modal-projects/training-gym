# workspace_setup — how a workspace is built

One routine (`prepare_workspace.sh`, sourced by the runners in `agents/`)
turns the repo's folders into one agent workspace. Folder to folder:

| source folder | becomes | how |
|---|---|---|
| `learning_agent_workspace/` | the workspace root | copied as-is (placeholders: empty `toolbox/`, `submission/` stubs, empty ledgers) |
| `instructions/` | `AGENTS.md` | stitched by `setup_agent_md.py`: objective, data access, methods, harness, tips, setup, rules blocks per the task config |
| `../task_configs/<task>.yaml` | (nothing — operator only) | the LAUNCH INPUT: `run_agent.sh task_configs/<task>.yaml`. Archetype, assets, toolbox selection, instruction overrides, session defaults. Never enters a workspace |
| `tasks/<task>/` | `task/` | assets copied per the config: corpus and dev gold as declared, `test.json` never |
| `toolbox_bank/` | `toolbox/` | the shared core, then the task's harness starters, data cards, and pinned packages; shelf docs resolved per method; catalog generated from what shipped |

`setup_agent_md.py` is the stitcher (also usable standalone to preview a
spec: `python3 workspace_setup/setup_agent_md.py --task fav2 --root .`).
The block files themselves live in `instructions/` — see its README for
the knobs.

## Task assets on HuggingFace

Git carries only the small tracked task files. The heavy and the held-out
assets (corpus/, dev.json, test.json) live in ONE private HF dataset
(`universal-learning-agent/tasks`), managed by `hf_tasks.py`:

    python3 workspace_setup/hf_tasks.py upload --task <T>   # after authoring a task
    python3 workspace_setup/hf_tasks.py fetch --task <T>    # fresh checkout: corpus + dev
    python3 workspace_setup/hf_tasks.py fetch --task <T> --with-test   # scoring operator only

`fetch` never downloads test.json unless `--with-test` is passed. The
token is `HUGGINGFACEHUB_API_TOKEN` in `.env`; the script refuses to
upload if the dataset repo is not private.
