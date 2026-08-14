"""ALFWorld env adapter — TextWorld text mode ONLY (no THOR/vision).

Scenario rows (tasks/alfworld/dev.json — generated once from the pinned
alfworld data by `modal run harness/rollout_modal.py::alfworld_splits`):

    {"id": "alfworld_dev_0001",
     "game_file": "json_2.1.1/valid_seen/pick_and_place_simple-.../trial_.../game.tw-pddl",
     "task_type": "pick_and_place_simple"}

`game_file` is RELATIVE to $ALFWORLD_DATA (baked into the Modal image by
`alfworld-download`; env pin in tasks/alfworld/task.yaml).

Episode protocol: `env.show_admissible` (task.yaml; pinned) controls whether
observations include the env's admissible-commands list (AlfredTWEnv exposes
it; AgentGym-style setups show it, ReAct-style setups don't — the choice moves
the base floor a lot, so it is config, not code). Reward is the env's own
sparse `won` signal: 1.0 on goal completion, 0.0 otherwise.

The policy is driven through agent.act() with the task's configured driver:
  tools  (default) NATIVE OpenAI tool calls against ALFWORLD_TOOLS below —
         the interface deployed agents actually use. Requires the endpoint be
         served with a tool-call parser.
  react  the ACTION:/FINAL: text protocol — same action space, works against
         any policy including text-only CLI references.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

OBS_CHARS = 2000     # per-turn transcript cap (the driver caps its own view too)
ADM_MAX = 60         # admissible-commands list cap (telecom-sized rooms are chatty)

# ---- native tool schema (driver: tools) ----------------------------------- #
# One function per ALFWorld verb with typed arguments, rendered back to the
# env's text command by _render(). This exercises the student's tool-calling
# pathway instead of prose parsing; the ACTION SPACE is identical either way
# (same verbs, same objects), so `tools` and `react` measure the same task
# through different interfaces.


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required, "additionalProperties": False}}}


_RECEP = {"type": "string",
          "description": "receptacle name with its number, e.g. 'shelf 1'"}
_OBJ = {"type": "string",
        "description": "object name with its number, e.g. 'keychain 4'"}

ALFWORLD_TOOLS = [
    _fn("go_to", "Move to a receptacle. You must be at a receptacle before "
        "interacting with it or its contents.", {"receptacle": _RECEP}, ["receptacle"]),
    _fn("open_receptacle", "Open a closed receptacle to reveal its contents.",
        {"receptacle": _RECEP}, ["receptacle"]),
    _fn("close_receptacle", "Close an open receptacle.",
        {"receptacle": _RECEP}, ["receptacle"]),
    _fn("take", "Pick up an object from the receptacle you are at.",
        {"object": _OBJ, "receptacle": _RECEP}, ["object", "receptacle"]),
    _fn("put", "Place a held object in or on the receptacle you are at.",
        {"object": _OBJ, "receptacle": _RECEP}, ["object", "receptacle"]),
    _fn("heat", "Heat a held object using an appliance (e.g. 'microwave 1').",
        {"object": _OBJ, "appliance": _RECEP}, ["object", "appliance"]),
    _fn("cool", "Cool a held object using an appliance (e.g. 'fridge 1').",
        {"object": _OBJ, "appliance": _RECEP}, ["object", "appliance"]),
    _fn("clean", "Clean a held object using an appliance (e.g. 'sinkbasin 1').",
        {"object": _OBJ, "appliance": _RECEP}, ["object", "appliance"]),
    _fn("use", "Use an object, e.g. turn on 'desklamp 1'.", {"object": _OBJ}, ["object"]),
    _fn("examine", "Look closely at an object or receptacle.",
        {"target": {"type": "string", "description": "object or receptacle name"}},
        ["target"]),
    _fn("look", "Look around the current location.", {}, []),
    _fn("inventory", "List what you are carrying.", {}, []),
]

_TEMPLATES = {
    "go_to": "go to {receptacle}",
    "open_receptacle": "open {receptacle}",
    "close_receptacle": "close {receptacle}",
    "take": "take {object} from {receptacle}",
    "put": "put {object} in/on {receptacle}",   # 'in/on' is ALFWorld's literal syntax
    "heat": "heat {object} with {appliance}",
    "cool": "cool {object} with {appliance}",
    "clean": "clean {object} with {appliance}",
    "use": "use {object}",
    "examine": "examine {target}",
    "look": "look",
    "inventory": "inventory",
}


def _render(name: str, args: dict) -> str | None:
    """(tool name, args) -> ALFWorld command string, or None if unmappable.
    A bad call becomes an observation, never a crash — the same treatment the
    env gives a nonsense command."""
    tpl = _TEMPLATES.get(name)
    if tpl is None:
        return None
    try:
        return tpl.format(**{k: str(v).strip() for k, v in (args or {}).items()})
    except KeyError:
        return None


def load_split(split_path) -> list[dict]:
    return json.loads(Path(split_path).read_text())


def _data_root() -> Path:
    return Path(os.environ.get("ALFWORLD_DATA",
                               "~/.cache/alfworld")).expanduser()


def _fmt(obs: str, infos: dict, show_admissible: bool) -> str:
    text = (obs or "").strip()
    if not show_admissible:
        return text
    adm = list(infos.get("admissible_commands") or [])
    if adm:
        shown = adm[:ADM_MAX]
        more = f" (+{len(adm) - ADM_MAX} more)" if len(adm) > ADM_MAX else ""
        text += "\n\nAdmissible actions: " + ", ".join(shown) + more
    return text


def run_episode(agent, row: dict, cfg: dict) -> dict:
    import textworld
    import textworld.gym
    try:  # strips TextWorld's internal name mangling from ALFRED entity names
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        wrappers = [AlfredDemangler()]
    except ImportError:
        wrappers = []

    game_file = _data_root() / row["game_file"]
    if not game_file.exists():
        raise FileNotFoundError(f"alfworld game not found: {game_file} "
                                "(is $ALFWORLD_DATA populated by alfworld-download?)")
    env_cfg = cfg.get("env") or {}
    max_steps = int(env_cfg.get("max_steps", 40))
    show_admissible = bool(env_cfg.get("show_admissible", True))
    infos_req = textworld.EnvInfos(won=True, admissible_commands=True)
    env_id = textworld.gym.register_game(str(game_file), infos_req,
                                         max_episode_steps=max_steps + 5,
                                         wrappers=wrappers)
    env = textworld.gym.make(env_id)
    try:
        obs, infos = env.reset()
        transcript: list[dict] = []
        state = {"won": False, "terminal": False}

        def step(action: str):
            o, _score, done, inf = env.step(action)
            state["won"] = bool(inf.get("won"))
            state["terminal"] = bool(done)
            text = _fmt(o, inf, show_admissible)
            transcript.append({"action": action, "observation": text[:OBS_CHARS]})
            return text, bool(done)

        def execute_text(action: str):
            return step(action)

        def execute_tool(name: str, args: dict):
            cmd = _render(name, args)
            if cmd is None:
                obs = (f"[invalid call] no such action {name!r} with arguments "
                       f"{args!r}. Use one of the provided tools with its "
                       "required arguments.")
                transcript.append({"action": f"{name}({args})", "observation": obs})
                return obs, False
            return step(cmd)

        sys_text = cfg.get("_sys_text", "")
        instruction = ((sys_text.strip() + "\n\n" if sys_text else "")
                       + _fmt(obs, infos, show_admissible))
        acfg = cfg.get("agent") or {}
        driver = acfg.get("driver", "tools")
        sampling = {"temperature": float(acfg.get("temperature", 0.0)),
                    "max_tokens": int(acfg.get("max_tokens", 2048))}
        if driver == "tools":
            res = agent.act(instruction, execute_tool, driver="tools",
                            max_turns=max_steps, tools=ALFWORLD_TOOLS,
                            extra_body=acfg.get("extra_body"), **sampling)
        else:
            res = agent.act(instruction, execute_text, driver=driver,
                            max_turns=max_steps, **sampling)
    finally:
        env.close()
    return {"reward": 1.0 if state["won"] else 0.0,
            "steps": len(transcript),
            "done": state["terminal"],
            "transcript": transcript,
            "info": {"game_file": row["game_file"],
                     "task_type": row.get("task_type", ""),
                     "won": state["won"],
                     "driver_summary": res.get("summary", "")}}
