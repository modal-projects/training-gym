"""In-process Berkeley Function Calling Leaderboard (BFCL) v3/v4 multi-turn environment.

BFCL grades agents against in-process Python state machines (``GorillaFileSystem``,
``TwitterAPI``, etc.): tool calls are Python expressions evaluated against live
instances, and grading is a state diff plus a response-subsequence check. This
module wraps ``bfcl_eval`` (imported lazily) into a ``step``/``evaluate`` shape.

- **Data** — :class:`BfclMultiTurnDataset` loads a multi-turn category and flattens
  per-turn ground-truth calls into an ordered sequence.
- **Environment** — :class:`BfclTurnEnvironment` holds fresh class instances per
  episode (no module-level caching) and reuses ``bfcl_eval``'s checkers.
- **Prompting** — :func:`build_prefix_messages` / :func:`tool_schemas_to_openai`
  mirror the Toolathlon helpers of the same name.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models.base import ToolCall

# Empty for interface parity with Toolathlon's DONE_TOOLS.
DONE_TOOLS: frozenset[str] = frozenset()


@dataclass
class Observation:
    text: str = ""
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: Observation
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalVerdict:
    passed: bool
    detail: str = ""
    harness_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

_JSON_TYPE_MAP = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "float": "number",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
}


def _bfcl_eval():
    """Import ``bfcl_eval`` lazily so the package isn't a hard dep of modal_training_gym."""
    try:
        import bfcl_eval
    except ImportError as e:
        raise ImportError(
            "This requires the `bfcl-eval` package (`uv pip install bfcl-eval`); "
            "see https://pypi.org/project/bfcl-eval/."
        ) from e
    return bfcl_eval


def _backend_mappings() -> tuple[dict[str, str], dict[str, str], list[str]]:
    from bfcl_eval.constants.executable_backend_config import (
        CLASS_FILE_PATH_MAPPING,
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
        STATELESS_CLASSES,
    )

    return CLASS_FILE_PATH_MAPPING, MULTI_TURN_FUNC_DOC_FILE_MAPPING, STATELESS_CLASSES


def _data_dir() -> str:
    bfcl_eval = _bfcl_eval()
    return os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Call-string <-> dict ─────────────────────────────────────────────────────


def parse_call_string(call: str) -> dict[str, Any]:
    """Parse a BFCL call string into ``{"name": str, "arguments": dict}``.

    Uses ``ast.literal_eval`` — never executes arbitrary code.
    """
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Not a call expression: {call!r}")
    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    arguments: dict[str, Any] = {
        f"_pos{i}": ast.literal_eval(arg) for i, arg in enumerate(node.args)
    }
    for kw in node.keywords:
        arguments[kw.arg] = ast.literal_eval(kw.value)
    return {"name": name, "arguments": arguments}


def _normalize_arguments(
    owner: Any, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ``_posN`` placeholders to keyword names via ``owner``'s method signature."""
    positional = sorted(
        ((k, v) for k, v in arguments.items() if k.startswith("_pos")),
        key=lambda kv: int(kv[0][4:]),
    )
    if not positional:
        return arguments
    try:
        param_names = [
            p for p in inspect.signature(getattr(owner, name)).parameters if p != "self"
        ]
    except (TypeError, ValueError):
        param_names = []
    normalized = dict(zip(param_names, (v for _, v in positional)))
    normalized.update({k: v for k, v in arguments.items() if not k.startswith("_pos")})
    return normalized


# ── Execution ────────────────────────────────────────────────────────────────


def _instantiate(
    class_name: str, initial_config: dict, *, long_context: bool = False
) -> Any:
    import importlib

    class_file_path_mapping, _, stateless_classes = _backend_mappings()
    module = importlib.import_module(class_file_path_mapping[class_name])
    instance = getattr(module, class_name)()
    if class_name not in stateless_classes:
        instance._load_scenario(
            deepcopy(initial_config.get(class_name, {})), long_context=long_context
        )
    return instance


def build_instances(
    involved_classes: list[str], initial_config: dict
) -> dict[str, Any]:
    """One fresh instance per involved class (safe for concurrent rollouts)."""
    return {name: _instantiate(name, initial_config) for name in involved_classes}


def _method_owner(instances: dict[str, Any], method_name: str) -> Any | None:
    if method_name.startswith("_"):
        return None
    for instance in instances.values():
        if hasattr(type(instance), method_name):
            return instance
    return None


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)
    return str(result)


def execute_call(instances: dict[str, Any], call: dict[str, Any]) -> tuple[str, bool]:
    """Run one ``{"name", "arguments"}`` call. Returns ``(result_text, is_error)``.

    Arguments are deep-copied before the call to avoid aliasing into instance state.
    """
    owner = _method_owner(instances, call["name"])
    if owner is None:
        return f"Error during execution: unknown function {call['name']!r}", True
    try:
        result = getattr(owner, call["name"])(**deepcopy(call.get("arguments") or {}))
        return _stringify(result), False
    except Exception as e:  # mirrors upstream's eval()-based catch-all
        return f"Error during execution: {e}", True


def replay(
    involved_classes: list[str], initial_config: dict, calls: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Fresh instances + observation text for each of ``calls``, executed in order.

    Mutates each ``call["arguments"]`` in place to resolve any ``_posN`` placeholders.
    """
    instances = build_instances(involved_classes, initial_config)
    observations = []
    for call in calls:
        owner = _method_owner(instances, call["name"])
        if owner is not None:
            call["arguments"] = _normalize_arguments(
                owner, call["name"], call.get("arguments") or {}
            )
        text, _is_error = execute_call(instances, call)
        observations.append(text)
    return instances, observations


# ── Tool schemas ─────────────────────────────────────────────────────────────


def to_json_schema(node: Any) -> Any:
    """Convert a BFCL func-doc schema fragment to JSON-Schema typing."""
    if isinstance(node, dict):
        out = {k: to_json_schema(v) for k, v in node.items() if k != "default"}
        if out.get("type") in _JSON_TYPE_MAP:
            out["type"] = _JSON_TYPE_MAP[out["type"]]
        return out
    if isinstance(node, list):
        return [to_json_schema(v) for v in node]
    return node


def tool_schemas_to_openai(tool_schemas: dict) -> list[dict]:
    """Convert BFCL function docs to OpenAI tools, normalizing BFCL type names."""
    tools = []
    for name in sorted(tool_schemas or {}):
        spec = tool_schemas[name]
        if isinstance(spec, dict) and ("parameters" in spec or "description" in spec):
            desc = spec.get("description", "")
            params = to_json_schema(spec.get("parameters", {}))
        else:
            desc, params = "", to_json_schema(spec)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def load_func_docs(
    involved_classes: list[str], excluded_function: list[str] | None = None
) -> dict:
    """BFCL function docs for ``involved_classes``, minus any ``excluded_function``."""
    _, doc_file_mapping, _ = _backend_mappings()
    excluded = set(excluded_function or [])
    data_dir = _data_dir()
    schemas: dict[str, dict] = {}
    for class_name in involved_classes:
        doc_file = doc_file_mapping.get(class_name)
        if not doc_file:
            continue
        for doc in _load_jsonl(os.path.join(data_dir, "multi_turn_func_doc", doc_file)):
            if doc["name"] in excluded:
                continue
            schemas[doc["name"]] = {
                "description": doc.get("description", ""),
                "parameters": to_json_schema(doc.get("parameters", {})),
            }
    return schemas


# ── Prompt-prefix reconstruction ────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
You are a tool-using agent completing a user's request with the function tools provided to you. Work one step at a time.

Rules:
- Make EXACTLY ONE tool call per turn. Emit only the tool call — no extra prose, narration, or markdown fences around it.
- Use only the tools provided to you, with their exact names. Do not invent tools, arguments, or file paths.
- Each user message may require several tool calls before the request is satisfied; keep calling tools until the request is complete, then stop calling tools.
- After each tool result, check whether it succeeded before continuing; do not blindly repeat a failed call with the same arguments.
Use the model's provided tool-calling interface."""


def default_system_prompt(tool_schemas: dict) -> str:
    """Behavioral rules only — the tool catalog travels via the chat template's
    ``tools=`` parameter (see :func:`tool_schemas_to_openai`), not prompt text."""
    return DEFAULT_SYSTEM_PROMPT


def build_prefix_messages(label: dict, K: int) -> list[dict]:
    """Reconstruct the chat-message prefix after the first ``K`` ground-truth calls."""
    observations = label.get("observations")
    if observations is None:
        _, observations = replay(
            label["involved_classes"],
            label["initial_config"],
            deepcopy(label["flattened_calls"]),
        )
    messages = [
        {
            "role": "system",
            "content": default_system_prompt(label.get("tool_schemas", {})),
        }
    ]
    shown = 0
    for turn in label["turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        for call in turn["calls"]:
            if shown >= K:
                return messages
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{shown}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                # Raw dict (not JSON string) for chat-template |items.
                                "arguments": call["arguments"],
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{shown}",
                    "content": str(observations[shown]),
                }
            )
            shown += 1
    return messages


def prefix_turn_index(label: dict, K: int) -> int:
    """Index of the latest user turn included in the step-``K`` prefix."""
    latest_turn = -1
    shown = 0
    for turn_index, turn in enumerate(label.get("turns", [])):
        latest_turn = turn_index
        for _call in turn.get("calls", []):
            if shown >= K:
                return turn_index
            shown += 1
    return latest_turn


def prune_prefix(messages: list[dict], max_messages: int) -> list[dict]:
    """Keep system + first user message and fill the rest with the most recent messages."""
    if max_messages <= 0:
        return []
    if len(messages) <= max_messages:
        return messages
    head = messages[:2]
    tail_n = max_messages - len(head)
    if tail_n <= 0:
        return messages[:max_messages]
    return head + messages[-tail_n:]


# ── Environment ──────────────────────────────────────────────────────────────


@dataclass
class BfclTurnEnvironment:
    """One live multi-turn BFCL episode, seeded by replaying the first ``K`` ground-truth calls."""

    label: dict
    instances: dict[str, Any] = field(default_factory=dict)
    exec_results: list[str] = field(default_factory=list)
    # Ground-truth calls already replayed to seed `instances`.
    K: int = 0

    def step(self, action: ToolCall) -> StepResult:
        call = {"name": action.name, "arguments": action.arguments or {}}
        text, is_error = execute_call(self.instances, call)
        self.exec_results.append(text)
        return StepResult(
            observation=Observation(text=text, is_error=is_error), done=False
        )

    def evaluate(self) -> EvalVerdict:
        """Grade via ``state_checker`` + ``response_checker`` against the full ground-truth trajectory."""
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            response_checker,
            state_checker,
        )

        ground_truth_instances, ground_truth_results = replay(
            self.label["involved_classes"],
            self.label["initial_config"],
            deepcopy(self.label["flattened_calls"]),
        )
        state_result = state_checker(self.instances, ground_truth_instances)
        if not state_result["valid"]:
            return EvalVerdict(passed=False, detail=state_result["error_message"])
        # Only check agent responses from K onward (prefix is identical by construction).
        response_result = response_checker(
            self.exec_results, ground_truth_results[self.K :], 0
        )
        if not response_result["valid"]:
            return EvalVerdict(passed=False, detail=response_result["error_message"])
        return EvalVerdict(passed=True)


def build_env(label: dict, K: int) -> BfclTurnEnvironment:
    """Fresh environment with state fast-forwarded through the first ``K`` ground-truth calls."""
    calls = deepcopy(label["flattened_calls"][:K])
    instances, _ = replay(label["involved_classes"], label["initial_config"], calls)
    return BfclTurnEnvironment(label=label, instances=instances, K=K)


# ── Episode runner ───────────────────────────────────────────────────────────


@dataclass
class BfclEpisodeResult:
    """Structured output from :func:`run_bfcl_episode`."""

    messages: list[dict]
    calls: list[dict[str, Any]]
    execution_successes: list[bool]
    verdict: EvalVerdict
    final_response: str = ""
    exit_reason: str = "max_turns"

    @property
    def first_call(self) -> dict[str, Any] | None:
        return self.calls[0] if self.calls else None


def run_bfcl_episode(
    label: dict,
    *,
    start_step: int,
    generate: Callable[[list[dict], list[dict]], dict],
    parse_response: Callable[[dict], tuple[str, list[ToolCall]]],
    max_turns: int,
    max_consecutive_errors: int = 3,
    observation_limit: int = 2000,
    log: Callable[[str], None] | None = None,
) -> BfclEpisodeResult:
    """Generate, execute, and grade one BFCL episode.

    ``generate`` receives the current OpenAI-style messages and tool schemas.
    ``parse_response`` converts its returned message into assistant text and
    normalized :class:`ToolCall` objects. Keeping those model-specific details
    in callbacks lets the BFCL lifecycle stay independent of a serving engine.
    """

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    messages = build_prefix_messages(label, start_step)
    tools = tool_schemas_to_openai(label.get("tool_schemas", {}))
    env = build_env(label, start_step)
    calls: list[dict[str, Any]] = []
    execution_successes: list[bool] = []
    final_response = ""
    exit_reason = "max_turns"
    consecutive_errors = 0
    current_turn = prefix_turn_index(label, start_step)

    for turn in range(max_turns):
        started_at = time.monotonic()
        message = generate(messages, tools)
        generation_seconds = time.monotonic() - started_at
        content, actions = parse_response(message)
        final_response = content
        emit(
            f"turn {turn} gen={generation_seconds:.1f}s "
            f"calls={[action.name for action in actions]}"
        )
        if not actions:
            next_turn = current_turn + 1
            turns = label.get("turns", [])
            if next_turn < len(turns):
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": turns[next_turn]["user"]},
                    ]
                )
                current_turn = next_turn
                consecutive_errors = 0
                emit(f"  advancing to user turn {current_turn}")
                continue
            exit_reason = "no_further_calls"
            break

        observations: list[str] = []
        stop = False
        for action in actions:
            calls.append({"name": action.name, "arguments": action.arguments or {}})
            try:
                step_result = env.step(action)
            except Exception as exc:
                emit(f"  execution error on {action.name}: {exc!r} — ending episode")
                execution_successes.append(False)
                exit_reason = "execution_error"
                stop = True
                break

            observation = step_result.observation
            execution_successes.append(not observation.is_error)
            observations.append(observation.text)
            emit(f"  exec {action.name} -> {'ERR' if observation.is_error else 'ok'}")
            consecutive_errors = consecutive_errors + 1 if observation.is_error else 0
            if step_result.done:
                exit_reason = "environment_done"
                stop = True
                break
            if (
                max_consecutive_errors > 0
                and consecutive_errors >= max_consecutive_errors
            ):
                exit_reason = "repeated_errors"
                stop = True
                break

        if stop:
            break

        call_ids = [f"call_t{turn}_{index}" for index in range(len(actions))]
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": action.arguments or {},
                        },
                    }
                    for call_id, action in zip(call_ids, actions)
                ],
            }
        )
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": observation[:observation_limit],
            }
            for call_id, observation in zip(call_ids, observations)
        )

    try:
        verdict = env.evaluate()
    except Exception as exc:
        emit(f"evaluate() failed: {exc!r} — marking failed")
        verdict = EvalVerdict(passed=False, detail=str(exc), harness_error=True)

    return BfclEpisodeResult(
        messages=messages,
        calls=calls,
        execution_successes=execution_successes,
        verdict=verdict,
        final_response=final_response,
        exit_reason=exit_reason,
    )


# ── Trajectory dataset ───────────────────────────────────────────────────────


def _first_user_text(turn_messages: list[dict]) -> str:
    for m in turn_messages:
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"])
    return ""


@dataclass(frozen=True)
class BfclMultiTurnConfig:
    """Which BFCL multi-turn category to load and how to carve a train/eval split.

    ``category`` is the bfcl_eval name (e.g. ``multi_turn_base``); the installed
    package's ``VERSION_PREFIX`` is prepended when resolving data files.
    """

    category: str = "multi_turn_base"
    # BFCL has no official split, so reserve the last N ids for eval.
    eval_tail: int = 30
    # Bound tool observations stored in each training prefix.
    obs_limit: int = 1500


def _category_filename(category: str) -> str:
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX

    return f"{VERSION_PREFIX}_{category}.json"


class BfclMultiTurnDataset(DatasetConfig):
    """Loads a BFCL multi-turn category from the installed ``bfcl_eval`` package.

    Last :attr:`config`.eval_tail ids are held out as eval.
    """

    input_key: str = "messages"
    label_key: str = "label"
    # JSON-lines output (the gym default is parquet).
    output_format: str = "jsonl"
    writes_eval_paths: bool = False

    def __init__(
        self,
        split: str = "train",
        config: BfclMultiTurnConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self._split = split
        self.hf_split = split
        self.config = config if config is not None else BfclMultiTurnConfig()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _entries(self) -> list[dict]:
        return _load_jsonl(
            os.path.join(_data_dir(), _category_filename(self.config.category))
        )

    def _ground_truths(self) -> dict[str, list[list[str]]]:
        path = os.path.join(
            _data_dir(),
            "possible_answer",
            _category_filename(self.config.category),
        )
        return {e["id"]: e["ground_truth"] for e in _load_jsonl(path)}

    def _ids_for_split(self, ids: list[str]) -> list[str]:
        tail = self.config.eval_tail
        if self._split == "eval":
            return ids[-tail:] if tail else []
        if self._split == "train":
            return ids[:-tail] if tail else list(ids)
        return ids

    def _make_row(self, entry: dict, ground_truth: list[list[str]]) -> dict:
        turns = [
            {
                "user": _first_user_text(turn_messages),
                "calls": [parse_call_string(c) for c in calls],
            }
            for turn_messages, calls in zip(entry["question"], ground_truth)
        ]
        flattened_calls = [c for turn in turns for c in turn["calls"]]
        _, observations = replay(
            entry["involved_classes"], entry["initial_config"], flattened_calls
        )
        label = {
            "task_id": entry["id"],
            "initial_config": entry["initial_config"],
            "involved_classes": entry["involved_classes"],
            "excluded_function": entry.get("excluded_function", []),
            "turns": [{"user": t["user"], "calls": t["calls"]} for t in turns],
            "flattened_calls": flattened_calls,
            "observations": [str(o)[: self.config.obs_limit] for o in observations],
            "tool_schemas": load_func_docs(
                entry["involved_classes"], entry.get("excluded_function")
            ),
            "total_steps": len(flattened_calls),
        }
        messages = [
            {
                "role": "system",
                "content": "You are a tool-calling agent. Output a JSON tool call.",
            },
            {"role": "user", "content": turns[0]["user"] if turns else ""},
        ]
        return {"messages": messages, "label": json.dumps(label)}

    def _load_split(self) -> list[dict]:
        entries_by_id = {e["id"]: e for e in self._entries()}
        gt_by_id = self._ground_truths()
        ids = self._ids_for_split(list(entries_by_id.keys()))
        return [
            self._make_row(entries_by_id[i], gt_by_id[i])
            for i in ids
            if i in entries_by_id and i in gt_by_id and gt_by_id[i] and any(gt_by_id[i])
        ]

    def load(self, split: str = "all") -> list[dict]:
        if split in ("train", "eval"):
            self._split = split
            self.hf_split = split
        return self._load_split()

    def prepare(self, path: str, eval_paths: dict | None = None) -> None:
        """Write this instance's split to ``path``. ``eval_paths`` is ignored."""
        del eval_paths  # train/eval are separate DatasetConfig instances
        rows = self._load_split()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
