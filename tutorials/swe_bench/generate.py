"""slime per-sample rollout hook (--custom-generate-function-path). One call = one episode:
boot the task sandbox, run the unmodified mini-swe harness through a token-recording model,
build a token-faithful training Sample, and grade it.
"""

import asyncio
import concurrent.futures
import logging
import random
import threading
import time
import uuid
from typing import Any

from .env import (
    SWE_INSTANCE_TEMPLATE,
    SWE_OBSERVATION_TEMPLATE,
    SWE_SYSTEM_TEMPLATE,
    SweEnvironment,
    SweEnvironmentConfig,
    grade_swe_patch,
)
from .environment import MiniSweEnvironmentAdapter
from .qwen3_model import Qwen3RecordingModel

logger = logging.getLogger("modal_training_gym.slime.swe_agent")

# Per-instance boot-failure tally. A task whose image won't build returns ABORTED and slime
# requeues the same group forever; after the cap we ship a masked reward-0 sample to drop it.
_boot_fails: dict[str, int] = {}


# Episodes are long I/O-bound chains; asyncio.to_thread caps at min(32, cpu+4) threads, which
# throttles in-flight episodes regardless of sglang_server_concurrency. Use a wide dedicated pool.
_episode_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _episode_executor(args) -> concurrent.futures.ThreadPoolExecutor:
    global _episode_pool
    if _episode_pool is None:
        with _pool_lock:
            if _episode_pool is None:
                from slime.utils.http_utils import get_rollout_num_engines

                n = getattr(
                    args, "sglang_server_concurrency", 128
                ) * get_rollout_num_engines(args)
                n = min(max(n, 32), 2048)
                _episode_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=n, thread_name_prefix="episode"
                )
                logger.info("agentic episode thread pool: max_workers=%d", n)
    return _episode_pool


def _run_episode(
    task: dict,
    tokenizer,
    sampling_params,
    router_url: str,
    limits: dict,
    session_id: str,
    abort_check=None,
) -> tuple[float, Qwen3RecordingModel, dict]:
    """Run stock mini-swe in a sandbox. Never raises; returns (reward, model, stats)."""
    from minisweagent.agents.default import DefaultAgent

    # Cold-start ramp: jitter each episode's start so the thundering herd of sandbox-boots + first queries
    # doesn't burst-overwhelm Modal's sandbox control plane (exec HTTPErrors ~1024 concurrent) or the sgl-
    # router (503s). Spreading the start over a window lets us run HIGH steady-state concurrency instead of
    # retreating to a low cap (GLM's fix). Held inside the sglang semaphore, so it also throttles the burst.
    if (ramp := limits.get("ramp_window", 0.0)) > 0:
        time.sleep(random.uniform(0.0, ramp))

    model = Qwen3RecordingModel(
        tokenizer,
        sampling_params,
        router_url,
        SWE_OBSERVATION_TEMPLATE,
        session_id,
        query_timeout=limits["query_timeout"],
        max_context_len=limits["max_context_len"],
        abort_check=abort_check,
    )
    patch, reward, solved, exit_status, grade_time = None, 0.0, 0.0, "none", 0.0
    harness_error = False
    sandbox = None
    t0 = time.perf_counter()
    try:
        # lifetime must cover the full agent run: the wall limit (episode_timeout) is checked BETWEEN steps,
        # so a slow FINAL turn can run another query_timeout past it → max episode ≈ episode_timeout +
        # query_timeout. Grading runs in its own fresh sandbox, so it's not this sandbox's concern. Using
        # grade_timeout (600) here under-sized it, reaping long episodes mid-final-turn ("already shut down").
        environment = SweEnvironment.create(
            task,
            config=SweEnvironmentConfig(
                lifetime=limits["episode_timeout"] + limits["query_timeout"] + 300,
                exec_timeout=limits["exec_timeout"],
                boot_retries=2,
            ),
            lifetime=limits["episode_timeout"] + limits["query_timeout"] + 300,
        )
        sandbox = MiniSweEnvironmentAdapter(environment)
        agent = DefaultAgent(
            model,
            sandbox,
            system_template=SWE_SYSTEM_TEMPLATE,
            instance_template=SWE_INSTANCE_TEMPLATE,
            step_limit=limits["max_steps"],
            cost_limit=0.0,
            wall_time_limit_seconds=limits["episode_timeout"],
        )
        sandbox.deadline = time.monotonic() + limits["episode_timeout"]
        try:
            exit_info = agent.run(task=task["problem_statement"]) or {}
        finally:
            sandbox.deadline = None
        exit_status = exit_info.get("exit_status", "?")
        patch = (
            exit_info.get("submission") or ""
        )  # agent's curated source-only patch (on Submitted)
        if not patch.strip():
            patch = environment.capture_patch()
    except Exception:
        logger.exception("episode failed (instance=%s)", task.get("instance_id"))
    finally:
        if sandbox is not None:
            sandbox.terminate()  # grading runs in a fresh, clean sandbox

    if patch is not None and not model.aborted:
        g0 = time.perf_counter()
        try:
            verdict = grade_swe_patch(
                task,
                patch,
                timeout=limits["grade_timeout"],
            )
            reward = float(verdict.passed)
            solved = float(verdict.passed)
            harness_error = verdict.harness_error
        except Exception:
            logger.exception("grading failed (instance=%s)", task.get("instance_id"))
            harness_error = True
        grade_time = time.perf_counter() - g0

    stats = {
        "turns": len(model.versions),  # productive (non-format-error) turns
        "hit_step_cap": float(
            len(model.versions) >= limits["max_steps"]
        ),  # ran out of turns (vs other exits)
        "n_calls": model.n_calls,  # total model calls; n_calls - turns = the format-error tax
        "format_errors": model.n_format_errors,
        "resumed_turns": model.resumed_turns,  # turns re-issued after a weight-sync abort
        "length_truncations": model.n_length_truncations,  # turns the model's output was cut at the per-turn cap
        "exit_status": exit_status,
        "solved": float(solved),
        "harness_error": harness_error,
        "gen_time": round(model.gen_time, 1),
        "exec_time": round(sandbox.exec_time if sandbox is not None else 0.0, 1),
        "exec_timeouts": sandbox.exec_timeouts if sandbox is not None else 0,
        "boot_time": round(sandbox.boot_time if sandbox is not None else 0.0, 1),
        "grade_time": round(grade_time, 1),
        "episode_time": round(time.perf_counter() - t0, 1),
        "output_tokens": sum(model.loss_mask),
        "response_tokens": len(model.tokens) - (model.prompt_len or len(model.tokens)),
        "reasoning_tokens": model.reasoning_tokens,
        "total_length": len(
            model.tokens
        ),  # full trajectory length → context utilization vs the 128k window
        "prefix_cache_hit": round(model.cached_tokens / model.input_tokens, 3)
        if model.input_tokens
        else 0.0,
        "gen_timestamp": time.time(),  # for sample-age (staleness) at train time
    }
    return reward, model, stats


def _ship_masked_sample(
    sample: Any, tokenizer, problem_statement: str, reason: str
) -> Any:
    """A valid but fully-masked reward-0 sample for a permanently-failing task: ships to leave
    the buffer, contributes no gradient (slime zeros the loss mask via remove_sample)."""
    from slime.utils.types import Sample

    ptoks = tokenizer.encode(problem_statement or "", add_special_tokens=False)[:512]
    eos = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else (ptoks[-1] if ptoks else 0)
    )
    sample.tokens = ptoks + [eos]
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.weight_versions = []
    sample.reward = 0.0
    sample.status = Sample.Status.COMPLETED
    sample.remove_sample = True
    sample.metadata = {
        **sample.metadata,
        "agentic": {"exit_status": reason, "turns": 0, "gen_timestamp": time.time()},
    }
    return sample


def _recycle_or_drop(args, sample, tokenizer, problem_statement, reason):
    """A status=ABORTED sample means 'requeue me'. The fully-async orchestrator honors that; the SYNC
    generate_rollout ships EVERY returned sample straight to training, where a status=ABORTED sample's
    reward=None crashes _post_process_rewards (torch.tensor(None)). So: async → ABORTED (recycle);
    sync → a masked reward-0 sample (valid reward, remove_sample=True so it contributes no gradient)."""
    from slime.utils.types import Sample

    if "fully_async" in getattr(args, "rollout_function_path", ""):
        sample.status = Sample.Status.ABORTED
        return sample
    return _ship_masked_sample(sample, tokenizer, problem_statement, reason)


async def generate(args, sample: Any, sampling_params, evaluation: bool = False):
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample

    state = GenerateState(args)
    if state.aborted:
        return _recycle_or_drop(
            args,
            sample,
            state.tokenizer,
            sample.prompt if isinstance(sample.prompt, str) else "",
            "StateAborted",
        )

    task = dict(sample.metadata)
    task["problem_statement"] = (
        sample.prompt if isinstance(sample.prompt, str) else task["problem_statement"]
    )
    limits = {
        "max_steps": getattr(args, "agentic_max_steps", 20),
        "episode_timeout": getattr(args, "agentic_episode_timeout", 1800),
        "exec_timeout": getattr(args, "agentic_exec_timeout", 120),
        "grade_timeout": getattr(args, "agentic_grade_timeout", 1800),
        "query_timeout": getattr(
            args, "agentic_query_timeout", 600
        ),  # per-turn cap; bounds hung generations
        "max_context_len": getattr(
            args, "rollout_max_context_len", 131072
        ),  # served window; per-turn gen cap
        "ramp_window": getattr(
            args, "agentic_ramp_window", 0.0
        ),  # cold-start stagger (s); spread the herd
    }
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    session_id = sample.session_id or str(
        uuid.uuid4()
    )  # pin an episode's turns to one worker (prefix cache)

    loop = asyncio.get_running_loop()
    kill_switch = (
        threading.Event()
    )  # hard-cap reaper: flips the abandoned episode thread's abort probe
    fut = loop.run_in_executor(
        _episode_executor(args),
        _run_episode,
        task,
        state.tokenizer,
        dict(sampling_params or state.sampling_params),
        router_url,
        limits,
        session_id,
        # per-turn abort probe: surplus episodes (sync abort window) and hard-cap orphans exit at their next
        # turn (RolloutAborted) instead of running to completion — the live state flag alone is unreliable
        # for orphans because state.reset() clears it at the next rollout while the thread is still alive.
        lambda: state.aborted or kill_switch.is_set(),
    )
    # Hard wall-cap on the whole episode. mini-swe's wall_time_limit only fires between turns, so a single
    # slow/streaming generation on a congested engine can overshoot it by hours, poisoning the batch with
    # extreme staleness. Cap it here from the rollout side and recycle; the orphan thread unwinds on its own.
    # A legitimate episode can use its wall limit, one final generation, and
    # one patched grading pass.
    hard_cap = (
        limits["episode_timeout"]
        + limits["query_timeout"]
        + limits["grade_timeout"]
        + 120
    )
    try:
        reward, model, stats = await asyncio.wait_for(fut, timeout=hard_cap)
    except asyncio.TimeoutError:
        kill_switch.set()  # reap the orphan thread at its next turn — else it outlives the abort window and
        # can inject requests into the next pause→flush_cache (engine never idle → flush timeout → crash)
        logger.warning(
            "episode exceeded hard cap %ds (instance=%s); recycling/dropping",
            hard_cap,
            task.get("instance_id"),
        )
        return _recycle_or_drop(
            args,
            sample,
            state.tokenizer,
            task.get("problem_statement", ""),
            "HardCapTimeout",
        )

    key = sample.label or task.get("instance_id") or ""

    if model.aborted:  # abort probe fired mid-episode → recycle (async) / drop (sync)
        return _recycle_or_drop(
            args,
            sample,
            state.tokenizer,
            task.get("problem_statement", ""),
            "RolloutAborted",
        )

    # No usable trajectory (boot failed / every turn rolled back) or the grading harness itself
    # failed: retry a few times for transient failures, then ship a masked sample so the group
    # leaves the buffer instead of recycling forever.
    if (
        model.prompt_len is None
        or len(model.tokens) <= model.prompt_len
        or stats["harness_error"]
    ):
        _boot_fails[key] = _boot_fails.get(key, 0) + 1
        if _boot_fails[key] <= getattr(args, "agentic_max_boot_retries", 3):
            return _recycle_or_drop(
                args,
                sample,
                state.tokenizer,
                task.get("problem_statement", ""),
                "BootRetry",
            )
        logger.warning(
            "instance %s unusable after %d boot failures; dropping",
            key,
            _boot_fails[key],
        )
        return _ship_masked_sample(
            sample, state.tokenizer, task.get("problem_statement", ""), "ImageUnusable"
        )

    _boot_fails.pop(key, None)
    sample.tokens = model.tokens
    sample.response_length = len(model.tokens) - model.prompt_len
    sample.response = state.tokenizer.decode(
        model.tokens[model.prompt_len :],
        skip_special_tokens=False,
    )
    sample.loss_mask = model.loss_mask[model.prompt_len :]
    sample.rollout_log_probs = model.logprobs[model.prompt_len :]
    sample.weight_versions = [v for v in model.versions if v is not None]
    sample.reward = reward
    sample.status = Sample.Status.COMPLETED
    sample.prefix_cache_info.cached_tokens = model.cached_tokens
    sample.prefix_cache_info.total_prompt_tokens = model.input_tokens
    sample.metadata = {**sample.metadata, "agentic": stats}
    return sample
