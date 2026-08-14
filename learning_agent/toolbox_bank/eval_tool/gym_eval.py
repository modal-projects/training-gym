#!/usr/bin/env python3
"""gym_eval — score a dev set on Modal through the Training Gym, on its dashboard.

`rubric_eval.py` gives you a number and a JSON file. This gives you the same
number, judged by the same canonical judge (gpt-5.6-luna via `judge_client`),
plus a dashboard row per question: the prompt, the task model's answer, its score,
its per-claim verdicts, and — if your harness returns a transcript — the whole
search conversation, rendered turn by turn. Two runs (base and trained) sit
side by side on the Evals tab, which is the comparison the margin *is*.

It serves the model itself: point it at an HF id or a `/out/models/<tag>/merged`
checkpoint and it brings up an SGLang endpoint through the gym, evaluates, and
tears nothing down (deployments idle out) so you can query it afterwards.

    # base, through your harness, on the task's dev set
    python3 toolbox/eval_tool/gym_eval.py --dev task/dev.json \
        --model "$TASK_MODEL" --harness harness.py:answer --label base

    # the checkpoint you just trained
    python3 toolbox/eval_tool/gym_eval.py --dev task/dev.json \
        --checkpoint /out/models/gym-grpo-1/merged --harness harness.py:answer --label trained

The harness callable is `fn(deployment, question) -> str | dict`. Return a dict
`{"answer": ..., "messages": [...]}` to get the transcript on the dashboard;
a bare string is treated as the answer. Evaluating without a harness is
allowed but it is not what `submission/eval.py` does — the numbers will not
transfer.

Needs `pip install -e toolbox/training_tool/training_gym` (the pinned SDK clone), Modal credentials, and `LEARNING_AGENT_JUDGE_URL`
for canonical judging. `--dry-run` needs none of them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

# Toolbox bootstrap (standard, identical in every CLI script): put the toolbox
# root on sys.path so absolute package imports resolve when run as a script.
_TOOLBOX_ROOT = next((str(p) for p in Path(__file__).resolve().parents if p.name in ("toolbox", "toolbox_bank")), None)
if _TOOLBOX_ROOT is None:
    raise SystemExit(f"toolbox bootstrap: no toolbox ancestor above {Path(__file__).resolve()}")
if _TOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, _TOOLBOX_ROOT)


def resolve_model_config(model_name: str):
    """Find the gym ModelConfig whose `model_name` matches this HF id."""
    from modal_training_gym.common import models

    for attr in dir(models):
        candidate = getattr(models, attr)
        configured = getattr(candidate, "model_name", None)
        if isinstance(configured, str) and configured.lower() == model_name.lower():
            return candidate()
    known = sorted({name for attr in dir(models)
                    if isinstance(name := getattr(getattr(models, attr), "model_name", None), str) and name})
    raise SystemExit(
        f"no gym ModelConfig for {model_name!r}. Known: {', '.join(known)}. "
        "Add one under modal_training_gym/common/models/ in the pinned clone.")


def load_rows(path: Path) -> list[dict]:
    """Read `{id, question, gold_answer, rubric}` rows (JSON array or JSONL)."""
    text = path.read_text()
    rows = (
        json.loads(text)
        if text.lstrip().startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    for i, row in enumerate(rows):
        if not row.get("question") or not row.get("rubric"):
            raise SystemExit(f"{path}:{i}: rows need 'question' and a non-empty 'rubric'")
    return rows


def load_callable(spec: str):
    """Import `path/to/file.py:function`."""
    if ":" not in spec:
        raise SystemExit(f"expected <file.py>:<function>, got {spec!r}")
    target, name = spec.rsplit(":", 1)
    file_path = Path(target).resolve()
    if not file_path.is_file():
        raise SystemExit(f"no such file: {file_path}")
    module_spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot import {file_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[file_path.stem] = module
    module_spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError:
        raise SystemExit(f"{target} has no attribute {name!r}") from None


def bootstrap_ci95(values: list[float], resamples: int = 10000, seed: int = 0) -> list[float]:
    """Seeded percentile bootstrap 95% CI — same construction as rubric_eval.py,
    so the two tools' intervals are comparable."""
    import random

    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return [means[int(0.025 * resamples)], means[int(0.975 * resamples)]]


def build_eval_fn(rows: list[dict], harness, n_votes: int, task: str | None):
    """One dashboard row per question: judged score + the evidence behind it."""
    from api_clients import judge_client
    from modal_training_gym import EvalRowResult

    by_question = {row["question"]: row for row in rows}

    def eval_fn(deployment, example) -> EvalRowResult:
        row = by_question[example["question"]]
        if harness is None:
            episode = {"answer": deployment.generate(row["question"])}
        else:
            produced = harness(deployment, row["question"])
            episode = produced if isinstance(produced, dict) else {"answer": produced}

        answer = episode.get("answer") or ""
        verdict = judge_client.judge_claims(
            row["question"],
            answer,
            row["rubric"],
            gold_answer=row.get("gold_answer"),
            task=task,
            n_votes=n_votes,
        )
        score = verdict["claim_score"]
        metadata = {
            # Numeric metadata keys become per-row metrics on the dashboard.
            "judge_votes": n_votes,
            "canonical": int(bool(verdict.get("canonical"))),
            "judge_model": verdict.get("model"),
            "per_claim": {
                cid: claim.get("final") for cid, claim in verdict["per_claim"].items()
            },
            "reference": row.get("gold_answer", ""),
            "row_id": row.get("id", ""),
        }
        if episode.get("messages"):
            # Rendered as a conversation in the dashboard's row inspector.
            metadata["trajectory_messages"] = episode["messages"]
        for key in ("turns", "searches", "tool_calls"):
            if key in episode:
                metadata[key] = episode[key]
        return EvalRowResult(
            # A row the judge could not score is a failure, not a zero — it is
            # recorded as one and excluded from the mean below.
            score=0.0 if score is None else float(score),
            prompt=row["question"],
            response=answer,
            metadata=metadata | {"judged": int(score is not None)},
        )

    return eval_fn


def lab_out_checkpoint(path: str):
    """Describe a `/out/models/<tag>/merged` directory as a gym Checkpoint.

    Trained weights live on the `lab-out` volume in HF format; the gym
    normally reads checkpoints off its own per-app volume, so the only thing
    it needs is which volume, mounted where.
    """
    from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType

    return Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name=Path(path).parent.name,
        path=path,
        timestamp=0.0,
        checkpoints_volume_name="lab-out",
        checkpoints_mount_path="/out",
    )


def serve(args):
    """Bring up an SGLang endpoint for the model or checkpoint under test."""
    from modal_training_gym import DeploymentConfig, list_checkpoints

    checkpoint = None
    if args.checkpoint:
        checkpoint = lab_out_checkpoint(args.checkpoint)
    elif args.training_run_id:
        found = list_checkpoints(args.training_run_id)
        if not found:
            raise SystemExit(f"no checkpoints for training run {args.training_run_id}")
        checkpoint = found[-1]

    return DeploymentConfig(
        model=resolve_model_config(args.model),
        checkpoint=checkpoint,
        app_name=f"lab-eval-{args.label}",
        served_model_name=args.label,
        unauthenticated=True,
    ).serve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Judge a dev set on the Training Gym dashboard with the canonical the operator judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dev", required=True, help="dev/test rows: {id, question, gold_answer, rubric}")
    ap.add_argument("--model", default=None, help="HF id of the base model (defaults to $TASK_MODEL)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint path on the lab-out volume to serve instead of "
                         "the base weights, e.g. /out/models/<tag>/merged")
    ap.add_argument("--training-run-id", default=None,
                    help="serve the last checkpoint of a gym training run instead "
                         "(skips the export to lab-out)")
    ap.add_argument("--harness", default=None,
                    help="fn(deployment, question) -> str | {answer, messages, ...} as <file.py>:<fn>")
    ap.add_argument("--label", default="task model", help="name for the deployment and the eval row")
    ap.add_argument("--task", default=None, help="task name, passed to the judge prompt")
    ap.add_argument("--votes", type=int, default=3, help="judge votes per claim (majority)")
    ap.add_argument("--concurrency", type=int, default=4, help="questions evaluated in parallel")
    ap.add_argument("--out", default=None, help="also write the results JSON here")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no Modal, no judge")
    args = ap.parse_args()

    import os

    args.model = args.model or os.environ.get("TASK_MODEL")
    if not args.model:
        raise SystemExit("--model or $TASK_MODEL is required (the architecture to serve)")

    rows = load_rows(Path(args.dev))
    print(f"dev: {len(rows)} questions from {args.dev}")
    if args.dry_run:
        served = args.checkpoint or args.training_run_id or args.model
        print(f"serve:   {served} as {args.label!r}")
        print(f"harness: {args.harness or 'NONE — bare chat, not what submission/eval.py does'}")
        print(f"judge:   gpt-5.6-luna, {args.votes} votes/claim")
        return

    if not args.harness:
        print("warning: no --harness; evaluating bare chat completion. "
              "the operator scores through the task model's full harness — these numbers will not transfer.",
              file=sys.stderr)

    from modal_training_gym import DatasetConfig, EvalConfig

    class DevSet(DatasetConfig):
        input_key = "question"
        label_key = "label"
        always_prepare = True
        rows: list[dict] = []

        def load(self, split="all"):
            return self.rows

    harness = load_callable(args.harness) if args.harness else None
    deployment = serve(args)
    result = EvalConfig(
        dataset=DevSet(rows=[{"question": row["question"]} for row in rows]),
        eval_fn=build_eval_fn(rows, harness, args.votes, args.task),
        prompt_column="question",
    ).evaluate(deployment, max_concurrency=args.concurrency)

    judged = [row.score for row in result.rows if row.metadata.get("judged")]
    unjudged = len(result.rows) - len(judged)
    mean = statistics.fmean(judged) if judged else float("nan")
    lo, hi = bootstrap_ci95(judged)
    print(f"[{args.label}] mean {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"n={len(judged)}" + (f"  ({unjudged} unjudged)" if unjudged else ""))
    print("dashboard: `training-gym open` → Evals")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "label": args.label,
            "mean": mean,
            "ci95": [lo, hi],
            "n": len(judged),
            "unjudged": unjudged,
            "per_question": [
                {"question": row.prompt, "answer": row.response,
                 "score": row.score, **row.metadata}
                for row in result.rows
            ],
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
