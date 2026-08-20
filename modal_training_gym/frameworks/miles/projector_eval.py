"""Held-out evaluation of a projector-only checkpoint.

A projector-only run trains supervised, with no rollout engine (see
:class:`~modal_training_gym.train_recipes.miles_recipe.glm_5_2.GLM_5_2_Projector_Recipe`),
so its loss is the only number it produces and miles' eval pass is rejected at
launch — the rollout function has nothing to generate with. That leaves the
question a projector run actually cares about unanswered: on proteins it never
trained on, does the frozen base read the projected vector well enough to name
the right answer?

This module answers it outside miles. It loads the projector checkpoint the run
wrote, rebuilds the module from the shape recorded in the checkpoint, loads the
same base model as a plain HF causal LM, and for each held-out row scores every
candidate answer by its mean token log-probability with the projected embedding
written into ``inputs_embeds`` at the row's position — the same merge the
training forward does. The prediction is the highest-scoring candidate.

Three numbers come out, and only their differences mean anything:

* the trained projector's accuracy,
* the same architecture untrained (identical init to what training started
  from), which is what "the projector contributes nothing" looks like, and
* the majority-class share of the eval split, which is what a model that
  ignores its input can score.

Results are written to the metadata volume as a normal
:class:`~modal_training_gym.common.eval.EvalResult`, so the run shows up in the
dashboard's evals view next to every other eval, with per-row predictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.ids import create_hash

#: Rows whose generated text is printed and stored for eyeballing. Generation is
#: much slower than scoring and adds nothing to the metric, so it is sampled.
_GENERATE_ROWS = 8

# Not ``HF_CACHE_PATH``: see the mount comment in ``ProjectorEval.run``.
_HF_CACHE_MOUNT = "/hf"


@dataclass
class ProjectorEvalRow:
    """One held-out row: what the model was asked, and what it answered."""

    prompt: str
    target: str
    prediction: str
    correct: bool
    scores: dict[str, float]
    generated: str = ""


@dataclass
class ProjectorEvalReport:
    """Accuracy of a projector checkpoint against its two baselines."""

    model_name: str
    checkpoint: str
    iteration: int
    classes: list[str]
    rows: list[ProjectorEvalRow] = field(default_factory=list)
    untrained_rows: list[ProjectorEvalRow] = field(default_factory=list)

    @staticmethod
    def _accuracy(rows: list[ProjectorEvalRow]) -> float:
        return sum(row.correct for row in rows) / len(rows) if rows else 0.0

    @property
    def accuracy(self) -> float:
        return self._accuracy(self.rows)

    @property
    def untrained_accuracy(self) -> float:
        return self._accuracy(self.untrained_rows)

    @property
    def majority_accuracy(self) -> float:
        """What predicting the eval split's most common class alone scores."""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.target] = counts.get(row.target, 0) + 1
        return max(counts.values()) / len(self.rows) if self.rows else 0.0

    def summary(self) -> str:
        if not self.classes:
            return "no classes"
        return (
            f"{self.checkpoint} (iteration {self.iteration}) on "
            f"{len(self.rows)} held-out rows, {len(self.classes)} classes: "
            f"trained {self.accuracy:.3f}, untrained {self.untrained_accuracy:.3f}, "
            f"majority class {self.majority_accuracy:.3f}, "
            f"uniform {1 / len(self.classes):.3f}"
        )

    def publish(self) -> str:
        """Save as an :class:`EvalResult` so the dashboard renders the run.

        Returns the ``eval_id``. Per-row predictions ride in ``metadata`` and the
        headline mean is accuracy, since each row scores 1 or 0.
        """
        from modal_training_gym.common.eval import (
            EvalConfigDurable,
            EvalResult,
            EvalRowResult,
        )

        eval_config_id = create_hash(
            "eval-config",
            "ProjectorEval",
            self.model_name,
            "projector_class_scoring",
            ",".join(self.classes),
        )
        EvalConfigDurable(
            eval_config_id=eval_config_id,
            dataset_name="projector-heldout",
            eval_fn_name="projector_class_scoring",
        ).save()
        result = EvalResult(
            eval_id=create_hash(
                "eval",
                eval_config_id,
                self.model_name,
                self.checkpoint,
                str(self.iteration),
            ),
            eval_config_id=eval_config_id,
            model_name=self.model_name,
            rows=[
                EvalRowResult(
                    score=1.0 if row.correct else 0.0,
                    response=row.prediction,
                    prompt=row.prompt,
                    metadata={
                        "target": row.target,
                        "generated": row.generated,
                        "scores": row.scores,
                        "checkpoint": self.checkpoint,
                        "iteration": self.iteration,
                        "untrained_accuracy": self.untrained_accuracy,
                        "majority_accuracy": self.majority_accuracy,
                    },
                )
                for row in self.rows
            ],
        )
        result.save()
        return result.eval_id


@dataclass
class ProjectorEval:
    """Run :func:`evaluate_projector` on Modal against a finished run's outputs.

    Takes the same three objects the training run took plus its
    ``training_run_id``, and reads what that run left on the volumes: the eval
    split the dataset's ``prepare()`` wrote, and the projector checkpoint the
    save hook wrote under the run's checkpoint directory. One GPU container, no
    cluster — the eval is a forward pass per (row, candidate) over a frozen
    base.
    """

    model: Any
    recipe: Any
    dataset: Any
    training_run_id: str
    gpu: str = ""
    timeout: int = 2 * 60 * 60
    checkpoint: str = ""

    def _volume_prefix(self) -> str:
        # Same derivation as build_miles_app, so this reads the volumes the run
        # wrote rather than creating empty ones next to them.
        return (
            getattr(self.recipe, "name", "")
            or f"miles-{type(self.recipe).__name__.lstrip('_').lower()}"
        )

    def _checkpoint_path(self) -> str:
        from modal_training_gym.train_recipes.base import CHECKPOINTS_PATH

        if self.checkpoint:
            return self.checkpoint
        save_dir = self.recipe.projector.save_dir or (
            f"{CHECKPOINTS_PATH}/{self.training_run_id}/projector"
        )
        return f"{save_dir}/projector_latest.pt"

    def _eval_path(self) -> str:
        from modal_training_gym.train_recipes.base import BaseTrainRecipe

        _train, eval_paths = BaseTrainRecipe._resolve_data_paths(self.dataset)
        if not eval_paths:
            raise TrainingGymConfigError(
                f"{type(self.dataset).__name__} writes no eval split, so there "
                "is nothing held out to evaluate on. Use a dataset whose "
                "prepare() materializes an eval path (writes_eval_paths=True)."
            )
        return eval_paths["eval"]

    def run(self, publish: bool = True) -> ProjectorEvalReport:
        import modal

        from modal_training_gym.common import hf_secrets
        from modal_training_gym.train_recipes.base import (
            CHECKPOINTS_PATH,
            DATA_PATH,
        )

        prefix = self._volume_prefix()
        image = (
            modal.Image.from_registry(self.recipe.docker_image)
            .entrypoint([])
            # The miles image ships a populated ``~/.cache/huggingface``, and
            # Modal refuses to mount a volume onto a non-empty path, so the
            # shared cache goes somewhere unused and HF_HOME is pointed at it.
            .env({"HF_HOME": _HF_CACHE_MOUNT})
            # The gym's id helper needs it and the miles image doesn't ship it,
            # so publishing to the dashboard fails at the last line otherwise.
            .pip_install("randomname")
            .add_local_python_source("modal_training_gym", copy=True)
        )
        app = modal.App(f"{prefix}-projector-eval", image=image)
        volumes = {
            _HF_CACHE_MOUNT: modal.Volume.from_name(
                "huggingface-cache", create_if_missing=True
            ),
            str(DATA_PATH): modal.Volume.from_name(
                f"{prefix}-data", create_if_missing=True
            ),
            str(CHECKPOINTS_PATH): modal.Volume.from_name(
                f"{prefix}-checkpoints", create_if_missing=True
            ),
        }
        gpu = (
            self.gpu or f"{self.recipe.gpu_type}:{self.recipe.actor_num_gpus_per_node}"
        )
        model_name = self.model.model_name
        checkpoint, eval_path = self._checkpoint_path(), self._eval_path()
        spec = self.recipe.projector

        @app.function(
            gpu=gpu,
            volumes=volumes,
            timeout=self.timeout,
            secrets=hf_secrets(),
            serialized=True,
            name="evaluate",
        )
        def evaluate() -> ProjectorEvalReport:
            report = evaluate_projector(
                model_name=model_name,
                checkpoint=checkpoint,
                eval_path=eval_path,
                embeddings_key=spec.embeddings_key,
                positions_key=spec.positions_key,
            )
            if publish:
                print(f"published eval {report.publish()}", flush=True)
            return report

        with app.run():
            return evaluate.remote()


def read_eval_rows(path: str, embeddings_key: str, positions_key: str) -> list[dict]:
    """Read the JSONL an ``EmbeddingProjectorDataset`` wrote for its eval split.

    The candidate answers are the distinct assistant turns in the file rather
    than a list passed in: the dataset already decided what the targets are, and
    a second declaration of them is a second thing to keep in sync.
    """
    rows: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages") or []
            metadata = record.get("metadata") or {}
            targets = [m for m in messages if m.get("role") == "assistant"]
            if not targets:
                raise TrainingGymConfigError(
                    f"{path} has a row with no assistant turn, so it declares no "
                    "answer to score against"
                )
            rows.append(
                {
                    "messages": [m for m in messages if m.get("role") != "assistant"],
                    "target": str(targets[-1].get("content", "")).strip(),
                    "embeddings": metadata[embeddings_key],
                    "positions": metadata[positions_key],
                }
            )
    if not rows:
        raise TrainingGymConfigError(f"{path} is empty")
    return rows


def load_projector(checkpoint: str, dtype: Any = None, trained: bool = True):
    """Rebuild the projector from ``checkpoint``'s recorded shape.

    ``trained=False`` returns the same architecture at the initialization the
    run started from — the baseline that isolates training from the mere
    presence of a vector in the embedding stream. The seed and output scale come
    from the checkpoint rather than from ``ProjectorSpec()``: a run that
    customized either started somewhere else, and comparing against the
    defaults' init would make the baseline a different model. Checkpoints
    written before those keys existed fall back to the defaults they were
    written under.
    """
    import torch

    from modal_training_gym.frameworks.miles.embedding_projector import (
        EmbeddingProjector,
        init_projector,
    )
    from modal_training_gym.frameworks.miles.projector_config import ProjectorSpec

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    projector = EmbeddingProjector(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
    )
    if trained:
        projector.load_state_dict(state["state_dict"])
    else:
        defaults = ProjectorSpec()
        init_projector(
            projector,
            int(config.get("init_seed", defaults.init_seed)),
            float(config.get("output_scale", defaults.output_scale)),
        )
    if dtype is not None:
        projector = projector.to(dtype)
    projector.eval()
    return projector, int(state.get("iteration", 0))


def _candidate_scores(model, tokenizer, prompt_ids, inputs_embeds, candidates):
    """Mean log-probability of each candidate answer after the prompt.

    Mean rather than sum: the candidates are class names of different token
    lengths ("nucleus" against "endoplasmic reticulum"), and a summed
    log-probability would rank by brevity as much as by fit.
    """
    import torch

    embed = model.get_input_embeddings()
    scores: dict[str, float] = {}
    for candidate in candidates:
        answer_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        answer = torch.tensor([answer_ids], device=inputs_embeds.device)
        full = torch.cat([inputs_embeds, embed(answer).to(inputs_embeds.dtype)], dim=1)
        with torch.no_grad():
            logits = model(
                inputs_embeds=full,
                attention_mask=torch.ones(full.shape[:2], device=full.device),
            ).logits
        # Position of the logits that predict the answer's first token: the last
        # prompt position.
        start = prompt_ids.shape[1] - 1
        logprobs = torch.log_softmax(
            logits[0, start : start + len(answer_ids)].float(), dim=-1
        )
        total = logprobs[torch.arange(len(answer_ids)), answer[0]].sum().item()
        scores[candidate] = total / len(answer_ids)
    return scores


def evaluate_projector(
    model_name: str,
    checkpoint: str,
    eval_path: str,
    embeddings_key: str,
    positions_key: str,
    generate_rows: int = _GENERATE_ROWS,
) -> ProjectorEvalReport:
    """Score a projector checkpoint on a held-out split. Runs on a GPU container."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_eval_rows(eval_path, embeddings_key, positions_key)
    classes = sorted({row["target"] for row in rows})
    if len(classes) < 2:
        raise TrainingGymConfigError(
            f"{eval_path} has a single distinct answer ({classes}), so accuracy "
            "on it says nothing"
        )
    print(f"{len(rows)} held-out rows, {len(classes)} classes: {classes}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    embed = model.get_input_embeddings()

    trained, iteration = load_projector(checkpoint, torch.bfloat16, trained=True)
    untrained, _ = load_projector(checkpoint, torch.bfloat16, trained=False)
    trained, untrained = trained.to(device), untrained.to(device)

    report = ProjectorEvalReport(
        model_name=model_name,
        checkpoint=checkpoint,
        iteration=iteration,
        classes=classes,
    )

    for index, row in enumerate(rows):
        rendered = tokenizer.apply_chat_template(
            row["messages"], add_generation_prompt=True, tokenize=False
        )
        prompt_ids = torch.tensor(
            [tokenizer(rendered, add_special_tokens=False)["input_ids"]], device=device
        )
        vectors = torch.tensor(row["embeddings"], dtype=torch.bfloat16, device=device)
        positions = list(row["positions"])
        if max(positions) >= prompt_ids.shape[1]:
            raise TrainingGymConfigError(
                f"row {index} places an embedding at token {max(positions)} of a "
                f"{prompt_ids.shape[1]}-token prompt"
            )

        for projector, bucket in (
            (trained, report.rows),
            (untrained, report.untrained_rows),
        ):
            inputs_embeds = embed(prompt_ids).clone()
            with torch.no_grad():
                projected = projector(vectors)
            for offset, position in enumerate(positions):
                inputs_embeds[0, position, :] = projected[offset].to(
                    inputs_embeds.dtype
                )
            scores = _candidate_scores(
                model, tokenizer, prompt_ids, inputs_embeds, classes
            )
            prediction = max(scores, key=lambda candidate: scores[candidate])
            generated = ""
            if bucket is report.rows and index < generate_rows:
                with torch.no_grad():
                    out = model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=torch.ones(
                            inputs_embeds.shape[:2], device=device
                        ),
                        max_new_tokens=12,
                        do_sample=False,
                    )
                generated = tokenizer.decode(out[0], skip_special_tokens=True)
            bucket.append(
                ProjectorEvalRow(
                    prompt=rendered,
                    target=row["target"],
                    prediction=prediction,
                    correct=prediction == row["target"],
                    scores=scores,
                    generated=generated,
                )
            )

        last = report.rows[-1]
        print(
            f"row {index}: target={last.target!r} prediction={last.prediction!r} "
            f"untrained={report.untrained_rows[-1].prediction!r}"
            + (f" generated={last.generated!r}" if last.generated else ""),
            flush=True,
        )

    print(report.summary(), flush=True)
    return report
