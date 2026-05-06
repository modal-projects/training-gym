from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from modal_training_gym.common.dataset import DatasetRow
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.deployment import ModelDeployment


EVAL_RESULTS_STORE_NAME = MetadataStore.EVAL_RESULTS.value


class EvalRowResult(BaseModel):
    score: float
    response: str = ""
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # metadata that user can inject about the evaluation result


EvalFn = Callable[[DatasetRow, str], EvalRowResult]


class EvalResult(BaseModel):
    eval_id: str
    created_at: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    rows: list[EvalRowResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def mean(self) -> float:
        return sum(r.score for r in self.rows) / self.total if self.total else 0.0

    def save(self) -> None:
        vol_put(MetadataStore.EVAL_RESULTS, self.eval_id, self.model_dump(mode="json"))

    @classmethod
    def from_id(cls, eval_id: str) -> "EvalResult":
        return cls.model_validate(vol_get(MetadataStore.EVAL_RESULTS, eval_id))

    @classmethod
    def list_results(cls) -> list["EvalResult"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_RESULTS)]


@dataclass
class EvalConfig:
    """Evaluate a deployed model on a dataset config.

    The dataset must expose ``load()`` and return iterable dict examples.
    """

    dataset: "DatasetConfig"
    eval_fn: EvalFn
    generate_kwargs: dict[str, Any] = field(default_factory=dict)

    def build_prompt(self, row: DatasetRow) -> str:
        input_column = getattr(self.dataset, "input_column", "")
        if not input_column:
            raise ValueError(
                "EvalConfig.build_prompt() requires dataset.input_column to be set."
            )
        raw = str(row[input_column])
        template = getattr(self.dataset, "prompt_template", "")
        if template:
            return template.format(input=raw)
        return raw

    def evaluate(
        self, deployment: "ModelDeployment", debug: bool = False
    ) -> EvalResult:
        import time
        from uuid import uuid4

        deployment.wait_until_ready()

        results: list[EvalRowResult] = []

        for idx, example in enumerate(self.dataset.load(), start=1):
            if debug:
                print(f"Evaluating example {idx}: {example}", flush=True)
            text = deployment.generate(
                self.build_prompt(example),
                **self.generate_kwargs,
            )
            result = self.eval_fn(example, text)
            if not result.response:
                result = EvalRowResult(
                    score=result.score,
                    response=text,
                    metadata=result.metadata,
                )
            if debug:
                print(
                    f"Finished example {idx}: "
                    f"response={result.response!r} score={result.score}",
                    flush=True,
                )
            results.append(result)

        result = EvalResult(
            eval_id=f"eval-{uuid4().hex[:12]}",
            created_at=int(time.time()),
            config={
                "deployment": {
                    "app_name": getattr(deployment.deployment_config, "app_name", ""),
                    "url": getattr(deployment, "url", ""),
                    "served_model_name": getattr(
                        deployment.deployment_config, "served_model_name", ""
                    ),
                    "model_name": getattr(
                        deployment.deployment_config.model, "model_name", ""
                    ),
                },
                "dataset": type(self.dataset).__name__,
                "eval_fn": getattr(self.eval_fn, "__name__", str(self.eval_fn)),
                "generate_kwargs": self.generate_kwargs,
            },
            rows=results,
        )
        result.save()
        return result
