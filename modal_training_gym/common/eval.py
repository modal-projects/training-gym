from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from modal_training_gym.common.dataset import DatasetRow
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.deployment import ModelDeployment


def _new_eval_config_id() -> str:
    from uuid import uuid4

    return f"eval-config-{uuid4().hex[:12]}"


def _callable_name(fn: Callable[..., Any]) -> str:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if name:
        return name
    return type(fn).__name__


class EvalConfigDurable(BaseModel):
    """JSON-serializable audit record for an :class:`EvalConfig`."""

    eval_config_id: str
    dataset_name: str
    eval_fn_name: str
    generate_kwargs: dict[str, Any] = Field(default_factory=dict)

    def save(self) -> None:
        vol_put(
            MetadataStore.EVAL_CONFIGS,
            self.eval_config_id,
            self.model_dump(mode="json"),
        )

    @classmethod
    def from_id(cls, eval_config_id: str) -> "EvalConfigDurable":
        return cls.model_validate(vol_get(MetadataStore.EVAL_CONFIGS, eval_config_id))

    @classmethod
    def list_configs(cls) -> list["EvalConfigDurable"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_CONFIGS)]


class EvalRowResult(BaseModel):
    """One evaluated row: score, response text, and optional metadata."""

    score: float
    response: str = ""
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )  # metadata that user can inject about the evaluation result


EvalFn = Callable[[DatasetRow, str], EvalRowResult]


class EvalResult(BaseModel):
    """Saved results for one evaluation run across a dataset."""

    eval_id: str
    eval_config_id: str
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC)
    )
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
    eval_config_id: str = field(default_factory=_new_eval_config_id)
    generate_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_durable(self) -> EvalConfigDurable:
        return EvalConfigDurable(
            eval_config_id=self.eval_config_id,
            dataset_name=type(self.dataset).__name__,
            eval_fn_name=_callable_name(self.eval_fn),
            generate_kwargs=self.generate_kwargs,
        )

    def save(self) -> EvalConfigDurable:
        durable = self.to_durable()
        durable.save()
        return durable

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
        from uuid import uuid4

        self.save()
        deployment.wait_until_ready()

        results: list[EvalRowResult] = []

        for idx, example in enumerate(self.dataset.load(), start=1):
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
            eval_config_id=self.eval_config_id,
            rows=results,
        )
        result.save()
        return result
