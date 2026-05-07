from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import datetime
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from modal_training_gym.common.dataset import DatasetRow
from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list, vol_put

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.deployment import ModelDeployment

EVAL_SUMMARY_STORE = MetadataStore.EVALS
EVAL_SUMMARY_KEY = "summary"
EVAL_SUMMARY_PAYLOAD_KEY = "summaries"


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
    prompt_column: str | None = None
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


class EvalSummary(BaseModel):
    eval_id: str
    eval_config_id: str
    created_at: datetime.datetime
    total: int
    mean: float

    @classmethod
    def list_summaries(cls) -> list["EvalSummary"]:
        try:
            payload = vol_get(EVAL_SUMMARY_STORE, EVAL_SUMMARY_KEY)
        except KeyError:
            return []
        summaries = (
            payload.get(EVAL_SUMMARY_PAYLOAD_KEY, [])
            if isinstance(payload, dict)
            else payload
        )
        if not isinstance(summaries, list):
            return []
        return [cls.model_validate(summary) for summary in summaries]

    @classmethod
    def save_summaries(cls, summaries: list["EvalSummary"]) -> None:
        vol_put(
            EVAL_SUMMARY_STORE,
            EVAL_SUMMARY_KEY,
            {
                EVAL_SUMMARY_PAYLOAD_KEY: [
                    summary.model_dump(mode="json") for summary in summaries
                ]
            },
        )


class EvalResult(BaseModel):
    """Saved results for one evaluation run across a dataset."""

    eval_id: str
    eval_config_id: str
    deployment_id: str
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

    def to_summary(self) -> EvalSummary:
        return EvalSummary(
            eval_id=self.eval_id,
            eval_config_id=self.eval_config_id,
            created_at=self.created_at,
            total=self.total,
            mean=self.mean,
        )

    def save(self) -> None:
        vol_put(MetadataStore.EVAL_RESULTS, self.eval_id, self.model_dump(mode="json"))
        summaries = EvalSummary.list_summaries()
        summaries_by_id = {summary.eval_id: summary for summary in summaries}
        summaries_by_id[self.eval_id] = self.to_summary()
        EvalSummary.save_summaries(
            sorted(
                summaries_by_id.values(),
                key=lambda summary: summary.created_at,
                reverse=True,
            )
        )

    @classmethod
    def from_id(cls, eval_id: str) -> "EvalResult":
        return cls.model_validate(vol_get(MetadataStore.EVAL_RESULTS, eval_id))

    @classmethod
    def list_results(cls) -> list["EvalResult"]:
        return [cls.model_validate(v) for v in vol_list(MetadataStore.EVAL_RESULTS)]



Response = str
EvalResponseFn = Callable[[DatasetRow, Response], EvalRowResult] # TOOD: bad name
EvalFn = Callable[["ModelDeployment", DatasetRow], EvalRowResult]


@dataclass
class EvalConfig:
    """Evaluate a deployed model on a dataset config.

    The dataset must expose ``load()`` and return iterable dict examples.
    """

    dataset: "DatasetConfig"
    eval_fn: EvalFn | None = None
    eval_response_fn: EvalResponseFn | None = None
    prompt_column: str | None = None
    eval_config_id: str | None = None
    generate_kwargs: dict[str, Any] = field(default_factory=dict)

    def _build_eval_fn(self, eval_response_fn: EvalResponseFn) -> EvalFn:
        def eval_fn(
            deployment: ModelDeployment,
            example: DatasetRow,
        ) -> EvalRowResult:
            text = deployment.generate(
                self.build_prompt(example),
                ensure_ready=False,
                **self.generate_kwargs,
            )
            result = eval_response_fn(example, text)
            return EvalRowResult(
                score=result.score,
                response=text,
                metadata=result.metadata,
            )
        return eval_fn

    def __post_init__(self):
        if self.eval_config_id is None:
            from uuid import uuid4

            class_name = type(self).__name__
            dataset_name = type(self.dataset).__name__
            eval_fn_name = _callable_name(self.eval_fn or self.eval_response_fn)
            self.eval_config_id = f"{class_name}.{dataset_name}.{eval_fn_name}.{uuid4().hex[:4]}"
        if self.eval_fn is None:
            assert self.eval_response_fn is not None, "eval_fn or eval_response_fn must be set"
            self.eval_fn = self._build_eval_fn(self.eval_response_fn)

    def to_durable(self) -> EvalConfigDurable:
        eval_callable = self.eval_response_fn if self.eval_response_fn is not None else self.eval_fn
        return EvalConfigDurable(
            eval_config_id=self.eval_config_id,
            dataset_name=type(self.dataset).__name__,
            eval_fn_name=_callable_name(eval_callable),
            prompt_column=self.prompt_column,
            generate_kwargs=self.generate_kwargs,
        )

    def save(self) -> EvalConfigDurable:
        durable = self.to_durable()
        durable.save()
        return durable

    def build_prompt(self, row: DatasetRow) -> str:
        prompt_column = (self.prompt_column or "").strip()
        input_column = getattr(self.dataset, "input_column", "")
        dataset_column = input_column if isinstance(input_column, str) else ""

        preferred_columns: list[str] = []
        if prompt_column:
            preferred_columns.append(prompt_column)
        if dataset_column and dataset_column not in preferred_columns:
            preferred_columns.append(dataset_column)
        for fallback in ("prompt", "input", "instruction", "question"):
            if fallback not in preferred_columns:
                preferred_columns.append(fallback)

        for column in preferred_columns:
            if column not in row:
                continue
            raw = str(row[column])
            if column in {prompt_column, dataset_column}:
                template = getattr(self.dataset, "prompt_template", "{input}") or "{input}"
                row_context = {
                    key: str(value)
                    for key, value in row.items()
                    if isinstance(key, str)
                }
                try:
                    return template.format(input=raw, **row_context)
                except (KeyError, ValueError):
                    return raw
            return raw

        raise ValueError(
            "EvalConfig.build_prompt() could not resolve a prompt column. "
            "Set EvalConfig.prompt_column or dataset.input_column, or include one of "
            "['prompt', 'input', 'instruction', 'question'] in dataset rows."
        )

    def evaluate(
        self,
        deployment: "ModelDeployment",
        debug: bool = False,
        max_concurrency: int = 1,
    ) -> EvalResult:
        from uuid import uuid4

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self.save()
        deployment.wait_until_ready()

        def _evaluate_indexed(item: tuple[int, DatasetRow]) -> tuple[int, EvalRowResult]:
            idx, example = item
            return idx, self.eval_fn(deployment, example)

        results: list[EvalRowResult] = []
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            indexed_results = executor.map(
                _evaluate_indexed,
                enumerate(self.dataset.load(), start=1),
            )
            for idx, result in indexed_results:
                if debug:
                    print(
                        f"Finished example {idx}: "
                        f"response={result.response!r} score={result.score}",
                        flush=True,
                    )
                results.append(result)

        result = EvalResult(
            eval_id=f"eval-{uuid4().hex[:12]}",
            deployment_id=deployment.deployment_id,
            eval_config_id=self.eval_config_id,
            rows=results,
        )
        result.save()
        return result
