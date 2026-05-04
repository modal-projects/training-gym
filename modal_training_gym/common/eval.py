from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.deployment import ModelDeployment


DatasetExample = dict[str, Any]

EVAL_RESULTS_STORE_NAME = "eval-results"


class EvalRowResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    passed: bool
    response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


EvalFn = Callable[[DatasetExample, str], EvalRowResult]


class EvalResult(BaseModel):
    eval_id: str
    accuracy: float
    correct: int
    total: int
    deployment_app_name: str = ""
    deployment_url: str = ""
    deployment_model_name: str = ""
    dataset_id: str = ""
    dataset_name: str = ""
    training_run_id: str = ""
    created_at: int = 0
    rows: list[EvalRowResult] = Field(default_factory=list)

    def save(self) -> None:
        from modal_training_gym.common.client import _post

        _post("/eval", self.model_dump())

    @classmethod
    def from_id(cls, eval_id: str) -> "EvalResult":
        from modal_training_gym.common.client import _get

        return cls.model_validate(_get(f"/evals/{eval_id}"))

    @classmethod
    def list_results(cls) -> list["EvalResult"]:
        from modal_training_gym.common.client import _get

        return [cls.model_validate(r) for r in _get("/evals")]


@dataclass
class EvalConfig:
    """Evaluate a deployed model on a dataset config.

    The dataset must expose ``load()`` and return iterable dict examples.
    """

    deployment: "ModelDeployment"
    dataset: "DatasetConfig"
    eval_fn: EvalFn
    temperature: float = 0.0
    generate_kwargs: dict[str, Any] = field(default_factory=dict)
    training_run_id: str = ""

    def build_prompt(self, example: DatasetExample) -> str:
        input_column = self.dataset.input_column
        if not input_column:
            raise ValueError(
                "EvalConfig requires dataset.input_column or an overridden build_prompt()."
            )
        return str(example[input_column])

    def evaluate(self, debug: bool = False, persist: bool = True) -> EvalResult:
        import time
        from uuid import uuid4

        self.deployment.wait_until_ready()
        rows = list(self.dataset.load())

        total = len(rows)
        results: list[EvalRowResult] = []

        for idx, example in enumerate(rows, start=1):
            if debug:
                print(f"Evaluating example {idx}/{total}: {example}", flush=True)
            text = self.deployment.generate(
                self.build_prompt(example),
                temperature=self.temperature,
                **self.generate_kwargs,
            )
            result = self.eval_fn(example, text)
            if not result.response:
                result = EvalRowResult(
                    score=result.score,
                    passed=result.passed,
                    response=text,
                    metadata=result.metadata,
                )
            if debug:
                print(
                    f"Finished example {idx}/{total}: "
                    f"response={result.response!r} score={result.score}",
                    flush=True,
                )
            results.append(result)

        correct = sum(int(result.passed) for result in results)
        score = sum(result.score for result in results)

        dataset_id = self._dataset_id()
        result = EvalResult(
            eval_id=f"eval-{uuid4().hex[:12]}",
            accuracy=score / total if total else 0.0,
            correct=correct,
            total=total,
            deployment_app_name=getattr(self.deployment, "app_name", ""),
            deployment_url=getattr(self.deployment, "url", ""),
            deployment_model_name=getattr(self.deployment, "served_model_name", ""),
            dataset_id=dataset_id,
            dataset_name=self._dataset_name(),
            training_run_id=self.training_run_id,
            created_at=int(time.time()),
            rows=results,
        )
        if persist:
            result.save()
        return result

    def _dataset_id(self) -> str:
        hf_repo = str(getattr(self.dataset, "hf_repo", "") or "")
        hf_split = str(getattr(self.dataset, "hf_split", "") or "")
        hf_config = str(getattr(self.dataset, "hf_config", "") or "")
        prompt_data = str(getattr(self.dataset, "prompt_data", "") or "")

        if hf_repo:
            parts = ["hf", hf_repo]
            if hf_config:
                parts.append(hf_config)
            if hf_split:
                parts.append(hf_split)
            return ":".join(parts)
        if prompt_data:
            return prompt_data
        return self.dataset.__class__.__name__

    def _dataset_name(self) -> str:
        hf_repo = str(getattr(self.dataset, "hf_repo", "") or "")
        if hf_repo:
            return hf_repo
        prompt_data = str(getattr(self.dataset, "prompt_data", "") or "")
        if prompt_data:
            return prompt_data
        return self.dataset.__class__.__name__
