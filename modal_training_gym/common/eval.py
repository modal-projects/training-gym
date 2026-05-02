from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelDeployment


DatasetExample = dict[str, Any]


@dataclass(frozen=True)
class EvalRowResult:
    score: float
    passed: bool
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


EvalFn = Callable[[DatasetExample, str], EvalRowResult]


@dataclass
class EvalResult:
    accuracy: float
    correct: int
    total: int
    rows: list[EvalRowResult] = field(default_factory=list)


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

    def build_prompt(self, example: DatasetExample) -> str:
        input_column = self.dataset.input_column
        if not input_column:
            raise ValueError(
                "EvalConfig requires dataset.input_column or an overridden build_prompt()."
            )
        return str(example[input_column])

    def evaluate(self, debug=False) -> EvalResult:
        self.deployment.wait_until_ready()
        rows = self.dataset.load()

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

        return EvalResult(
            accuracy=score / total if total else 0.0,
            correct=correct,
            total=total,
            rows=results,
        )
