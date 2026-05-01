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
    startup_timeout_seconds: int = 20 * 60
    startup_poll_seconds: int = 5

    def build_prompt(self, example: DatasetExample) -> str:
        input_column = self.dataset.input_column
        if not input_column:
            raise ValueError(
                "EvalConfig requires dataset.input_column or an overridden build_prompt()."
            )
        return str(example[input_column])

    def _wait_until_ready(self, client, debug: bool) -> None:
        import time

        started = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            try:
                client.models.list()
                if debug:
                    print(
                        f"vLLM endpoint is ready after {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                return
            except Exception as exc:
                elapsed = time.monotonic() - started
                if elapsed >= self.startup_timeout_seconds:
                    raise TimeoutError(
                        f"Timed out waiting {self.startup_timeout_seconds}s for "
                        f"{self.deployment.url} to become ready."
                    ) from exc
                if debug:
                    print(
                        f"Waiting for vLLM endpoint to start "
                        f"({elapsed:.1f}s elapsed, attempt {attempt}): {exc}",
                        flush=True,
                    )
                time.sleep(self.startup_poll_seconds)

    def evaluate(self, debug=False) -> EvalResult:
        from openai import OpenAI

        client = OpenAI(
            base_url=f"{self.deployment.url.rstrip('/')}/v1",
            api_key="not-needed",
        )
        self._wait_until_ready(client, debug)
        rows = self.dataset.load()

        total = len(rows)
        results: list[EvalRowResult] = []

        for idx, example in enumerate(rows, start=1):
            if debug:
                print(f"Evaluating example {idx}/{total}: {example}", flush=True)
            response = client.chat.completions.create(
                model=self.deployment.served_model_name,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": self.build_prompt(example),
                    }
                ],
            )
            text = response.choices[0].message.content or ""
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
