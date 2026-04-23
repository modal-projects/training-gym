"""LLM-as-judge reward helper, shared across training frameworks.

Posts a rubric prompt to a running OpenAI-compatible chat-completions
endpoint (typically a vLLM server deployed separately on Modal), parses the
first numeric token out of the reply, and returns a score normalized to
[0, 1]. Subclass and override `build_prompt()` to define the rubric for
your task.

Pure client-side — this module does not deploy or manage the judge server.
Stand up the endpoint however you prefer (vLLM on Modal, SGLang, an
external service) and pass its base URL to the constructor.

Example:

    class HaikuStyleJudge(LlmJudge):
        def build_prompt(self, prompt: str, response: str, **kw) -> str:
            return (
                f"Rate this haiku about '{prompt}' from 0-10 "
                f"on poetic quality and relevance:\\n{response}\\n"
                f"Output ONLY a single number."
            )

    judge = HaikuStyleJudge(
        model_name="qwen3-4b",
        base_url="http://localhost:8000",
        max_score=10,
    )
    async with aiohttp.ClientSession() as session:
        score = await judge.score(session, topic, poem)  # in [0, 1]
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp


class LlmJudge:
    """LLM-as-judge client for an OpenAI-compatible chat-completions endpoint.

    Subclasses implement `build_prompt()` with the task-specific rubric; the
    base class handles the HTTP call, response parsing, and normalization.
    Any error (network, bad JSON, no number found) yields 0.0 rather than
    raising, so one misbehaving rollout cannot abort a training step.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_score: float = 10.0,
        max_tokens: int = 100,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.max_score = max_score
        self.max_tokens = max_tokens

    def build_prompt(self, prompt: str, response: str, **kwargs: Any) -> str:
        """Return the judge prompt for one (prompt, response) pair.

        Subclasses must override. `kwargs` carry per-call context (e.g. a
        reference answer, a few-shot exemplar) the rubric may need.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.build_prompt must be overridden"
        )

    async def score(
        self,
        session: "aiohttp.ClientSession",
        prompt: str,
        response: str,
        **kwargs: Any,
    ) -> float:
        """Score one (prompt, response) pair; returns a float in [0, 1]."""
        judge_prompt = self.build_prompt(prompt, response, **kwargs)
        try:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"content-type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "max_tokens": self.max_tokens,
                },
            ) as resp:
                if resp.status != 200:
                    print(f"LlmJudge HTTP {resp.status}: {await resp.text()}")
                    return 0.0
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                match = re.search(r"(\d+(?:\.\d+)?)", raw)
                if match is None:
                    return 0.0
                raw_score = float(match.group(1))
                return min(max(raw_score, 0.0), self.max_score) / self.max_score
        except Exception as e:
            print(f"LlmJudge error: {e}")
            return 0.0
