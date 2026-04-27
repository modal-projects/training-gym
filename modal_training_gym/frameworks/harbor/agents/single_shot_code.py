"""Single-shot code agent for Harbor code-generation tasks.

Loaded on the remote Miles image where ``harbor`` is installed. Makes one
OpenAI-compatible chat completion call, extracts code from the response,
and writes it to ``/workspace/solution.py`` in the sandbox environment.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import requests
from harbor.agents.base import BaseAgent

from modal_training_gym.frameworks.harbor.trajectory import (
    TrajectoryTurn,
    parse_thinking,
)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_IM_START_RE = re.compile(r"<\|im_start\|>assistant\s*", re.IGNORECASE)
_IM_END_RE = re.compile(r"<\|im_end\|>")
_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_python_code(text: str) -> str:
    """Strip chat template artifacts and code fences from model output."""
    text = _IM_START_RE.sub("", text)
    text = _THINK_RE.sub("", text)
    text = _IM_END_RE.sub("", text)

    fenced = _CODE_FENCE_RE.findall(text)
    if fenced:
        text = "\n\n".join(fenced)

    return text.strip()


def code_size_bytes(code: str) -> int:
    return len(code.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


class SingleShotCodeAgent(BaseAgent):
    """Harbor agent that generates code via a single LLM API call."""

    @staticmethod
    def name() -> str:
        return "single-shot-code-agent"

    def __init__(
        self,
        logs_dir,
        model_name: str | None = None,
        *,
        base_url: str = "http://127.0.0.1:30000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_sec: float = 90.0,
        system_prompt: str = "Return only valid Python code. No explanation.",
        target_path: str = "/workspace/solution.py",
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.system_prompt = system_prompt
        self.target_path = target_path

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        pass

    def _chat_completion(self, instruction: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name or "model",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ],
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()

    async def run(self, instruction, environment, context) -> None:
        t_start = time.time()
        result = await asyncio.to_thread(self._chat_completion, str(instruction))
        t_end = time.time()

        raw_content = result["choices"][0]["message"]["content"]
        code = extract_python_code(raw_content)

        local_path = Path(self.logs_dir) / "solution.py"
        local_path.write_text(code)

        await environment.upload_file(str(local_path), self.target_path)

        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get(
            "reasoning_tokens", 0
        )

        reasoning, response_content = parse_thinking(raw_content)

        context.metadata["trajectory_messages"] = [
            TrajectoryTurn(role="user", content=str(instruction)).to_dict(),
            TrajectoryTurn(
                role="assistant",
                content=response_content,
                reasoning=reasoning,
                duration_sec=t_end - t_start,
                token_usage={
                    "reasoning": reasoning_tokens,
                    "answer": max(0, completion_tokens - reasoning_tokens),
                },
                t_start=t_start,
                t_end=t_end,
            ).to_dict(),
        ]
        context.metadata["n_input_tokens"] = prompt_tokens
        context.metadata["n_output_tokens"] = completion_tokens
        context.metadata["candidate_bytes"] = code_size_bytes(code)
        context.metadata["raw_response_text"] = raw_content
        context.metadata["written_content"] = code
