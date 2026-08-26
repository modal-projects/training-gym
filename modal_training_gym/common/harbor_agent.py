from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    class BaseAgent:
        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            **kwargs,
        ) -> None:
            self.logs_dir = logs_dir
else:
    try:
        from harbor.agents.base import BaseAgent
    except ModuleNotFoundError as exc:
        if exc.name != "harbor":
            raise

        class BaseAgent:
            def __init__(
                self,
                logs_dir: Path,
                model_name: str | None = None,
                **kwargs,
            ) -> None:
                self.logs_dir = logs_dir


class TrainingGymResponseAgent(BaseAgent):
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        response: str,
        candidate_path: str,
        candidate_command: str,
        **kwargs,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._response = response
        self._candidate_path = candidate_path
        self._candidate_command = candidate_command

    @staticmethod
    def name() -> str:
        return "training-gym-response"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        candidate_file = self.logs_dir / Path(self._candidate_path).name
        candidate_file.write_text(self._response, encoding="utf-8")
        await environment.upload_file(candidate_file, self._candidate_path)

        command = self._candidate_command.format(
            candidate_path=shlex.quote(self._candidate_path)
        )
        result = await environment.exec(command=command)
        context.metadata = {
            "candidate_path": self._candidate_path,
            "candidate_return_code": result.return_code,
            "candidate_stdout": result.stdout,
            "candidate_stderr": result.stderr,
        }
