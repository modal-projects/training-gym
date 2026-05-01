from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common import GPUType

@dataclass
class DeployConfig:
    """vLLM serving deployment configuration.

    Attach to a ``ModelConfig`` via ``deploy=DeployConfig(...)`` to
    control how ``serve()`` deploys the model. When ``None``, ``serve()``
    infers GPU type and count from the model's training presets.

    ## Fields

    gpu : GPUType | None
        GPU type for the serving container. When ``None``, inferred
        from the model's training or framework presets. Default ``None``.
    n_gpu : int | None
        Number of GPUs (tensor-parallel degree for vLLM). When ``None``,
        inferred from the model's presets. Default ``None``.
    extra_vllm_args : list[str]
        Additional CLI args passed to ``vllm serve``. Default ``[]``.
    environment_name : str | None
        Modal environment to deploy into. Default ``None``.
    deploy_strategy : str
        Modal deployment strategy. Default ``"rolling"``.
    """

    gpu: "GPUType | None" = None
    n_gpu: int | None = None
    extra_vllm_args: list[str] | None = None
    environment_name: str | None = None
    deploy_strategy: str = "rolling"


@dataclass(frozen=True)
class ModelDeployment:
    """A deployed model endpoint.

    Returned by ``ModelConfig.serve()``. Carries the resolved
    ``DeployConfig`` alongside the live endpoint URL so callers can
    inspect what was actually deployed.
    """

    app: "App"
    app_name: str
    served_model_name: str
    url: str
    deploy: DeployConfig

    def wait_until_ready(self) -> None:
        import time
        import requests
        while True:
            try:
                requests.get(f"{self.url}/v1/models")
                return
            except Exception:
                time.sleep(1)

    def generate(self, prompt: str) -> str:
        import requests
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json={"model": self.served_model_name, "messages": [{"role": "user", "content": prompt}]},
        )
        return response.json()["choices"][0]["message"]["content"]
