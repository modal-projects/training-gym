from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common import GPUType


@dataclass
class DeployConfig:
    """Serving deployment configuration (vLLM or SGLang).

    Attach to a ``ModelConfig`` via ``deploy=DeployConfig(...)`` to
    control how ``serve()`` deploys the model. When ``None``, ``serve()``
    infers GPU type and count from the model's training presets.

    ## Fields

    backend : ``"vllm"`` | ``"sglang"``
        Inference engine to use. Default ``"vllm"``.
    gpu : GPUType | None
        GPU type for the serving container. When ``None``, inferred
        from the model's training or framework presets. Default ``None``.
    n_gpu : int | None
        Number of GPUs (tensor-parallel degree). When ``None``,
        inferred from the model's presets. Default ``None``.
    extra_vllm_args : list[str] | None
        Additional CLI args passed to ``vllm serve``. Only used when
        ``backend="vllm"``. Default ``None``.
    extra_sglang_args : list[str] | None
        Additional CLI args passed to ``sglang.launch_server``. Only
        used when ``backend="sglang"``. Default ``None``.
    sglang_image_tag : str | None
        Docker image tag for SGLang. When ``None``, the default in
        ``build_sglang_serve_app`` is used. Default ``None``.
    environment_name : str | None
        Modal environment to deploy into. Default ``None``.
    deploy_strategy : str
        Modal deployment strategy. Default ``"rolling"``.
    """

    backend: Literal["vllm", "sglang"] = "vllm"
    gpu: "GPUType | None" = None
    n_gpu: int | None = None
    extra_vllm_args: list[str] | None = None
    extra_sglang_args: list[str] | None = None
    sglang_image_tag: str | None = None
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
                print(f"Waiting for {self.url} to be ready...")

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        self.wait_until_ready()
        body = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        response = requests.post(f"{self.url}/v1/chat/completions", json=body)
        return response.json()["choices"][0]["message"]["content"]
