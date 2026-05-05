"""Deployment config and model deployment handle.

``DeploymentConfig`` composes a ``ModelConfig`` with a serving recipe
(``VllmRecipe`` or ``SglangRecipe``) and exposes ``serve()`` to deploy
the model on Modal.

``ModelDeployment`` is the handle returned by ``serve()`` — it carries
the live endpoint URL and convenience methods for generation and eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from modal_training_gym.common.models import ModelConfig
    from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe
    from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe

DEPLOYMENTS_STORE_NAME = "deployments"


@dataclass
class DeploymentConfig:
    """Deploy a model behind a serving engine.

    When ``recipe`` is ``None``, defaults to ``SglangRecipe()``.
    """

    model: "ModelConfig"
    recipe: "VllmRecipe | SglangRecipe | None" = None

    app_name: str | None = None
    served_model_name: str | None = None

    def serve(self) -> "ModelDeployment":
        """Build, deploy, and return a ``ModelDeployment`` handle."""
        import modal

        from modal_training_gym.deploy_recipes.base import DeployRecipeType
        from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe

        recipe = self.recipe or SglangRecipe()
        model = self.model

        model_path = model.model_path or model.model_name
        if not model_path:
            raise ValueError(
                f"{type(model).__name__} has no model path to serve. "
                "Set model_path or model_name."
            )

        default_name_src = model.model_name or model_path
        default_slug = (
            default_name_src.rstrip("/").split("/")[-1].replace("_", "-").lower()
        )
        resolved_app_name = self.app_name or f"{default_slug}-serve"
        resolved_served_model_name = self.served_model_name or default_slug
        checkpoints_volume = getattr(model, "checkpoints_volume_name", None)
        checkpoints_mount_path = getattr(model, "checkpoints_mount_path", None)

        if recipe.recipe_type == DeployRecipeType.SGLANG:
            from modal_training_gym.common.serve_sglang import (
                build_sglang_serve_app,
            )

            app = build_sglang_serve_app(
                recipe=recipe,
                app_name=resolved_app_name,
                model_path=model_path,
                served_model_name=resolved_served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
            )
        else:
            from modal_training_gym.common.serve_vllm import build_vllm_serve_app

            app = build_vllm_serve_app(
                recipe=recipe,
                app_name=resolved_app_name,
                model_path=model_path,
                served_model_name=resolved_served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
            )

        env_name = recipe.environment_name
        strategy = recipe.deploy_strategy
        app.deploy(
            environment_name=env_name,
            strategy=strategy,
        )

        if recipe.recipe_type == DeployRecipeType.SGLANG:
            serve_cls = modal.Cls.from_name(
                resolved_app_name,
                "Server",
                environment_name=env_name,
            )
            urls = serve_cls._experimental_get_flash_urls()
            url = urls[0] if urls else None
        else:
            serve_fn = modal.Function.from_name(
                resolved_app_name,
                "serve",
                environment_name=env_name,
            )
            url = serve_fn.get_web_url()
        if not url:
            raise RuntimeError(
                f"Deployed {resolved_app_name!r} but no web URL was returned."
            )
        return ModelDeployment(
            app=app,
            app_name=resolved_app_name,
            served_model_name=resolved_served_model_name,
            url=url,
        )


class ModelDeployment(BaseModel):
    """A deployed model endpoint.

    Returned by ``DeploymentConfig.serve()``. Carries the live endpoint
    URL and convenience methods for generation and evaluation.
    """

    model_config = ConfigDict(frozen=True)

    app_name: str
    served_model_name: str
    url: str

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        self.wait_until_ready()
        body = {
            "model": self.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        resp = requests.post(f"{self.url}/v1/chat/completions", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


    def __post_init__(self) -> None:
        self.save()

    def save(self) -> None:
        from modal_training_gym.common import vol_put

        vol_put(DEPLOYMENTS_STORE_NAME, self.app_name, self.model_dump(mode="json"))

    @classmethod
    def list_deployments(cls) -> list["ModelDeployment"]:
        from modal_training_gym.common import vol_list

        return [cls.model_validate(d) for d in vol_list(DEPLOYMENTS_STORE_NAME)]

    def wait_until_ready(self, timeout: int = 600) -> None:
        import time

        import requests

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.url}/v1/models", timeout=10)
                if resp.ok:
                    data = resp.json()
                    if data.get("data"):
                        return
                    print(f"Waiting for {self.url} (no models loaded yet)...")
                else:
                    print(f"Waiting for {self.url} (status {resp.status_code})...")
            except requests.ConnectionError:
                print(f"Waiting for {self.url} to be ready...")
            except Exception:
                print(f"Waiting for {self.url} (not ready yet)...")
            time.sleep(5)
        raise TimeoutError(f"{self.url} not ready after {timeout}s")
