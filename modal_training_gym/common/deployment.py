"""Deployment config and model deployment handle.

``DeploymentConfig`` composes a ``ModelConfig`` with a serving recipe
(``VllmRecipe`` or ``SglangRecipe``) and exposes ``serve()`` to deploy
the model on Modal.

``ModelDeployment`` is the handle returned by ``serve()`` — it carries
the live endpoint URL and convenience methods for generation and eval.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from modal_training_gym.common.models import ModelConfig
from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe
from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe
from modal_training_gym.utils.metadata import MetadataStore, vol_list, vol_put

DEPLOYMENTS_STORE_NAME = MetadataStore.DEPLOYMENTS.value


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
        if not self.app_name:
            self.app_name = f"{default_slug}-serve"
        if not self.served_model_name:
            self.served_model_name = default_slug
        checkpoints_volume = getattr(model, "checkpoints_volume_name", None)
        checkpoints_mount_path = getattr(model, "checkpoints_mount_path", None)

        if recipe.recipe_type == DeployRecipeType.SGLANG:
            from modal_training_gym.deploy_recipes.sglang_recipe.serve_sglang import (
                build_sglang_serve_app,
            )

            app = build_sglang_serve_app(
                recipe=recipe,
                app_name=self.app_name,
                model_path=model_path,
                served_model_name=self.served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
            )
        else:
            from modal_training_gym.deploy_recipes.vllm_recipe.serve_vllm import (
                build_vllm_serve_app,
            )

            app = build_vllm_serve_app(
                recipe=recipe,
                app_name=self.app_name,
                model_path=model_path,
                served_model_name=self.served_model_name,
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
                self.app_name,
                "Server",
                environment_name=env_name,
            )
            urls = serve_cls._experimental_get_flash_urls()
            url = urls[0] if urls else None
        else:
            serve_fn = modal.Function.from_name(
                self.app_name,
                "serve",
                environment_name=env_name,
            )
            url = serve_fn.get_web_url()
        if not url:
            raise RuntimeError(
                f"Deployed {self.app_name!r} but no web URL was returned."
            )
        return ModelDeployment(
            deployment_config=self,
            url=url,
        )


class ModelDeployment(BaseModel):
    """A deployed model endpoint.

    Returned by ``DeploymentConfig.serve()``. Carries the live endpoint
    URL and convenience methods for generation and evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deployment_config: DeploymentConfig
    # TODO: Add Modal app id
    url: str

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        self.wait_until_ready()
        body = {
            "model": self.deployment_config.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        resp = requests.post(f"{self.url}/v1/chat/completions", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def __post_init__(self) -> None:
        self.save()

    def save(self) -> None:
        vol_put(
            MetadataStore.DEPLOYMENTS,
            self.deployment_config.app_name,
            self.model_dump(mode="json"),
        )

    @classmethod
    def list_deployments(cls) -> list["ModelDeployment"]:
        return [cls.model_validate(d) for d in vol_list(MetadataStore.DEPLOYMENTS)]

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
