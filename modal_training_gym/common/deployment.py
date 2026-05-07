"""Deployment config and model deployment handle.

``DeploymentConfig`` composes a ``ModelConfig`` with a serving recipe
(``VllmRecipe`` or ``SglangRecipe``) and exposes ``serve()`` to deploy
the model on Modal.

``ModelDeployment`` is the handle returned by ``serve()`` — it carries
the live endpoint URL and convenience methods for generation and eval.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import inspect
import uuid

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType, convert_checkpoint_to_hf
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe
from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe
from modal_training_gym.deploy_recipes.base import DeployRecipeType
from modal.experimental import list_deployed_apps
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_list,
    vol_put,
    vol_upsert_summary_item,
)

DEPLOYMENTS_STORE_NAME = MetadataStore.DEPLOYMENTS.value


@dataclass
class DeploymentConfig:
    """Deploy a model behind a serving engine.

    When ``recipe`` is ``None``, defaults to ``SglangRecipe()``.
    """

    model: ModelConfig
    checkpoint: Checkpoint | None = None
    recipe: VllmRecipe | SglangRecipe | None = None

    app_name: str | None = None
    served_model_name: str | None = None

    def _checkpoints_volume_name(self) -> str | None:
        if self.checkpoint is not None and self.checkpoint.checkpoints_volume_name:
            return self.checkpoint.checkpoints_volume_name
        return getattr(self.model, "checkpoints_volume_name", None)

    def _checkpoints_mount_path(self) -> str | None:
        if self.checkpoint is not None and self.checkpoint.checkpoints_mount_path:
            return self.checkpoint.checkpoints_mount_path
        return getattr(self.model, "checkpoints_mount_path", None)

    def _resolved_model_path(self) -> str:
        if self.checkpoint is not None:
            checkpoint_path = getattr(self.checkpoint, "path", "")
            if checkpoint_path:
                return checkpoint_path

        model_path = self.model.model_path or self.model.model_name
        if not model_path:
            raise ValueError(
                f"{type(self.model).__name__} has no model path to serve. "
                "Set model_path or model_name."
            )
        return model_path

    def serve(self) -> "ModelDeployment":
        """Build, deploy, and return a ``ModelDeployment`` handle."""

        recipe = self.recipe or SglangRecipe()
        model = self.model

        if (
            self.checkpoint is not None
            and self.checkpoint.checkpoint_type == CheckpointType.megatron
        ):
            self.checkpoint = convert_checkpoint_to_hf(
                checkpoint=self.checkpoint,
                model=model,
                recipe=recipe,
            )

        model_path = self._resolved_model_path()

        default_name_src = model.model_name or model_path
        default_slug = (
            default_name_src.rstrip("/").split("/")[-1].replace("_", "-").lower()
        )
        if not self.app_name:
            self.app_name = f"{default_slug}-serve"
        if not self.served_model_name:
            self.served_model_name = default_slug
        checkpoints_volume = self._checkpoints_volume_name()
        checkpoints_mount_path = self._checkpoints_mount_path()

        if isinstance(recipe, SglangRecipe):
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
        elif isinstance(recipe, VllmRecipe):
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
        else:
            raise ValueError(f"Unsupported deploy recipe: {type(recipe).__name__}")

        env_name = recipe.environment_name
        strategy = recipe.deploy_strategy
        app.deploy(
            environment_name=env_name,
            strategy=strategy,
        )

        if recipe.recipe_type == DeployRecipeType.SGLANG:
            server = getattr(app, "Server", None)
            if server is None and hasattr(app, "registered_classes"):
                server = app.registered_classes.get("Server")
            if server is None:
                raise RuntimeError(
                    f"Deployed {self.app_name!r} but could not resolve SGLang Server class handle."
                )
            urls = server._experimental_get_flash_urls()
            url = urls[0] if urls else None
        else:
            url = app.serve.get_web_url()
        modal_app_id = app.app_id
        if not url:
            raise RuntimeError(
                f"Deployed {self.app_name!r} but no web URL was returned."
            )
        if not modal_app_id:
            raise RuntimeError(
                f"Deployed {self.app_name!r} but no Modal app id was returned."
            )
        deployment = ModelDeployment(
            deployment_config=self,
            modal_app_id=modal_app_id,
            url=url,
        )
        deployment.save()
        return deployment


class DeploymentStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    INACTIVE = "inactive"


class ModelDeployment(BaseModel):
    """A deployed model endpoint.

    Returned by ``DeploymentConfig.serve()``. Carries the live endpoint
    URL, Modal app id, and convenience methods for generation and evaluation.
    """

    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(arbitrary_types_allowed=True)
    deployment_config: DeploymentConfig

    modal_app_id: str
    url: str

    @field_validator("deployment_config", mode="before")
    @classmethod
    def _parse_deployment_config(cls, value: object) -> DeploymentConfig | object:
        if not isinstance(value, dict):
            return value

        deployment_config = dict(value)
        model = deployment_config.get("model")
        if isinstance(model, dict):
            deployment_config["model"] = ModelConfig(**model)
        return DeploymentConfig(**deployment_config)

    @field_serializer("deployment_config")
    def _serialize_deployment_config(self, value: DeploymentConfig) -> dict[str, object]:
        model = value.model
        return {
            "model": {
                "model_name": getattr(model, "model_name", ""),
                "model_path": getattr(model, "model_path", None),
                "checkpoints_volume_name": getattr(
                    model,
                    "checkpoints_volume_name",
                    None,
                ),
                "checkpoints_mount_path": getattr(
                    model,
                    "checkpoints_mount_path",
                    None,
                ),
            },
            "app_name": value.app_name,
            "served_model_name": value.served_model_name,
        }

    def generate(self, prompt: str, ensure_ready: bool = True, **kwargs) -> str:
        import time

        import requests

        if ensure_ready:
            self.wait_until_ready()
        body = {
            "model": self.deployment_config.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        transient_status_codes = {502, 503, 504}
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=body,
                    timeout=120,
                )
                if resp.status_code in transient_status_codes and attempt < max_attempts:
                    print(
                        f"Transient generation error {resp.status_code} from {self.url}; "
                        f"retrying ({attempt}/{max_attempts})..."
                    )
                    self.wait_until_ready(timeout=120)
                    time.sleep(min(2 * attempt, 5))
                    continue
                resp.raise_for_status()
                message = resp.json()["choices"][0]["message"]
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if content is None:
                    return message.get("reasoning_content", "")
                return str(content)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= max_attempts:
                    raise
                print(
                    f"Transient generation transport error from {self.url}: {exc}; "
                    f"retrying ({attempt}/{max_attempts})..."
                )
                self.wait_until_ready(timeout=120)
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(f"Failed to generate from {self.url} after {max_attempts} attempts")

    def save(self) -> None:
        payload = self.model_dump(mode="json")
        vol_put(
            MetadataStore.DEPLOYMENTS,
            self.deployment_id,
            payload,
        )
        vol_upsert_summary_item(
            MetadataStore.DEPLOYMENTS_SUMMARY,
            payload,
            item_id_key="deployment_id",
            sort_key=lambda item: (
                str(item.get("deployment_config", {}).get("app_name", "")),
                str(item.get("deployment_id", "")),
            ),
            reverse=True,
        )

    @property
    def status(self) -> str:
        import requests

        try:
            response = requests.get(f"{self.url}/v1/models", timeout=5)
            if response.ok and response.json().get("data"):
                return DeploymentStatus.READY.value
        except Exception:
            pass

        try:
            apps = list_deployed_apps()
            if inspect.isawaitable(apps):
                apps = asyncio.run(apps)
            for app in apps:
                if (
                    app.app_id == self.modal_app_id
                    or app.name == self.deployment_config.app_name
                ):
                    return DeploymentStatus.INITIALIZING.value
        except Exception:
            pass

        return DeploymentStatus.INACTIVE.value

    @classmethod
    def list_deployments(cls) -> dict[str, "ModelDeployment"]:
        return {d["deployment_id"]: cls.model_validate(d) for d in vol_list(MetadataStore.DEPLOYMENTS)}

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
