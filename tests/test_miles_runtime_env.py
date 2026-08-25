"""Ray runtime env construction for miles jobs."""

from __future__ import annotations

from modal_training_gym.frameworks.miles import launcher
from modal_training_gym.frameworks.miles.launcher import build_ray_runtime_env
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


class _FakeImage:
    def __init__(self):
        self.commands: list[str] = []

    @classmethod
    def from_registry(cls, _docker_image: str) -> "_FakeImage":
        return cls()

    def entrypoint(self, _entrypoint: list[str]) -> "_FakeImage":
        return self

    def run_commands(self, *commands: str) -> "_FakeImage":
        self.commands.extend(commands)
        return self

    def env(self, _environment: dict[str, str]) -> "_FakeImage":
        return self


def test_multinode_image_reinstalls_matching_rdma_runtime(monkeypatch):
    monkeypatch.setattr(launcher, "Image", _FakeImage)
    recipe = MilesRecipe(colocate=False, rollout_num_gpus=8)

    image = launcher._build_miles_base_image(recipe)

    assert launcher.RDMA_RUNTIME_INSTALL_COMMAND in image.commands


def test_single_node_image_keeps_base_rdma_runtime(monkeypatch):
    monkeypatch.setattr(launcher, "Image", _FakeImage)

    image = launcher._build_miles_base_image(MilesRecipe())

    assert launcher.RDMA_RUNTIME_INSTALL_COMMAND not in image.commands


def test_ld_library_path_comes_from_the_container(monkeypatch):
    """Workers get the container's linker path, behind the system lib dir."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/wheel/nvidia/lib"
    )
    assert env_vars["MASTER_ADDR"] == "10.0.0.1"
    assert env_vars["no_proxy"] == "127.0.0.1,10.0.0.1"


def test_recipe_environment_still_wins(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/from/container")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        metric_env={},
        environment={
            "LD_LIBRARY_PATH": "/from/recipe",
            "PYTHONPATH": "/root/Megatron-LM/",
        },
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/from/recipe"
    assert env_vars["PYTHONPATH"] == "/root/Megatron-LM/"


def test_unset_container_path_yields_only_the_system_lib_dir(monkeypatch):
    """No empty entry, which the loader would read as the working directory."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu"


def test_system_lib_dir_is_not_duplicated(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", metric_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib"
    )


def test_metric_env_is_preserved(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        metric_env={"WANDB_RUN_ID": "abc", "WANDB_RESUME": "allow"},
        environment={},
    )["env_vars"]

    assert env_vars["WANDB_RUN_ID"] == "abc"
    assert env_vars["WANDB_RESUME"] == "allow"
