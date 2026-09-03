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


def test_image_applies_final_weight_sync_patch(monkeypatch):
    """The patch that stops miles syncing weights into idle engines after the
    final save is baked into every miles image."""
    monkeypatch.setattr(launcher, "Image", _FakeImage)

    commands = "\n".join(launcher._build_miles_base_image(MilesRecipe()).commands)

    assert launcher._PATCH_SKIP_FINAL_WEIGHT_SYNC_B64 in commands


def test_flight_recorder_dump_is_scoped_per_run(monkeypatch):
    """The dump prefix a recipe sets is rewritten under the run id in the Ray
    env, so concurrent or retried runs never overwrite each other's dumps."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        metric_env={},
        environment={"TORCH_FR_DUMP_TEMP_FILE": "/checkpoints/nccl_fr/rank_"},
        extra_env={"TRAINING_GYM_TRAINING_RUN_ID": "run-abc123"},
    )["env_vars"]

    assert (
        env_vars["TORCH_FR_DUMP_TEMP_FILE"] == "/checkpoints/nccl_fr/run-abc123/rank_"
    )


def test_flight_recorder_prefix_absent_when_recipe_sets_none(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        metric_env={},
        environment={},
        extra_env={"TRAINING_GYM_TRAINING_RUN_ID": "run-abc123"},
    )["env_vars"]
    assert "TORCH_FR_DUMP_TEMP_FILE" not in env_vars
    assert launcher.flight_recorder_prefix({}, "run-abc123") == ""


def test_nemotron_recipe_arms_the_flight_recorder():
    from modal_training_gym.train_recipes.miles_recipe import (
        Nemotron3_Ultra_550B_A55B_Recipe,
    )

    env = Nemotron3_Ultra_550B_A55B_Recipe().environment
    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert int(env["TORCH_NCCL_TRACE_BUFFER_SIZE"]) > 0
    # Heartbeat must be far shorter than the 60 min collective timeout, or the
    # dump never fires before the run gives up.
    assert int(env["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"]) < 60 * 60
    assert env["TORCH_FR_DUMP_TEMP_FILE"].startswith("/checkpoints/")


def test_miles_recipes_are_pinned_off_aws():
    """Every miles recipe inherits the base-class cloud pin; an explicit
    cloud= still overrides it."""
    from modal_training_gym.train_recipes.miles_recipe import (
        Nemotron3_Ultra_550B_A55B_Recipe,
    )

    assert MilesRecipe().cloud == "oci"
    assert Nemotron3_Ultra_550B_A55B_Recipe().cloud == "oci"
    assert Nemotron3_Ultra_550B_A55B_Recipe(cloud="aws").cloud == "aws"
