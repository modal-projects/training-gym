"""Ray runtime env construction for miles jobs."""

from __future__ import annotations

from modal_training_gym.frameworks.miles.launcher import build_ray_runtime_env


def test_ld_library_path_comes_from_the_container(monkeypatch):
    """Workers get the container's linker path, behind the system lib dir."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", wandb_env={}, environment={}
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
        wandb_env={},
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
        head_addr="10.0.0.1", wandb_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu"


def test_system_lib_dir_is_not_duplicated(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib")

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1", wandb_env={}, environment={}
    )["env_vars"]

    assert env_vars["LD_LIBRARY_PATH"] == (
        "/usr/lib/x86_64-linux-gnu:/wheel/nvidia/lib"
    )


def test_wandb_env_is_preserved(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    env_vars = build_ray_runtime_env(
        head_addr="10.0.0.1",
        wandb_env={"WANDB_RUN_ID": "abc", "WANDB_RESUME": "allow"},
        environment={},
    )["env_vars"]

    assert env_vars["WANDB_RUN_ID"] == "abc"
    assert env_vars["WANDB_RESUME"] == "allow"
