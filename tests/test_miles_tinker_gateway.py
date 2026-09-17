"""TinkerGateway launcher: the Modal app that hosts Miles' ``serve_tinker.py``.

Covers the parts that need no GPU: the served recipe honors the invariants
``serve_tinker.py`` asserts (``load == hf_checkpoint``, checkpoint root on the
checkpoints Volume), the Ray job runs ``serve_tinker.py`` rather than
``train.py``, and the Modal server is authenticated on the gateway port.
"""

import asyncio
import shlex
import socket
import threading
from dataclasses import replace
from unittest.mock import patch

import pytest

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import Qwen3_30B
from modal_training_gym.frameworks.miles.tinker_gateway import (
    TINKER_HEALTH_PATH,
    TINKER_PORT,
    TINKER_SDK_VERSION,
    TinkerGateway,
    build_tinker_gateway_app,
    build_tinker_gateway_cmd,
    forward_port,
    gateway_checkpoint_root,
    gateway_recipe,
    require_tinker_gateway,
    wait_for_gateway_ready,
)
from modal_training_gym.train_recipes.miles_recipe import (
    MilesRecipe,
    Qwen3_30B_A3B_Tinker_Recipe,
)

HF_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/abc"


def _gateway(**overrides) -> MilesRecipe:
    params = dict(
        multi_lora_n_adapters=2,
        lora_rank=8,
        colocate=False,
        actor_num_gpus_per_node=1,
        rollout_num_gpus=1,
        miles_model_name="qwen3-30B-A3B",
    )
    params.update(overrides)
    return MilesRecipe(**params)


def _served(miles: MilesRecipe | None = None, **kwargs) -> MilesRecipe:
    miles = miles or _gateway()
    root = gateway_checkpoint_root(
        miles, app_name="demo-tinker", checkpoints_mount_path="/checkpoints"
    )
    return gateway_recipe(
        miles, model=Qwen3_30B(), hf_path=HF_PATH, checkpoint_root=root, **kwargs
    )


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_ordinary_recipes_are_rejected() -> None:
    with pytest.raises(TrainingGymConfigError, match="multi_lora_n_adapters"):
        require_tinker_gateway(MilesRecipe())
    with pytest.raises(TrainingGymConfigError, match="multi_lora_n_adapters"):
        build_tinker_gateway_cmd(MilesRecipe(), model=Qwen3_30B(), miles_root="/r")
    with pytest.raises(TrainingGymConfigError, match="multi_lora_n_adapters"):
        TinkerGateway.launch(MilesRecipe())


def test_checkpoint_root_defaults_under_app_on_checkpoints_volume() -> None:
    root = gateway_checkpoint_root(
        _gateway(), app_name="demo-tinker", checkpoints_mount_path="/checkpoints"
    )
    assert root == "/checkpoints/demo-tinker/tinker"


def test_explicit_checkpoint_root_must_live_on_checkpoints_volume() -> None:
    ok = gateway_checkpoint_root(
        _gateway(tinker_checkpoint_root="/checkpoints/shared/tinker"),
        app_name="demo-tinker",
        checkpoints_mount_path="/checkpoints",
    )
    assert ok == "/checkpoints/shared/tinker"
    with pytest.raises(TrainingGymConfigError):
        gateway_checkpoint_root(
            _gateway(tinker_checkpoint_root="/tmp/tinker"),
            app_name="demo-tinker",
            checkpoints_mount_path="/checkpoints",
        )


def test_served_recipe_pins_serve_tinker_invariants() -> None:
    served = _served()
    assert served.load == served.hf_checkpoint == HF_PATH
    assert served.tinker_checkpoint_root == "/checkpoints/demo-tinker/tinker"
    # serve_tinker.py derives <save>/tinker itself; keep both spellings agreeing.
    assert served.save == "/checkpoints/demo-tinker"
    assert served.tinker_server_host == "0.0.0.0"
    assert served.tinker_server_port == TINKER_PORT
    assert served.tinker_base_model == "Qwen/Qwen3-30B-A3B"


def test_served_recipe_keeps_explicit_tinker_settings() -> None:
    served = _served(
        _gateway(
            tinker_server_port=9000,
            tinker_base_model="my-base",
            tinker_train_mlp=False,
        )
    )
    assert served.tinker_server_port == 9000
    assert served.tinker_base_model == "my-base"
    assert served.tinker_train_mlp is False


def test_conflicting_hf_checkpoint_is_rejected() -> None:
    with pytest.raises(TrainingGymConfigError, match="exactly one frozen base"):
        _served(_gateway(hf_checkpoint="/some/other/model"))
    # A matching explicit hf_checkpoint is fine.
    assert _served(_gateway(hf_checkpoint=HF_PATH)).load == HF_PATH


def test_gateway_cmd_runs_serve_tinker_not_train() -> None:
    served = _served()
    cmd = build_tinker_gateway_cmd(served, model=Qwen3_30B(), miles_root="/root/miles")
    argv = shlex.split(shlex.split(cmd)[-1])
    assert "python3 /root/miles/serve_tinker.py" in cmd
    assert "/root/miles/train.py" not in cmd
    assert "/root/miles/train_async.py" not in cmd
    assert (
        _flag_value(argv, "--load") == _flag_value(argv, "--hf-checkpoint") == HF_PATH
    )
    assert _flag_value(argv, "--multi-lora-n-adapters") == "2"
    assert _flag_value(argv, "--tinker-server-port") == str(TINKER_PORT)
    assert (
        _flag_value(argv, "--tinker-checkpoint-root")
        == "/checkpoints/demo-tinker/tinker"
    )
    assert _flag_value(argv, "--tinker-base-model") == "Qwen/Qwen3-30B-A3B"
    assert "--tinker-train-attn" in argv


def _capture_server_kwargs(recipe: MilesRecipe, **launch_kwargs) -> dict:
    """Build the gateway app against a fake ``modal.App`` and return the
    ``@app.server(...)`` kwargs and app tags."""
    captured: dict = {}

    class FakeApp:
        def __init__(self, name, *, tags=None, **_kwargs):
            captured["app_name"] = name
            captured["tags"] = tags

        def server(self, **kwargs):
            captured["server"] = kwargs
            return lambda cls: cls

    with (
        patch("modal.App", FakeApp),
        patch("modal.Volume"),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.hf_secrets",
            return_value=[],
        ),
        patch(
            "modal_training_gym.frameworks.miles.launcher.build_miles_source_image"
        ) as build_image,
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.resolve_checkpoint_volumes",
            return_value=("demo-checkpoints", "/checkpoints", object()),
        ),
    ):
        build_tinker_gateway_app(
            miles=recipe, model=Qwen3_30B(), app_name="demo-tinker", **launch_kwargs
        )
    image = build_image.return_value.add_local_python_source.return_value
    image.uv_pip_install.assert_called_once_with(f"tinker=={TINKER_SDK_VERSION}")
    return captured


def test_gateway_app_is_authenticated_server_on_tinker_port() -> None:
    captured = _capture_server_kwargs(Qwen3_30B_A3B_Tinker_Recipe())
    server = captured["server"]
    assert server["port"] == TINKER_PORT
    assert server["unauthenticated"] is False
    assert server["gpu"] == "H100:8"
    assert server["min_containers"] == server["max_containers"] == 1
    assert set(map(str, server["volumes"])) == {
        "/root/.cache/huggingface",
        "/checkpoints",
    }
    assert captured["tags"]["_modal_job_type"] == "serving"
    assert captured["tags"]["_modal_framework"] == "miles-tinker"


def test_gateway_app_can_opt_out_of_proxy_auth() -> None:
    captured = _capture_server_kwargs(
        Qwen3_30B_A3B_Tinker_Recipe(), unauthenticated=True
    )
    assert captured["server"]["unauthenticated"] is True


def test_gateway_app_uses_recipe_port_and_multi_node_cluster() -> None:
    recipe = replace(
        Qwen3_30B_A3B_Tinker_Recipe(),
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        tinker_server_port=9000,
    )
    assert recipe.total_nodes == 2
    server = _capture_server_kwargs(recipe)["server"]
    assert server["port"] == 9000
    # One container per Miles node; Modal requires the pool to be a multiple
    # of the cluster size.
    assert server["min_containers"] == server["max_containers"] == 2
    assert server["experimental_options"] == {"efa_enabled": True}


def test_gateway_app_rejects_ordinary_recipe() -> None:
    with pytest.raises(TrainingGymConfigError, match="multi_lora_n_adapters"):
        build_tinker_gateway_app(
            miles=MilesRecipe(), model=Qwen3_30B(), app_name="demo-tinker"
        )


def test_wait_for_gateway_ready_surfaces_job_death() -> None:
    with pytest.raises(RuntimeError, match="exited before serving: boom"):
        wait_for_gateway_ready(port=1, timeout=30, is_dead=lambda: "boom")


def test_wait_for_gateway_ready_times_out() -> None:
    with pytest.raises(TimeoutError):
        wait_for_gateway_ready(port=1, timeout=0, is_dead=lambda: None)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_forward_port_relays_to_target() -> None:
    target = socket.socket()
    target.bind(("127.0.0.1", 0))
    target.listen()
    target_port = target.getsockname()[1]
    listen_port = _free_port()

    def echo_once() -> None:
        conn, _ = target.accept()
        with conn:
            conn.sendall(conn.recv(1024).upper())

    threading.Thread(target=echo_once, daemon=True).start()

    loop = asyncio.new_event_loop()
    task = loop.create_task(forward_port(listen_port, "127.0.0.1", target_port))

    async def roundtrip() -> bytes:
        for _ in range(50):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", listen_port)
                break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            raise AssertionError("forwarder never started listening")
        writer.write(b"ping")
        await writer.drain()
        data = await reader.read(1024)
        writer.close()
        return data

    try:
        assert loop.run_until_complete(asyncio.wait_for(roundtrip(), 5)) == b"PING"
    finally:
        task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        loop.close()
        target.close()


def test_handle_exposes_health_url_and_sdk_pin() -> None:
    gw = TinkerGateway(
        app_name="demo-tinker",
        model=Qwen3_30B(),
        recipe=_gateway(),
        base_model="Qwen/Qwen3-30B-A3B",
        n_slots=2,
        checkpoint_root="/checkpoints/demo-tinker/tinker",
        checkpoints_volume_name="demo-checkpoints",
        url="https://demo--tinker.modal.run/",
    )
    assert gw.health_url == f"https://demo--tinker.modal.run{TINKER_HEALTH_PATH}"
    assert gw.unauthenticated is False
    assert TINKER_SDK_VERSION == "0.26.2"
