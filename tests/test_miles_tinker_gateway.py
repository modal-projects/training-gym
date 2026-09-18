"""TinkerGateway launcher: the Modal app that hosts Miles' ``serve_tinker.py``.

Covers the parts that need no GPU: the served recipe honors the invariants
``serve_tinker.py`` asserts (``load == hf_checkpoint``, checkpoint root on the
checkpoints Volume), the Ray job runs ``serve_tinker.py`` rather than
``train.py``, and the Modal server is authenticated on the gateway port.
"""

import asyncio
import contextlib
import shlex
import socket
import threading
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import Qwen3_30B
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.run_summary import build_run_summary
from modal_training_gym.common.status import MilesStatus
from modal_training_gym.frameworks.miles.tinker_gateway import (
    TINKER_HEALTH_PATH,
    TINKER_PORT,
    TINKER_SDK_VERSION,
    TINKER_TIMING_PATCH_COMMAND,
    GatewayRunRecord,
    TinkerGateway,
    build_tinker_gateway_app,
    build_tinker_gateway_cmd,
    create_gateway_training_run,
    forward_port,
    gateway_checkpoint_root,
    gateway_recipe,
    gateway_training_run_id,
    mark_gateway_run_failed,
    mark_gateway_run_stopped,
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
    image.uv_pip_install.return_value.run_commands.assert_called_once_with(
        TINKER_TIMING_PATCH_COMMAND
    )
    return captured


def test_timing_patch_is_applied_after_source_overlays() -> None:
    # The base64 payload is the patcher itself; it runs after miles_git_ref
    # checkouts so a source overlay can't drop the instrumentation.
    assert "base64 -d | python3" in TINKER_TIMING_PATCH_COMMAND
    assert "forward_backward" in TINKER_TIMING_PATCH_COMMAND


def _persist_stub(saved: list[TrainingRun], tokens: dict) -> contextlib.ExitStack:
    """Stub the metadata Volume: saves append to ``saved``; ``from_id`` returns
    a copy of the most recent save (what the dashboard would have stored)."""

    def _from_id(run_id: str, **_: object) -> TrainingRun:
        stored = next(r for r in reversed(saved) if r.training_run_id == run_id)
        return TrainingRun.model_validate(stored.model_dump(mode="json"))

    stack = contextlib.ExitStack()
    stack.enter_context(patch("modal_training_gym.cli.setup.ensure_dashboard_deployed"))
    stack.enter_context(
        patch(
            "modal_training_gym.common.config.get_framework_status_url",
            return_value="https://gym.test/api/framework-status",
        )
    )
    stack.enter_context(
        patch.object(
            TrainingRun,
            "save",
            autospec=True,
            side_effect=lambda run: saved.append(run),
        )
    )
    stack.enter_context(
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.vol_put",
            side_effect=lambda store, key, value: tokens.__setitem__(key, value),
        )
    )
    stack.enter_context(patch.object(TrainingRun, "from_id", side_effect=_from_id))
    return stack


def _create_run(recipe: MilesRecipe | None = None, **kwargs):
    saved: list[TrainingRun] = []
    tokens: dict = {}
    with _persist_stub(saved, tokens):
        run, record = create_gateway_training_run(
            recipe or Qwen3_30B_A3B_Tinker_Recipe(),
            Qwen3_30B(),
            app_name="demo-tinker",
            checkpoint_root="/checkpoints/demo-tinker/tinker",
            checkpoints_volume_name="demo-checkpoints",
            unauthenticated=False,
            **kwargs,
        )
    return run, record, saved, tokens


def test_gateway_run_is_a_running_miles_run_without_dataset() -> None:
    run, record, saved, tokens = _create_run()

    assert saved == [run]
    assert run.training_run_id == record.training_run_id
    assert run.dataset_id == ""
    assert run.framework.value == "miles"
    assert run.status is TrainingRunStatus.RUNNING
    assert run.framework_status is MilesStatus.INITIALIZING
    assert run.config["dataset"] == {}
    assert run.config["entrypoint"] == "serve_tinker.py"
    assert run.config["model"]["model_name"] == "Qwen/Qwen3-30B-A3B"
    assert run.config["recipe"]["multi_lora_n_adapters"] == 4
    assert run.app_name == "demo-tinker"
    assert run.metadata is not None
    assert run.metadata["run_type"] == "tinker_gateway"
    assert run.metadata["n_slots"] == 4
    assert run.metadata["base_model"] == "Qwen/Qwen3-30B-A3B"
    assert run.metadata["checkpoint_root"] == "/checkpoints/demo-tinker/tinker"
    assert run.source_model is not None
    assert run.source_model.model_name == "Qwen/Qwen3-30B-A3B"
    assert run.metadata["checkpoints_volume_name"] == "demo-checkpoints"
    assert run.metadata["unauthenticated"] is False
    assert run.metadata["gateway_url"] == ""


def test_gateway_run_token_is_stored_for_the_dashboard() -> None:
    _run, record, _saved, tokens = _create_run()

    assert record.framework_status_url == "https://gym.test/api/framework-status"
    assert len(record.framework_status_token) >= 32
    assert tokens == {record.training_run_id: {"token": record.framework_status_token}}


def test_gateway_run_ids_are_unique_per_launch() -> None:
    recipe, model = Qwen3_30B_A3B_Tinker_Recipe(), Qwen3_30B()
    a = gateway_training_run_id(recipe, model, app_name="demo-tinker")
    b = gateway_training_run_id(recipe, model, app_name="demo-tinker")
    assert a != b


def test_gateway_run_summary_is_ready_with_empty_dataset() -> None:
    run, _record, _saved, _tokens = _create_run()
    run.modal_app_id = "ap-123"
    run.metadata = {**(run.metadata or {}), "gateway_url": "https://gw.modal.run"}
    run.framework_status = MilesStatus.SERVING
    run.metadata["framework_progress"] = {"phase": "serving", "is_active": True}

    summary = build_run_summary(run.model_dump(mode="json"))

    assert summary.status == "running"
    assert summary.display_status == "ready"
    assert summary.display_stage == "Serving Tinker API"
    assert summary.dataset == ""
    assert summary.recipe == "miles-tinker"
    assert summary.model == "Qwen/Qwen3-30B-A3B"


def test_gateway_run_summary_is_pending_until_serving() -> None:
    run, _record, _saved, _tokens = _create_run()
    run.modal_app_id = "ap-123"
    for status in (MilesStatus.DOWNLOAD_MODEL, MilesStatus.INITIALIZING):
        run.framework_status = status
        summary = build_run_summary(run.model_dump(mode="json"))
        assert summary.display_status == "pending"
        assert summary.recipe == "miles-tinker"


def test_stopped_gateway_run_summary_is_stopped() -> None:
    run, _record, _saved, _tokens = _create_run()
    run.status = TrainingRunStatus.STOPPED
    summary = build_run_summary(run.model_dump(mode="json"))
    assert summary.display_status == "stopped"


def test_ordinary_run_summary_is_not_marked_ready() -> None:
    run, _record, _saved, _tokens = _create_run()
    run.metadata = {}
    summary = build_run_summary(run.model_dump(mode="json"))
    assert summary.display_status == "pending"
    assert summary.recipe == "miles"


def test_mark_gateway_run_stopped_terminalizes_running_run_once() -> None:
    run, _record, _saved, _tokens = _create_run()
    run.started_at = int(run.started_at) - 120
    saved: list[TrainingRun] = []
    with (
        patch.object(TrainingRun, "from_id", return_value=run),
        patch.object(
            TrainingRun, "save", autospec=True, side_effect=lambda r: saved.append(r)
        ),
    ):
        mark_gateway_run_stopped(run.training_run_id)
        mark_gateway_run_stopped(run.training_run_id)

    assert len(saved) == 1
    assert run.status is TrainingRunStatus.STOPPED
    assert run.ended_at == run.completed_at
    assert run.duration_seconds is not None and run.duration_seconds >= 120


def test_mark_gateway_run_stopped_leaves_failed_run_alone() -> None:
    run, _record, _saved, _tokens = _create_run()
    run.status = TrainingRunStatus.FAILED
    with (
        patch.object(TrainingRun, "from_id", return_value=run),
        patch.object(TrainingRun, "save", autospec=True) as save,
    ):
        mark_gateway_run_stopped(run.training_run_id)
    save.assert_not_called()
    assert run.status is TrainingRunStatus.FAILED


def test_launch_records_run_then_binds_app_id_and_url() -> None:
    saved: list[TrainingRun] = []
    tokens: dict = {}

    class FakeApp:
        app_id = "ap-gateway"

        def deploy(self, *, environment_name=None):
            pass

        TinkerGatewayServer = MagicMock(
            **{"get_url.return_value": "https://gw.modal.run/"}
        )

    with (
        _persist_stub(saved, tokens),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.resolve_checkpoint_volumes",
            return_value=("demo-checkpoints", "/checkpoints", object()),
        ),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.build_tinker_gateway_app",
            return_value=FakeApp(),
        ) as build_app,
    ):
        gw = TinkerGateway.launch(Qwen3_30B_A3B_Tinker_Recipe(), model=Qwen3_30B())

    record = build_app.call_args.kwargs["run_record"]
    assert isinstance(record, GatewayRunRecord)
    assert record.training_run_id == gw.training_run_id
    assert record.framework_status_token == tokens[gw.training_run_id]["token"]

    # Saved once before deploy (so the containers can post status) and once
    # after, with the Modal app + URL bound.
    assert [r.training_run_id for r in saved] == [gw.training_run_id] * 2
    final = saved[-1]
    assert final.modal_app_id == "ap-gateway"
    assert final.modal_app_url == gw.modal_app_url
    assert final.metadata is not None
    assert final.metadata["gateway_url"] == "https://gw.modal.run"
    assert final.status is TrainingRunStatus.RUNNING
    assert gw.url == "https://gw.modal.run"


def test_launch_binds_app_without_reverting_container_status_updates() -> None:
    saved: list[TrainingRun] = []
    tokens: dict = {}

    class FakeApp:
        app_id = "ap-gateway"
        TinkerGatewayServer = MagicMock(
            **{"get_url.return_value": "https://gw.modal.run"}
        )

        def deploy(self, *, environment_name=None):
            # The head container posts ``serving`` before deploy() returns.
            stored = TrainingRun.model_validate(saved[-1].model_dump(mode="json"))
            stored.framework_status = MilesStatus.SERVING
            saved.append(stored)

    with (
        _persist_stub(saved, tokens),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.resolve_checkpoint_volumes",
            return_value=("demo-checkpoints", "/checkpoints", object()),
        ),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.build_tinker_gateway_app",
            return_value=FakeApp(),
        ),
    ):
        TinkerGateway.launch(Qwen3_30B_A3B_Tinker_Recipe(), model=Qwen3_30B())

    final = saved[-1]
    assert final.modal_app_id == "ap-gateway"
    assert final.framework_status is MilesStatus.SERVING


def test_launch_stops_live_app_and_fails_run_when_finalization_raises() -> None:
    saved: list[TrainingRun] = []
    tokens: dict = {}

    class FakeApp:
        app_id = "ap-gateway"
        TinkerGatewayServer = MagicMock(**{"get_url.return_value": ""})

        def deploy(self, *, environment_name=None):
            pass

    with (
        _persist_stub(saved, tokens),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.resolve_checkpoint_volumes",
            return_value=("demo-checkpoints", "/checkpoints", object()),
        ),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.build_tinker_gateway_app",
            return_value=FakeApp(),
        ),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.subprocess.run"
        ) as run_cmd,
        pytest.raises(RuntimeError, match="no web URL"),
    ):
        TinkerGateway.launch(
            Qwen3_30B_A3B_Tinker_Recipe(), model=Qwen3_30B(), environment_name="dev"
        )

    run_cmd.assert_called_once_with(
        ["modal", "app", "stop", "-y", "--env", "dev", "ap-gateway"], check=False
    )
    assert saved[-1].status is TrainingRunStatus.FAILED
    assert saved[-1].error_message is not None
    assert "no web URL" in saved[-1].error_message


def test_mark_gateway_run_failed_records_ray_error() -> None:
    run, _record, saved, _tokens = _create_run()
    with _persist_stub(saved, {}):
        mark_gateway_run_failed(run.training_run_id, "Ray job status FAILED")
        mark_gateway_run_failed(run.training_run_id, "again")
    assert saved[-1].status is TrainingRunStatus.FAILED
    assert saved[-1].error_message == "Ray job status FAILED"
    assert sum(r.status is TrainingRunStatus.FAILED for r in saved) == 1


def test_launch_marks_run_failed_when_app_build_raises() -> None:
    saved: list[TrainingRun] = []
    with (
        _persist_stub(saved, {}),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.resolve_checkpoint_volumes",
            return_value=("demo-checkpoints", "/checkpoints", object()),
        ),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.build_tinker_gateway_app",
            side_effect=ValueError("bad overlay"),
        ),
        pytest.raises(ValueError, match="bad overlay"),
    ):
        TinkerGateway.launch(Qwen3_30B_A3B_Tinker_Recipe(), model=Qwen3_30B())

    assert saved[-1].status is TrainingRunStatus.FAILED
    assert saved[-1].error_message == "ValueError: bad overlay"


def test_create_run_marks_run_failed_when_token_write_raises() -> None:
    saved: list[TrainingRun] = []
    with (
        _persist_stub(saved, {}),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.vol_put",
            side_effect=OSError("volume unavailable"),
        ),
        pytest.raises(OSError, match="volume unavailable"),
    ):
        create_gateway_training_run(
            Qwen3_30B_A3B_Tinker_Recipe(),
            Qwen3_30B(),
            app_name="demo-tinker",
            checkpoint_root="/checkpoints/demo-tinker/tinker",
            checkpoints_volume_name="demo-checkpoints",
            unauthenticated=False,
        )

    assert saved[0].status is TrainingRunStatus.RUNNING
    assert saved[-1].status is TrainingRunStatus.FAILED


def test_launch_marks_run_failed_when_deploy_raises() -> None:
    saved: list[TrainingRun] = []
    tokens: dict = {}

    class FakeApp:
        def deploy(self, *, environment_name=None):
            raise RuntimeError("no capacity")

    with (
        _persist_stub(saved, tokens),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.resolve_checkpoint_volumes",
            return_value=("demo-checkpoints", "/checkpoints", object()),
        ),
        patch(
            "modal_training_gym.frameworks.miles.tinker_gateway.build_tinker_gateway_app",
            return_value=FakeApp(),
        ),
        pytest.raises(RuntimeError, match="no capacity"),
    ):
        TinkerGateway.launch(Qwen3_30B_A3B_Tinker_Recipe(), model=Qwen3_30B())

    assert saved[-1].status is TrainingRunStatus.FAILED
    assert saved[-1].error_message == "RuntimeError: no capacity"
    assert saved[-1].ended_at is not None


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
