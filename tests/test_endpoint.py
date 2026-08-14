"""Unit tests for the managed Modal Endpoint wrapper in ``common/endpoint.py``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import modal
import pytest

from modal_training_gym.common import endpoint as endpoint_module
from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType
from modal_training_gym.common.endpoint import Endpoint
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig


class _FakeClock:
    """Deterministic stand-in for the ``time`` module used by endpoint.py.

    ``sleep()`` advances the monotonic clock instead of blocking, so polling
    loops and their timeouts resolve immediately and can be asserted on.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _FakeResponse:
    def __init__(
        self, status_code: int = 200, message: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._message = message or {"role": "assistant", "content": "ok"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://ws--ep.modal.run")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": self._message}]}


class _FakeModalCli:
    """Records ``modal endpoint create`` invocations and scripts ``get_url()``."""

    DEFAULT_URL = "https://ws--ep-test.modal.run"

    def __init__(self, urls: list[str | None | BaseException] | None) -> None:
        self.commands: list[list[str]] = []
        self.run_kwargs: list[dict[str, Any]] = []
        self.servers: list[tuple[str, str, str | None]] = []
        self._urls = urls

    def run(self, command: list[str], **kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        self.run_kwargs.append(kwargs)
        return SimpleNamespace(returncode=0)

    def from_name(
        self, app_name: str, cls_name: str, environment_name: str | None = None
    ) -> SimpleNamespace:
        self.servers.append((app_name, cls_name, environment_name))
        return SimpleNamespace(get_url=self.get_url)

    def get_url(self) -> str | None:
        if self._urls is None:
            return self.DEFAULT_URL
        value = self._urls.pop(0) if self._urls else None
        if isinstance(value, BaseException):
            raise value
        return value

    @property
    def last_command(self) -> list[str]:
        return self.commands[-1]

    def flag_value(self, flag: str) -> str:
        return self.last_command[self.last_command.index(flag) + 1]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(endpoint_module, "time", fake)
    return fake


@pytest.fixture
def fake_modal_cli(monkeypatch: pytest.MonkeyPatch, clock: _FakeClock):
    def _install(urls: list[str | None | BaseException] | None = None) -> _FakeModalCli:
        cli = _FakeModalCli(urls)
        monkeypatch.setattr(endpoint_module.subprocess, "run", cli.run)
        monkeypatch.setattr(
            endpoint_module,
            "modal",
            SimpleNamespace(
                Server=SimpleNamespace(from_name=cli.from_name),
                exception=modal.exception,
            ),
        )
        return cli

    return _install


def _endpoint(*, requires_proxy_auth: bool = False) -> Endpoint:
    return Endpoint(
        "https://ws--ep.modal.run",
        endpoint_name="my-ft",
        model_name="model",
        requires_proxy_auth=requires_proxy_auth,
    )


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen3-4B", ModelConfig(model_name="Qwen/Qwen3-4B")]
)
def test_launch_accepts_strings_and_model_configs(
    fake_modal_cli, model: ModelConfig | str
) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch(model, unauthenticated=True)

    assert cli.flag_value("--model") == "Qwen/Qwen3-4B"
    assert endpoint.model_name == "Qwen/Qwen3-4B"


def test_launch_creates_a_public_endpoint(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)

    assert cli.last_command[1:5] == ["-m", "modal", "endpoint", "create"]
    assert cli.flag_value("--model") == "Qwen/Qwen3-4B"
    assert cli.flag_value("--name") == endpoint.endpoint_name
    assert "--unauthenticated" in cli.last_command
    assert "--env" not in cli.last_command
    assert "--routing-region" not in cli.last_command
    assert "--custom-volume-name" not in cli.last_command
    assert cli.run_kwargs[-1] == {"check": True, "timeout": 120}
    assert endpoint.model_name == "Qwen/Qwen3-4B"
    assert endpoint.requires_proxy_auth is False


def test_launch_omits_the_unauthenticated_flag_for_proxy_auth(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=False)

    assert "--unauthenticated" not in cli.last_command
    assert endpoint.requires_proxy_auth is True


def test_launch_forwards_environment_and_routing_region(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    Endpoint.launch(
        "Qwen/Qwen3-4B",
        unauthenticated=True,
        environment="dev",
        routing_region="us-east",
    )

    assert cli.flag_value("--env") == "dev"
    assert cli.flag_value("--routing-region") == "us-east"
    assert cli.servers[-1][2] == "dev"


def _checkpoint(
    path: str = "/checkpoints/run-1/iter_10_hf",
    mount_path: str = "/checkpoints",
) -> Checkpoint:
    return Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name=path.rstrip("/").rsplit("/", 1)[-1],
        path=path,
        timestamp=0.0,
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path=mount_path,
    )


@pytest.mark.parametrize(
    ("path", "mount_path", "expected"),
    [
        ("/checkpoints/run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints/run-1/iter_10_hf/", "/checkpoints", "run-1/iter_10_hf"),
        ("/data/ckpt/iter_5_hf", "/data/ckpt", "iter_5_hf"),
        ("run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints", "/checkpoints", ""),
    ],
)
def test_checkpoint_path_relative_to_volume_strips_the_mount_prefix(
    path: str, mount_path: str, expected: str
) -> None:
    assert _checkpoint(path, mount_path).path_relative_to_volume == expected


@pytest.mark.parametrize(
    ("path", "mount_path", "expected"),
    [
        ("/checkpoints/run-1/iter_10_hf", "/checkpoints", "run-1/iter_10_hf"),
        ("/checkpoints/run-1/iter_10_hf/", "/checkpoints", "run-1/iter_10_hf"),
        ("/data/ckpt/iter_5_hf", "/data/ckpt", "iter_5_hf"),
    ],
)
def test_launch_mounts_the_checkpoint_relative_to_the_volume_root(
    fake_modal_cli, path: str, mount_path: str, expected: str
) -> None:
    cli = fake_modal_cli()

    Endpoint.launch(
        "Qwen/Qwen3-4B", _checkpoint(path, mount_path), unauthenticated=True
    )

    assert cli.flag_value("--custom-volume-name") == "gym-checkpoints"
    assert cli.flag_value("--custom-volume-path") == expected


def test_launch_converts_megatron_checkpoints_before_create(
    fake_modal_cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = fake_modal_cli()
    megatron = Checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_10",
        path="/checkpoints/run-1/iter_10",
        timestamp=0.0,
        training_run_id="run-1",
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    converted = _checkpoint("/checkpoints/run-1/iter_10_hf")
    seen: dict[str, Any] = {}

    def convert(checkpoint, model, **kwargs):
        seen["checkpoint"] = checkpoint
        seen["model"] = model
        return converted

    monkeypatch.setattr(endpoint_module, "convert_megatron_checkpoint_to_hf", convert)

    Endpoint.launch("Qwen/Qwen3-4B", megatron, unauthenticated=True)

    assert seen["checkpoint"] is megatron
    assert seen["model"].model_name == "Qwen/Qwen3-4B"
    assert cli.flag_value("--custom-volume-path") == "run-1/iter_10_hf"


def test_launch_leaves_hf_checkpoints_unchanged(
    fake_modal_cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = fake_modal_cli()
    checkpoint = _checkpoint()
    seen: list[tuple[Checkpoint, Checkpoint]] = []
    original = endpoint_module.convert_megatron_checkpoint_to_hf

    def convert(checkpoint, model, **kwargs):
        result = original(checkpoint, model)
        seen.append((checkpoint, result))
        return result

    monkeypatch.setattr(endpoint_module, "convert_megatron_checkpoint_to_hf", convert)

    Endpoint.launch("Qwen/Qwen3-4B", checkpoint, unauthenticated=True)

    assert seen == [(checkpoint, checkpoint)]
    assert cli.flag_value("--custom-volume-path") == "run-1/iter_10_hf"


def test_launch_skips_conversion_for_hub_models(
    fake_modal_cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_modal_cli()

    def convert(*args, **kwargs):
        raise AssertionError("hub launch converted a checkpoint")

    monkeypatch.setattr(endpoint_module, "convert_megatron_checkpoint_to_hf", convert)

    Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)


def test_launch_derives_stable_names_from_the_serving_spec(fake_modal_cli) -> None:
    fake_modal_cli()

    public = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=True)
    repeated = Endpoint.launch(
        ModelConfig(model_name="Qwen/Qwen3-4B"), unauthenticated=True
    )
    authenticated = Endpoint.launch("Qwen/Qwen3-4B", unauthenticated=False)
    other_model = Endpoint.launch("Qwen/Qwen3-8B", unauthenticated=True)
    other_region = Endpoint.launch(
        "Qwen/Qwen3-4B", unauthenticated=True, routing_region="us-east"
    )
    checkpointed = Endpoint.launch("Qwen/Qwen3-4B", _checkpoint(), unauthenticated=True)
    other_checkpoint = Endpoint.launch(
        "Qwen/Qwen3-4B",
        _checkpoint("/checkpoints/run-1/iter_20_hf"),
        unauthenticated=True,
    )

    assert public.endpoint_name.startswith("training-gym-")
    assert public.endpoint_name == repeated.endpoint_name
    assert (
        len(
            {
                public.endpoint_name,
                authenticated.endpoint_name,
                other_model.endpoint_name,
                other_region.endpoint_name,
                checkpointed.endpoint_name,
                other_checkpoint.endpoint_name,
            }
        )
        == 6
    )


def test_launch_uses_an_explicit_endpoint_name(fake_modal_cli) -> None:
    cli = fake_modal_cli()

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B", endpoint_name="my-ft", unauthenticated=True
    )

    assert endpoint.endpoint_name == "my-ft"
    assert cli.flag_value("--name") == "my-ft"
    assert cli.servers[-1][:2] == ("ep-my-ft", "Server")


def test_launch_polls_until_the_server_publishes_a_url(
    fake_modal_cli, clock: _FakeClock
) -> None:
    fake_modal_cli(
        [
            modal.exception.NotFoundError("no server yet"),
            None,
            "https://ws--ep-my-ft.modal.run/",
        ]
    )

    endpoint = Endpoint.launch(
        "Qwen/Qwen3-4B", endpoint_name="my-ft", unauthenticated=True
    )

    assert endpoint.url == "https://ws--ep-my-ft.modal.run"
    assert clock.slept == [1, 1]


def test_launch_times_out_when_no_url_is_published(
    fake_modal_cli, clock: _FakeClock
) -> None:
    fake_modal_cli([])

    with pytest.raises(TimeoutError, match="my-ft"):
        Endpoint.launch(
            "Qwen/Qwen3-4B",
            endpoint_name="my-ft",
            unauthenticated=True,
            wait_timeout_sec=5,
        )

    assert clock.now >= 5


def test_headers_are_empty_for_public_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_KEY", "wk-test")
    monkeypatch.setenv("MODAL_SECRET", "ws-test")

    assert _endpoint()._headers() == {}


def test_headers_carry_proxy_credentials_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_KEY", "wk-test")
    monkeypatch.setenv("MODAL_SECRET", "ws-test")

    assert _endpoint(requires_proxy_auth=True)._headers() == {
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
    }


def test_headers_require_configured_proxy_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoint_module, "modal_proxy_auth_headers", lambda: {})

    with pytest.raises(TrainingGymConfigError, match="MODAL_KEY and MODAL_SECRET"):
        _endpoint(requires_proxy_auth=True)._headers()


def test_wait_until_ready_polls_the_model_route(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    responses = [_FakeResponse(503), _FakeResponse(404), _FakeResponse()]
    requests: list[tuple[str, dict[str, Any]]] = []

    def _get(url: str, **kwargs: Any) -> _FakeResponse:
        requests.append((url, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(endpoint_module.httpx, "get", _get)

    _endpoint().wait_until_ready(timeout=60)

    assert requests[0][0] == "https://ws--ep.modal.run/v1/models"
    assert requests[0][1]["headers"] == {}
    assert clock.slept == [2, 2]


def test_wait_until_ready_sends_proxy_headers_when_required(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx,
        "get",
        lambda *_, **kwargs: captured.update(kwargs) or _FakeResponse(),
    )

    _endpoint(requires_proxy_auth=True).wait_until_ready(timeout=60)

    assert captured["headers"]["Modal-Key"] == "wk"


@pytest.mark.parametrize("status_code", [401, 403])
def test_wait_until_ready_reports_rejected_proxy_credentials(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, status_code: int
) -> None:
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx, "get", lambda *_, **__: _FakeResponse(status_code)
    )

    with pytest.raises(RuntimeError, match="rejected proxy authentication"):
        _endpoint(requires_proxy_auth=True).wait_until_ready(timeout=60)


def test_wait_until_ready_surfaces_unexpected_statuses(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    monkeypatch.setattr(
        endpoint_module.httpx, "get", lambda *_, **__: _FakeResponse(418)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _endpoint().wait_until_ready(timeout=60)


def test_wait_until_ready_times_out_and_keeps_the_last_error(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    def _get(*_: Any, **__: Any) -> _FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(endpoint_module.httpx, "get", _get)

    with pytest.raises(TimeoutError, match="my-ft") as excinfo:
        _endpoint().wait_until_ready(timeout=6)

    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)
    assert clock.slept == [2, 2, 2]


def test_chat_posts_the_conversation_and_returns_the_assistant_message(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(message={"role": "assistant", "content": "hi"})

    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    message = _endpoint().chat([{"role": "user", "content": "hello"}], temperature=0)

    assert message == {"role": "assistant", "content": "hi"}
    assert captured["url"] == "https://ws--ep.modal.run/v1/chat/completions"
    assert captured["json"] == {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0,
    }
    assert captured["headers"] == {}


def test_chat_serializes_tool_arguments_without_mutating_messages(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(message={"role": "assistant", "content": ""})

    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"key": "value"},
                    },
                }
            ],
        }
    ]

    _endpoint().chat(messages)

    assert captured["json"]["messages"][0]["tool_calls"][0]["function"][
        "arguments"
    ] == ('{"key": "value"}')
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"key": "value"}


def test_chat_sends_proxy_headers_when_required(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx,
        "post",
        lambda *_, **kwargs: captured.update(kwargs) or _FakeResponse(),
    )

    _endpoint(requires_proxy_auth=True).chat([])

    assert captured["headers"]["Modal-Key"] == "wk"


def test_chat_requires_proxy_credentials_before_posting(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    def _post(*_: Any, **__: Any) -> _FakeResponse:
        raise AssertionError("chat must not post without proxy credentials")

    monkeypatch.setattr(endpoint_module, "modal_proxy_auth_headers", lambda: {})
    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    with pytest.raises(TrainingGymConfigError, match="Proxy authentication requires"):
        _endpoint(requires_proxy_auth=True).chat([])


def test_chat_retries_transient_statuses(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    responses = [_FakeResponse(503), _FakeResponse(429), _FakeResponse()]
    monkeypatch.setattr(
        endpoint_module.httpx, "post", lambda *_, **__: responses.pop(0)
    )

    assert _endpoint().chat([])["content"] == "ok"
    assert clock.slept == [2, 4]


def test_chat_raises_after_the_last_transient_attempt(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    monkeypatch.setattr(
        endpoint_module.httpx, "post", lambda *_, **__: _FakeResponse(503)
    )

    with pytest.raises(httpx.HTTPStatusError):
        _endpoint().chat([], max_attempts=2)

    assert clock.slept == [2]


def test_chat_retries_request_errors_then_reraises(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    attempts: list[int] = []

    def _post(*_: Any, **__: Any) -> _FakeResponse:
        attempts.append(len(attempts) + 1)
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(endpoint_module.httpx, "post", _post)

    with pytest.raises(httpx.ConnectError):
        _endpoint().chat([], max_attempts=3)

    assert len(attempts) == 3
    assert clock.slept == [2, 4]


@pytest.mark.parametrize("status_code", [401, 403])
def test_chat_reports_rejected_proxy_credentials(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, status_code: int
) -> None:
    monkeypatch.setattr(
        endpoint_module,
        "modal_proxy_auth_headers",
        lambda: {"Modal-Key": "wk", "Modal-Secret": "ws"},
    )
    monkeypatch.setattr(
        endpoint_module.httpx, "post", lambda *_, **__: _FakeResponse(status_code)
    )

    with pytest.raises(RuntimeError, match="Proxy credentials were rejected"):
        _endpoint(requires_proxy_auth=True).chat([])
