import pytest
from pydantic import ValidationError

from modal_training_gym import TrackioConfig, WandbConfig
from modal_training_gym.common.metrics import (
    metric_cli_fields,
    metric_runtime_env,
    metrics_metadata,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


_SLIME_KW = dict(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    rollout_num_gpus_per_engine=1,
    num_rollout=1,
    rollout_batch_size=4,
    rollout_max_response_len=256,
    rollout_temperature=1.0,
    save_interval=1,
)


def _flags(args: list[str]) -> dict[str, str | bool]:
    flags: dict[str, str | bool] = {}
    index = 0
    while index < len(args):
        flag = args[index]
        value = True
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            value = args[index + 1]
            index += 1
        flags[flag] = value
        index += 1
    return flags


def test_wandb_positional_arguments_keep_original_order() -> None:
    config = WandbConfig("project", "entity", "group")

    assert config.project == "project"
    assert config.entity == "entity"
    assert config.group == "group"


def test_wandb_config_works_through_metrics_field() -> None:
    recipe = SlimeRecipe(
        **_SLIME_KW,
        metrics=WandbConfig(project="training", key="local-key"),
    )

    flags = _flags(recipe.cli_args())

    assert flags["--use-wandb"] is True
    assert flags["--wandb-key"] == "local-key"


def test_recipe_rejects_old_wandb_field() -> None:
    with pytest.raises(ValidationError, match="wandb"):
        SlimeRecipe(
            **_SLIME_KW,
            wandb=WandbConfig(project="training"),
        )


def test_trackio_runtime_env_and_metadata_are_not_secret_bearing() -> None:
    config = TrackioConfig(
        project="training",
        group="smoke",
        space_id="modal/training-gym",
        server_url="https://trackio.example.com?write_token=secret",
    )

    env = metric_runtime_env(config, run_id="run-1")
    metadata = metrics_metadata(config, run_id="run-1")

    assert env["TRAINING_GYM_METRICS_PROVIDER"] == "trackio"
    assert "WANDB_RUN_ID" not in env
    assert "WANDB_RESUME" not in env
    assert env["TRACKIO_SPACE_ID"] == "modal/training-gym"
    assert metadata["provider"] == "trackio"
    assert metadata["label"] == "Trackio"
    assert metadata["url"] == "https://huggingface.co/spaces/modal/training-gym"
    assert "secret" not in repr(metadata)


def test_metric_cli_fields_for_wandb_include_key() -> None:
    fields = metric_cli_fields(WandbConfig(project="training", key="local-key"))

    assert fields["wandb_key"] == "local-key"


def test_redact_env_values_hides_credentials_in_runtime_env() -> None:
    from modal_training_gym.common.launcher_utils import (
        REDACTED,
        redact_env_values,
    )

    env = {
        "env_vars": {
            "WANDB_RUN_ID": "run-1",
            "HF_TOKEN": "hf_credential_value_1234567890",
            "TRACKIO_WRITE_TOKEN": "write-token",
            "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN": "status-token",
            "MAX_TOKENS_PER_GPU": "4096",
        }
    }

    redacted = redact_env_values(env)["env_vars"]

    assert redacted["WANDB_RUN_ID"] == "run-1"
    assert redacted["HF_TOKEN"] == REDACTED
    assert redacted["TRACKIO_WRITE_TOKEN"] == REDACTED
    assert redacted["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] == REDACTED
    assert redacted["MAX_TOKENS_PER_GPU"] == "4096"


def test_redact_env_values_hides_credentials_in_url_query_strings() -> None:
    from modal_training_gym.common.launcher_utils import redact_env_values

    env = {
        "env_vars": {
            "TRACKIO_SERVER_URL": (
                "https://trackio.example.com/path?write_token=secret&mode=live"
            )
        }
    }

    url = redact_env_values(env)["env_vars"]["TRACKIO_SERVER_URL"]

    assert "secret" not in url
    assert "mode=live" in url


def test_redact_cli_command_hides_wandb_key() -> None:
    from modal_training_gym.common.launcher_utils import (
        REDACTED,
        redact_cli_command,
    )

    cmd = (
        "python3 train.py --use-wandb --wandb-project training --wandb-group g "
        "--disable-wandb-random-suffix --wandb-key live-key "
        "--max-tokens-per-gpu 4096"
    )

    redacted = redact_cli_command(cmd)

    assert "live-key" not in redacted
    assert f"--wandb-key {REDACTED}" in redacted
    assert "--wandb-project training" in redacted
    assert "--max-tokens-per-gpu 4096" in redacted


def test_record_metric_attempt_seeds_from_legacy_wandb_attempts() -> None:
    from modal_training_gym.common.framework import Framework
    from modal_training_gym.common.run import TrainingRun, record_metric_attempt

    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
        metadata={
            "wandb_attempts": [
                {
                    "attempt": 1,
                    "entity": "modal",
                    "project": "training",
                    "run_id": "run-1",
                }
            ]
        },
    )

    record_metric_attempt(
        run,
        provider="wandb",
        label="W&B",
        entity="modal",
        project="training",
        group="",
        run_id="run-1-a2",
        attempt_count=2,
    )

    attempts = run.metadata["metrics_attempts"]
    assert [a["run_id"] for a in attempts] == ["run-1", "run-1-a2"]


def test_redact_env_values_hides_url_userinfo_credentials() -> None:
    from modal_training_gym.common.launcher_utils import redact_env_values

    env = {"TRACKIO_SERVER_URL": "https://user:token-value@trackio.example.com/path"}

    url = redact_env_values(env)["TRACKIO_SERVER_URL"]

    assert "token-value" not in url
    assert "trackio.example.com/path" in url


def test_redact_env_values_hides_bare_userinfo_token() -> None:
    from modal_training_gym.common.launcher_utils import redact_env_values

    env = {"TRACKIO_SERVER_URL": "https://hf_abc123@trackio.example.com/"}

    url = redact_env_values(env)["TRACKIO_SERVER_URL"]

    assert "hf_abc123" not in url
    assert "trackio.example.com" in url


def test_redact_cli_command_hides_secrets_in_json_flag_values() -> None:
    import shlex

    from modal_training_gym.common.launcher_utils import (
        REDACTED,
        redact_cli_command,
    )

    cmd = "python3 train.py --train-env-vars " + shlex.quote(
        '{"HF_TOKEN": "hf_secretvalue", "NCCL_DEBUG": "INFO"}'
    )

    redacted = redact_cli_command(cmd)

    assert "hf_secretvalue" not in redacted
    assert REDACTED in redacted
    assert "NCCL_DEBUG" in redacted
    assert "INFO" in redacted


def test_redact_cli_command_hides_secrets_in_bash_wrapped_json_flags() -> None:
    import shlex

    from modal_training_gym.common.launcher_utils import (
        REDACTED,
        redact_cli_command,
    )

    inner = "python3 train.py --wandb-key live-key --train-env-vars " + shlex.quote(
        '{"HF_TOKEN": "hf_secretvalue", "NCCL_DEBUG": "INFO"}'
    )
    cmd = "bash -c " + shlex.quote(inner)

    redacted = redact_cli_command(cmd)

    assert "hf_secretvalue" not in redacted
    assert "live-key" not in redacted
    assert REDACTED in redacted
    assert "NCCL_DEBUG" in redacted


def test_redact_cli_command_preserves_shell_syntax() -> None:
    import shlex

    from modal_training_gym.common.launcher_utils import (
        REDACTED,
        redact_cli_command,
    )

    inner = (
        'source /root/qwen3.sh && torchrun train.py "${MODEL_ARGS[@]}" '
        "--num-layers 36 --wandb-key live-key"
    )
    cmd = "bash -c " + shlex.quote(inner)

    redacted = redact_cli_command(cmd)

    assert redacted == "bash -c " + shlex.quote(inner.replace("live-key", REDACTED))


def test_redact_cli_command_hides_secret_values_starting_with_dashes() -> None:
    from modal_training_gym.common.launcher_utils import (
        REDACTED,
        redact_cli_command,
    )

    redacted = redact_cli_command("train.py --wandb-key --odd-secret-- --other x")

    assert "--odd-secret--" not in redacted
    assert f"--wandb-key {REDACTED}" in redacted
    assert "--other x" in redacted


def test_redact_cli_command_redacts_urls_in_generic_flag_values() -> None:
    from modal_training_gym.common.launcher_utils import redact_cli_command

    cmd = (
        "python train.py "
        "--custom-config-path 'https://user:tok@host/cfg.yaml?write_token=abc' "
        "--data-path /data/train.jsonl"
    )

    redacted = redact_cli_command(cmd)

    assert "tok" not in redacted.replace("write_token", "")
    assert "abc" not in redacted
    assert "--data-path /data/train.jsonl" in redacted


def test_trackio_tokens_delivered_via_inline_secret_not_runtime_env(
    monkeypatch,
) -> None:
    import modal

    from modal_training_gym.common.metrics import (
        inline_metric_secrets,
        named_metric_secrets,
    )

    config = TrackioConfig(
        project="p",
        token="hf_tok",
        write_token="wtok",
        server_url="https://user:tok@trackio.internal",
    )

    env = metric_runtime_env(config, run_id="run-1")
    assert "HF_TOKEN" not in env
    assert "TRACKIO_WRITE_TOKEN" not in env
    # A credential-bearing server URL rides the inline secret, not runtime env.
    assert "TRACKIO_SERVER_URL" not in env
    assert config.inline_secret_env()["TRACKIO_SERVER_URL"] == config.server_url
    plain = TrackioConfig(project="p", server_url="https://trackio.internal")
    assert (
        metric_runtime_env(plain, run_id="r")["TRACKIO_SERVER_URL"] == plain.server_url
    )

    class FakeSecret:
        def __init__(self, name: str = "", env: dict | None = None) -> None:
            self.name = name
            self.env = env

        @classmethod
        def from_name(cls, name: str) -> "FakeSecret":
            return cls(name=name)

        @classmethod
        def from_dict(cls, env: dict) -> "FakeSecret":
            return cls(env=env)

        def hydrate(self) -> None:
            pass

    monkeypatch.setattr(modal, "Secret", FakeSecret)

    secrets = [*named_metric_secrets(config), *inline_metric_secrets(config)]
    assert {
        "TRACKIO_SERVER_URL": config.server_url,
        "HF_TOKEN": "hf_tok",
        "TRACKIO_WRITE_TOKEN": "wtok",
    } in [s.env for s in secrets]


def test_named_metric_secrets_missing_secret_tolerated_only_for_default_name(
    monkeypatch,
) -> None:
    import modal

    from modal_training_gym.common.metrics import named_metric_secrets

    class FakeSecret:
        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        def from_name(cls, name: str) -> "FakeSecret":
            return cls(name)

        def hydrate(self) -> None:
            raise RuntimeError("secret not found")

    monkeypatch.setattr(modal, "Secret", FakeSecret)

    # Missing default-name secret is tolerated (optional provider).
    assert named_metric_secrets(TrackioConfig(project="p")) == []
    # An explicitly configured name must exist, so it is returned unhydrated
    # and a missing secret fails loudly at launch.
    custom = TrackioConfig(project="p", modal_secret_name="my-trackio-secret")
    secrets = named_metric_secrets(custom)
    assert [s.name for s in secrets] == ["my-trackio-secret"]


def test_redact_url_credentials_handles_schemeless_urls() -> None:
    from modal_training_gym.common.launcher_utils import redact_url_credentials

    url = redact_url_credentials("user:secret-pass@trackio.internal/path")

    assert "secret-pass" not in url
    assert "trackio.internal/path" in url
    assert not url.startswith("schemeless://")
    # Plain values with "@" but no credential shape are left untouched.
    assert redact_url_credentials("joe@example.com") == "joe@example.com"


def test_trackio_server_url_credentials_not_persisted() -> None:
    config = TrackioConfig(
        project="training",
        server_url="https://user:secret-pass@trackio.internal/view?x=1",
    )

    url = config.to_metadata()["url"]

    assert "secret-pass" not in url
    assert "@" not in url
    assert url == "https://trackio.internal/view"


def test_trackio_dashboard_url_credentials_not_persisted() -> None:
    config = TrackioConfig(
        project="training",
        dashboard_url="https://user:secret-pass@dash.example.com/view?write_token=tok",
    )

    url = config.to_metadata()["url"]

    assert "secret-pass" not in url
    assert "tok" not in url.rsplit("write_token=", 1)[-1].split("&")[0]
    assert "@" not in url
    assert url.startswith("https://dash.example.com/view")


def test_trackio_shim_command_is_a_single_line() -> None:
    from modal_training_gym.common.metrics import _TRACKIO_WANDB_SHIM_COMMAND

    assert "\n" not in _TRACKIO_WANDB_SHIM_COMMAND
    assert _TRACKIO_WANDB_SHIM_COMMAND.startswith("echo ")
    assert _TRACKIO_WANDB_SHIM_COMMAND.endswith(" | base64 -d | python3")


def test_recipe_metrics_field_preserves_subclass_instance() -> None:
    recipe = SlimeRecipe(
        **_SLIME_KW,
        metrics=WandbConfig(project="training", group="smoke"),
    )

    assert type(recipe.metrics) is WandbConfig
    assert recipe.metrics.project == "training"
    assert recipe.metrics.group == "smoke"
