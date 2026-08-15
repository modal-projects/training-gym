from __future__ import annotations

import pytest

from modal_training_gym.common import run_summary as run_summary_module
from modal_training_gym.common import tracker as tracker_module
from modal_training_gym.common.run_summary import (
    build_run_summaries,
    build_run_summary,
)

CUSTOM_PROJECT_URL = "https://metrics.example.com/?project={project}"


@pytest.fixture(autouse=True)
def default_tracker(monkeypatch):
    """These assert wandb.ai links, so the developer's own tracker env must not leak in."""
    for name in (
        tracker_module.LABEL_ENV,
        tracker_module.RUN_URL_TEMPLATE_ENV,
        tracker_module.PROJECT_URL_TEMPLATE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    tracker_module.tracker_config.cache_clear()
    yield
    tracker_module.tracker_config.cache_clear()


def _run(**overrides):
    run = {
        "training_run_id": "run-1",
        "framework": "slime",
        "status": "running",
        "modal_app_id": "ap-123",
        "created_at": 100,
        "started_at": 110,
        "updated_at": 150,
        "config": {
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "dataset": {"hf_repo": "openai/gsm8k"},
            "recipe": {
                "gpu_type": "H100",
                "actor_num_nodes": 2,
                "actor_num_gpus_per_node": 8,
            },
            "wandb": {
                "entity": "modal",
                "project": "training",
                "group": "smoke",
            },
            "lr": 1e-6,
            "global_batch_size": 32,
        },
        "metadata": {
            "group_id": "sweep-a",
            "framework_progress": {
                "phase": "generate_rollouts",
                "current": 4,
                "total": 10,
                "unit": "step",
                "is_active": True,
                "rollout_id": 3,
                "updated_at": 149,
            },
            "latest_rollout": {
                "rollout_id": 3,
                "mean": 0.75,
                "total": 16,
                "created_at": 148,
            },
            "group_tags": {
                "group_id": "sweep-a",
                "axes": ["recipe.lr"],
                "overrides": {"recipe.lr": 1e-6},
                "tags": [],
            },
            "attempt_count": 2,
            "last_attempt_status": "running",
            "last_attempt_started_at": 140,
            "resumed_from_checkpoint": True,
            "resume_checkpoint_name": "iter_0000003",
            "resume_from_iteration": 3,
            "wandb_attempts": [
                {
                    "attempt": 2,
                    "entity": "modal",
                    "project": "training",
                    "run_id": "run-1-a2",
                }
            ],
        },
    }
    run.update(overrides)
    return run


def _result(**overrides):
    result = {
        "training_run_id": "run-1",
        "app_name": "training-app",
        "checkpoint_dir": "/checkpoints/run-1",
        "model_config": {
            "model_name": "Qwen/Qwen3-4B",
            "model_path": "/checkpoints/run-1/iter_0000004",
        },
        "wandb_entity": "modal",
        "wandb_project": "training",
        "wandb_training_run_id": "run-1-a2",
    }
    result.update(overrides)
    return result


def test_legacy_value_helpers_unwrap_and_normalize_values():
    wrapped = {"value": "legacy", "source": "old-store"}

    assert run_summary_module._unwrap(wrapped) == "legacy"
    assert run_summary_module._text(wrapped) == "legacy"
    assert run_summary_module._text(None) == ""
    assert run_summary_module._mapping({"value": {"key": "value"}}) == {"key": "value"}
    assert run_summary_module._mapping("not-a-mapping") == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        (123.5, 123),
        ("123", 123),
        ("123.5", 123),
        ({"value": "123.5", "source": "legacy"}, 123),
        ("2025-01-02T03:04:05Z", 1735787045),
        ("2025-01-02T03:04:05", 1735787045),
        ("", 0),
        ("not-a-timestamp", 0),
        (None, 0),
        (True, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        ("1e309", 0),
    ],
)
def test_timestamp_normalization(value, expected):
    assert run_summary_module._timestamp(value) == expected


def test_number_and_integer_helpers_apply_fallbacks():
    assert run_summary_module._number({"value": "2.5"}) == 2.5
    assert run_summary_module._number("invalid", default=7) == 7
    assert run_summary_module._number(float("inf"), default=7) == 7
    assert run_summary_module._optional_int("4.0") == 4
    assert run_summary_module._optional_int("4.25") is None
    assert run_summary_module._optional_int("") is None
    assert run_summary_module._optional_int("invalid") is None
    assert run_summary_module._integer("invalid", default=7) == 7


def test_url_helpers_match_old_frontend_behavior():
    assert run_summary_module._modal_app_url("") is None
    assert run_summary_module._modal_app_url("ap-123") == "https://modal.com/id/ap-123"
    assert (
        run_summary_module._modal_app_url("https://modal.com/apps/example")
        == "https://modal.com/apps/example"
    )
    assert run_summary_module._wandb_url(None, "project", "run") is None
    assert (
        run_summary_module._wandb_url("my entity", "project/name", "run id")
        == "https://wandb.ai/my%20entity/project%2Fname/runs/run%20id"
    )
    assert (
        run_summary_module._wandb_url("entity", "project", "")
        == "https://wandb.ai/entity/project"
    )


def test_config_summary_fallbacks_and_wandb_defaults():
    summary = run_summary_module._config_summary(
        {
            "model": {"model_name": "model"},
            "dataset": {
                "hf_repo": "repo/dataset",
                "prompt_data": "prompts.jsonl",
                "name": "fallback-name",
            },
            "preset": {"gpu_type": "A100"},
            "recipe": {"gpu_type": "H100"},
            "wandb": {
                "entity": "entity",
                "project": "project",
                "group": "group",
                "run_id": "abcdefgh",
            },
        }
    )

    assert summary.model_name == "model"
    assert summary.dataset_name == "repo/dataset"
    assert summary.dataset_prompt_data == "prompts.jsonl"
    assert summary.gpu_type == "H100"
    assert summary.lr == 0
    assert summary.global_batch_size == 0
    assert summary.wandb_training_run_id == "abcdefgh"
    assert summary.wandb_url == "https://wandb.ai/entity/project/runs/abcdefgh"
    assert run_summary_module._config_summary(None) == {}


def test_progress_rollout_and_resume_helpers_handle_missing_and_invalid_values():
    assert run_summary_module._framework_progress({}) is None
    progress = run_summary_module._framework_progress(
        {
            "framework_progress": {
                "current": "invalid",
                "total": 0,
                "unit": "",
                "is_active": "true",
                "rollout_id": "2.5",
                "step_id": "3",
                "updated_at": "2025-01-02T03:04:05Z",
            }
        }
    )
    assert progress.current is None
    assert progress.total is None
    assert progress.unit == "step"
    assert progress.is_active is None
    assert progress.rollout_id is None
    assert progress.step_id == 3
    assert progress.updated_at == 1735787045

    assert run_summary_module._latest_rollout({}) is None
    rollout = run_summary_module._latest_rollout(
        {
            "latest_rollout": {
                "rollout_id": "invalid",
                "mean": "invalid",
                "total": "3.5",
                "created_at": "100.5",
            }
        }
    )
    assert rollout.rollout_id == 0
    assert rollout.mean == 0
    assert rollout.total == 0
    assert rollout.created_at == 100

    assert run_summary_module._resume_state({"attempt_count": 1}) is None
    resume = run_summary_module._resume_state(
        {
            "attempt_count": 1,
            "resume_checkpoint_path": "/checkpoints/iter_1",
            "resume_from_iteration": "1.5",
            "last_attempt_started_at": "100.5",
        }
    )
    assert resume.resumed_from_checkpoint is True
    assert resume.resume_from_iteration is None
    assert resume.last_attempt_started_at == 100


def test_group_tag_helper_synthesizes_tags_from_overrides():
    assert run_summary_module._group_tags({}, "") is None

    tags = run_summary_module._group_tags(
        {
            "group_tags": {
                "overrides": {
                    "recipe.actor_num_nodes": 2,
                    "config.lr": 1e-6,
                }
            }
        },
        "group-1",
    )

    assert tags.group_id == "group-1"
    assert tags.axes == ["recipe.actor_num_nodes", "config.lr"]
    assert [(tag.key, tag.label, tag.value) for tag in tags.tags] == [
        ("recipe.actor_num_nodes", "actor num nodes", 2),
        ("config.lr", "lr", 1e-6),
    ]


def test_wandb_attempt_helpers_skip_invalid_links_and_dedupe_by_url():
    links = run_summary_module._wandb_attempt_links(
        {
            "wandb_attempts": [
                None,
                {"attempt": 1, "project": "missing-entity"},
                {
                    "attempt": "2",
                    "entity": "entity",
                    "project": "project",
                    "run_id": "run-2",
                },
            ]
        }
    )
    duplicate = run_summary_module.WandbLink(
        label="duplicate",
        url=links[0].url,
    )
    other = run_summary_module.WandbLink(
        label="other",
        url="https://wandb.ai/entity/project/runs/other",
    )

    assert links[0].label == "W&B a2"
    assert links[0].attempt == 2
    assert run_summary_module._dedupe_links(links, [duplicate, other]) == [
        links[0],
        other,
    ]


def test_missing_identity_uses_stable_fallbacks():
    summary = build_run_summary(
        {
            "modal_app_id": "ap-fallback",
            "created_at": "invalid",
            "config": None,
        },
        fallback_index=3,
    )

    assert summary.training_run_id == "unknown-run-ap-fallback-0-3"
    assert summary.run_id == summary.training_run_id
    assert summary.status == "running"
    assert summary.framework == "(untagged)"
    assert summary.created_at == 0
    assert summary.started_at == 0
    assert summary.updated_at == 0
    assert summary.config_summary == {}


def test_build_current_summary_joins_result_and_derives_public_fields():
    summary = build_run_summary(
        _run(framework_status="generate_rollouts"),
        _result(),
    )

    assert summary.training_run_id == "run-1"
    assert summary.run_id == "run-1"
    assert summary.status == "running"
    assert summary.display_status == "completed"
    assert summary.display_stage == "Generating rollouts"
    assert summary.model == "Qwen/Qwen3-4B"
    assert summary.dataset == "openai/gsm8k"
    assert summary.recipe == "slime"
    assert summary.group_id == "sweep-a"
    assert summary.framework_progress.current == 4
    assert summary.framework_progress.is_active is True
    assert summary.latest_rollout.mean == 0.75
    assert summary.has_train_result is True
    assert summary.train_result.checkpoint_dir == "/checkpoints/run-1"
    assert summary.modal_app_url == "https://modal.com/id/ap-123"
    assert summary.resume_state.attempt_count == 2
    assert summary.group_tags.tags[0].key == "recipe.lr"
    assert [link.label for link in summary.wandb_links] == ["W&B a2", "W&B"]


def test_a_custom_tracker_relabels_and_relinks_recorded_runs(monkeypatch):
    """Links are derived at read time, so switching backends also repairs the
    runs recorded before the switch."""
    monkeypatch.setenv(tracker_module.LABEL_ENV, "metrics")
    monkeypatch.setenv(
        tracker_module.RUN_URL_TEMPLATE_ENV,
        "https://metrics.example.com/?project={project}&run={run_id}",
    )
    monkeypatch.setenv(
        tracker_module.PROJECT_URL_TEMPLATE_ENV,
        "https://metrics.example.com/?project={project}",
    )
    tracker_module.tracker_config.cache_clear()

    run = _run(framework_status="generate_rollouts")
    run["config"]["wandb"]["run_id"] = "run-1"

    summary = build_run_summary(run, _result())

    assert [link.label for link in summary.wandb_links] == ["metrics a2", "metrics"]
    assert summary.wandb_links[0].url == (
        "https://metrics.example.com/?project=training&run=run-1-a2"
    )
    assert summary.config_summary.wandb_url == (
        "https://metrics.example.com/?project=training&run=run-1"
    )


def test_a_tracker_without_entities_still_links(monkeypatch):
    """An entity is required only by templates that name one. A backend whose
    URLs have no notion of an entity must still link a run that never recorded
    one."""
    monkeypatch.setenv(
        tracker_module.RUN_URL_TEMPLATE_ENV,
        "https://metrics.example.com/?project={project}&run={run_id}",
    )
    tracker_module.tracker_config.cache_clear()

    run = _run()
    run["config"]["wandb"] = {  # no entity
        "project": "training",
        "group": "g",
        "run_id": "recorded-id",
    }
    run["metadata"].pop("wandb_attempts")

    summary = build_run_summary(run, None)

    assert summary.config_summary.wandb_url == (
        "https://metrics.example.com/?project=training&run=recorded-id"
    )


@pytest.mark.parametrize(
    "template",
    [
        "https://metrics.example.com/?project={project}&run={run_id}",
        # A template needing only the run id must not turn the training run id
        # into a link for a run that never logged metrics.
        "https://metrics.example.com/runs/{run_id}",
    ],
)
@pytest.mark.parametrize(
    "wandb_config",
    # One per rejection reason; the full set of shapes `_identifier` has to cope
    # with is enumerated against `_identifier` itself, below.
    [{}, {"project": "   "}, {"project": 0}],
)
def test_a_run_with_no_tracker_metadata_gets_no_link(
    monkeypatch, template, wandb_config
):
    monkeypatch.setenv(tracker_module.RUN_URL_TEMPLATE_ENV, template)
    tracker_module.tracker_config.cache_clear()

    run = _run()
    run["config"]["wandb"] = wandb_config
    run["metadata"].pop("wandb_attempts")

    summary = build_run_summary(run, None)

    assert summary.config_summary.wandb_training_run_id == ""
    assert summary.config_summary.wandb_url is None
    assert summary.wandb_links == []


def test_a_recorded_run_id_links_even_without_a_project(monkeypatch):
    """`WandbConfig.project` defaults to "", but a recorded run id is the run's
    real identity: it must be linked, not discarded along with the project."""
    monkeypatch.setenv(
        tracker_module.RUN_URL_TEMPLATE_ENV, "https://metrics.example.com/runs/{run_id}"
    )
    tracker_module.tracker_config.cache_clear()

    run = _run()
    run["config"]["wandb"] = {"project": "", "run_id": "real-id"}
    run["metadata"].pop("wandb_attempts")

    summary = build_run_summary(run, None)

    assert summary.config_summary.wandb_training_run_id == "real-id"
    assert (
        summary.config_summary.wandb_url == "https://metrics.example.com/runs/real-id"
    )


def test_tracker_identifiers_unwrap_historical_shapes_and_reject_non_strings():
    """These feed URLs, so `_text`'s "stringify anything" is wrong for them: it
    would put a Python repr in an href and let a broken record pass for a run."""
    cyclic: dict[str, object] = {}
    cyclic["value"] = cyclic

    assert run_summary_module._identifier("  spaced  ") == "spaced"
    assert run_summary_module._identifier({"value": "wrapped"}) == "wrapped"
    for rejected in (
        None,
        0,
        1,
        False,
        True,
        [],
        ["id"],
        {},
        {"source": "legacy"},
        {"value": {"value": "nested"}},  # unwrapped once; the rest isn't a string
        cyclic,  # unwrapping until a string would never return
    ):
        assert run_summary_module._identifier(rejected) == ""


def test_build_historical_summary_accepts_aliases_wrappers_and_iso_timestamps():
    historical = {
        "run_id": "legacy-1",
        "framework": {"value": "miles"},
        "status": {"value": "completed"},
        "created_at": "2025-01-02T03:04:05Z",
        "ended_at": "1735787105",
        "config": {
            "model": {"model_name": {"value": "legacy/model"}},
            "dataset": {"prompt_data": "legacy.jsonl"},
            "preset": {"gpu_type": "A100"},
        },
    }

    summary = build_run_summary(historical)

    assert summary.training_run_id == "legacy-1"
    assert summary.framework == "miles"
    assert summary.status == "completed"
    assert summary.model == "legacy/model"
    assert summary.dataset == "legacy.jsonl"
    assert summary.config_summary.gpu_type == "A100"
    assert summary.created_at == 1735787045
    assert summary.completed_at == 1735787105
    assert summary.duration_seconds == 60


def test_missing_result_keeps_run_available():
    summary = build_run_summaries([_run()], train_results=None)[0]

    assert summary.status == "running"
    assert summary.display_status == "pending"
    assert summary.has_train_result is False
    assert summary.train_result is None


def test_display_stage_marks_inactive_gpu_stage_as_queued():
    run = _run(
        framework_status="download_model",
        metadata={
            "framework_progress": {
                "is_active": False,
            }
        },
    )

    summary = build_run_summary(run)

    assert summary.display_stage == "Queuing for GPU — Downloading model"


def test_build_summaries_dedupes_by_id_and_prefers_newest_record():
    older = _run(created_at=90, updated_at=90)
    newer = _run(created_at=200, updated_at=210, framework_status="training")

    summaries = build_run_summaries([older, newer], [_result()])

    assert len(summaries) == 1
    assert summaries[0].created_at == 200
    assert summaries[0].framework_status == "training"


def test_missing_config_and_urls_match_old_frontend_defaults():
    summary = build_run_summary(
        _run(config=None, modal_app_id=""),
        _result(
            wandb_entity=None,
            wandb_project=None,
            wandb_training_run_id=None,
        ),
    )

    assert summary.config_summary == {}
    assert summary.modal_app_url is None
    assert summary.train_result.wandb_url is None


def test_config_defaults_and_fractional_integer_fields_are_rejected():
    run = _run(
        config={},
        metadata={
            "framework_progress": {"current": "1.5", "total": "2.5"},
            "latest_rollout": {"rollout_id": "1.5", "total": "3.5"},
            "attempt_count": "2.5",
            "resume_from_iteration": "1.5",
        },
    )

    summary = build_run_summary(run)

    assert summary.config_summary.lr == 0
    assert summary.config_summary.wandb_url is None
    assert summary.framework_progress.current is None
    assert summary.framework_progress.total is None
    assert summary.latest_rollout.rollout_id == 0
    assert summary.latest_rollout.total == 0
    assert summary.resume_state is None


def test_negative_duration_is_recalculated():
    summary = build_run_summary(
        _run(started_at=110, ended_at=125, duration_seconds=-10)
    )

    assert summary.duration_seconds == 15


def test_wrappers_with_additional_keys_are_unwrapped():
    summary = build_run_summary(
        _run(
            framework={"value": "miles", "source": "legacy"},
            created_at="100.5",
            modal_app_id="https://modal.com/apps/example",
        )
    )

    assert summary.framework == "miles"
    assert summary.created_at == 100
    assert summary.modal_app_url == "https://modal.com/apps/example"


def test_malformed_records_do_not_hide_valid_runs():
    summaries = build_run_summaries(
        [
            _run(training_run_id="valid-run"),
            _run(
                training_run_id="invalid-run",
                step_times={"1": {"phase": {"not": "an integer"}}},
            ),
            None,  # type: ignore[list-item]
        ],
        [None],  # type: ignore[list-item]
    )

    assert [summary.training_run_id for summary in summaries] == ["valid-run"]


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [({"value": "wrapped-id"}, "wrapped-id"), (0, "")],
)
def test_an_attempt_link_carries_a_normalized_run_id(monkeypatch, recorded, expected):
    """The link payload carries run_id for the UI, so a broken attempt record must
    not surface as "0" or "{'value': 'x'}" on an otherwise working project link."""
    monkeypatch.setenv(tracker_module.PROJECT_URL_TEMPLATE_ENV, CUSTOM_PROJECT_URL)
    tracker_module.tracker_config.cache_clear()

    run = _run()
    run["config"]["wandb"] = {}
    run["metadata"]["wandb_attempts"] = [
        {"attempt": 1, "project": "training", "run_id": recorded}
    ]

    links = build_run_summary(run, None).wandb_links

    assert [link.url for link in links] == [
        "https://metrics.example.com/?project=training"
    ]
    assert links[0].run_id == expected


def test_a_record_with_no_recorded_run_id_links_to_the_project_not_a_guess():
    """Every launcher records the id it passed to the tracker, so a config
    without one predates that — and such runs started with no WANDB_RUN_ID, so
    the tracker minted an id this side never saw. Deriving one from the training
    run id would link to a run that does not exist."""
    run = _run(training_run_id="electric-batter-6362579afd91")
    run["config"]["wandb"] = {"entity": "e", "project": "p"}
    run["metadata"].pop("wandb_attempts")

    summary = build_run_summary(run, None)

    assert summary.config_summary.wandb_training_run_id == ""
    assert summary.config_summary.wandb_url == "https://wandb.ai/e/p"
