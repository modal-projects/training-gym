from __future__ import annotations

import pytest

from modal_training_gym.common import tracker as tracker_module
from modal_training_gym.common.tracker import TrackerConfig

CUSTOM_RUN = "https://metrics.example.com/?project={project}&run={run_id}"
CUSTOM_PROJECT = "https://metrics.example.com/?project={project}"


@pytest.fixture(autouse=True)
def clean_tracker_env(monkeypatch):
    """Defaults must be tested against an empty environment, not the developer's."""
    for name in (
        tracker_module.LABEL_ENV,
        tracker_module.RUN_URL_TEMPLATE_ENV,
        tracker_module.PROJECT_URL_TEMPLATE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    tracker_module.tracker_config.cache_clear()
    yield
    tracker_module.tracker_config.cache_clear()


# ── Defaults reproduce the wandb.ai links ───────────────────────────────────


def test_defaults_match_the_wandb_urls_they_replaced():
    config = tracker_module.tracker_config()

    assert config.label == "W&B"
    assert (
        config.url(entity="entity", project="project", run_id="run")
        == "https://wandb.ai/entity/project/runs/run"
    )
    # No run id yet -> project-level fallback.
    assert (
        config.url(entity="entity", project="project")
        == "https://wandb.ai/entity/project"
    )
    # Neither default template can render without an entity.
    assert config.url(project="project", run_id="run") is None
    assert config.url() is None


@pytest.mark.parametrize(
    "value",
    # One per character class that percent-encoding has to survive: a path
    # separator, query/fragment delimiters, a literal percent, non-ASCII, and a
    # value that is empty once stripped.
    ["project/name", "a?b#c", "100%", "ünïcode", "  "],
)
def test_default_urls_encode_identifiers_wherever_they_appear(value):
    """The wandb.ai URLs are spelled out here rather than derived from the
    templates, so a change to the substitution machinery cannot quietly move
    them. Every identifier is fully encoded: no value can add a path segment or
    a query of its own."""
    from urllib.parse import quote

    def expected(entity: str, project: str, run_id: str) -> str | None:
        entity, project, run_id = entity.strip(), project.strip(), run_id.strip()
        if not entity or not project:
            return None
        base = f"https://wandb.ai/{quote(entity, safe='')}/{quote(project, safe='')}"
        return f"{base}/runs/{quote(run_id, safe='')}" if run_id else base

    config = tracker_module.tracker_config()
    for entity, project, run_id in (
        (value, "project", "run"),
        ("entity", value, "run"),
        ("entity", "project", value),
    ):
        assert config.url(entity=entity, project=project, run_id=run_id) == expected(
            entity, project, run_id
        )


# ── Custom backends ─────────────────────────────────────────────────────────


def test_custom_templates_need_only_the_fields_they_reference(monkeypatch):
    monkeypatch.setenv(tracker_module.LABEL_ENV, "metrics")
    monkeypatch.setenv(tracker_module.RUN_URL_TEMPLATE_ENV, CUSTOM_RUN)
    monkeypatch.setenv(tracker_module.PROJECT_URL_TEMPLATE_ENV, CUSTOM_PROJECT)
    tracker_module.tracker_config.cache_clear()

    config = tracker_module.tracker_config()

    assert config.label == "metrics"
    # No entity anywhere, which the wandb.ai templates would have required.
    assert (
        config.url(project="proj", run_id="abcdefgh")
        == "https://metrics.example.com/?project=proj&run=abcdefgh"
    )
    assert config.url(project="proj") == "https://metrics.example.com/?project=proj"
    assert config.url() is None


def test_values_cannot_inject_extra_query_parameters():
    config = TrackerConfig(run_url_template=CUSTOM_RUN, project_url_template=None)

    assert (
        config.url(project="a&b=c", run_id="d e")
        == "https://metrics.example.com/?project=a%26b%3Dc&run=d%20e"
    )


def test_group_is_substitutable():
    config = TrackerConfig(
        run_url_template="https://metrics.example.com/{group}/{run_id}",
        project_url_template=None,
    )

    assert (
        config.url(group="sweep", run_id="r") == "https://metrics.example.com/sweep/r"
    )
    assert config.url(run_id="r") is None  # missing group, no fallback configured


# ── Partial configuration must not mix backends ─────────────────────────────


def test_setting_one_template_disables_the_other_default(monkeypatch):
    """Configuring one URL template must not leave the other pointing at wandb.ai:
    a half-finished set of templates produces no link, never a mixed backend."""
    monkeypatch.setenv(tracker_module.LABEL_ENV, "metrics")
    monkeypatch.setenv(tracker_module.RUN_URL_TEMPLATE_ENV, CUSTOM_RUN)
    tracker_module.tracker_config.cache_clear()

    config = tracker_module.tracker_config()

    assert config.project_url_template is None
    assert config.url(entity="e", project="p", run_id="r").startswith(
        "https://metrics.example.com/"
    )
    # Run id unknown: no project fallback rather than a wandb.ai link.
    assert config.url(entity="e", project="p") is None


def test_an_invalid_custom_template_disables_that_link_not_falls_back(monkeypatch):
    monkeypatch.setenv(tracker_module.RUN_URL_TEMPLATE_ENV, "https://x.example/{oops}")
    monkeypatch.setenv(tracker_module.PROJECT_URL_TEMPLATE_ENV, CUSTOM_PROJECT)
    tracker_module.tracker_config.cache_clear()

    config = tracker_module.tracker_config()

    assert config.run_url_template is None
    assert (
        config.url(project="p", run_id="r") == "https://metrics.example.com/?project=p"
    )


def test_label_alone_renames_without_moving_the_links(monkeypatch):
    monkeypatch.setenv(tracker_module.LABEL_ENV, "Internal W&B")
    tracker_module.tracker_config.cache_clear()

    config = tracker_module.tracker_config()

    assert config.label == "Internal W&B"
    assert config.run_url_template == tracker_module.DEFAULT_RUN_URL_TEMPLATE


# ── Rejected templates ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("template", "why"),
    [
        ("https://x.example/{nope}", "unknown field"),
        ("https://x.example/{project!r}", "conversion"),
        ("https://x.example/{project:>8}", "format spec"),
        ("https://x.example/{project:{group}}", "nested format spec"),
        ("https://x.example/{}", "positional field"),
        ("https://x.example/{unbalanced", "malformed"),
        ("https://x.example/fixed", "no fields, so it can't identify a run"),
        ("javascript:alert({project})", "non-http scheme"),
        ("/relative/{project}", "no scheme"),
        ("https://user@/{project}", "netloc but no host"),
        # Run metadata must not be able to choose the link's host.
        ("https://{project}/x", "templated host"),
    ],
)
def test_unrenderable_templates_are_rejected_with_a_warning(
    monkeypatch, capsys, template, why
):
    monkeypatch.setenv(tracker_module.RUN_URL_TEMPLATE_ENV, template)
    tracker_module.tracker_config.cache_clear()

    config = tracker_module.tracker_config()

    assert config.run_url_template is None, why
    assert "[tracker]" in capsys.readouterr().err


def test_render_never_raises_even_if_a_template_slips_through():
    """`url()` is called while building every run summary; the /api/runs route
    turns any exception there into an empty run list."""
    config = TrackerConfig(
        run_url_template="https://x.example/{project!r}",
        project_url_template="https://x.example/{project}",
    )

    assert config.url(entity="e", project="p", run_id="r") == "https://x.example/p"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_url_template": "https://metrics.example.com/runs/{run_id}"},
        {"project_url_template": "https://metrics.example.com/{project}"},
    ],
)
def test_both_templates_must_be_given_explicitly(kwargs):
    """Neither template may default to wandb.ai: that would let a caller set one
    and silently keep W&B's other half, which is the mixing tracker_config()
    exists to prevent."""
    with pytest.raises(TypeError):
        TrackerConfig(**kwargs)
