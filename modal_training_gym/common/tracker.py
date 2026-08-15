"""Where the dashboard's per-run metric links point, and what they're called.

Training frameworks reach Weights & Biases through the ``wandb`` package, so a
deployment can put a wandb-compatible backend behind that import — a
self-hosted W&B server, or a shim that forwards to another tracker. The runs
are then logged somewhere other than wandb.ai, and the dashboard's links are
dead ends.

The dashboard never stores those URLs: it derives them at read time from the
``(entity, project, group, run_id)`` tuple the launcher already records per
attempt. That makes the link target a rendering choice, set by env vars on the
dashboard app:

    TRAINING_GYM_TRACKER_LABEL                 # "W&B"
    TRAINING_GYM_TRACKER_RUN_URL_TEMPLATE      # a single run
    TRAINING_GYM_TRACKER_PROJECT_URL_TEMPLATE  # fallback when the run is unknown

Unset, the defaults reproduce wandb.ai links exactly. Because nothing is baked
into the stored records, pointing an existing dashboard at another backend also
relabels and relinks the runs recorded before the switch.

Setting *either* URL template switches off both W&B defaults: whichever one you
don't set produces no link rather than silently falling back to wandb.ai, so a
half-finished or mistyped set of templates can't mix the two backends.

``TRACKER_LABEL`` is independent of that, and on its own only renames the
links — it does not move them. Relabelling wandb.ai (an internal name for it,
say) is the point; if you meant to move the links too, set a template.

Templates are ``str.format`` strings over :data:`TEMPLATE_FIELDS`, and a
template renders only when every field it references is non-empty — which is
what makes ``PROJECT_URL_TEMPLATE`` a fallback for a run whose id isn't known
yet, and what lets a backend that has no notion of an entity leave ``{entity}``
out. The fields a template names are therefore also the evidence it needs: a
``{group}``-only template links any run with a recorded group, which is the
point of writing one. Values are percent-encoded, so a backend that keys runs off a query
parameter works as well as one that uses path segments::

    TRAINING_GYM_TRACKER_LABEL=metrics
    TRAINING_GYM_TRACKER_RUN_URL_TEMPLATE=https://metrics.example.com/?project={project}&run={run_id}
    TRAINING_GYM_TRACKER_PROJECT_URL_TEMPLATE=https://metrics.example.com/?project={project}

This is one tracker per dashboard, not per run: it describes where *this*
deployment's training containers log.
"""

from __future__ import annotations

import os
import string
import sys
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import quote, urlparse

LABEL_ENV = "TRAINING_GYM_TRACKER_LABEL"
RUN_URL_TEMPLATE_ENV = "TRAINING_GYM_TRACKER_RUN_URL_TEMPLATE"
PROJECT_URL_TEMPLATE_ENV = "TRAINING_GYM_TRACKER_PROJECT_URL_TEMPLATE"

DEFAULT_LABEL = "W&B"
DEFAULT_RUN_URL_TEMPLATE = "https://wandb.ai/{entity}/{project}/runs/{run_id}"
DEFAULT_PROJECT_URL_TEMPLATE = "https://wandb.ai/{entity}/{project}"

# Substitutable fields — what the launcher records per attempt. Keep in sync
# with the keyword arguments of TrackerConfig.url().
TEMPLATE_FIELDS = frozenset({"entity", "project", "group", "run_id"})

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_FORMATTER = string.Formatter()


def _warn(message: str) -> None:
    print(f"[tracker] {message}", file=sys.stderr, flush=True)


def _fields(template: str) -> set[str]:
    """Fields ``template`` references, or ``ValueError`` if it isn't renderable.

    Deliberately narrow: only bare ``{field}`` references to known names. A
    conversion (``{project!r}``), a format spec (``{project:>8}``), a nested
    spec (``{project:{group}}``) or a positional field all parse fine here but
    blow up — or resolve a name we never supply — at format time, so they are
    rejected up front rather than at render time.
    """
    found: set[str] = set()
    for _, name, spec, conversion in _FORMATTER.parse(template):
        if name is None:
            continue
        if conversion is not None:
            raise ValueError(f"conversion {'!' + conversion!r} is not supported")
        if spec:
            raise ValueError(f"format spec {spec!r} is not supported")
        if name not in TEMPLATE_FIELDS:
            raise ValueError(
                f"unknown field {name!r}; supported: {sorted(TEMPLATE_FIELDS)}"
            )
        found.add(name)
    if not found:
        raise ValueError("template references no fields, so it can't identify a run")
    return found


def _accept(template: str, env_name: str) -> str | None:
    """Return ``template`` if it is safe to render, else warn and return None.

    A bad template is a config mistake, not a reason to take the dashboard
    down, so the affected link is dropped and everything else keeps working.
    The scheme and host checks matter because the result goes straight into an
    ``href``: an operator-supplied ``javascript:`` template would otherwise be
    a self-XSS foothold on a dashboard that holds workspace credentials, and
    ``https://`` alone renders a link that goes nowhere.
    """
    try:
        fields = _fields(template)
        parsed = urlparse(template)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise ValueError(f"scheme must be http or https, got {parsed.scheme!r}")
        # hostname, not netloc: "https://:/x" and "https://user@/x" have a
        # netloc but no actual host, and render links that go nowhere.
        if not parsed.hostname:
            raise ValueError("no host")
        if "{" in parsed.netloc:
            # The operator fixes the host; run metadata only fills the path and
            # query. Otherwise a run's project name chooses where the link goes.
            raise ValueError("host must be a literal, not a template field")
        # Prove it formats before any run summary depends on it.
        template.format(**dict.fromkeys(fields, "x"))
    except (ValueError, KeyError, IndexError) as exc:
        _warn(f"ignoring {env_name}: {exc}")
        return None
    return template


def _render(template: str | None, values: dict[str, str]) -> str | None:
    """Percent-encode ``values`` into ``template``; None if any is empty."""
    if not template:
        return None
    try:
        fields = _fields(template)
        if any(not values.get(field, "") for field in fields):
            return None
        return template.format(
            **{field: quote(values[field], safe="") for field in fields}
        )
    except Exception:  # never let a link break run-summary construction
        return None


@dataclass(frozen=True)
class TrackerConfig:
    # Both templates are required, with no default: defaulting one to wandb.ai
    # would let a caller configure a custom run URL and silently keep W&B's
    # project URL, which is the backend mixing tracker_config() rules out.
    run_url_template: str | None
    project_url_template: str | None
    label: str = DEFAULT_LABEL

    def url(
        self,
        *,
        entity: str = "",
        project: str = "",
        group: str = "",
        run_id: str = "",
    ) -> str | None:
        """Link to this run, falling back to its project, else None."""
        values = {
            "entity": entity.strip(),
            "project": project.strip(),
            "group": group.strip(),
            "run_id": run_id.strip(),
        }
        return _render(self.run_url_template, values) or _render(
            self.project_url_template, values
        )


@lru_cache(maxsize=1)
def tracker_config() -> TrackerConfig:
    """Tracker config for this process, read from the environment once.

    Tests that patch the environment should call ``tracker_config.cache_clear()``.
    """
    run_template = os.environ.get(RUN_URL_TEMPLATE_ENV, "").strip()
    project_template = os.environ.get(PROJECT_URL_TEMPLATE_ENV, "").strip()
    if run_template or project_template:
        # Custom backend: never mix in a wandb.ai default for the other one.
        run_url = _accept(run_template, RUN_URL_TEMPLATE_ENV) if run_template else None
        project_url = (
            _accept(project_template, PROJECT_URL_TEMPLATE_ENV)
            if project_template
            else None
        )
    else:
        run_url, project_url = DEFAULT_RUN_URL_TEMPLATE, DEFAULT_PROJECT_URL_TEMPLATE
    return TrackerConfig(
        label=os.environ.get(LABEL_ENV, "").strip() or DEFAULT_LABEL,
        run_url_template=run_url,
        project_url_template=project_url,
    )
