"""List-field metadata and filtering for training-run summaries."""

from __future__ import annotations

from collections.abc import Iterable

from modal_training_gym.common.run_summary import RunSummary


def run_list_field_metadata() -> dict[str, dict[str, object]]:
    """Return list metadata from selected ``RunSummary`` fields."""
    metadata: dict[str, dict[str, object]] = {}
    for field_name, field in RunSummary.model_fields.items():
        extra = (
            field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        )
        if not extra.get("list"):
            continue
        metadata[field_name] = {
            "label": field.title or field_name.replace("_", " ").title(),
            "filterable": bool(extra.get("filterable")),
            "timestamp": bool(extra.get("timestamp")),
        }
    return metadata


def filter_run_summaries(
    summaries: Iterable[RunSummary],
    *,
    filters: dict[str, str] | None = None,
    since: int | None = None,
    limit: int | None = None,
) -> list[RunSummary]:
    """Filter summaries by their configured list fields and update recency."""
    metadata = run_list_field_metadata()
    active_filters = {
        name: value.strip().casefold()
        for name, value in (filters or {}).items()
        if metadata.get(name, {}).get("filterable") and value.strip()
    }

    selected: list[RunSummary] = []
    for summary in summaries:
        if since is not None and max(summary.created_at, summary.updated_at) < since:
            continue
        if any(
            str(getattr(summary, name)).casefold() != expected
            for name, expected in active_filters.items()
        ):
            continue
        selected.append(summary)

    selected.sort(
        key=lambda summary: (summary.updated_at, summary.run_id),
        reverse=True,
    )
    if limit is not None:
        selected = selected[:limit]
    return selected
