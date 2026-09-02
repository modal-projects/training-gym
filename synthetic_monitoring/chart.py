import io
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from scripts.validate_model_configs import ValidationResult

_STEP_SUFFIX = re.compile(r" \(step \d+\)$")
_LANE_ORDER = {None: 0, "actor": 1, "rollout": 2}

HISTORY_DICT_NAME = "gym-synmon-timing-baselines"

_GRAY_00 = "#181818"
_GRAY_08 = "#222222"
_GRAY_15 = "#272727"
_GRAY_20 = "#2f2f2f"
_GRAY_30 = "#464646"
_GRAY_40 = "#747474"
_GRAY_50 = "#a3a3a3"
_GRAY_70 = "#d1d1d1"
_GREEN_50 = "#6ac345"
_GREEN_70 = "#63cd93"
_GREEN_80 = "#7fee64"
_RED_75 = "#f87171"
_YELLOW_60 = "#d1c05f"
_BLUE_60 = "#79a4c4"
_PHASE_COLORS = (
    _BLUE_60,
    _GREEN_50,
    _YELLOW_60,
    _RED_75,
    _GREEN_70,
    _GRAY_50,
    _GRAY_40,
    "#c4a27a",
    _GRAY_70,
)


@dataclass(frozen=True)
class RunPoint:
    ts: float
    timings: dict[str, float]
    training_run_id: str
    total_duration_s: float

    @classmethod
    def from_validation_result(
        cls, result: "ValidationResult", *, ts: float | None = None
    ) -> "RunPoint":
        timings: dict[str, float] = {}
        step_times = result.step_times or {}
        substep_times = result.substep_times or {}
        for step_key in sorted(step_times, key=int):
            ordered_substeps = sorted(
                substep_times[step_key].items(), key=lambda item: item[1]["start"]
            )
            for name, entry in ordered_substeps:
                # duration_s sums busy time across overlapping invocations, so
                # a concurrent phase can exceed its wall time; say so in the key.
                key = name.replace(")", ", busy)") if entry["concurrent"] else name
                timings[f"{key} (step {step_key})"] = entry["duration_s"]
            timings[f"Step {step_key}"] = step_times[step_key]["duration_s"]
        return cls(
            ts=time.time() if ts is None else ts,
            timings=timings,
            training_run_id=result.training_run_id,
            total_duration_s=result.total_duration_s,
        )


def append_history(
    model_name: str, point: RunPoint, *, environment_name: str
) -> list[RunPoint]:
    history = modal.Dict.from_name(
        HISTORY_DICT_NAME, create_if_missing=True, environment_name=environment_name
    )
    points = [RunPoint(**item) for item in history.get(model_name, [])]
    if point.training_run_id and any(
        p.training_run_id == point.training_run_id for p in points
    ):
        return points
    points.append(point)
    points.sort(key=lambda p: (p.ts, p.training_run_id))
    history[model_name] = [asdict(p) for p in points]
    return points


def _panel_title(label: str) -> str:
    name = _STEP_SUFFIX.sub("", label).replace("_", " ")
    return name[:1].upper() + name[1:]


def _lane(title: str) -> int:
    match = re.search(r"\((actor|rollout)\b", title)
    return _LANE_ORDER[match.group(1) if match else None]


def _substep_series(history: list[RunPoint]) -> dict[str, list[float | None]]:
    names: list[str] = []
    seen: set[str] = set()
    for point in history:
        for label in point.timings:
            if label.startswith("Step "):
                continue
            pretty = _panel_title(label)
            if pretty not in seen:
                seen.add(pretty)
                names.append(pretty)
    names.sort(key=_lane)
    series: dict[str, list[float | None]] = {name: [] for name in names}
    for point in history:
        for name in names:
            total = 0.0
            found = False
            for label, value in point.timings.items():
                if label.startswith("Step "):
                    continue
                if _panel_title(label) == name:
                    found = True
                    total += float(value)
            series[name].append(total if found else None)
    return series


def _style_history_axes(ax) -> None:
    ax.set_facecolor(_GRAY_08)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRAY_20, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.tick_params(colors=_GRAY_50, length=3, width=0.8)
    ax.xaxis.label.set_color(_GRAY_50)
    ax.yaxis.label.set_color(_GRAY_50)
    ax.title.set_color(_GRAY_70)
    for spine in ax.spines.values():
        spine.set_color(_GRAY_30)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("time (UTC)")


def render_timing_history_chart(
    history: list[RunPoint],
    *,
    model_name: str,
) -> bytes:
    from matplotlib import pyplot
    from matplotlib.gridspec import GridSpec

    series = _substep_series(history)
    tick_labels = [
        datetime.fromtimestamp(p.ts, tz=timezone.utc).strftime("%m-%d %H:%M")
        for p in history
    ]
    xs = list(range(len(history)))

    names = list(series)
    ncols = 3
    nrows_sub = (len(names) + ncols - 1) // ncols
    nrows = nrows_sub + 1

    fig_height_in = 2.6 * nrows + 1.1
    fig = pyplot.figure(figsize=(13.5, fig_height_in))
    fig.patch.set_facecolor(_GRAY_00)
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        height_ratios=[1.0] * nrows_sub + [1.2],
        hspace=0.62,
        wspace=0.32,
        # Fixed headroom in inches so the suptitle clears the first axes title
        # whether there are zero substep rows or several.
        top=1 - 0.9 / fig_height_in,
        bottom=0.10,
        left=0.06,
        right=0.98,
    )
    fig.suptitle(
        f"{model_name} (n={len(history)} runs)",
        color=_GRAY_70,
        fontsize=13,
        fontweight="medium",
    )

    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
        for x, y in zip(xs, series[name], strict=True):
            if y is None:
                continue
            ax.bar(x, y, color=color, width=0.72, zorder=2)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("seconds")
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=7)
        _style_history_axes(ax)

    ax = fig.add_subplot(gs[nrows_sub, :])
    for x, point in zip(xs, history, strict=True):
        ax.bar(x, point.total_duration_s, color=_GREEN_80, width=0.55, zorder=2)
    ax.set_title("Total")
    ax.set_ylabel("seconds")
    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=8)
    _style_history_axes(ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=_GRAY_00)
    pyplot.close(fig)
    return buf.getvalue()
