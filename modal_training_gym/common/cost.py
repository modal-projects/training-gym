"""Cost estimation helpers for Modal GPU training runs.

Provides per-GPU hourly rates (Modal on-demand pricing) and a simple
``estimate_cost`` function that tutorials and configs can use to give
users a cost expectation before they launch a run.

Prices are point-in-time snapshots — see https://modal.com/pricing for
the latest rates.
"""

from __future__ import annotations

# Modal on-demand GPU prices in USD per hour (as of 2025-06).
# Source: https://modal.com/pricing
GPU_HOURLY_PRICES: dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "L40S": 1.95,
    "A100": 2.50,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
    "B300": 6.25,  # same as B200 at launch; update when pricing diverges
}


def estimate_cost(
    gpu_type: str,
    n_gpus: int,
    hours: float,
) -> float:
    """Estimate Modal compute cost for a training run.

    Parameters
    ----------
    gpu_type:
        GPU model string (must be a key in ``GPU_HOURLY_PRICES``).
    n_gpus:
        Total number of GPUs across all nodes.
    hours:
        Estimated wall-clock runtime in hours.

    Returns
    -------
    float
        Estimated cost in USD (GPU cost only — excludes CPU/memory).

    Raises
    ------
    KeyError
        If *gpu_type* is not in the price table.
    """
    price = GPU_HOURLY_PRICES[gpu_type]
    return price * n_gpus * hours


def format_cost_range(
    gpu_type: str,
    n_gpus: int,
    smoke_minutes: float,
    full_hours: float = 1.0,
) -> str:
    """Return a human-readable cost string for tutorial docs.

    Example output::

        ~$0.33 (smoke, ~5 min) — ~$3.95/hr (full run) on 1×A10
    """
    smoke_cost = estimate_cost(gpu_type, n_gpus, smoke_minutes / 60)
    full_cost = estimate_cost(gpu_type, n_gpus, full_hours)
    gpu_label = f"{n_gpus}×{gpu_type}" if n_gpus > 1 else gpu_type
    return (
        f"~${smoke_cost:.2f} (smoke, ~{smoke_minutes:.0f} min) — "
        f"~${full_cost:.2f}/hr (full run) on {gpu_label}"
    )
