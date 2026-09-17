"""Client-side helpers for a Modal-hosted Miles Tinker gateway.

The gateway speaks the Tinker protocol, so the official ``tinker`` SDK (pinned
to :data:`TINKER_SDK_VERSION`, the version Miles is tested against) and the
`tinker-cookbook <https://github.com/thinking-machines-lab/tinker-cookbook>`_
recipes work against it unchanged. What this module adds is the Modal-specific
glue the SDK cannot know about:

- **Two layers of auth.** The gateway is deployed behind Modal proxy auth by
  default, so every request carries ``Modal-Key`` / ``Modal-Secret`` headers
  (:func:`gateway_headers`). *Inside* that, the gateway identifies tenants by
  the Tinker API key — each distinct key owns its own adapters and sampler
  snapshots (:func:`tenant_api_key`).
- **A fixed adapter layout.** Which module groups carry LoRA weights (attention,
  MLP, unembedding) and the maximum rank are set when the server starts, from
  the recipe's ``tinker_train_*`` / ``lora_rank`` fields. The gateway rejects
  ``create_lora_training_client`` calls that disagree; :func:`lora_config_kwargs`
  derives matching arguments from the recipe so clients fail locally instead.
- **Supervised datums.** :func:`cross_entropy_datums` tokenizes plain text into
  the next-token-prediction ``Datum`` shape the ``cross_entropy`` loss expects.

The ``tinker`` SDK is an optional dependency: install it with
``pip install tinker==0.26.2`` (it is not pulled in by ``modal-training-gym``).
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from modal_training_gym.common.config import modal_proxy_auth_headers
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.frameworks.miles.tinker_gateway import (
    TINKER_SDK_VERSION,
    TinkerGateway,
)
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    import tinker
    from tinker import types
    from transformers import PreTrainedTokenizerBase

#: Prefix the SDK requires on API keys. The gateway treats the whole key as
#: the tenant identity, so anything after the prefix is free-form.
TENANT_KEY_PREFIX = "tml-"

#: The gateway's default ``max_datums_per_request``; larger batches are
#: rejected up front rather than partially processed.
MAX_DATUMS_PER_REQUEST = 1024


def installed_sdk_version() -> str | None:
    """Version of the ``tinker`` SDK importable here, or ``None``."""
    try:
        return version("tinker")
    except PackageNotFoundError:
        return None


def require_sdk() -> None:
    """Fail with an install hint if the pinned ``tinker`` SDK is missing.

    A different installed version is tolerated with a warning: the gateway is
    tested against exactly :data:`TINKER_SDK_VERSION` and the wire protocol
    is young, so mismatches are the first thing to suspect on odd errors.
    """
    installed = installed_sdk_version()
    if installed is None:
        raise TrainingGymConfigError(
            "the tinker SDK is not installed; run "
            f"`pip install tinker=={TINKER_SDK_VERSION}` (the version the Miles "
            "gateway is tested against)."
        )
    if installed != TINKER_SDK_VERSION:
        print(
            f"[tinker] warning: tinker=={installed} is installed but the gateway "
            f"is tested against tinker=={TINKER_SDK_VERSION}"
        )


def tenant_api_key(tenant: str) -> str:
    """Tinker API key that the gateway maps to ``tenant``.

    The SDK insists on the ``tml-`` prefix; the gateway hashes the full key
    into the tenant id that owns adapters, futures and ``tinker://`` paths.
    Keys already carrying the prefix are returned unchanged.
    """
    tenant = tenant.strip()
    if not tenant:
        raise TrainingGymConfigError("tenant must be a non-empty string")
    if tenant.startswith(TENANT_KEY_PREFIX):
        return tenant
    return f"{TENANT_KEY_PREFIX}{tenant}"


def gateway_headers(gateway: TinkerGateway) -> dict[str, str]:
    """Modal proxy-auth headers a client must send to reach ``gateway``.

    Empty for an ``unauthenticated=True`` gateway. Otherwise the
    ``MODAL_KEY`` / ``MODAL_SECRET`` pair (or the pair saved by
    ``training-gym setup``) is required, and its absence is reported here
    rather than as an HTTP 401 on the first SDK call.
    """
    if gateway.unauthenticated:
        return {}
    headers = modal_proxy_auth_headers()
    if not headers:
        raise TrainingGymConfigError(
            f"{gateway.app_name!r} requires Modal proxy auth but MODAL_KEY / "
            "MODAL_SECRET are not set; export the wk-/ws- pair (or run "
            "`training-gym setup`), or launch with unauthenticated=True."
        )
    return headers


def lora_config_kwargs(
    recipe: MilesRecipe, *, rank: int | None = None
) -> dict[str, int | bool]:
    """``create_lora_training_client`` arguments matching the server layout.

    Returns ``rank`` plus ``train_attn`` / ``train_mlp`` / ``train_unembed``
    copied from the recipe's ``tinker_train_*`` fields. ``rank`` defaults to
    the recipe's ``lora_rank`` (the slot capacity) and may be lower, never
    higher — the gateway would reject the request.
    """
    if not recipe.is_tinker_gateway:
        raise TrainingGymConfigError(
            f"{type(recipe).__name__} does not serve a Tinker gateway "
            "(multi_lora_n_adapters is unset)"
        )
    max_rank = recipe.lora_rank
    if max_rank is None:
        raise TrainingGymConfigError(
            f"{type(recipe).__name__}.lora_rank is unset; the gateway needs it"
        )
    if rank is None:
        rank = max_rank
    if not 1 <= rank <= max_rank:
        raise TrainingGymConfigError(
            f"rank={rank} is outside [1, {max_rank}]; the gateway's slots were "
            f"allocated for lora_rank={max_rank}"
        )
    return {
        "rank": rank,
        "train_attn": recipe.tinker_train_attn,
        "train_mlp": recipe.tinker_train_mlp,
        "train_unembed": recipe.tinker_train_unembed,
    }


def service_client(
    gateway: TinkerGateway,
    *,
    tenant: str,
    default_headers: dict[str, str] | None = None,
    **kwargs: object,
) -> tinker.ServiceClient:
    """A ``tinker.ServiceClient`` for ``tenant`` pointed at ``gateway``.

    Sets ``base_url`` to the gateway, ``api_key`` to
    :func:`tenant_api_key(tenant) <tenant_api_key>` and folds the Modal
    proxy-auth headers into ``default_headers``. Extra keyword arguments go
    to ``tinker.ServiceClient`` unchanged.
    """
    require_sdk()
    import tinker

    headers = gateway_headers(gateway) | (default_headers or {})
    return tinker.ServiceClient(
        base_url=gateway.url,
        api_key=tenant_api_key(tenant),
        default_headers=headers,
        **kwargs,
    )


def cross_entropy_datums(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str]
) -> list[types.Datum]:
    """Next-token-prediction datums for ``forward_backward(loss_fn="cross_entropy")``.

    Each text becomes one ``Datum`` whose ``model_input`` is the token
    sequence minus its last token and whose ``target_tokens`` are the same
    sequence shifted by one, with unit ``weights`` (every position counts
    equally). Texts that tokenize to fewer than two tokens are rejected.
    """
    from tinker import types

    if len(texts) > MAX_DATUMS_PER_REQUEST:
        raise TrainingGymConfigError(
            f"{len(texts)} texts exceeds the gateway's {MAX_DATUMS_PER_REQUEST} "
            "datums per forward_backward request; split the batch."
        )
    datums = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            raise TrainingGymConfigError(
                f"{text!r} tokenizes to {len(tokens)} token(s); need at least two"
            )
        datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "weights": [1.0] * (len(tokens) - 1),
                },
            )
        )
    return datums


def target_token_count(datums: Sequence[types.Datum]) -> int:
    """Number of supervised positions across ``datums``.

    ``forward_backward`` reports ``metrics["loss:sum"]``; dividing by this
    gives the mean per-token loss the cookbook logs.
    """
    total = 0
    for datum in datums:
        shape = datum.loss_fn_inputs["target_tokens"].shape
        if not shape:
            raise TrainingGymConfigError("target_tokens must be a 1-D tensor")
        total += shape[0]
    return total


__all__ = [
    "MAX_DATUMS_PER_REQUEST",
    "TENANT_KEY_PREFIX",
    "TINKER_SDK_VERSION",
    "TinkerGateway",
    "cross_entropy_datums",
    "gateway_headers",
    "installed_sdk_version",
    "lora_config_kwargs",
    "require_sdk",
    "service_client",
    "target_token_count",
    "tenant_api_key",
]
