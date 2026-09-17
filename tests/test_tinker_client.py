"""Client-side glue for a Modal-hosted Tinker gateway (``modal_training_gym.tinker``).

No GPU and no gateway: these cover the pure helpers (tenant keys, proxy-auth
headers, LoRA layout derivation) and, when the optional ``tinker`` SDK is
installed, the datum construction and ``ServiceClient`` wiring.
"""

from unittest.mock import patch

import pytest

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import Qwen3_30B
from modal_training_gym.frameworks.miles.tinker_gateway import (
    TINKER_SDK_VERSION,
    TinkerGateway,
)
from modal_training_gym.tinker import (
    MAX_DATUMS_PER_REQUEST,
    gateway_headers,
    installed_sdk_version,
    lora_config_kwargs,
    require_sdk,
    tenant_api_key,
)
from modal_training_gym.train_recipes.miles_recipe import (
    MilesRecipe,
    Qwen3_30B_A3B_Tinker_Recipe,
)

tinker = pytest.importorskip("tinker") if installed_sdk_version() else None
needs_sdk = pytest.mark.skipif(tinker is None, reason="tinker SDK not installed")


def _gateway(unauthenticated: bool = False, **recipe_overrides) -> TinkerGateway:
    recipe = Qwen3_30B_A3B_Tinker_Recipe(**recipe_overrides)
    return TinkerGateway(
        app_name="demo-tinker",
        model=Qwen3_30B(),
        recipe=recipe,
        base_model="Qwen/Qwen3-30B-A3B",
        n_slots=recipe.multi_lora_n_adapters or 0,
        checkpoint_root="/checkpoints/demo-tinker/tinker",
        checkpoints_volume_name="training-gym-checkpoints",
        unauthenticated=unauthenticated,
        url="https://demo-tinker.modal.run",
    )


class _WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return [hash(word) % 1000 for word in text.split()]


# --- tenant keys -------------------------------------------------------------


@pytest.mark.parametrize(
    ("tenant", "expected"),
    [
        ("alice", "tml-alice"),
        ("  alice ", "tml-alice"),
        ("tml-already", "tml-already"),
    ],
)
def test_tenant_api_key_adds_sdk_prefix(tenant, expected):
    assert tenant_api_key(tenant) == expected


@pytest.mark.parametrize("tenant", ["", "   "])
def test_tenant_api_key_rejects_empty(tenant):
    with pytest.raises(TrainingGymConfigError, match="non-empty"):
        tenant_api_key(tenant)


# --- proxy auth --------------------------------------------------------------


def test_gateway_headers_forwards_modal_proxy_pair():
    env = {"MODAL_KEY": "wk-test", "MODAL_SECRET": "ws-test"}
    with (
        patch.dict("os.environ", env),
        patch("modal_training_gym.common.config.load_proxy_auth"),
    ):
        assert gateway_headers(_gateway()) == {
            "Modal-Key": "wk-test",
            "Modal-Secret": "ws-test",
        }


def test_gateway_headers_requires_pair_for_authenticated_gateway():
    with (
        patch.dict("os.environ", {"MODAL_KEY": "", "MODAL_SECRET": ""}),
        patch("modal_training_gym.common.config.load_proxy_auth"),
    ):
        with pytest.raises(TrainingGymConfigError, match="MODAL_KEY / MODAL_SECRET"):
            gateway_headers(_gateway())


def test_gateway_headers_empty_for_unauthenticated_gateway():
    with patch.dict("os.environ", {"MODAL_KEY": "", "MODAL_SECRET": ""}):
        assert gateway_headers(_gateway(unauthenticated=True)) == {}


# --- LoRA layout -------------------------------------------------------------


def test_lora_config_kwargs_defaults_to_full_rank_and_recipe_groups():
    recipe = Qwen3_30B_A3B_Tinker_Recipe(tinker_train_unembed=False)
    assert lora_config_kwargs(recipe) == {
        "rank": recipe.lora_rank,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }


def test_lora_config_kwargs_allows_lower_rank():
    assert lora_config_kwargs(Qwen3_30B_A3B_Tinker_Recipe(), rank=4)["rank"] == 4


@pytest.mark.parametrize("rank", [0, 33])
def test_lora_config_kwargs_rejects_rank_outside_slot_capacity(rank):
    with pytest.raises(TrainingGymConfigError, match=r"outside \[1, 32\]"):
        lora_config_kwargs(Qwen3_30B_A3B_Tinker_Recipe(), rank=rank)


def test_lora_config_kwargs_rejects_non_gateway_recipe():
    with pytest.raises(TrainingGymConfigError, match="multi_lora_n_adapters"):
        lora_config_kwargs(MilesRecipe(lora_rank=8))


def test_gateway_lora_config_kwargs_uses_served_recipe():
    gateway = _gateway(tinker_train_mlp=False)
    assert gateway.lora_config_kwargs(rank=16) == {
        "rank": 16,
        "train_attn": True,
        "train_mlp": False,
        "train_unembed": True,
    }


# --- SDK presence ------------------------------------------------------------


def test_require_sdk_points_at_pinned_version_when_missing():
    with patch("modal_training_gym.tinker.installed_sdk_version", return_value=None):
        with pytest.raises(
            TrainingGymConfigError, match=f"tinker=={TINKER_SDK_VERSION}"
        ):
            require_sdk()


def test_require_sdk_warns_on_version_mismatch(capsys):
    with patch("modal_training_gym.tinker.installed_sdk_version", return_value="0.1.0"):
        require_sdk()
    assert TINKER_SDK_VERSION in capsys.readouterr().out


def test_service_client_needs_sdk():
    with patch("modal_training_gym.tinker.installed_sdk_version", return_value=None):
        with pytest.raises(TrainingGymConfigError, match="not installed"):
            _gateway(unauthenticated=True).service_client(tenant="alice")


# --- with the SDK installed -----------------------------------------------------


@needs_sdk
def test_service_client_sets_url_tenant_key_and_proxy_headers():
    env = {"MODAL_KEY": "wk-test", "MODAL_SECRET": "ws-test"}
    with (
        patch.dict("os.environ", env),
        patch("modal_training_gym.common.config.load_proxy_auth"),
    ):
        client = _gateway().service_client(
            tenant="alice", default_headers={"X-Extra": "1"}
        )
    kwargs = client._holder_kwargs
    assert kwargs["base_url"] == "https://demo-tinker.modal.run"
    assert kwargs["api_key"] == "tml-alice"
    assert kwargs["default_headers"] == {
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
        "X-Extra": "1",
    }


@needs_sdk
def test_cross_entropy_datums_shift_targets_by_one():
    from modal_training_gym.tinker import cross_entropy_datums, target_token_count

    tokenizer = _WordTokenizer()
    datums = cross_entropy_datums(tokenizer, ["a b c d", "e f"])

    first, second = datums
    tokens = tokenizer.encode("a b c d")
    assert first.model_input.to_ints() == tokens[:-1]
    assert first.loss_fn_inputs["target_tokens"].data == tokens[1:]
    assert first.loss_fn_inputs["weights"].data == [1.0, 1.0, 1.0]
    assert second.model_input.length == 1
    assert target_token_count(datums) == 4


@needs_sdk
def test_cross_entropy_datums_rejects_single_token_text():
    from modal_training_gym.tinker import cross_entropy_datums

    with pytest.raises(TrainingGymConfigError, match="at least two"):
        cross_entropy_datums(_WordTokenizer(), ["solo"])


@needs_sdk
def test_cross_entropy_datums_rejects_oversized_batch():
    from modal_training_gym.tinker import cross_entropy_datums

    texts = ["a b"] * (MAX_DATUMS_PER_REQUEST + 1)
    with pytest.raises(TrainingGymConfigError, match="datums per forward_backward"):
        cross_entropy_datums(_WordTokenizer(), texts)
