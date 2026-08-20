from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import pytest
from modal._object import _Object
from modal._utils.async_utils import synchronizer

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.frameworks.slime import launcher
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


_MODAL_SERIALIZED_FUNCTION_LIMIT_BYTES = 64 * 1024
_STRONG_SERIALIZED_FUNCTION_BUDGET_BYTES = 48 * 1024
_REQUEST_ENVIRONMENT_KEY = "DRIFT_ASYNC_RL_COMPLETION_NEUTRALIZER_REQUEST_B64_ZLIB"


def _request_bearing_recipe() -> SlimeRecipe:
    # Match the size of the frozen CVB request-bearing environment value that
    # exposed Modal's registration limit.  The value is already compressed and
    # encoded before it reaches Training Gym, so an incompressible fixture is
    # unnecessary here: cloudpickle stores this string verbatim.
    image_env = {
        _REQUEST_ENVIRONMENT_KEY: "x" * 40_876,
        "DRIFT_ASYNC_RL_COMPLETION_RUNTIME_PREFLIGHT": "1",
        "DRIFT_ASYNC_RL_REPLAY_IN_PROCESS_RETRIES": "0",
    }
    return SlimeRecipe(
        gpu_type="H200",
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        rollout_num_gpus_per_engine=1,
        num_rollout=4,
        rollout_batch_size=256,
        rollout_max_response_len=256,
        rollout_temperature=1.0,
        save_interval=1,
        image_env=image_env,
        max_retries=0,
        max_attempts=1,
    )


def _closure_cell(function: Any, name: str) -> Any:
    raw = function.info.raw_f
    cells = dict(zip(raw.__code__.co_freevars, raw.__closure__ or (), strict=True))
    return cells[name]


def _fake_hydrate_captured_modal_objects(functions: Mapping[str, Any]) -> None:
    objects: dict[int, _Object] = {}
    for function in functions.values():
        raw = function.info.raw_f
        for cell in raw.__closure__ or ():
            candidate = synchronizer._translate_in(cell.cell_contents)
            if isinstance(candidate, _Object):
                objects[id(candidate)] = candidate
    for ordinal, candidate in enumerate(objects.values()):
        candidate._hydrate(
            f"vo-serialization-budget-{ordinal:08d}",
            object(),
            None,
        )


def test_request_image_environment_is_baked_once_but_not_serialized_in_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _request_bearing_recipe()
    original_image_env = dict(recipe.image_env)
    image_environment_calls: list[dict[str, str]] = []
    original_env = launcher.Image.env

    def record_image_environment(
        image: launcher.Image,
        variables: Mapping[str, str | None],
    ) -> launcher.Image:
        image_environment_calls.append(
            {key: value for key, value in variables.items() if value is not None}
        )
        return original_env(image, variables)

    # App construction must remain completely local: named-secret hydration is
    # irrelevant to function packaging and would otherwise contact Modal.
    monkeypatch.setattr(launcher, "hf_secrets", lambda: [])
    monkeypatch.setattr(launcher, "resolve_caller_context", lambda: (None, None))
    monkeypatch.setattr(launcher.Image, "env", record_image_environment)

    app = launcher.build_slime_app(
        training_run_id="cvb-serialization-budget",
        slime=recipe,
        model=Qwen3_4B(),
        dataset=HuggingFaceDataset(
            hf_repo="openai/gsm8k",
            input_column="question",
            output_column="answer",
        ),
        name="cvb-serialization-budget",
    )

    functions = app.registered_functions
    assert set(functions) == {
        "convert_checkpoint",
        "download",
        "prepare_dataset",
        "train",
    }
    request_environment_calls = [
        value for value in image_environment_calls if _REQUEST_ENVIRONMENT_KEY in value
    ]
    assert request_environment_calls == [original_image_env]
    assert recipe.image_env == original_image_env

    train_recipe = _closure_cell(functions["train"], "slime").cell_contents
    conversion_recipe = _closure_cell(
        functions["convert_checkpoint"], "slime"
    ).cell_contents
    assert train_recipe is conversion_recipe
    assert train_recipe is not recipe
    assert train_recipe.image_env == {}
    assert {
        key: value for key, value in vars(train_recipe).items() if key != "image_env"
    } == {key: value for key, value in vars(recipe).items() if key != "image_env"}

    _fake_hydrate_captured_modal_objects(functions)
    assert all(function.info.is_serialized() for function in functions.values())
    serialized_sizes = {
        tag: len(function.info.serialized_function())
        for tag, function in functions.items()
    }
    assert all(
        size <= _STRONG_SERIALIZED_FUNCTION_BUDGET_BYTES
        for size in serialized_sizes.values()
    ), serialized_sizes

    # Prove that this exact fixture reproduces the original registration
    # defect if the request-bearing image environment is put back into the
    # shared recipe closure.
    bloated_recipe = copy.copy(train_recipe)
    object.__setattr__(bloated_recipe, "image_env", original_image_env)
    train_recipe_cell = _closure_cell(functions["train"], "slime")
    train_recipe_cell.cell_contents = bloated_recipe
    try:
        assert (
            len(functions["train"].info.serialized_function())
            > _MODAL_SERIALIZED_FUNCTION_LIMIT_BYTES
        )
    finally:
        train_recipe_cell.cell_contents = train_recipe
