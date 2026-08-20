"""Projector-only training: the path from a recipe field to miles' CLI and data.

The container-side halves (provider, forward merge, checkpoint hook) need torch
and a GPU, so what is pinned here is the wiring the launcher owns: the spec
reaching the containers, the flags that make the run supervised and engine-free,
the guards against configurations that silently train nothing, and the dataset
columns miles reads.
"""

import json

import pytest
from pydantic import ValidationError

from modal_training_gym import (
    GLM_5_2,
    GLM_5_2_5Layer,
    EmbeddingProjectorDataset,
    GLM_5_2_5Layer_Projector_Recipe,
    GLM_5_2_Projector_Recipe,
    ProjectorSpec,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.frameworks.miles.projector_config import (
    ARGS_KEY,
    PROVIDER_PATH,
    ROLLOUT_PATH,
    SAVE_HOOK_PATH,
    from_miles_args,
    should_save_projector,
)
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def _flags(args: list[str]) -> dict[str, str]:
    return {
        args[i]: (
            args[i + 1]
            if i + 1 < len(args) and not args[i + 1].startswith("--")
            else ""
        )
        for i in range(len(args))
        if args[i].startswith("--")
    }


def _dataset() -> EmbeddingProjectorDataset:
    return EmbeddingProjectorDataset(
        rows=[
            {
                "messages": [
                    {"role": "user", "content": "protein?"},
                    {"role": "assistant", "content": "kinase"},
                ],
                "embeddings": [[0.0] * 4, [1.0] * 4],
                "positions": [2, 3],
            }
        ]
    )


def test_spec_reaches_containers_through_extra_config():
    recipe = GLM_5_2_5Layer_Projector_Recipe(
        projector=ProjectorSpec(input_dim=4, hidden_dim=8, num_layers=1)
    )
    payload = recipe.extra_config[ARGS_KEY]
    assert payload["input_dim"] == 4 and payload["num_layers"] == 1
    # miles sets every extra_config key on args, which is how the provider and
    # the checkpoint hook read the spec back without new CLI flags.
    spec = from_miles_args(_FakeArgs(payload))
    assert spec.input_dim == 4
    assert spec.embeddings_key == "projector_embeddings"


class _FakeArgs:
    def __init__(self, payload: dict) -> None:
        setattr(self, ARGS_KEY, payload)


def test_missing_spec_is_a_clear_error():
    with pytest.raises(ValueError, match=ARGS_KEY):
        from_miles_args(_FakeArgs.__new__(_FakeArgs))


@pytest.mark.parametrize("container", ["extra_config", "custom_config"])
def test_spec_is_also_found_nested_under_the_config_containers(container: str):
    """Mirrors the gym's other in-container readers, which try both spellings.

    The pinned image flattens ``extra_config`` onto ``args``; a plumbing change
    that kept the dict nested must not silently lose the projector.
    """
    args = _FakeArgs.__new__(_FakeArgs)
    setattr(args, container, {ARGS_KEY: ProjectorSpec(input_dim=7).to_args_dict()})
    assert from_miles_args(args).input_dim == 7


def test_frozen_base_cannot_be_left_uninitialized():
    """``hf_checkpoint`` is the only base-weight source once ref_load is rejected.

    Bridge mode falls back to ``args.load = args.ref_load or args.hf_checkpoint``,
    so an empty one would train the projector against random base weights.
    """
    with pytest.raises(ValidationError, match="hf_checkpoint"):
        GLM_5_2_Projector_Recipe(hf_checkpoint="")
    with pytest.raises(ValidationError, match="bridge"):
        GLM_5_2_Projector_Recipe(megatron_to_hf_mode="dist")
    assert GLM_5_2_Projector_Recipe().hf_checkpoint == "zai-org/GLM-5.2"


def test_overrides_assigned_after_construction_are_still_rejected():
    """Pydantic dataclasses do not re-validate on assignment, but callers assign.

    ``scripts/validate_model_configs.py`` sets ``eval_interval`` on the recipe
    the backend built, which would otherwise reach the rollout function's hard
    raise mid-run.
    """
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    recipe.eval_interval = 5
    with pytest.raises(TrainingGymConfigError, match="cannot evaluate"):
        recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer())

    recipe.eval_interval = None
    recipe.save_interval = 1
    # Writing the unchanged frozen base is wasteful, not wrong: warn, don't fail.
    with pytest.warns(UserWarning, match="frozen base"):
        recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer())


def test_recipe_emits_supervised_engine_free_flags():
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    flags = _flags(recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer()))

    assert flags["--custom-model-provider-path"] == PROVIDER_PATH
    assert flags["--rollout-function-path"] == ROLLOUT_PATH
    assert flags["--loss-type"] == "sft_loss"
    assert "--debug-train-only" in flags
    assert "--disable-compute-advantages-and-returns" in flags
    # No LoRA anywhere: the whole point of freezing the base instead.
    assert "--lora-rank" not in flags
    # The projector spec travels in the YAML config, never as a flag.
    assert "--projector" not in flags


def test_the_fields_this_adds_to_the_base_recipe_change_no_other_recipe():
    """The projector's five new ``MilesRecipe`` fields are opt-in, not defaults.

    They default to values ``cli_args`` skips, so every miles recipe that existed
    before this change emits the command line it emitted before — which is the
    only thing keeping a projector-shaped flag out of a Qwen or Moonlight run.
    """
    from modal_training_gym.common.models.qwen3_5_4b import Qwen3_5_4B
    from modal_training_gym.train_recipes.miles_recipe import Qwen3_5_4b_Miles_Recipe

    recipe = Qwen3_5_4b_Miles_Recipe()
    flags = _flags(recipe.cli_args(dataset=_dataset(), model=Qwen3_5_4B()))
    for flag in (
        "--custom-model-provider-path",
        "--loss-type",
        "--num-epoch",
        "--debug-train-only",
        "--disable-compute-advantages-and-returns",
    ):
        assert flag not in flags


def test_save_hook_runs_through_the_gyms_phase_reporting_wrapper():
    """Dashboard phase/substep timing must survive the projector's own hook."""
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    flags = _flags(recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer()))
    assert flags["--custom-megatron-before-train-step-hook-path"].startswith(
        "modal_training_gym.frameworks.miles.phase_reporting"
    )
    assert (
        recipe.extra_config["training_gym_custom_megatron_before_train_step_hook_path"]
        == SAVE_HOOK_PATH
    )


def test_an_unwired_step_hook_fails_at_model_construction():
    """Without the hook there is no grad all-reduce and no checkpoint at all."""
    from types import SimpleNamespace

    from modal_training_gym.frameworks.miles.projector_config import (
        require_projector_step_hook,
    )

    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    flags = _flags(recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer()))
    wired = SimpleNamespace(
        custom_megatron_before_train_step_hook_path=flags[
            "--custom-megatron-before-train-step-hook-path"
        ],
        extra_config=dict(recipe.extra_config or {}),
    )
    require_projector_step_hook(wired)

    wired.extra_config.pop(
        "training_gym_custom_megatron_before_train_step_hook_path", None
    )
    with pytest.raises(ValueError, match="unreduced gradients and no checkpoints"):
        require_projector_step_hook(wired)


def test_only_the_attention_backend_that_keeps_the_gradient_is_allowed():
    """tilelang's fused sparse-MLA backward hands the embeddings NaN dq.

    Verified on 8xH200: finite forward and finite grads down to the query
    projection, then NaN out of ``sparse_mla_bwd`` — harmless for the LoRA runs
    upstream ships, fatal for a projector whose gradient comes from there. The
    working pairing is unrecomputable, so recompute is rejected with it.
    """
    for override in (
        {"dsa_attention_backend": "tilelang", "qkv_format": "thd"},
        {"qkv_format": "thd"},
    ):
        with pytest.raises(ValidationError, match='dsa_attention_backend="megatron"'):
            GLM_5_2_Projector_Recipe(**override)
    with pytest.raises(ValidationError, match="recompute_granularity=None"):
        GLM_5_2_Projector_Recipe(recompute_granularity="full")
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    assert (recipe.dsa_attention_backend, recipe.qkv_format) == ("megatron", "bshd")
    recipe.recompute_granularity = "full"
    with pytest.raises(TrainingGymConfigError, match="recompute_granularity=None"):
        recipe.validate_model_parallelism(GLM_5_2_5Layer())


def test_lora_is_rejected():
    # Pydantic wraps the validator's error; the message is what a user reads.
    with pytest.raises(ValidationError, match="lora_rank must be unset"):
        GLM_5_2_Projector_Recipe(lora_rank=16)


def test_pipeline_parallelism_is_rejected():
    """PP>1 would give later stages an optimizer over an empty parameter set."""
    recipe = GLM_5_2_Projector_Recipe(pipeline_model_parallel_size=2)
    with pytest.raises(TrainingGymConfigError, match="pipeline_model_parallel_size=1"):
        recipe.validate_model_parallelism(GLM_5_2())


def test_context_parallelism_is_rejected():
    """CP shards the sequence its own way; the merge rebases only TP/SP."""
    recipe = GLM_5_2_Projector_Recipe(context_parallel_size=2)
    with pytest.raises(TrainingGymConfigError, match="context_parallel_size=1"):
        recipe.validate_model_parallelism(GLM_5_2())


def test_tensor_parallelism_without_sequence_parallelism_is_rejected():
    """Summing the replicated projector's grads over TP assumes disjoint shards."""
    recipe = GLM_5_2_Projector_Recipe(sequence_parallel=False)
    with pytest.raises(TrainingGymConfigError, match="sequence_parallel=True"):
        recipe.validate_model_parallelism(GLM_5_2())
    # TP=1 has nothing to sum over, so the combination is fine there — at a
    # batch that still covers the 64 data-parallel ranks TP=1 leaves.
    GLM_5_2_Projector_Recipe(
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        rollout_batch_size=64,
        global_batch_size=64,
    ).validate_model_parallelism(GLM_5_2())


def test_distributed_optimizer_is_rejected():
    """It replaces the main_grad buffers the projector's TP sum reads whole."""
    with pytest.raises(ValidationError, match="use_distributed_optimizer=False"):
        GLM_5_2_Projector_Recipe(use_distributed_optimizer=True)
    assert GLM_5_2_Projector_Recipe().use_distributed_optimizer is False


def test_projector_starts_at_the_bases_embedding_scale():
    """A unit-std LayerNorm output is ~100x the scale of a token embedding."""
    spec = ProjectorSpec()
    assert spec.output_scale == 0.01
    assert spec.to_args_dict()["output_scale"] == 0.01
    with pytest.raises(ValidationError, match="output_scale=0.0 must be positive"):
        ProjectorSpec(output_scale=0.0)


def test_renaming_the_positions_column_is_rejected():
    """Miles offsets packed-batch keys only when they end in ``_positions``."""
    with pytest.raises(ValidationError, match="must end in '_positions'"):
        ProjectorSpec(positions_key="proj_pos")
    assert ProjectorSpec(positions_key="protein_positions").positions_key


def test_megatron_checkpoint_loading_is_rejected():
    """The projector is a submodule, so base state dicts lack its keys."""
    with pytest.raises(ValidationError, match="resumes through projector.load"):
        GLM_5_2_Projector_Recipe(load="/checkpoints/glm")
    with pytest.raises(ValidationError, match="resumes through projector.load"):
        GLM_5_2_Projector_Recipe(ref_load="/checkpoints/glm")


def test_numbered_checkpoints_do_not_depend_on_the_predicted_step_count():
    """``train_iters`` is a prediction; only numbered files may depend on it."""
    total = 10
    # A run that outlives the prediction keeps one numbered file, not one per
    # step from then on. Every applied step refreshes projector_latest.pt.
    assert [
        s for s in range(1, 14) if should_save_projector(s, s, total, save_interval=0)
    ] == [total]
    assert [
        s for s in range(1, 11) if should_save_projector(s, s, 10, save_interval=4)
    ] == [4, 8, 10]
    # The shipped defaults: one save, at the end of the run.
    recipe = GLM_5_2_Projector_Recipe()
    total = (
        recipe.num_rollout
        * recipe.rollout_batch_size
        * recipe.n_samples_per_prompt
        // recipe.global_batch_size
    )
    assert [
        s
        for s in range(1, total + 1)
        if should_save_projector(s, s, total, recipe.projector.save_interval)
    ] == [total]
    # An interval of 0 turns periodic saves off, but not the final one.
    assert not should_save_projector(3, 3, total, 0)
    assert should_save_projector(total, total, total, 0)
    # A skipped update (gradient overflow) consumed one of the run's steps, so
    # the last attempt still saves what the applied steps produced.
    assert should_save_projector(total - 1, total, total, 0)
    # ...but a run where nothing ever applied has no trained adapter to write.
    assert not should_save_projector(0, total, total, 0)


def test_expert_parallel_size_must_divide_the_model_scripts_experts():
    """GLM-5.2 has no ModelArchitecture, so the gym's own EP preflight is a no-op."""
    with pytest.raises(TrainingGymConfigError, match="256 routed experts"):
        GLM_5_2_Projector_Recipe(
            expert_model_parallel_size=24
        ).validate_model_parallelism(GLM_5_2())
    for recipe in (GLM_5_2_Projector_Recipe(), GLM_5_2_5Layer_Projector_Recipe()):
        recipe.validate_model_parallelism(GLM_5_2())


def test_a_step_must_have_a_sample_for_every_data_parallel_rank():
    """Both shipped shapes, and a rejection when a topology outgrows the batch."""
    for recipe in (GLM_5_2_Projector_Recipe(), GLM_5_2_5Layer_Projector_Recipe()):
        recipe.validate_model_parallelism(GLM_5_2())
    with pytest.raises(TrainingGymConfigError, match="16 data-parallel rank"):
        GLM_5_2_Projector_Recipe(
            rollout_batch_size=8, global_batch_size=8
        ).validate_model_parallelism(GLM_5_2())


def test_no_eval_pass_is_configured():
    """The rollout raises on evaluation, so the run must not schedule one."""
    assert GLM_5_2_Projector_Recipe().skip_eval_before_train is True
    with pytest.raises(ValidationError, match="cannot evaluate"):
        GLM_5_2_Projector_Recipe(eval_interval=5)


def test_synthetic_validation_data_is_regenerated_per_run():
    """The on-volume path is class-derived, so stale rows would be reused."""
    assert EmbeddingProjectorDataset.synthetic(n_rows=2, input_dim=4).always_prepare


def test_disk_reservation_survives_caller_supplied_train_kwargs():
    recipe = GLM_5_2_Projector_Recipe(train_function_kwargs={"timeout": 60})
    assert recipe.train_function_kwargs["timeout"] == 60
    assert recipe.train_function_kwargs["ephemeral_disk"] > 0


def test_base_recipe_lookup():
    assert isinstance(MilesRecipe.get_base_recipe(GLM_5_2()), GLM_5_2_Projector_Recipe)
    assert isinstance(
        MilesRecipe.get_base_recipe(GLM_5_2_5Layer()),
        GLM_5_2_5Layer_Projector_Recipe,
    )


def test_dataset_writes_embeddings_into_the_metadata_column(tmp_path):
    dataset = _dataset()
    out = str(tmp_path / "train.jsonl")
    dataset.prepare(out)
    dataset.validate_prepared(out)
    row = json.loads(open(out).readline())
    assert row["metadata"]["projector_positions"] == [2, 3]
    assert len(row["metadata"]["projector_embeddings"]) == 2
    # The conversation stays a message list: the SFT loss mask is built by
    # splitting it, which a rendered string would not allow.
    assert isinstance(row["messages"], list)


def test_dataset_rejects_mismatched_embeddings_and_positions():
    with pytest.raises(TrainingGymConfigError, match="position"):
        EmbeddingProjectorDataset(
            rows=[
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "embeddings": [[0.0], [1.0]],
                    "positions": [1],
                }
            ]
        )


def test_offloading_the_train_model_is_rejected():
    """Offloading zeroes the parameter buffer, which holds only the projector.

    Megatron allocates it in a torch_memory_saver region with no backup, and
    miles' ``sleep()`` pauses every tag, so a projector-only run would wake up
    training weights that read exactly zero (verified on 8xH200).
    """
    with pytest.raises(ValidationError, match="no_offload_train=True"):
        GLM_5_2_Projector_Recipe(no_offload_train=False)
    assert GLM_5_2_Projector_Recipe().no_offload_train is True
    recipe = GLM_5_2_5Layer_Projector_Recipe(projector=ProjectorSpec(input_dim=4))
    recipe.no_offload_train = False
    with pytest.raises(TrainingGymConfigError, match="no_offload_train=True"):
        recipe.cli_args(dataset=_dataset(), model=GLM_5_2_5Layer())


def test_a_zeroed_projector_is_caught_at_the_first_forward():
    """Zeroed weights write exactly-zero rows: trained, but on no signal at all."""
    torch = pytest.importorskip("torch")
    from modal_training_gym.frameworks.miles.embedding_projector import (
        EmbeddingProjector,
        check_projector_weights,
        init_projector,
    )

    projector = EmbeddingProjector(input_dim=4, hidden_dim=8, output_dim=6)
    init_projector(projector, seed=0, output_scale=0.01)
    stats = check_projector_weights(projector)
    assert "mlp.0.weight rms=" in stats and "norm.weight rms=0.01" in stats
    out = projector(torch.zeros(2, 4))
    assert torch.isfinite(out).all() and float(out.abs().max()) > 0.0

    with torch.no_grad():
        for param in projector.parameters():
            param.zero_()
    with pytest.raises(ValueError, match="no_offload_train=True"):
        check_projector_weights(projector)
    # Every row it would merge is exactly zero, whatever the encoder produced.
    assert float(projector(torch.ones(2, 4)).abs().max()) == 0.0
