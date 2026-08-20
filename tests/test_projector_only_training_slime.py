"""Projector-only training on slime: the path from a recipe field to slime's CLI.

The container-side halves (provider, forward merge, position rebasing,
checkpoint hook) need torch and a GPU, so what is pinned here is the wiring the
launcher owns: the spec reaching the containers, the flags that make the run
supervised and engine-free, and the guards against configurations that would
silently train nothing or train on a wrong gradient.

The miles side of the same feature is pinned in
``tests/test_projector_only_training.py``.
"""

import pytest
from pydantic import ValidationError

from modal_training_gym import (
    EmbeddingProjectorDataset,
    ProjectorSpec,
    Qwen3_6_35B,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.train import _merge_recipe
from modal_training_gym.frameworks.slime.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_training_gym.frameworks.slime.projector_config import (
    ARGS_KEY,
    PROVIDER_PATH,
    ROLLOUT_PATH,
    SAVE_HOOK_PATH,
    TRAINABLE_PARAM_PATTERNS,
    from_slime_args,
    should_save_projector,
)
from modal_training_gym.train_recipes.slime_recipe import (
    Qwen3_6_35b_Projector_Recipe,
    Qwen3_6_35b_Recipe,
    SlimeRecipe,
)


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
    return EmbeddingProjectorDataset.synthetic(n_rows=2, input_dim=4)


def _cli_flags(recipe: Qwen3_6_35b_Projector_Recipe) -> dict[str, str]:
    return _flags(recipe.cli_args(dataset=_dataset(), model=Qwen3_6_35B()))


class _FakeArgs:
    def __init__(self, payload: dict) -> None:
        setattr(self, ARGS_KEY, payload)


def test_spec_reaches_containers_through_extra_config():
    recipe = Qwen3_6_35b_Projector_Recipe(
        projector=ProjectorSpec(input_dim=4, hidden_dim=8, num_layers=1)
    )
    payload = recipe.extra_config[ARGS_KEY]
    assert payload["input_dim"] == 4 and payload["num_layers"] == 1
    # slime sets every extra_config key on args, which is how the provider, the
    # rollout function and the checkpoint hook read the spec back without new
    # CLI flags.
    spec = from_slime_args(_FakeArgs(payload))
    assert spec.input_dim == 4
    assert spec.embeddings_key == "projector_embeddings"


def test_missing_spec_is_a_clear_error():
    with pytest.raises(ValueError, match=ARGS_KEY):
        from_slime_args(_FakeArgs.__new__(_FakeArgs))


def test_recipe_emits_supervised_engine_free_flags():
    flags = _cli_flags(Qwen3_6_35b_Projector_Recipe(projector=ProjectorSpec()))

    assert flags["--custom-model-provider-path"] == PROVIDER_PATH
    assert flags["--rollout-function-path"] == ROLLOUT_PATH
    assert flags["--loss-type"] == "sft_loss"
    assert "--debug-train-only" in flags
    assert "--disable-compute-advantages-and-returns" in flags
    # Freezing happens in the provider; this states the same intent in slime's
    # own terms and is what its freeze wrapper re-applies to the built model.
    assert flags["--only-train-params-name-list"] == TRAINABLE_PARAM_PATTERNS[0]
    # No LoRA anywhere: the whole point of freezing the base instead.
    assert "--lora-rank" not in flags
    # The projector spec travels in the YAML config, never as a flag.
    assert "--projector" not in flags


def test_other_slime_recipes_command_lines_are_untouched():
    """The supervised flags are opt-in: an RL recipe still emits neither."""
    flags = _flags(
        Qwen3_6_35b_Recipe().cli_args(dataset=_dataset(), model=Qwen3_6_35B())
    )
    assert "--loss-type" not in flags
    assert "--loss-mask-type" not in flags
    assert "--debug-train-only" not in flags


def test_no_eval_pass_is_configured():
    """The rollout function raises on evaluation, so none may be scheduled."""
    assert Qwen3_6_35b_Projector_Recipe().eval_interval is None
    assert "--eval-interval" not in _cli_flags(Qwen3_6_35b_Projector_Recipe())


def test_save_hook_runs_through_the_gyms_phase_reporting_wrapper():
    """Dashboard phase/substep timing must survive the projector's own hook."""
    flags = _cli_flags(Qwen3_6_35b_Projector_Recipe())
    assert flags["--custom-megatron-before-train-step-hook-path"].startswith(
        "modal_training_gym.frameworks.slime.phase_reporting"
    )
    assert (
        Qwen3_6_35b_Projector_Recipe().extra_config[
            "training_gym_custom_megatron_before_train_step_hook_path"
        ]
        == SAVE_HOOK_PATH
    )


def test_base_weights_come_from_the_rl_recipes_conversion():
    """Training at PP1 reshards the RL recipe's PP2 base, so it converts once.

    The conversion layout must stay pinned even though training is PP1: slime's
    converter re-derives PP from the world size when passed PP=1, which at TP2
    asks Megatron for 4 ranks in a 2-rank job.
    """
    rl, projector = Qwen3_6_35b_Recipe(), Qwen3_6_35b_Projector_Recipe()
    assert projector.pipeline_model_parallel_size == 1
    assert projector.ref_load == rl.ref_load
    assert projector.conversion_tensor_model_parallel_size == 2
    assert projector.conversion_pipeline_model_parallel_size == 2
    assert projector.ref_load.endswith("tp2pp2")

    nodes, procs, conversion_args = get_checkpoint_conversion_policy(
        projector, model=Qwen3_6_35B()
    )
    # The converter's job must have exactly TP*PP ranks, or Megatron asserts on
    # a world size that does not match the model layout it was given.
    assert nodes * procs == 4
    assert "--tensor-model-parallel-size 2" in conversion_args
    assert "--pipeline-model-parallel-size 2" in conversion_args


def test_pipeline_parallelism_is_rejected():
    """PP>1 would give later stages an optimizer over an empty parameter set."""
    with pytest.raises(TrainingGymConfigError, match="pipeline_model_parallel_size=1"):
        Qwen3_6_35b_Projector_Recipe(
            pipeline_model_parallel_size=2
        ).validate_model_parallelism(Qwen3_6_35B())


def test_context_parallelism_is_rejected():
    """CP chunks the sequence its own way; the merge rebases only TP/SP."""
    with pytest.raises(TrainingGymConfigError, match="context_parallel_size=1"):
        Qwen3_6_35b_Projector_Recipe(
            context_parallel_size=2
        ).validate_model_parallelism(Qwen3_6_35B())


def test_tensor_parallelism_without_sequence_parallelism_is_rejected():
    """Summing the replicated projector's grads over TP assumes disjoint shards."""
    with pytest.raises(TrainingGymConfigError, match="sequence_parallel=True"):
        Qwen3_6_35b_Projector_Recipe(
            sequence_parallel=False
        ).validate_model_parallelism(Qwen3_6_35B())
    # The shipped shape passes its own preflight, EP included.
    Qwen3_6_35b_Projector_Recipe().validate_model_parallelism(Qwen3_6_35B())


def test_distributed_optimizer_is_rejected():
    """Its reduce-scattered main_grad is not what the TP sum may read."""
    # Pydantic wraps the validator's error; the message is what a user reads.
    with pytest.raises(ValidationError, match="use_distributed_optimizer=False"):
        Qwen3_6_35b_Projector_Recipe(use_distributed_optimizer=True)


def test_rl_objectives_are_rejected():
    """No engines run, so a policy loss or KL would score the dataset's tokens."""
    with pytest.raises(ValidationError, match="sft_loss"):
        Qwen3_6_35b_Projector_Recipe(loss_type="policy_loss")
    with pytest.raises(ValidationError, match="KL"):
        Qwen3_6_35b_Projector_Recipe(use_kl_loss=True)
    with pytest.raises(ValidationError, match="KL"):
        Qwen3_6_35b_Projector_Recipe(kl_coef=0.01)


def test_megatron_never_writes_the_frozen_base():
    """Only the hook writes; Megatron's own interval must stay out of the run."""
    recipe = Qwen3_6_35b_Projector_Recipe()
    assert recipe.save_interval > recipe.num_rollout
    assert recipe.no_save_optim is True


def test_a_finished_run_always_leaves_a_projector_checkpoint():
    """The run's last optimizer step saves even when the interval misses it."""
    recipe = Qwen3_6_35b_Projector_Recipe()
    total = (
        recipe.num_rollout
        * recipe.rollout_batch_size
        * recipe.n_samples_per_prompt
        // recipe.global_batch_size
    )
    saves = [
        s
        for s in range(1, total + 1)
        if should_save_projector(s, s, total, recipe.projector.save_interval)
    ]
    assert saves and saves[-1] == total
    # A skipped update (gradient overflow) consumed one of the run's steps, so
    # the last attempt still saves what the applied steps produced...
    assert should_save_projector(total - 1, total, total, 0)
    # ...but a run where nothing ever applied has no trained adapter to write.
    assert not should_save_projector(0, total, total, 0)


def test_projector_training_stays_opt_in():
    """Qwen3.6-35B's default recipe is still the RL one."""
    assert type(SlimeRecipe.get_base_recipe(Qwen3_6_35B())) is Qwen3_6_35b_Recipe


def test_preset_merge_keeps_the_projector_recipe_class():
    """The merge fills unset fields from the RL preset without demoting to it.

    ``get_base_recipe`` returns the RL recipe, so rebuilding the merged recipe
    as the preset's class would drop ``projector`` and every guard above.
    """
    merged = _merge_recipe(
        Qwen3_6_35b_Recipe(),
        Qwen3_6_35b_Projector_Recipe(
            projector=ProjectorSpec(input_dim=4, num_layers=1)
        ),
    )
    assert type(merged) is Qwen3_6_35b_Projector_Recipe
    assert merged.projector.input_dim == 4
    assert merged.pipeline_model_parallel_size == 1
    assert merged.loss_type == "sft_loss"
    assert merged.extra_config[ARGS_KEY]["input_dim"] == 4


def test_validation_registry_dispatches_the_projector_recipe_by_name():
    """``check -m Qwen3.6-35B-A3B-Projector`` has to reach the projector path.

    The registry keys on the ``ModelConfig``, and this model's slime base recipe
    is the RL one, so the entry carries ``projector=True`` and the backend picks
    the recipe from its own table. Dispatch-only: it is a 35B MoE on a node.
    """
    from modal_training_gym.common.models.validation import (
        _ValidationConfig,
        Framework,
    )
    from scripts.validation_backends import build_recipe_and_dataset

    config = _ValidationConfig.find("Qwen3.6-35B-A3B-Projector")
    assert config.projector and not config.run_on_pr
    assert config.framework is Framework.SLIME

    recipe, dataset = build_recipe_and_dataset(
        config.framework, config.model_config(), 3, projector=config.projector
    )
    assert type(recipe) is Qwen3_6_35b_Projector_Recipe
    # No public dataset ships encoder embeddings, so the rows are synthetic and
    # sized to the projector's input width: enough for every step of the run.
    assert dataset.synthetic_input_dim == recipe.projector.input_dim
    assert dataset.synthetic_rows == recipe.rollout_batch_size * 3


def test_a_projector_entry_without_a_recipe_is_rejected():
    """A registry entry for a model with no projector recipe must not run RL."""
    from modal_training_gym.common.errors import TrainingGymConfigError
    from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
    from scripts.validation_backends import build_recipe_and_dataset
    from modal_training_gym.common.models.validation import Framework

    with pytest.raises(TrainingGymConfigError, match="no projector-only slime"):
        build_recipe_and_dataset(Framework.SLIME, Qwen3_4B(), 1, projector=True)


def test_projector_keys_are_hidden_whatever_prefix_megatron_asks_for():
    """The base checkpoint predates the projector, so its keys must not appear.

    Megatron's checkpointing calls ``sharded_state_dict()`` with the default
    empty prefix, but ``prefix`` is the signature's first positional argument
    and a caller passing one prefixes every key — matching the bare name would
    then leave the projector in and fail the load of a fine base checkpoint.
    """
    pytest.importorskip("torch")  # the framework module is import-time torch
    from modal_training_gym.frameworks.slime.embedding_projector import (
        _hide_projector_from_megatron_checkpoints,
    )

    class FakeModel:
        def sharded_state_dict(self, prefix=""):
            return {
                f"{prefix}decoder.layers.0.mlp.weight": object(),
                f"{prefix}embedding_projector.mlp.0.weight": object(),
            }

    model = FakeModel()
    _hide_projector_from_megatron_checkpoints(model)

    assert list(model.sharded_state_dict()) == ["decoder.layers.0.mlp.weight"]
    assert list(model.sharded_state_dict("module.")) == [
        "module.decoder.layers.0.mlp.weight"
    ]
    assert list(model.sharded_state_dict(prefix="module.")) == [
        "module.decoder.layers.0.mlp.weight"
    ]
