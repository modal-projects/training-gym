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
    ProteinLocalizationDataset,
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


def test_the_labelled_synthetic_task_puts_the_answer_only_in_the_embedding():
    """What makes the loss curve mean something: the prompt cannot answer it.

    Every row shares one prompt and the target follows from the embedding alone,
    so a falling loss is the projector carrying information rather than the base
    reading the question. The control deletes exactly that information and
    nothing else.
    """
    rows = EmbeddingProjectorDataset.synthetic_classification(
        n_rows=8, input_dim=16, n_classes=4
    ).rows
    prompts = {r["messages"][0]["content"] for r in rows}
    targets = [r["messages"][1]["content"] for r in rows]
    assert len(prompts) == 1
    assert len(set(targets)) == 4
    # One vector per class, identical across the rows sharing a class.
    per_class = {t: tuple(r["embeddings"][0]) for t, r in zip(targets, rows)}
    assert len(set(per_class.values())) == 4
    assert all(any(v) for v in per_class.values())

    control = EmbeddingProjectorDataset.synthetic_classification(
        n_rows=8, input_dim=16, n_classes=4, zero_embeddings=True
    ).rows
    assert [r["messages"] for r in control] == [r["messages"] for r in rows]
    assert all(not any(r["embeddings"][0]) for r in control)


def test_the_labelled_synthetic_task_rejects_more_classes_than_it_has_words():
    dataset = EmbeddingProjectorDataset.synthetic_classification(
        n_rows=2, input_dim=4, n_classes=99
    )
    with pytest.raises(TrainingGymConfigError, match="class words"):
        dataset.rows


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


def test_a_hook_call_without_an_optimizer_fails_rather_than_skipping():
    """The saver owns the grad all-reduce, so skipping it trains partial grads.

    The pinned image passes the optimizer, and the GPU runs installed the saver
    and wrote checkpoints; if a future image stops doing so, the run has to stop
    rather than quietly train tensor-parallel-partial gradients and produce no
    adapter.
    """
    pytest.importorskip("torch")
    from modal_training_gym.frameworks.miles.embedding_projector import (
        save_projector_checkpoint,
    )

    with pytest.raises(RuntimeError, match="without an optimizer"):
        save_projector_checkpoint(
            object(), rollout_id=0, step_id=0, model=[], optimizer=None
        )


def test_the_untrained_baseline_starts_where_the_run_started(tmp_path):
    """A custom init has to survive into the eval, or the baseline is a different model.

    ``load_projector(trained=False)`` is the "the projector contributes nothing"
    number, and it only means that if it inits from the seed and scale the run
    used rather than from ``ProjectorSpec()``'s.
    """
    torch = pytest.importorskip("torch")
    from modal_training_gym.frameworks.miles.embedding_projector import (
        EmbeddingProjector,
        init_projector,
        write_projector_checkpoint,
    )
    from modal_training_gym.frameworks.miles.projector_eval import load_projector

    spec = ProjectorSpec(
        input_dim=4, hidden_dim=8, output_dim=6, init_seed=7, output_scale=0.25
    )
    projector = EmbeddingProjector(input_dim=4, hidden_dim=8, output_dim=6)
    init_projector(projector, spec.init_seed, spec.output_scale)
    expected = {k: v.clone() for k, v in projector.state_dict().items()}
    write_projector_checkpoint(spec, str(tmp_path), projector, step=3, numbered=False)

    baseline, iteration = load_projector(
        str(tmp_path / "projector_latest.pt"), trained=False
    )
    assert iteration == 3
    for name, param in baseline.state_dict().items():
        assert torch.equal(param, expected[name]), name


# ── Real biological data: ESM-2 over DeepLoc ──────────────────────────────────


def test_protein_dataset_turns_deeploc_locations_into_answer_words():
    """The target has to be words, since the model answers by generating them.

    DeepLocMulti's label is ``"Endoplasmic.reticulum,M"``; DeepLocBinary's is a
    bare ``"M"``/``"S"``. Neither is something a decoder should be asked to emit.
    """
    dataset = ProteinLocalizationDataset()
    assert dataset._class_word("Endoplasmic.reticulum,M") == "endoplasmic reticulum"
    assert dataset._class_word("Nucleus,U") == "nucleus"
    assert dataset._class_word("M") == "membrane"
    assert dataset._class_word("S") == "soluble"


def test_protein_dataset_holds_its_eval_split_out_by_construction():
    """Train and eval come from DeepLoc's own disjoint splits, not a resample."""
    dataset = ProteinLocalizationDataset()
    assert dataset.train_split != dataset.eval_split
    assert dataset.always_prepare, "stale rows on the volume would be reused"
    train_path, eval_paths = MilesRecipe._resolve_data_paths(dataset)
    assert eval_paths and eval_paths["eval"] != train_path


def test_protein_dataset_rejects_an_encoder_the_projector_is_not_shaped_for():
    """Caught while materializing data, not after a cluster is up."""
    dataset = ProteinLocalizationDataset(input_dim=1536)
    dataset._encode = lambda sequences: [[0.0] * 640 for _ in sequences]  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TrainingGymConfigError, match="640-wide"):
        dataset._split_rows("test", 1)


def test_protein_rows_carry_the_embedding_in_the_column_miles_reads():
    dataset = ProteinLocalizationDataset(rows=[])
    row = dataset._to_row(
        {
            "messages": [
                {"role": "user", "content": dataset.prompt},
                {"role": "assistant", "content": "nucleus"},
            ],
            "embeddings": [[0.5] * 640],
            "positions": [1],
            "label": "nucleus",
        }
    )
    assert row["metadata"]["projector_embeddings"] == [[0.5] * 640]
    assert row["metadata"]["projector_positions"] == [1]


def test_the_held_out_eval_reads_targets_and_embeddings_back(tmp_path):
    """The eval's candidate set is the file's own answers, not a second list."""
    from modal_training_gym.frameworks.miles.projector_eval import read_eval_rows

    path = tmp_path / "eval.jsonl"
    dataset = ProteinLocalizationDataset(rows=[])
    with open(path, "w") as f:
        for word in ("nucleus", "membrane"):
            f.write(
                json.dumps(
                    dataset._to_row(
                        {
                            "messages": [
                                {"role": "user", "content": dataset.prompt},
                                {"role": "assistant", "content": word},
                            ],
                            "embeddings": [[0.25] * 640],
                            "positions": [1],
                            "label": word,
                        }
                    )
                )
                + "\n"
            )

    rows = read_eval_rows(str(path), "projector_embeddings", "projector_positions")
    assert [r["target"] for r in rows] == ["nucleus", "membrane"]
    # The answer is removed from what the model is shown, or accuracy is free.
    assert all(m["role"] != "assistant" for r in rows for m in r["messages"])
    assert rows[0]["embeddings"] == [[0.25] * 640] and rows[0]["positions"] == [1]


def test_the_report_states_both_baselines_it_has_to_beat():
    from modal_training_gym.frameworks.miles.projector_eval import (
        ProjectorEvalReport,
        ProjectorEvalRow,
    )

    def row(target: str, prediction: str) -> ProjectorEvalRow:
        return ProjectorEvalRow(
            prompt="p",
            target=target,
            prediction=prediction,
            correct=target == prediction,
            scores={},
        )

    report = ProjectorEvalReport(
        model_name="m",
        checkpoint="c",
        iteration=40,
        classes=["nucleus", "membrane"],
        rows=[
            row("nucleus", "nucleus"),
            row("nucleus", "nucleus"),
            row("membrane", "membrane"),
            row("membrane", "nucleus"),
        ],
        untrained_rows=[row("nucleus", "nucleus")] + [row("membrane", "nucleus")] * 3,
    )
    assert report.accuracy == 0.75
    assert report.untrained_accuracy == 0.25
    assert report.majority_accuracy == 0.5
    assert "trained 0.750" in report.summary()


def test_publishing_the_report_reaches_the_dashboard_store(monkeypatch):
    """The eval only counts if it lands in the dashboard, so publish is exercised.

    Both of the bugs this catches were invisible until a GPU eval had already
    finished: ``publish()`` is the last line of a multi-minute remote call, so a
    wrong ``create_hash`` arity there costs a whole eval to discover.
    """
    from modal_training_gym.frameworks.miles.projector_eval import (
        ProjectorEvalReport,
        ProjectorEvalRow,
    )

    written: dict[str, dict] = {}
    monkeypatch.setattr(
        "modal_training_gym.common.eval.vol_put",
        lambda store, key, value, **kw: written.__setitem__(key, value),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.eval.vol_get",
        lambda store, key, **kw: written.get(key, {}),
    )

    report = ProjectorEvalReport(
        model_name="zai-org/GLM-5.2",
        checkpoint="/checkpoints/run/projector/projector_latest.pt",
        iteration=150,
        classes=["nucleus", "membrane"],
        rows=[
            ProjectorEvalRow(
                prompt="p",
                target="nucleus",
                prediction="nucleus",
                correct=True,
                scores={"nucleus": -1.0, "membrane": -2.0},
            )
        ],
    )
    eval_id = report.publish()
    assert written[eval_id]["rows"][0]["score"] == 1.0
    assert written[eval_id]["model_name"] == "zai-org/GLM-5.2"


def test_the_eval_resolves_the_held_out_split_the_run_wrote(monkeypatch):
    """``ProjectorEval`` reads the training run's paths, not paths of its own."""
    from modal_training_gym.frameworks.miles.projector_eval import ProjectorEval
    from modal_training_gym.train_recipes.miles_recipe.glm_5_2 import (
        GLM_5_2_5Layer_Projector_Recipe,
    )

    recipe = GLM_5_2_5Layer_Projector_Recipe(num_rollout=4)
    dataset = ProteinLocalizationDataset()
    ev = ProjectorEval(
        model=GLM_5_2_5Layer(),
        recipe=recipe,
        dataset=dataset,
        training_run_id="some-run-id",
    )
    assert ev._eval_path().endswith(".jsonl")
    assert ev._checkpoint_path() == (
        "/checkpoints/some-run-id/projector/projector_latest.pt"
    )
