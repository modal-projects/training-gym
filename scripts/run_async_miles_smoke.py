"""Minimal async MILES smoke run on Qwen3-0.6B.

Mirrors the slime validator (``validate_model_configs.run_base_training_on_slime``):
same gsm8k dataset and ``rm_type="deepscaler"``, but on ``MilesRecipe`` with
``async_mode=True`` so rollout generation overlaps training. The goal is to
prove the async rollout path works end-to-end, not to train a useful model.

    uv run scripts/run_async_miles_smoke.py
"""

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import Qwen3_0_6B
from modal_training_gym.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


class Gsm8kDataset(HuggingFaceDataset):
    """Same gsm8k prompts the slime validator uses (``pick_dataset``).

    Defined here rather than imported so cloudpickle inlines it from
    ``__main__`` into the containers.
    """

    hf_repo = "openai/gsm8k"
    hf_config = "main"
    input_column = "question"
    output_column = "answer"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True

    def load(self, split: str = "all"):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds.map(lambda r: {"answer": r["answer"].split("####")[-1].strip()})


def build_train_config() -> TrainConfig:
    model = Qwen3_0_6B()
    recipe = MilesRecipe(
        gpu_type="H100",
        # Miles' train_async.py asserts `not colocate`, and Modal clustered
        # functions take whole nodes, so the smallest async shape is one node
        # of trainer GPUs plus one node of rollout GPUs.
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        async_mode=True,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=32,
        rollout_max_response_len=512,
        max_tokens_per_gpu=4096,
        rm_type="deepscaler",
        eval_interval=None,
        skip_eval_before_train=True,
        save_interval=100,
    )
    return TrainConfig(model=model, dataset=Gsm8kDataset(n_rows=10), recipe=recipe)


def main() -> None:
    result = build_train_config().train()
    print(f"Training run id: {result.training_run_id}")


if __name__ == "__main__":
    main()
