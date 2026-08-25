from modal_training_gym import (
    HuggingFaceDataset,
    Qwen3_4B,
    Qwen3_4b_Recipe,
    TrainConfig,
)


class MathDataset(HuggingFaceDataset):
    hf_repo = "zhuzilin/dapo-math-17k"
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True


def main() -> None:
    config = TrainConfig(
        model=Qwen3_4B(),
        dataset=MathDataset(n_rows=120),
        recipe=Qwen3_4b_Recipe(
            gpu_type="H100",
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=1,
            colocate=True,
            num_rollout=1,
            n_samples_per_prompt=4,
            rollout_batch_size=8,
            rollout_max_response_len=2048,
            max_tokens_per_gpu=4096,
            sglang_mem_fraction_static=0.6,
            rm_type="deepscaler",
        ),
    )
    run = config.launch()
    print(run.training_run_id)


if __name__ == "__main__":
    main()
