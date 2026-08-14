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
    result = TrainConfig(
        model=Qwen3_4B(),
        dataset=MathDataset(n_rows=120),
        recipe=Qwen3_4b_Recipe(
            rm_type="deepscaler",
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=1,
            n_samples_per_prompt=4,
            rollout_batch_size=8,
            rollout_max_response_len=2048,
            max_tokens_per_gpu=4096,
            sglang_mem_fraction_static=0.6,
        ),
    ).train()
    print(result.training_run_id)


if __name__ == "__main__":
    main()
