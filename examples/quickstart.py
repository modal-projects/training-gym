from modal_training_gym import (
    HuggingFaceDataset,
    Qwen3_4B,
    Qwen3_4B_Recipe,
    TrainConfig,
)


def main() -> None:
    config = TrainConfig(
        model=Qwen3_4B(),
        dataset=HuggingFaceDataset(
            "zhuzilin/dapo-math-17k",
            hf_split="train[:120]",
            input_column="prompt",
            output_column="label",
            input_format="messages",
        ),
        recipe=Qwen3_4B_Recipe(
            rm_type="deepscaler",
        ),
    )
    run = config.launch()
    print(run.training_run_id)


if __name__ == "__main__":
    main()
