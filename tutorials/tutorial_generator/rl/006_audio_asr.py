"""Tutorial source for `006_audio_asr` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 2×H100",
    "summary": "Automatic Speech Recognition (Audio)",
    "difficulty": "Intermediate",
    "order": 39,
    "api_classes": [
        "CustomDeployment",
        "MultimodalDataset",
        "Qwen3_ASR_1_7B",
        "Qwen3_ASR_1_7b_Recipe",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Transcribing speech at a fraction of frontier costs

    We've shown before that when it comes to speech transcription,
    open models are [100x faster and 100x cheaper](https://modal.com/blog/fast-cheap-batch-transcription)
    than proprietary APIs, and open models still occupy
    [the top spots](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
    But there's no reason to stop there: we can achieve state-of-the-art
    performance by post-training open models to get even lower word error rates (WER).
    As an example, we show how to post-train
    [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on the 
    [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
    dataset.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run --with soundfile --with librosa --with jiwer --with datasets \\
        python tutorials/rl/006_audio_asr/006_audio_asr.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main\n"
    "if importlib.util.find_spec('librosa') is None:\n"
    "    %uv pip install -q soundfile librosa jiwer datasets"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        CustomDeployment,
        MultimodalDataset,
        Qwen3_ASR_1_7B,
        Qwen3_ASR_1_7b_Recipe,
        TrainConfig,
        list_checkpoints,
    )

@markdown
def _deploy_base_intro():
    """
    ## Deploy the base model

    Since audio models are not yet supported on
    [Endpoints](https://modal.com/docs/guide/endpoints), we use a
    [CustomDeployment](https://gym.modal.dev/reference/deployment/customdeployment/)
    to deploy the base and trained models.
    """


@code
def _deploy_base():
    model = Qwen3_ASR_1_7B()
    base_deployment = CustomDeployment.launch(
        model,
        unauthenticated=True,
        recreate_if_existing=True
    )
    base_deployment.wait_until_ready(timeout=15 * 60)
    print(f"base model deployed to {base_deployment.url}")


@markdown
def _score_fn_intro():
    """
    ## Define a scoring function

    As mentioned before, we measure capability by lower WER, so that's what we'll use.
    We can use the `jiwer` library to calculate this so we don't have to ourselves.
    """


@code
def _score_fn():
    async def score_transcript(response: str, label: str) -> float:
        import jiwer

        if not label:
            return 0.0
        return -float(jiwer.wer(label, response))



@markdown
def _dataset_intro():
    """
    ## Get the dataset

    Since this dataset contains audio files, we create a `MultimodalDataset`
    to pass the audio clips to rollouts. We do some pre-processing with
    `soundfile` and store as base64 inline for demonstration purposes.
    In a production use case, you'd likely instead store by reference.
    """


@code
def _dataset():
    class LibriSpeechASRDataset(MultimodalDataset):
        modality = "audio"
        hf_repo = "hf-internal-testing/librispeech_asr_dummy"
        hf_config = "clean"
        hf_split = "validation"
        always_prepare = True
        apply_chat_template = False  # ensures the data URI is valid throughout the rollout

        def __init__(self, **kwargs):
            super().__init__(rows=[], **kwargs)

        def _build_rows(self) -> list[dict]:
            import base64 as b64
            import io

            import soundfile as sf
            from datasets import Audio, load_dataset

            ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
            ds = ds.cast_column("audio", Audio(decode=False))  # decode with soundfile instead of torchcodec
            rows = []
            for ex in ds:
                audio = ex["audio"]
                data = (
                    audio["bytes"]
                    if audio.get("bytes")
                    else open(audio["path"], "rb").read()
                )
                arr, sr = sf.read(io.BytesIO(data))
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format="WAV")
                data_uri = "data:audio/wav;base64," + b64.b64encode(
                    buf.getvalue()
                ).decode("ascii")
                rows.append(
                    {
                        self.input_key: "<audio>\nTranscribe the speech to text. Respond with only the transcript.",
                        self.media_column: [data_uri],
                        self.label_key: ex["text"].lower().strip(),
                    }
                )
            return rows

        def load(self, split: str = "all") -> list[dict]:
            return self._build_rows()

        def prepare(self, path, eval_paths=None):
            rows = self._build_rows()
            self._write_jsonl(rows, path)
            if eval_paths:
                for eval_path in eval_paths.values():
                    self._write_jsonl(rows, eval_path)

    train_dataset = LibriSpeechASRDataset(hf_split="validation[:8]")
    eval_dataset = LibriSpeechASRDataset(hf_split="validation[8:16]")


@notebook_only
@code
def _dataset_peek():
    df = eval_dataset.to_pandas()
    print(f"{len(df)} rows")
    df.head(5)


@markdown
def _eval_base_intro():
    """
    ## Evaluate the base model

    Let's get our baseline measure of performance.
    """


@code
def _eval_base():
    def run_eval(deployment, max_concurrency: int = 2) -> float:
        from concurrent.futures import ThreadPoolExecutor
        import base64
        import io

        import requests
        import soundfile as sf


        deployment.wait_until_ready(timeout=15 * 60)

        def _score_one(example):
            data_uri = example["audios"][0]
            reference = (example["label"] or "").lower().strip()
            b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
            arr, sr = sf.read(io.BytesIO(base64.b64decode(b64)))

            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            buf.seek(0)
            resp = requests.post(
                f"{deployment.url}/v1/audio/transcriptions",
                files={"file": ("clip.wav", buf, "audio/wav")},
                data={
                    "model": deployment.served_model_name,
                    "temperature": "0.0",
                },
                timeout=120,
            )
            resp.raise_for_status()
            hypothesis = (resp.json().get("text") or "").lower().strip()
            wer = score_transcript(hypothesis, reference)

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            wers = list(executor.map(_score_one, eval_dataset.load()))
        return sum(wers) / len(wers) if wers else float("nan")

    print("running base model evaluation...")
    base_mean = run_eval(base_deployment)
    print(f"average WER: {base_mean:.1%}")


@markdown
def _rm_fn_intro():
    """
    ## Creating a reward function

    To make our scoring function a reward function, we must return the 
    negative WER so that lower WER leads to higher rewards.
    """


@code
def _rm_fn():
    async def wer_rm(args, sample, **kwargs) -> float:
        wer = score_transcript(sample.response, sample.label)
        return -wer


@markdown
def _train_intro():
    """
    ## Begin training

    There are many ASR-specific changes to the default framework recipes such as
    the transcription rollout, padded (bshd) batches, and the many-samples/high-temperature
    settings that surface reward variance. To not pass the burden of specifying onto you,
    we created `Qwen3_ASR_1_7b_Recipe` so that you can focus on training.
    """


@code
def _train():
    train_run = TrainConfig(
        model=Qwen3_ASR_1_7B(),
        dataset=train_dataset,
        recipe=Qwen3_ASR_1_7b_Recipe(
            gpu_type="H100",
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            tensor_model_parallel_size=1,
            sequence_parallel=False,       
            rollout_num_gpus=2,
            rollout_num_gpus_per_engine=1,
            custom_rm_function=wer_rm,
        ),
    )
    train_result = train_run.train()
    print(f"run id: {train_result.training_run_id}")


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained checkpoint

    Let's run the same eval on the trained checkpoint.
    """

@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"checkpoint: {checkpoint.path}")

    trained_deployment = CustomDeployment.launch(
        model,
        checkpoint,
        unauthenticated=True,
        recreate_if_existing=True
    )
    trained_deployment.wait_until_ready(timeout=15 * 60)
    print(f"checkpoint deployed to {trained_deployment.url}")

    print("running checkpoint evaluation...")
    trained_mean = run_eval(trained_deployment)
    print(f"average WER: {trained_mean:.1f}")
