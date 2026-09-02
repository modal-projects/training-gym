# ---
# order: 6
# deps: jiwer, requests, soundfile
# ---
#
# # Speech transcription that's better, faster, and cheaper
#
# We've shown before that when it comes to speech transcription,
# open models are [100x faster and 100x cheaper](https://modal.com/blog/fast-cheap-batch-transcription)
# than proprietary APIs, and open models still occupy
# [the top spots](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
# in terms of WER. But there's no reason to stop there: we can achieve state-of-the-art
# performance by post-training open models to redefine your task's Pareto frontier.
# As an example, we show how to post-train
# [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on the 
# [hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
# dataset.

import jiwer
import requests
import soundfile as sf
from datasets import Audio, load_dataset

import base64
import io
from concurrent.futures import ThreadPoolExecutor

from modal_training_gym import (
    CustomDeployment,
    MultimodalDataset,
    Qwen3_ASR_1_7B,
    Qwen3_ASR_1_7B_Recipe,
    TrainConfig,
)

# ## Deploy the base model
#
# Since audio models are not yet supported on
# [Endpoints](https://modal.com/docs/guide/endpoints), we use a
# [CustomDeployment](https://gym.modal.dev/reference/customdeployment)
# to deploy the base and trained models.

model = Qwen3_ASR_1_7B()
base_deployment = CustomDeployment.launch(
    model,
    unauthenticated=True,
)
base_deployment.wait_until_ready(timeout=15 * 60)
print(f"base model deployed to {base_deployment.url}")

# ## Define a scoring function
#
# As mentioned before, we measure capability by lower WER, so that's what we'll use.
# We can use the `jiwer` library to calculate this so we don't have to ourselves.

def score_transcript(response: str, label: str) -> float:
    response = (response or "").lower().strip()
    label = (label or "").lower().strip()
    if not label:
        return 0.0
    return float(jiwer.wer(label, response))

# ## Get the dataset
#
# Since this dataset contains audio files, we create a `MultimodalDataset`
# to pass the audio clips to rollouts. We do some pre-processing with
# `soundfile` and store as base64 inline for demonstration purposes.
# In a production use case, you'd likely instead store references and
# resolve them in a custom `generate` function.

class LibriSpeechASRDataset(MultimodalDataset):
    modality = "audio"
    hf_repo = "hf-internal-testing/librispeech_asr_dummy"
    hf_config = "clean"
    hf_split = "validation"
    always_prepare = True
    apply_chat_template = False  # ensures the data URI is valid throughout the rollout

    def load(self) -> list[dict]:
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
            data_uri = "data:audio/wav;base64," + base64.b64encode(
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

train_dataset = LibriSpeechASRDataset(hf_split="validation[:8]")

eval_dataset = LibriSpeechASRDataset(hf_split="validation[8:16]")

# ## Evaluate the base model
#
# Let's get our baseline measure of performance.

def run_eval(deployment, max_concurrency: int = 2) -> float:
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
        return score_transcript(hypothesis, reference)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        wers = list(executor.map(_score_one, eval_dataset.load()))
    return sum(wers) / len(wers) if wers else float("nan")

print("running base model evaluation...")
base_mean = run_eval(base_deployment)
print(f"average WER: {base_mean:.1%}")

# ## Creating a reward function
#
# To make our scoring function a reward function, we must return the
# negative WER so that lower WER leads to higher rewards.

async def wer_rm(args, sample, **kwargs) -> float:
    return -score_transcript(sample.response, sample.label)

# ## Begin training
#
# There are many ASR-specific changes to the default framework recipes such as
# the transcription rollout, padded (bshd) batches, and the many-samples/high-temperature
# settings that surface reward variance. To not pass the burden of specifying onto you,
# we created `Qwen3_ASR_1_7B_Recipe` so that you can focus on training.

config = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_ASR_1_7B_Recipe(
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
run = config.launch()
print(f"run id: {run.training_run_id}")

# ## Evaluate the trained checkpoint
#
# Let's run the same eval on the trained checkpoint.

result = run.result()
checkpoint = result.checkpoints()[-1]
print(f"checkpoint: {checkpoint.path}")

trained_deployment = CustomDeployment.launch(
    model,
    checkpoint,
    unauthenticated=True,
)
trained_deployment.wait_until_ready(timeout=15 * 60)
print(f"checkpoint deployed to {trained_deployment.url}")

print("running checkpoint evaluation...")
trained_mean = run_eval(trained_deployment)
print(f"average WER: {trained_mean:.1%}")
