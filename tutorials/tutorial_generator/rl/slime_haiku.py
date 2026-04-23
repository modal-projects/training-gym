"""Tutorial source for `slime_haiku` — parsed by generate_tutorial.py.

Qwen3-4B GRPO on haiku poems. Demonstrates a custom async reward model
that combines deterministic structure scoring (5-7-5 syllable count) with
an optional LLM-as-judge style score. The judge is built on
`modal_training_gym.common.llm_judge.LlmJudge`; the reference reward
implementation lives in `modal_training_gym.common.haiku_reward`. In the
notebook output, the tutorial writes an editable local `haiku.py` so users
can modify the reward inline; in the plain `.py` output, the tutorial falls
back to the packaged module.
"""

TUTORIAL_METADATA = {
    'framework': '`slime`',
    'cluster_shape': '1 × 8×H100',
    'summary': 'Qwen3-4B GRPO on haiku poems — structure score + LLM judge',
    'difficulty': 'Intermediate',
    'order': 15,
    'api_classes': [
        'SlimeConfig', 'ModalConfig', 'DatasetConfig',
        'Qwen3_4B', 'WandbConfig', 'LlmJudge',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-4B GRPO on Haiku with SLIME on Modal

    This tutorial teaches [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
    to write 5-7-5 haiku poems about Modal-flavored topics. The training
    algorithm is **GRPO** (Group Relative Policy Optimization, the method
    popularized by DeepSeek-R1): for each prompt, the model generates a
    group of candidate responses, each response is scored by a reward
    model, and the policy is updated to prefer responses that scored
    above the group's mean.

    The interesting piece is the reward model, which has two halves:

    1. **Structure** — deterministic, cheap, and always on. A 5-7-5
       syllable check via CMUdict. No network, no GPU, runs inside the
       training loop.
    2. **Style** — an LLM-as-judge score. A separately-deployed vLLM
       server rates the poem on relevance, poetic quality, and Modal
       vocabulary usage. Optional: activated only when `LLM_JUDGE_URL`
       is set in the training environment, so the tutorial is runnable
       end-to-end without standing up a second service.

    Both pieces are wired into SLIME through its `custom_rm_path` hook.
    In notebooks, the tutorial writes a local `haiku.py` file that SLIME
    imports; from the CLI, the same config falls back to the packaged
    `modal_training_gym.common.haiku_reward` module.

    The workflow has four stages:

    | Stage | What it does | Where |
    |---|---|---|
    | `download_model` | Pulls `Qwen/Qwen3-4B` into the HF cache volume | 1×H100 |
    | `prepare_dataset` | Downloads `statworx/haiku` and writes train/test parquet | CPU |
    | `train` | GRPO training loop over the haiku prompts | 1×8×H100, colocated |
    | `serve_app` (separate) | Hosts the finished checkpoint via vLLM + Flash | 1×H100 |

    Invoke any function on the returned training `app` via `modal run`
    from the CLI, or interactively with `app.run()` + `.remote()` from a
    notebook.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@markdown
def _explain_imports():
    """
    ## Imports

    Three import groups:

    - **`modal`** — needed only for the interactive cells below
      (`modal.enable_output()`, `app.run()`). The training app itself is
      built by the launcher, so the tutorial body never touches the
      Modal SDK directly.
    - **`haiku` / `modal_training_gym.common.haiku_reward`** — the custom
      reward module. In notebooks, the next cell writes a local
      `haiku.py` file so you can edit the rubric inline and ship it to
      the training image with `local_python_sources=["haiku"]`. In the
      plain `.py` tutorial, the same code falls back to the packaged
      reference module at `modal_training_gym/common/haiku_reward.py`.
    - **`modal_training_gym.*`** — shared containers (`DatasetConfig`,
      `Qwen3_4B`, `WandbConfig`), the vLLM serving helper, and SLIME's
      framework-specific config + launcher classes. `DATA_PATH` is the
      canonical mount point (`/data`) for the shared data volume on
      every SLIME container.
    """


@notebook_only
@code
def _add_tutorial_dir_to_path():
    import sys, os
    # `modal run` adds the script's directory to sys.path automatically;
    # notebooks don't, so the local `haiku.py` we write below wouldn't
    # be importable without this.
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())


@notebook_only
@markdown
def _write_reward_module_intro():
    """
    In the notebook version, make the reward module explicit by writing
    a real `haiku.py` file in the current working directory. Edit this
    cell if you want to try a different scoring rubric or add your own
    custom reward logic.
    """


@notebook_only
@shell('''%%writefile haiku.py
"""Notebook-local haiku reward module for the `slime_haiku` tutorial."""

from __future__ import annotations

import os
import re
from typing import Any

from modal_training_gym.common.llm_judge import LlmJudge

_CMUDICT: dict = {}


def _get_cmudict() -> dict:
    """Lazy-load NLTK's CMUdict; cached on first call per process."""
    import nltk
    from nltk.corpus import cmudict as nltk_cmudict

    if not _CMUDICT:
        nltk.download("cmudict", quiet=True)
        _CMUDICT.update(nltk_cmudict.dict())
    return _CMUDICT


def _count_syllables_for_word(word: str, cmudict: dict) -> int:
    original = word
    word = word.lower().strip()
    phones = cmudict.get(word)
    if phones:
        return len([p for p in phones[0] if p[-1].isdigit()])
    if original.isupper() and 2 <= len(original) <= 6 and original.isalpha():
        return sum(3 if c == "w" else 1 for c in original.lower())
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _diff_syllables(text: str, target: int, cmudict: dict) -> int:
    words = re.findall(r"[a-zA-Z]+", text)
    total = sum(_count_syllables_for_word(w, cmudict) for w in words)
    return abs(total - target)


def _segment_haiku_lines(response: str) -> list[str]:
    if "/" in response:
        lines = [line.strip() for line in response.split("/")]
    elif ". " in response:
        lines = [line.strip() for line in response.split(". ")]
    else:
        lines = [line.strip() for line in response.split("\\n")]
    return [line for line in lines if line]


def score_haiku_structure(response: str, cmudict: dict) -> float:
    """Score in [0, 1]: 1/4 for 3 lines, 1/4 per line with correct syllables."""
    lines = _segment_haiku_lines(response)
    score = 0.25 if len(lines) == 3 else 0.0
    for i, target in enumerate([5, 7, 5]):
        if i < len(lines):
            diff = _diff_syllables(lines[i], target, cmudict)
            if diff == 0:
                score += 0.25
            elif diff == 1:
                score += 0.125
    return score


class HaikuStyleJudge(LlmJudge):
    """`LlmJudge` with a haiku rubric: relevance + poetic quality + vocab."""

    def build_prompt(self, prompt: str, response: str, **kwargs: Any) -> str:
        return (
            "You are evaluating a haiku poem.\\n\\n"
            f"Topic: {prompt}\\n"
            f"Response:\\n{response}\\n\\n"
            "Score 0-10 on relevance to the topic and poetic quality. "
            "single number (0-10), nothing else."
        )


def _extract_topic(sample) -> str:
    """Recover the haiku topic ("cat", "modal", ...) from a SLIME sample."""
    for attr in ("question", "prompt"):
        val = getattr(sample, attr, None)
        if isinstance(val, str) and val:
            return val
    data = getattr(sample, "data", None) or {}
    for key in ("question", "prompt"):
        val = data.get(key) if hasattr(data, "get") else None
        if isinstance(val, str) and val:
            return val
    return ""


async def haiku_rm(args, sample, **kwargs) -> float:
    """Async SLIME reward: structure plus optional LLM style score."""
    import aiohttp

    cmudict = _get_cmudict()
    structure = score_haiku_structure(sample.response, cmudict)

    judge_url = os.environ.get("LLM_JUDGE_URL", "")
    if not judge_url or structure < 1.0:
        return structure

    judge = HaikuStyleJudge(
        model_name=os.environ.get("LLM_JUDGE_MODEL", "qwen3-4b"),
        base_url=judge_url,
        max_score=10,
    )
    async with aiohttp.ClientSession() as session:
        style = await judge.score(session, _extract_topic(sample), sample.response)
    return structure + style
''')
def _write_reward_module():
    pass


@code
def _imports():
    import modal

    try:
        import haiku as _local_haiku
    except ImportError:
        _local_haiku = None

    if _local_haiku is not None and hasattr(_local_haiku, "haiku_rm"):
        haiku = _local_haiku
        CUSTOM_RM_PATH = "haiku.haiku_rm"
        LOCAL_PYTHON_SOURCES = ["haiku"]
    else:
        from modal_training_gym.common import haiku_reward as haiku

        CUSTOM_RM_PATH = "modal_training_gym.common.haiku_reward.haiku_rm"
        LOCAL_PYTHON_SOURCES = []

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.common.serve_vllm import build_vllm_serve_app
    from modal_training_gym.frameworks.slime import ModalConfig, SlimeConfig
    from modal_training_gym.frameworks.slime.config import DATA_PATH


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    SLIME reads training data from parquet files on a shared volume. A
    `DatasetConfig` subclass specifies two things:

    1. **Where the files live** (`prompt_data`, `eval_prompt_data`) —
       paths under `DATA_PATH` (`/data` inside every SLIME container).
    2. **How to produce them** (`prepare()`) — runs once inside the
       `prepare_dataset` Modal function with the data volume mounted
       read-write, and should write the parquet files atomically.

    The class attrs also map to SLIME CLI flags:

    - `input_key="messages"` — column name SLIME reads the chat
      conversation from.
    - `label_key="label"` — column holding the ground-truth poem (used
      only by the optional LLM judge to grade relative quality; the
      structure-only reward ignores it).
    - `apply_chat_template=True` — ask SLIME to run the tokenizer's
      chat template over `messages` before rollout, matching what the
      model was pretrained with.
    - `rm_type="async_rm"` — the reward model is an async Python
      function rather than a built-in (`math`, `deepscaler`, …) or a
      remote HTTP service.

    Inside `prepare()` we:

    1. Download
       [`statworx/haiku`](https://huggingface.co/datasets/statworx/haiku),
       a dataset of ~7k haiku poems tagged with a `keywords` topic.
    2. Turn each row into a chat conversation: a short system prompt
       that asks for a haiku, plus a user prompt like
       `"Write me a haiku about cat."`.
    3. Also precompute the tokenized `prompt` string via the model's
       chat template — SLIME uses that as the exact rollout input.
    4. Hold out the last 20% (capped at 1000 rows) as a test split so
       `eval_prompt_data` has something to score against during
       training.
    """


@code
def _define_dataset():
    class HaikuDataset(DatasetConfig):
        input_key = "messages"
        label_key = "label"
        apply_chat_template = True
        rollout_shuffle = True
        rm_type = "async_rm"

        def __init__(self, data_path, hf_checkpoint):
            # `data_path` is DATA_PATH on whichever container is mounting
            # the shared volume — /data on SLIME containers. Stashing it
            # plus the HF model id lets prepare() run standalone later.
            self._data_path = str(data_path)
            self._hf_checkpoint = hf_checkpoint
            self.prompt_data = f"{self._data_path}/haiku/train.parquet"
            self.eval_prompt_data = ["haiku", f"{self._data_path}/haiku/test.parquet"]

        def prepare(self):
            import os

            from datasets import load_dataset
            from transformers import AutoTokenizer

            # Use the base-model tokenizer to render the chat template;
            # SLIME will re-tokenize at rollout, but pre-rendering the
            # prompt string here lets us log exactly what the model sees.
            tokenizer = AutoTokenizer.from_pretrained(self._hf_checkpoint)
            ds = load_dataset("statworx/haiku")

            system_prompt = (
                "You are a haiku poet. You will be given a prompt and you "
                "will need to write a haiku about the prompt."
            )

            def to_chat(example):
                # statworx/haiku rows look like
                # {"keywords": "Cat", "text": "<reference haiku>"}.
                # We turn `keywords` into a user question and keep the
                # reference poem under `label` for the judge to consume.
                keyword = example["keywords"].lower()
                question = f"Write me a haiku about {keyword}."
                messages = [
                    {"content": system_prompt, "role": "system"},
                    {"content": question, "role": "user"},
                ]
                return {
                    "question": question,
                    "label": example["text"],
                    "messages": messages,
                    "prompt": tokenizer.apply_chat_template(
                        messages, tokenize=False, enable_thinking=False
                    ),
                }

            # Hold out the last 20% (≤ 1000 rows) as the test split so
            # `eval_prompt_data` has fresh prompts the policy never sees
            # during rollout.
            train_ds = ds["train"]
            test_size = min(1000, int(len(train_ds) * 0.2))
            test_ds = train_ds.select(range(len(train_ds) - test_size, len(train_ds)))
            train_ds = train_ds.select(range(len(train_ds) - test_size))

            os.makedirs(f"{self._data_path}/haiku", exist_ok=True)
            train_ds.map(to_chat, remove_columns=["keywords"]).to_parquet(
                f"{self._data_path}/haiku/train.parquet"
            )
            test_ds.map(to_chat, remove_columns=["keywords"]).to_parquet(
                f"{self._data_path}/haiku/test.parquet"
            )


@markdown
def _explain_reward():
    """
    ## Reward model

    GRPO needs a scalar reward for every rollout. The notebook writes a
    local `haiku.py` module; the CLI version uses the packaged
    `modal_training_gym.common.haiku_reward` module from
    `modal_training_gym/common/haiku_reward.py`. Both expose the same
    `haiku_rm`, an **async** reward function matching the signature
    SLIME expects:

    ```python
    async def haiku_rm(args, sample, **kwargs) -> float:
        ...
    ```

    - `args` is the SLIME CLI namespace (rarely needed).
    - `sample` exposes `.response` (the model's generation), `.prompt`
      / `.question` (the input), and `.label` (the reference answer,
      when available).
    - Return value is the reward. Higher is better; GRPO subtracts the
      group mean so absolute magnitudes are less important than
      consistency across samples.

    `haiku_rm` composes two scoring components:

    - **`score_haiku_structure(response, cmudict)`** — pure-Python check
      that returns a value in `[0, 1]`: a quarter-point for having
      three lines, plus up to a quarter per line for hitting the 5-7-5
      syllable target (half-credit for off-by-one). Uses NLTK's CMUdict
      for pronunciation; falls back to a vowel-group heuristic for
      out-of-dict words.
    - **`HaikuStyleJudge(LlmJudge)`** — subclass of the shared
      `modal_training_gym.common.llm_judge.LlmJudge`. The base class
      handles the boring part (POST to an OpenAI-compatible
      chat-completions endpoint, regex-parse a number out of the reply,
      normalize to `[0, 1]`). The subclass overrides `build_prompt()`
      with the haiku-specific rubric: relevance + poetic quality + Modal
      vocabulary usage.

    The two components are combined with **curriculum-style gating** —
    the style score is evaluated only when the structure score is
    perfect (`1.0`). Early in training the model has to learn 5-7-5
    first; once it does, the style score kicks in and rewards
    thematically interesting poems. Total reward is therefore
    `structure + style ∈ [0, 2]`.

    Gating also makes the style path **opt-in**: it only fires when
    `LLM_JUDGE_URL` is set in the training environment. Without a judge
    deployed, the reward degrades gracefully to structure-only and the
    run still progresses — convenient for smoke-testing this tutorial.
    """


@markdown
def _explain_config():
    """
    ## Define the experiment

    `SlimeConfig(...)` wires the model, dataset, reward, and cluster
    shape into a single object. Fields that aren't set here fall back to
    SLIME's defaults (optimizer, Megatron parallelism, GRPO clipping,
    etc.) — see the
    [`slime_gsm8k`](../../rl/slime_gsm8k/slime_gsm8k.ipynb) tutorial
    for a fully explicit example.

    Key choices for this run:

    - `rm_type="async_rm"` + `custom_rm_path=CUSTOM_RM_PATH` — when the
      notebook-created `haiku.py` is present, SLIME imports
      `haiku.haiku_rm`; otherwise it falls back to
      `modal_training_gym.common.haiku_reward.haiku_rm`.
    - `local_python_sources=LOCAL_PYTHON_SOURCES` on `ModalConfig` —
      ships the notebook-authored `haiku.py` into the training image
      when present. The packaged fallback already lives under
      `modal_training_gym`, so it doesn't need an extra local source.
    - `apply_chat_template_kwargs='{"enable_thinking": false}'` —
      Qwen3's chat template has a `enable_thinking` toggle; we disable
      it so rollouts produce poems, not chain-of-thought traces.
    - `save_interval=10`, `eval_interval=20` — checkpoint every 10
      steps, evaluate every 20.
    """


@code
def _define_config():
    base_model = Qwen3_4B()
    my_training_run = SlimeConfig(
        model=base_model,
        dataset=HaikuDataset(DATA_PATH, base_model.model_name),
        wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-haiku"),
        ref_load=base_model.model_name,
        megatron_to_hf_mode="bridge",

        rm_type="async_rm",
        custom_rm_path=CUSTOM_RM_PATH,

        num_rollout=50,
        apply_chat_template_kwargs='{"enable_thinking": false}',

        eval_interval=20,
        n_samples_per_eval_prompt=8,

        save="/checkpoints/qwen3-4b-haiku",
        save_interval=10,

        modal=ModalConfig(
            gpu="H100",
            local_python_sources=LOCAL_PYTHON_SOURCES,
            image_run_commands=[
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ],
        ),
    )


@markdown
def _build_section():
    """
    ## Build the Modal app

    `build_app()` returns a `modal.App` with four functions defined
    against the right volumes, secrets, and GPU spec:

    - `download_model` — pulls the HF checkpoint into the
      `huggingface-cache` volume (1×H100, 2 hour timeout).
    - `prepare_dataset` — runs `HaikuDataset.prepare()` against the
      `slime-data` volume (CPU).
    - `convert_checkpoint` — no-op in bridge mode; for `megatron_to_hf_mode
      ="raw"` experiments it'd do the HF → torch_dist conversion.
    - `train` — the actual GRPO run on a Modal `@clustered(n_nodes)`
      cluster (1×8×H100 here), with Ray brought up in-container via
      `ModalRayCluster`.

    Every function closes over the `my_training_run` config above, so the
    app is fully determined by that single object.
    """


@code
def _build_app():
    app = my_training_run.build_app()


@markdown
def _run_section():
    """
    ## Run training
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI, run the three stages in order. `download_model` and
    `prepare_dataset` are one-shots — once the HF weights and parquet
    files are on their volumes, reruns skip the download. `train` is
    long-running; `--detach` keeps it going after you close the terminal:

    ```bash
    uv run modal run tutorials/rl/slime_haiku/slime_haiku.py::app.download_model
    uv run modal run tutorials/rl/slime_haiku/slime_haiku.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/slime_haiku/slime_haiku.py::app.train
    ```

    To enable the LLM-judge style score, set `LLM_JUDGE_URL` (and
    optionally `LLM_JUDGE_MODEL`) on the train function's environment —
    either by editing the `environment` dict in `ModalConfig`, or via a
    Modal secret attached to the `train` function. Omit it entirely for
    the structure-only variant that this tutorial ships with by default.
    """


@notebook_only
@markdown
def _run_interactive():
    """
    In a notebook, `app.run()` opens an ephemeral app context and
    `.remote()` invokes a remote function inside it. `modal.enable_output()`
    streams container logs into the notebook so you can watch the run.
    Put one stage per cell to checkpoint your progress:
    """


@notebook_only
@code
def _invoke_download_model():
    with modal.enable_output():
        with app.run():
            app.download_model.remote()


@notebook_only
@code
def _invoke_prepare_dataset():
    with modal.enable_output():
        with app.run():
            app.prepare_dataset.remote()


@notebook_only
@code
def _invoke_train():
    with modal.enable_output():
        with app.run():
            app.train.remote()


@markdown
def _serve_section():
    """
    ## Serve the trained model

    `build_vllm_serve_app(...)` is the post-training counterpart to
    `build_app()`: a thin wrapper around the canonical
    [Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference)
    that returns a **second** Modal app, fully independent of the
    training app. It registers one `@modal.web_server`-decorated
    function that mounts the checkpoints volume, runs
    `vllm serve <model_path>`, and exposes the standard OpenAI
    chat-completions API at
    `https://<workspace>--<app_name>-serve.modal.run`.

    Because it's a separate app, you `modal deploy` it (not
    `modal run`) — that gives you a long-lived URL you can point a
    playground UI or eval harness at. Modal scales the replica to zero
    after `scaledown_window` of idleness.

    ### Pointing at the right checkpoint

    `model_path` must point at a directory containing an **HF-format**
    checkpoint (`config.json` plus weight shards). SLIME bridge mode
    writes Megatron torch_dist checkpoints under `save` — pick out an
    HF-compatible subdirectory once training completes, or run a
    torch_dist → HF conversion if your config didn't emit one. Inspect
    what the training left behind with:

    ```bash
    modal volume ls slime-checkpoints qwen3-4b-haiku/
    ```

    and point `model_path` at whichever directory has a `config.json`.
    The reference
    [qwen3-haiku `convert_torch_dist_to_hf.py`](https://github.com/modal-labs/qwen3-haiku/blob/joy/initial-setup/tools/convert_torch_dist_to_hf.py)
    is a worked example of the conversion flow if you need it.
    """


@code
def _build_serve_app():
    serve_app = build_vllm_serve_app(
        # Modal app name — determines the deployed URL shape:
        # https://<workspace>--serve-haiku-model-serve.modal.run
        app_name="serve-haiku-model",
        # Absolute path inside the container to an HF-format checkpoint
        # directory (must contain config.json). Update after inspecting
        # the slime-checkpoints volume — the exact subdirectory depends
        # on slime's bridge-mode output layout. (To smoke-test the
        # serving pipeline before training, swap this for the HF repo
        # id "Qwen/Qwen3-4B" and drop the `checkpoints_volume` arg
        # below; vLLM will download the base model itself.)
        model_path="/checkpoints/qwen3-4b-haiku",
        # The `model` field callers pass in chat-completions requests.
        served_model_name="qwen3-4b-haiku",
        # Mount the training checkpoints volume; without this,
        # /checkpoints wouldn't exist in the serving container.
        checkpoints_volume="slime-checkpoints",
        gpu="H100",
        n_gpu=1,
        # Qwen3's reasoning-parser separates <think>...</think> traces
        # from the final response on the server side. --enforce-eager
        # skips CUDA-graph capture (faster cold starts, slightly slower
        # inference — ideal for scale-from-zero endpoints).
        extra_vllm_args=["--enforce-eager", "--reasoning-parser", "qwen3"],
    )


@py_only
@markdown
def _serve_cli():
    """
    Deploy the serving app separately from the training app:

    ```bash
    uv run modal deploy tutorials/rl/slime_haiku/slime_haiku.py::serve_app
    ```

    The URL is deterministic once deployed:
    `https://<workspace>--serve-haiku-model-serve.modal.run`. Query it
    with any OpenAI-compatible client — the endpoint speaks the
    standard chat-completions API:

    ```bash
    curl https://<workspace>--serve-haiku-model-serve.modal.run/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
            "model": "qwen3-4b-haiku",
            "messages": [{"role": "user", "content": "Write me a haiku about cat."}],
            "temperature": 0.0
          }'
    ```

    To swap checkpoints, edit `model_path` above and redeploy — Modal
    will spin up a fresh container pointed at the new weights.
    """


@notebook_only
@markdown
def _serve_interactive():
    """
    From a notebook, `serve_app.deploy()` does exactly what
    `modal deploy` does — creates a long-lived deployment rather than an
    ephemeral run. `modal.enable_output()` streams the deploy logs so
    you can see when vLLM finishes loading the weights.
    """


@notebook_only
@code
def _invoke_deploy_serve():
    with modal.enable_output():
        serve_app.deploy()
