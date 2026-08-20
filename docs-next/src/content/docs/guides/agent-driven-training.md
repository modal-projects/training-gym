---
order: 0
title: Agent-driven training
description: Install the Training Gym skills and let a coding agent configure, prove, monitor, and promote a training run with the CLI.
---

# Agent-driven training

The Training Gym is more than an API for writing training configurations. The
package ships a bundle of agent skills that teach a coding agent the whole RL
lifecycle — configure, prove, smoke test, diagnose, promote — and a
`training-gym` CLI that supplies the evidence each of those decisions needs:
run status, reward history, logs, and rollout traces.

This guide walks that loop end to end using one real objective:
**post-training a model to answer in rhyme**. The skill provides the training
judgment; the CLI provides the evidence.

## The CLI

The `training-gym` CLI is installed with the package:

```bash
pip install git+https://github.com/modal-projects/training-gym.git@main
training-gym --help
```

Two command groups matter for this loop. `training-gym skills` installs the
agent skills into your project, and `training-gym run` inspects training runs
through your deployed dashboard. The
[CLI reference](/reference/cli/) documents every command and flag.

## Install the agent skills

Install the skill bundle into the current project:

```bash
training-gym skills install
```

The command finds the nearest Git repository and copies each bundled skill to
`.agents/skills/<skill-name>`, then links `.claude/skills/<skill-name>` to the
canonical copy for Claude compatibility. The bundle contains
`agent-driven-training`, `example-validation`, `modal-infrastructure`,
`model-support`, and `training-gym-overview`.

`agent-driven-training` is the skill that owns the lifecycle. It tells the
agent to:

- ask before making choices that change model behavior or GPU cost
- catch dataset and reward bugs locally before they waste GPU time
- scale up only after cheaper, shorter test runs show the training pipeline is
  healthy
- inspect actual model outputs to verify that reward reflects the intended
  behavior
- investigate suspicious reward trends, so metric exploits are not mistaken for
  learning

The other skills cover the neighboring work the agent may need: validating
examples, operating raw Modal infrastructure, adding support for a new model,
and navigating the Training Gym codebase itself.

## Give the agent an objective

The example we'll walk through here is with a relatively ambiguous prompt:

> can you post-train a model to rhyme in its output

The agent proposes a model, dataset, reward function, and cluster shape, then
asks you to confirm the choices that affect behavior and cost before it writes
anything. For the session described here it selected **Qwen3-4B**,
`tatsu-lab/alpaca`, and a reward that grants rhyme credit only when the
response rhymes *and* stays relevant to the question.

## Read the configuration it writes

The Training Gym gives the agent framework primitives and lifecycle guidance; what
comes back is ordinary Python that you can inspect, edit, and run. The rhyming
configuration is a single file, shown here in four parts.

### The dataset

Alpaca rows that carry an extra `input` field are dropped — the prompt template
passes only `instruction`, so those rows would ask an unanswerable question.
Reference answers are length-bounded to keep the later embedding comparison
meaningful, since a two-word label has no topic to match.

```python
from modal_training_gym import HuggingFaceDataset

SYSTEM_PROMPT = (
    "You are a poet who answers every question in rhyme. Answer the question "
    "correctly and completely, but write the entire answer as verse: at least "
    "four lines, one clause per line, with line endings that rhyme in couplets "
    "(AABB). Do not write any prose, preamble, or explanation outside the verse."
)


class RhymeInstructionDataset(HuggingFaceDataset):
    """Self-contained Alpaca instructions; the label is the reference answer."""

    hf_repo = "tatsu-lab/alpaca"
    input_column = "instruction"
    output_column = "output"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True
    system_prompt = SYSTEM_PROMPT
    prompt_template = "{input}"

    def load(self, split: str = "all"):
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        ds = ds.filter(
            lambda r: not r["input"].strip() and 60 <= len(r["output"]) <= 600
        )
        if self.n_rows:
            ds = ds.select(range(min(self.n_rows, len(ds))))
        return ds
```

### The rhyme scorer

Rhyme is scored deterministically from CMU pronunciation data: take the
phonemes from each line's last stressed vowel onward and compare line endings
under both AABB and ABAB. Two details are anti-gaming, and both were added
because the agent saw the model exploit their absence:

- a word never rhymes with itself, so repeating one end word cannot score a
  perfect scheme
- the score is multiplied by the share of distinct end words and the share of
  lines with real substance, so a stack of one-word lines earns little

```python
import re

_CMUDICT: dict = {}
_VOWELS = ("A", "E", "I", "O", "U")


def _cmudict() -> dict:
    if not _CMUDICT:
        import nltk
        from nltk.corpus import cmudict

        nltk.download("cmudict", quiet=True)
        _CMUDICT.update(cmudict.dict())
    return _CMUDICT


def _strip_thinking(text: str) -> str:
    """Drop a ``<think>`` block and any stray markdown bullets/numbering."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _lines(text: str) -> list[str]:
    return [line.strip() for line in _strip_thinking(text).split("\n") if line.strip()]


def _end_word(line: str) -> str:
    words = re.findall(r"[a-zA-Z']+", line)
    return words[-1].lower().strip("'") if words else ""


def rhyme_tail(word: str) -> tuple:
    """Phonemes from the last stressed vowel onward, stress markers removed.

    Falls back to the last three letters for words the dictionary doesn't know
    (names, coinages), which is a decent orthographic proxy.
    """
    if not word:
        return ()
    phones = _cmudict().get(word)
    if not phones:
        return ("~", word[-3:])
    seq = phones[0]
    stressed = [i for i, p in enumerate(seq) if p[-1] in ("1", "2")]
    if stressed:
        start = stressed[-1]
    else:
        vowels = [i for i, p in enumerate(seq) if p[0] in _VOWELS]
        start = vowels[-1] if vowels else 0
    return tuple(re.sub(r"\d", "", p) for p in seq[start:])


def words_rhyme(a: str, b: str) -> bool:
    """True when two *different* words share a rhyme tail."""
    if not a or not b or a == b:
        return False
    return rhyme_tail(a) == rhyme_tail(b)


def _scheme_score(end_words: list[str], offset: int) -> float:
    """Fraction of rhyming pairs: offset 1 = AABB, offset 2 = ABAB."""
    pairs = []
    for start in range(0, len(end_words) - offset, 2 * offset):
        for k in range(offset):
            i, j = start + k, start + k + offset
            if j < len(end_words):
                pairs.append((end_words[i], end_words[j]))
    if not pairs:
        return 0.0
    return sum(words_rhyme(a, b) for a, b in pairs) / len(pairs)


def score_rhyme(response: str) -> float:
    """Rhyme quality of a response in ``[0, 1]``."""
    lines = _lines(response)
    if len(lines) < 2:
        return 0.0
    end_words = [_end_word(line) for line in lines]
    if not any(end_words):
        return 0.0

    scheme = max(_scheme_score(end_words, 1), _scheme_score(end_words, 2))
    distinct = len({w for w in end_words if w}) / len(end_words)
    substantial = sum(
        len(re.findall(r"[a-zA-Z']+", line)) >= 3 for line in lines
    ) / len(lines)
    length_factor = min(1.0, len(lines) / 4)
    return scheme * distinct * substantial * length_factor
```

### The relevance gate

Rhyme alone is trivially satisfiable by ignoring the question, so a small
sentence-embedding model compares the response to the reference answer. It is
baked into the training image (see `image_overlay` below) and loaded once per
worker, with a token-overlap fallback if the load ever fails.

Relevance *gates* rhyme rather than merely adding to it: below the gate
threshold, no amount of rhyming earns credit.

```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIR = "/opt/rhyme-embedder"

_EMBEDDER: dict = {}


def _embedder():
    """Mean-pooling MiniLM loaded once per worker from the baked image dir."""
    if not _EMBEDDER:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBED_DIR)
        model = AutoModel.from_pretrained(EMBED_DIR)
        model.eval()
        _EMBEDDER["tokenizer"] = tokenizer
        _EMBEDDER["model"] = model
        _EMBEDDER["torch"] = torch
        print(f"[rhyme_rm] embedder ready: {EMBED_MODEL}")
    return _EMBEDDER


def embed(texts: list[str]) -> list:
    """L2-normalized mean-pooled sentence embeddings."""
    parts = _embedder()
    torch = parts["torch"]
    batch = parts["tokenizer"](
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        out = parts["model"](**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def _lexical_overlap(a: str, b: str) -> float:
    """Token-F1 fallback used only if the embedder fails to load."""
    ta = {w for w in re.findall(r"[a-z']+", a.lower()) if len(w) > 3}
    tb = {w for w in re.findall(r"[a-z']+", b.lower()) if len(w) > 3}
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def score_relevance(response: str, reference: str) -> float:
    """Topical agreement with the reference answer, rescaled to ``[0, 1]``."""
    response = _strip_thinking(response)
    reference = (reference or "").strip()
    if not response or not reference:
        return 0.0
    try:
        vectors = embed([response, reference])
        cosine = float((vectors[0] * vectors[1]).sum())
    except Exception as exc:  # noqa: BLE001
        print(f"[rhyme_rm] embedder unavailable ({exc}); using lexical overlap")
        cosine = _lexical_overlap(response, reference)
    return max(0.0, min(1.0, (cosine - 0.10) / 0.45))


def rhyme_reward(response: str, reference: str) -> float:
    """Gated rhyme quality plus a smaller standalone relevance term."""
    rhyme = score_rhyme(response)
    relevance = score_relevance(response, reference)
    gate = min(1.0, relevance / 0.4)
    return gate * rhyme + 0.3 * relevance


async def rhyme_rm(args, sample, **kwargs) -> float:
    from modal_training_gym import Qwen3_4B

    model = Qwen3_4B()
    response = model.parse_response(getattr(sample, "response", "") or "")
    reference = getattr(sample, "label", "") or ""
    return rhyme_reward(response.content or "", str(reference))
```

`rhyme_rm` is the function Training Gym calls: an async reward that reads
`sample.response` and `sample.label` and returns a float. It is passed to the
recipe as `custom_rm_function`, documented on
[`SlimeRecipe`](/reference/training/slimerecipe/).

### The training config

The config is built by a function rather than declared once, so the same code
serves the one-step proof, the smoke test, and the full run by varying three
arguments. `capture_trace=True` is what makes `run trace` useful later.

```python
from modal_training_gym import Qwen3_4B, TrainConfig
from modal_training_gym.train_recipes.slime_recipe import Qwen3_4b_Recipe


def _image_overlay(image):
    return image.run_commands(
        "uv pip install --system 'nltk>=3.8.0'",
        "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
        # Download through a scratch cache so the image does not leave files
        # where the shared Hugging Face Volume needs to mount.
        "HF_HOME=/tmp/hf-build HF_HUB_CACHE=/tmp/hf-build "
        'python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{EMBED_MODEL}', local_dir='{EMBED_DIR}')\"",
        "rm -rf /tmp/hf-build /root/.cache/huggingface",
    )


def build_config(*, num_rollout: int, n_rows: int, save_interval: int) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_4B(),
        dataset=RhymeInstructionDataset(n_rows=n_rows),
        recipe=Qwen3_4b_Recipe(
            custom_rm_function=rhyme_rm,
            num_rollout=num_rollout,
            rollout_batch_size=16,
            n_samples_per_prompt=8,
            rollout_max_response_len=1024,
            rollout_temperature=1.0,
            save_interval=save_interval,
            eval_interval=None,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            capture_trace=True,
            trace_sample_limit=16,
            image_overlay=_image_overlay,
        ),
    )
```

Before spending any GPU time, the skill requires a local preflight: a compile
and import check, dataset formatting exercised on representative rows, and the
reward function tested on correct, non-rhyming, off-topic, repeated-word,
malformed, and empty responses. The predicted answer has to come from the model
response — never from a prompt or reference field.

## Prove one step

A one-step run (`num_rollout=1`) catches configuration, data, and reward
failures before you commit to a longer horizon. `launch()` starts a detached
Modal app and returns the run ID used for everything that follows:

```python
training_run = build_config(num_rollout=1, n_rows=512, save_interval=1)
run = training_run.launch()
print(f"training_run_id: {run.training_run_id}")
```

The proof passes only when the run completes one rollout, records a nonempty
reward, produces no traceback, and yields samples that make sense. Startup can
take tens of minutes, so a run that has not reached step 1 yet is not a stuck
run.

## Watch the run with the CLI

`run list` confirms the run was recorded, and is how you recover an ID after
closing a terminal or notebook:

```bash
training-gym run list --since 2h
```

`run get --verbose` is the normal progress check. It reports the current stage
and step alongside reward history and rollout summaries:

```bash
training-gym run get <run-id> --verbose
```

Reach for logs when `run get` reports a failure or its step stops advancing:

```bash
training-gym run logs <run-id>
training-gym run logs <run-id> --follow
training-gym run logs <run-id> --search "checkpoint"
```

Reward is only a proxy, so read what the model actually wrote before promoting
anything. The dry run previews the download; the second command writes traces
beneath `./traces/<run-id>/`:

```bash
training-gym run trace <run-id> --out ./traces --dry-run
training-gym run trace <run-id> --out ./traces --yes
```

Every one of these commands takes `--json`, which is what the agent uses when
monitoring on its own. The human-readable output is easier to read when you are
following along.

## Catch reward hacking before you scale

Trace inspection is not a formality. In the session below, reward rose steadily
and looked healthy — and the traces showed the model gaming the rhyme signal
with repeated end words and undersized lines. The agent tightened the scorer
(the distinct-end-word and substance factors above), reran the cheap stage, and
re-read real samples before promoting.

<video controls playsinline width="100%">
  <source src="/agent-driven-training-rhyme.mp4" type="video/mp4">
  <source src="https://gym.modal.dev/agent-driven-training-rhyme.mp4" type="video/mp4">
  <a href="https://gym.modal.dev/agent-driven-training-rhyme.mp4">Watch the agent-driven training demo.</a>
</video>

## Smoke test, then promote

The agent uses the same ladder at every scale, with a fresh run ID at each
rung:

1. **Prove one step** — one completed rollout, nonempty reward, no traceback,
   sensible samples.
2. **Smoke test** — about ten steps, then inspect the reward trajectory plus
   baseline, transition, and recent traces. Change one setting at a time and
   repeat with a new run.
3. **Promote** — launch the full horizon only once the signal is informative
   and the responses are genuinely improving.

A full run is not a commitment to spend its whole horizon. If reward flattens,
declines, or stops being informative, stopping early beats letting a healthy
but ineffective job finish by default.

## Results

The promoted rhyme run finished in 46 minutes:

- Rhyme score: 0.475 → 0.908
- Answers above the rhyme threshold: 27% → 84%
- Non-rhyming answers: 35/128 → 0/128
- Relevance: 0.877 → 0.886 (essentially flat)

That last line is the one that matters. Relevance held steady while rhyme
climbed, which is the evidence that the model learned to rhyme *in addition to*
answering the question, rather than trading one for the other.

## What the loop gives you

Installing the skills gives your agent a runbook for the full training
lifecycle. Instead of stopping once it has generated a configuration, the agent
preflights locally, proves one step, smoke tests, diagnoses problems, and
promotes only a healthy run.

The CLI is what makes those judgments possible. Run status and logs distinguish
slow startup from failure, reward history shows whether learning is happening,
and rollout traces reveal whether the model is improving or merely gaming the
reward.

## Related

- [CLI reference](/reference/cli/) — every `training-gym` command and flag.
- [The observability dashboard](/guides/tools/observability-dashboard/) — the
  same run data in a browser, including per-step timing profiles.
- [`TrainConfig`](/reference/training/trainconfig/) and
  [`SlimeRecipe`](/reference/training/slimerecipe/) — the configuration surface
  the agent writes against.
- [`HuggingFaceDataset`](/reference/core/huggingfacedataset/) — the dataset base
  class used above.
