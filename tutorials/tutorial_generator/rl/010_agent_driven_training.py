# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `010_agent_driven_training` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Post-train Qwen3-4B from one plain-language prompt with an agent",
    "difficulty": "Intermediate",
    "order": 70,
    "api_classes": [
        "HuggingFaceDataset",
        "Qwen3_4B",
        "TrainConfig",
        "SlimeRecipe",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # From one prompt to a trained model

    This entire training project started with one sentence:

    > can you post train a model to rhyme in its output

    That was enough for the agent to turn a rough idea into a working RL
    experiment on Modal. It chose **Qwen3-4B**, designed a reward that balances
    rhyme with relevance, prepared a dataset, wrote the GRPO configuration,
    debugged the remote environment, and managed the run from a one-step proof
    through full training.

    The result: answers meeting the rhyme threshold rose from **27% to 84%**,
    while relevance held steady. The model learned to rhyme *in addition to*
    answering the question—not instead of it.

    This tutorial shows both what the agent built and, more importantly, how it
    used Training Gym's observability to prove that such a simple prompt
    produced a real behavior change.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run locally (your machine drives the Modal GPU workers):

    ```
    cd training-gym
    uv sync
    uv run tutorials/rl/010_agent_driven_training/010_agent_driven_training.py
    ```

    To detach and watch it from the Modal dashboard instead:

    ```
    uv run modal run -d tutorials/rl/010_agent_driven_training/010_agent_driven_training.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main\n"
    "if importlib.util.find_spec('nltk') is None:\n"
    "    %uv pip install -q nltk"
)
def _notebook_setup():
    pass


@markdown
def _confirmations():
    """
    ## Turn one sentence into a training objective

    The initial prompt was intentionally underspecified. The agent did not need
    a finished reward function or training recipe from the user; it only asked
    for the few choices that materially affect behavior and cost:

    - **Behavior:** rhyme in every answer, not only when explicitly requested.
    - **Model:** Qwen3-4B.
    - **Topology:** one node with 8×H100 GPUs, as resolved by the validated
      Qwen3-4B recipe.
    - **Content quality:** reward rhyme only when the response remains relevant
      to the question.

    With those answers, the agent had enough of a specification to implement
    and validate the complete training workflow below.
    """


@markdown
def _code_summary():
    """
    ## Dataset

    We use `tatsu-lab/alpaca`, keeping only self-contained instructions whose
    optional `input` field is empty. Each row provides an instruction for the
    rollout and a reference answer for the relevance reward. The agent checked
    representative formatted rows locally before launching a training run.
    """


@code
def _imports_and_constants():
    import re

    from modal_training_gym import HuggingFaceDataset, Qwen3_4B, TrainConfig
    from modal_training_gym.train_recipes.slime_recipe import Qwen3_4b_Recipe

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_DIR = "/opt/rhyme-embedder"

    SYSTEM_PROMPT = (
        "You are a poet who answers every question in rhyme. Answer the question "
        "correctly and completely, but write the entire answer as verse: at least "
        "four lines, one clause per line, with line endings that rhyme in couplets "
        "(AABB). Do not write any prose, preamble, or explanation outside the verse."
    )


@code
def _dataset():
    class RhymeInstructionDataset(HuggingFaceDataset):
        """Self-contained Alpaca instructions; the label is the reference answer.

        Rows carrying an extra ``input`` field are dropped — the prompt template
        only passes ``instruction``, so those rows would ask an unanswerable
        question. Reference answers are length-bounded to keep the embedding
        comparison meaningful (a two-word label has no topic to match).
        """

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


@markdown
def _reward_intro():
    """
    ## Reward function

    The agent combined a deterministic rhyme score with an embedding-based
    relevance score. Relevance gates rhyme so unrelated verse cannot win, and
    anti-gaming checks penalize repeated end words and one-word lines.

    Before spending GPU time, the agent exercised the reward on correct,
    non-rhyming, off-topic, repeated-word, malformed, and empty responses. The
    full implementation is below.
    """


@code
def _rhyme_scoring():
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
        return [
            line.strip() for line in _strip_thinking(text).split("\n") if line.strip()
        ]

    def _end_word(line: str) -> str:
        words = re.findall(r"[a-zA-Z']+", line)
        return words[-1].lower().strip("'") if words else ""

    def rhyme_tail(word: str) -> tuple:
        """Phonemes from the last stressed vowel onward, stress markers removed.

        Falls back to the last three letters for words the dictionary doesn't
        know (names, coinages), which is a decent orthographic proxy.
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
        """True when two *different* words share a rhyme tail.

        A word never rhymes with itself — otherwise repeating one end word would
        score a perfect rhyme scheme.
        """
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
        """Rhyme quality of a response in ``[0, 1]``.

        Combines the best-fitting rhyme scheme with two anti-gaming factors:
        the share of distinct end words, and the share of lines with real
        substance (three or more words).
        """
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


@code
def _relevance_scoring():
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
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
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


@code
def _combined_reward():
    def rhyme_reward(response: str, reference: str) -> float:
        """Gated rhyme quality plus a smaller standalone relevance term."""
        rhyme = score_rhyme(response)
        relevance = score_relevance(response, reference)
        gate = min(1.0, relevance / 0.4)
        return gate * rhyme + 0.3 * relevance

    async def rhyme_rm(args, sample, **kwargs) -> float:
        model = Qwen3_4B()
        response = model.parse_response(getattr(sample, "response", "") or "")
        reference = getattr(sample, "label", "") or ""
        return rhyme_reward(response.content or "", str(reference))


@markdown
def _reward_details():
    """
    ## Training

    The agent assembled the validated Qwen3-4B recipe, custom reward, dataset,
    and image dependencies into one `TrainConfig`. The important process
    detail is that this same configuration and cluster shape are reused at
    every stage; only the rollout horizon changes.
    """


@code
def _config_excerpt():
    def _image_overlay(image):
        return image.run_commands(
            "uv pip install --system 'nltk>=3.8.0'",
            "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            # Download through a scratch cache so the image does not leave
            # files where the shared Hugging Face Volume needs to mount.
            "HF_HOME=/tmp/hf-build HF_HUB_CACHE=/tmp/hf-build "
            'python -c "from huggingface_hub import snapshot_download; '
            f"snapshot_download('{EMBED_MODEL}', local_dir='{EMBED_DIR}')\"",
            "rm -rf /tmp/hf-build /root/.cache/huggingface",
        )

    def build_config(
        *, num_rollout: int, n_rows: int, save_interval: int
    ) -> TrainConfig:
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


@markdown
def _how_it_works():
    """
    ## The agent owns the training loop

    Writing the configuration is only the beginning. The repository's
    `agent-driven-training` skill has the agent own four stages:

    1. **Configure and preflight.** Implement the dataset and reward, then test
       formatting and adversarial reward cases locally.
    2. **Prove one step.** Launch a fresh one-rollout job. Advance only when it
       records a nonempty reward, completes without a traceback, and produces
       sensible traces.
    3. **Smoke test.** Run about ten steps with the same topology. Check step
       timing, reward spread, and whether the policy is exploiting the scorer.
    4. **Promote.** Start a fresh full run only after the smoke test is healthy.
       Continue monitoring and stop early if the reward becomes flat,
       suspicious, or unstable.

    Modal supplies remote GPU workers, image builds, networking, and
    persistent volumes. Training Gym handles cluster bring-up, checkpointing,
    and run metadata. The agent chooses the next stage from observed evidence.
    """


@markdown
def _monitoring():
    """
    ## Watch the agent prove it worked

    After each launch, the agent records the new training run ID and monitors
    that run directly. It does not treat a live launcher process—or a quiet
    terminal during model startup—as proof that training is healthy.

    If the run ID is not visible in the launch output, list recent runs:

    ```bash
    training-gym run list --since 1h
    ```

    Poll the run summary throughout startup and training:

    ```bash
    training-gym run get <run-id> --verbose
    ```

    This reports the current stage and step, latest reward, reward trajectory,
    rollout timing, Modal app ID, and terminal status. For an automated
    monitor, the agent uses `--json` and treats an empty response or CLI error
    as an observability failure—not as a training event.

    When progress stops or an error appears, stream the run logs:

    ```bash
    training-gym run logs <run-id> --follow
    ```

    The recipe captures sample traces so the agent can inspect generated
    answers and reward execution before promotion:

    ```bash
    training-gym run trace <run-id> --out ./traces --dry-run
    training-gym run trace <run-id> --out ./traces --yes
    ```

    The same data is available visually with `training-gym open`. The
    dashboard shows stage timing, reward distributions, individual rollouts,
    and logs. The agent uses both interfaces: the CLI for repeatable monitoring
    and the dashboard for quickly spotting collapsed rewards, slow phases, or
    suspicious high-scoring samples.
    """


@markdown
def _run_proof_intro():
    """
    ## Prove one step

    This tutorial defaults to the cheapest meaningful stage: one rollout.
    It still starts the full Qwen3-4B cluster, so model and worker startup can
    take several minutes.

    After the proof passes, change `num_rollout` to 10 for a fresh smoke test.
    Promote to the full horizon only after checking reward spread and rollout
    traces.
    """


@code
def _run_proof():
    training_run = build_config(num_rollout=1, n_rows=512, save_interval=1)
    train_result = training_run.train()
    print(f"training_run_id: {train_result.training_run_id}")


@markdown
def _success():
    """
    ## What one prompt produced

    The agent promoted this training job only after each stage passed:

    - **Proof — `grouchy-ellipsis-453dfe77b375`:** `run get --verbose`
      reported 1/1 steps complete with mean reward 0.785. The 128 samples had
      useful reward spread, the prompts and responses were intact, and the
      checkpoint was present.
    - **Smoke — `fast-discriminator-1680dee16483`:** all 10 steps completed
      without errors. First- and last-step traces showed rhyme improving
      without relevance falling, so the reward was not being hacked.
    - **Full — `overcast-gauge-2d52c4e2b271`:** the agent monitored milestone
      steps and reassessed at the midpoint before allowing all 100 steps to
      finish.

    The full run completed in 46 minutes. Mean reward rose from **0.738 to
    1.174** (maximum 1.30). Rhyme score rose from **0.475 to 0.908**, answers
    above the rhyme threshold rose from **27% to 84%**, and non-rhyming answers
    fell from **35/128 to 0/128**.

    Most importantly, relevance held steady (**0.877 to 0.886**) while rhyme
    improved. That is the evidence that the model learned to rhyme *in
    addition to* answering the question. The agent also confirmed that every
    Modal app stopped after final checkpoint synchronization.
    """


@markdown
def _lessons():
    """
    ## Debugging lessons

    The complete trace includes mistakes as well as the successful run:

    - A launch was accidentally backgrounded twice and its process group was
      torn down.
    - A build-time Hugging Face download polluted
      `/root/.cache/huggingface`, so Modal could not mount the shared model
      cache there. The fixed image downloads through a scratch cache and
      removes the mount-point contents before startup.
    - A CPU verification app crash-looped because it imported a local-only
      module that was unavailable in the container. Reading the logs revealed
      that this was failure, not slow progress.

    In each case, the next useful action came from checking run state and logs,
    not from waiting longer. Agent-driven training is a lifecycle rather than
    a single code-generation step: the agent must distinguish startup from a
    crash loop, verify rewards on real traces, clean up failed apps, and change
    course when observability contradicts its assumptions.

    One caveat remains: this scorer measures rhyme, not poetic quality.
    Awkward filler can still score well. A production objective could add an
    LLM judge or a separate fluency reward, followed by the same prove, smoke,
    and promotion process.
    """
