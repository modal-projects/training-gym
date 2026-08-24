# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `001_learning_agent` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100 (+ 1×H100 teacher)",
    "summary": "The Learning Agent Bench loop on the gym: acquire a corpus, author your own exam, train a searching student with GRPO, and read the margin off the dashboard",
    "difficulty": "Advanced",
    "order": 20,
    "api_classes": [
        "DatasetConfig",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "ModelDeployment",
        "Qwen3_4B",
        "Qwen3_8B",
        "SlimeRecipe",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # A learning agent on the gym

    [LAB](https://github.com/modal-projects/learning-agent) (Learning Agent
    Bench) asks a different question than most agent benchmarks: not *can an
    agent use tools*, but **can an agent make a model learn**. An agent is
    dropped into a domain it has never seen and must come back with a student
    model that answers expert questions better than the untrained one. Nothing
    is delegated — the agent acquires the corpus, manufactures its own training
    signal, builds the harness the student answers through, and invents the
    evaluation it steers by. There is no verifiable reward and no human in the
    loop. The only number that counts is the **margin**: trained student minus
    base student, on questions neither of them was trained on.

    This tutorial runs that loop end to end on Modal, with the Training Gym as
    the instrument you watch it through:

    1. **Acquire a corpus** the student was never trained on, and pin it.
    2. **Author the exam** — a teacher reads the corpus and writes questions,
       gold answers, and weighted-claim rubrics, split into `train` / `dev` /
       `test` pools from *disjoint* passages.
    3. **Build the harness** — the student answers by searching the corpus
       (`grep` / `read`), not from memory. The same harness is used in
       training rollouts and in evaluation, which is the rule LAB enforces.
    4. **Train** with GRPO (slime) on a deterministic reward.
    5. **Score** base and trained students with an n-vote rubric judge, and
       report the margin with a seeded bootstrap CI — on `dev`, and once, at
       the very end, on the locked `test` split.

    Throughout, every number and every trajectory reports to the gym
    dashboard: reward curves, per-claim scores, the student's search
    transcripts, step timings, and both evals side by side.

    **Why the split between reward and judge?** LAB's rule of thumb is that an
    LLM judge belongs in the dev-signal path (a handful of calls, seconds
    each) and a deterministic scorer belongs in the reward path (thousands of
    calls per step). We follow it exactly: the teacher judges evals,
    arithmetic scores rollouts. The two disagree — that gap is itself a
    measurement, and the dashboard is where you read it.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/agent/001_learning_agent/001_learning_agent.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    import hashlib
    import json
    import random
    import re
    import shutil
    import subprocess
    import tempfile
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import modal

    from modal_training_gym import (
        DatasetConfig,
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        Qwen3_4B,
        Qwen3_8B,
        SlimeRecipe,
        TrainConfig,
        list_checkpoints,
    )


# ── 0. The track ─────────────────────────────────────────────────────────


@markdown
def _track_intro():
    """
    ## Pick a track

    LAB grades the same loop at three levels of hand-holding, and the track is
    the single most important thing about a result — a number from `easy` and
    a number from `hard` are not comparable:

    | track | corpus | dev set | harness |
    |---|---|---|---|
    | `easy` | given | **given**, graded | given |
    | `medium` | given | **you author it** | given |
    | `hard` | **you acquire it** | **you author it** | **you build it** |

    This tutorial runs `hard`: nothing is handed over. Two of those columns are
    real switches below — whether we fetch the corpus ourselves, and whether
    the student answers through a harness we build or closed-book. The dev set
    is authored by the teacher on every track here, because there is no
    operator standing by to hand us a graded one; on `easy` you would replace
    that step with the file you were given.
    """


@code
def _track():
    TRACK = "hard"

    TRACK_RULES = {
        "easy": {"acquire_corpus": False, "build_harness": False},
        "medium": {"acquire_corpus": False, "build_harness": False},
        "hard": {"acquire_corpus": True, "build_harness": True},
    }
    RULES = TRACK_RULES[TRACK]


# ── 1. Acquire the corpus ────────────────────────────────────────────────


@markdown
def _corpus_intro():
    """
    ## Acquire the corpus, then pin it

    A learning task needs a domain the student does not already know. We use
    the [DSPy](https://github.com/stanfordnlp/dspy) documentation at a pinned
    tag: recent enough that a 4B model has no reliable knowledge of it, small
    enough to fetch in seconds.

    Pinning matters more than it looks. The corpus is the *content* of
    everything downstream — questions, rubrics, reward, judge — so a corpus
    that moves makes two runs incomparable even though every number still
    prints. LAB writes a content tree-hash of each corpus into
    `bench/pins.json` and refuses to score on drift; here we compute the same
    kind of hash and carry it into every eval record, so the dashboard row
    tells you which corpus produced it.
    """


@code
def _acquire_corpus():
    CORPUS_REPO = "https://github.com/stanfordnlp/dspy"
    CORPUS_TAG = "3.0.3"
    CORPUS_SUBDIR = "docs/docs"
    CORPUS_DIR = Path("/tmp/learning_agent_corpus")

    def fetch_corpus() -> Path:
        """Sparse-clone just the docs tree at the pinned tag."""
        if CORPUS_DIR.exists():
            shutil.rmtree(CORPUS_DIR)
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "--branch", CORPUS_TAG, CORPUS_REPO, str(CORPUS_DIR),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", CORPUS_SUBDIR],
            cwd=CORPUS_DIR,
            check=True,
            capture_output=True,
        )
        return CORPUS_DIR / CORPUS_SUBDIR

    def tree_hash(root: Path) -> str:
        """Content hash over the corpus: path + bytes, in sorted path order."""
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.md")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def chunk_document(text: str, *, max_chars: int = 1400) -> list[str]:
        """Split on markdown headings, then pack sections up to `max_chars`."""
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)  # frontmatter
        sections = re.split(r"\n(?=#{1,3} )", text)
        chunks, buffer = [], ""
        for section in sections:
            if len(buffer) + len(section) <= max_chars:
                buffer += section
            else:
                if buffer.strip():
                    chunks.append(buffer.strip())
                buffer = section[:max_chars]
        if buffer.strip():
            chunks.append(buffer.strip())
        return chunks

    def load_chunks(root: Path, *, min_chars: int = 600) -> list[dict]:
        chunks = []
        for path in sorted(root.rglob("*.md")):
            for i, chunk in enumerate(chunk_document(path.read_text(errors="ignore"))):
                if len(chunk) >= min_chars:
                    chunks.append(
                        {"source": f"{path.relative_to(root)}#{i}", "text": chunk}
                    )
        return chunks

    corpus_root = CORPUS_DIR / CORPUS_SUBDIR
    if RULES["acquire_corpus"] or not corpus_root.exists():
        corpus_root = fetch_corpus()
    corpus_pin = tree_hash(corpus_root)
    all_chunks = load_chunks(corpus_root)
    print(f"Corpus: {CORPUS_REPO}@{CORPUS_TAG} — {len(all_chunks)} chunks")
    print(f"Corpus pin: sha256:{corpus_pin[:16]}…")


@markdown
def _split_intro():
    """
    ## Split the corpus before writing a single question

    The most common way to fake a learning result is to measure on questions
    the model was trained on. Splitting *questions* is not enough — two
    questions written from the same passage are near-duplicates, so a train
    question can teach the exact passage a test question grades.

    So we split the **passages** first, into three disjoint pools, and only
    then write questions from each. `dev` steers the loop and can be looked at
    as often as you like. `test` is looked at **once**, at the very end. That
    is the discipline that makes the final margin mean anything.
    """


@code
def _split_pools():
    N_TRAIN_CHUNKS, N_DEV_CHUNKS, N_TEST_CHUNKS = 96, 24, 24

    def stride_sample(chunks: list[dict], limit: int) -> list[dict]:
        """Stride instead of slice so a sample spans the whole doc tree."""
        stride = max(1, len(chunks) // limit)
        return chunks[::stride][:limit]

    wanted = N_TRAIN_CHUNKS + N_DEV_CHUNKS + N_TEST_CHUNKS
    pool = stride_sample(all_chunks, wanted)
    random.Random(0).shuffle(pool)  # seeded: the split reproduces exactly

    train_chunks = pool[:N_TRAIN_CHUNKS]
    dev_chunks = pool[N_TRAIN_CHUNKS:N_TRAIN_CHUNKS + N_DEV_CHUNKS]
    test_chunks = pool[N_TRAIN_CHUNKS + N_DEV_CHUNKS:wanted]

    print(
        f"passages — train {len(train_chunks)} / dev {len(dev_chunks)} / "
        f"test {len(test_chunks)} (disjoint)"
    )


# ── 2. Manufacture the training signal ───────────────────────────────────


@markdown
def _teacher_intro():
    """
    ## Deploy the teacher

    One model does two jobs in this loop: it writes the exam (once, up front)
    and it judges answers (three times: base on dev, trained on dev, trained
    on test). It never touches the reward path and never appears in the
    student's answer path — LAB forbids external models there, and so do we.

    `DeploymentConfig.serve()` gives us an OpenAI-compatible endpoint and
    registers the deployment on the dashboard's *Deployments* tab.

    We hold the handles in a small dict rather than plain module globals. That
    is not a style preference: the tutorial generator hoists any assignment a
    function refers to up to module scope, so a bare `teacher = ….serve()`
    would deploy an 8B model the moment the file is *imported*, before the
    secret check below has had a chance to fail fast.
    """


@code
def _serve_teacher():
    ENDPOINTS: dict = {}

    ENDPOINTS["teacher"] = DeploymentConfig(
        model=Qwen3_8B(),
        app_name="learning-agent-teacher",
        served_model_name="qwen3-8b-teacher",
        unauthenticated=True,
    ).serve()
    ENDPOINTS["teacher"].wait_until_ready(timeout=3000)
    print(f"Teacher URL: {ENDPOINTS['teacher'].url}")


@markdown
def _qa_intro():
    """
    ## Write the questions, gold answers, and rubrics

    This is the step LAB's `medium` and `hard` tracks force on the agent: with
    no graded dev set in the workspace, it must build its own compass.

    We ask the teacher, for each passage, for one question answerable *only*
    from that passage, a gold answer, and 2–4 weighted claims a correct answer
    must contain. That triple is LAB's label shape:

    ```json
    {"question": "...",
     "label": {"gold_answer": "...",
               "rubric": [{"claim_id": "c1", "weight": 2, "statement": "..."}]}}
    ```

    The rubric is what makes an unverifiable domain scorable: prose can't be
    string-matched, but *claims present in the prose* can be counted. The
    reward function and the judge both read this same field, from opposite
    ends of the cost spectrum.

    Generation is best-effort and strict about what it keeps. A row with no
    claims would be accepted by a naive check (`all([])` is `True`) and then
    score a hard zero forever — poisoning the reward with a question that
    cannot be answered well and dragging both eval means down with it.
    """


@code
def _generate_qa():
    QA_PROMPT = (
        "You are writing an exam about a documentation corpus.\n\n"
        "Read the passage and write ONE question that can be answered only by "
        "someone who has read it. Then write the gold answer, and 2-4 rubric "
        "claims: short, independently checkable statements that a correct "
        "answer must contain. Weight each claim 1-3 by importance.\n\n"
        "Return ONLY a JSON object, no prose, no code fences:\n"
        '{{"question": "...", "gold_answer": "...", "rubric": '
        '[{{"claim_id": "c1", "weight": 2, "statement": "..."}}]}}\n\n'
        "Passage (from {source}):\n---\n{text}\n---"
    )

    def parse_json_object(text: str) -> dict | None:
        """Pull the first JSON object out of a model response."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def is_well_formed(row: dict | None) -> bool:
        if not isinstance(row, dict):
            return False
        if not (row.get("question") and row.get("gold_answer")):
            return False
        rubric = row.get("rubric")
        # `bool(rubric)`: an empty rubric is unscorable, not merely unusual.
        return isinstance(rubric, list) and bool(rubric) and all(
            isinstance(claim, dict) and claim.get("statement") for claim in rubric
        )

    def write_exam_item(deployment: ModelDeployment, chunk: dict) -> dict | None:
        response = deployment.generate(
            QA_PROMPT.format(source=chunk["source"], text=chunk["text"]),
            ensure_ready=False,
            temperature=0.7,
            max_tokens=1024,
            chat_template_kwargs={"enable_thinking": False},
        )
        row = parse_json_object(response)
        if not is_well_formed(row):
            return None
        return {
            "question": row["question"].strip(),
            "gold_answer": row["gold_answer"].strip(),
            "rubric": [
                {
                    "claim_id": str(claim.get("claim_id") or f"c{i}"),
                    "weight": float(claim.get("weight") or 1),
                    "statement": str(claim["statement"]).strip(),
                }
                for i, claim in enumerate(row["rubric"], start=1)
            ],
            "source": chunk["source"],
        }

    def write_exam(chunks: list[dict]) -> list[dict]:
        teacher = ENDPOINTS["teacher"]
        with ThreadPoolExecutor(max_workers=8) as executor:
            rows = executor.map(lambda c: write_exam_item(teacher, c), chunks)
        return [row for row in rows if row is not None]

    raw_train = write_exam(train_chunks)
    raw_dev = write_exam(dev_chunks)
    raw_test = write_exam(test_chunks)
    for name, raw, chunks in (
        ("train", raw_train, train_chunks),
        ("dev", raw_dev, dev_chunks),
        ("test", raw_test, test_chunks),
    ):
        print(f"{name}: {len(raw)}/{len(chunks)} passages produced a usable item")


@markdown
def _hygiene_intro():
    """
    ## Hygiene: dedup, then decontaminate

    LAB ships this as a hard gate (`data_tool/pool/dedup_decontam`) and it is
    the least glamorous, most load-bearing step in the loop.

    **Dedup** protects the *training* signal. GRPO learns from the spread of
    rewards *within a group of answers to one prompt*; two copies of the same
    question do not add information, they just spend rollout budget.

    **Decontamination** protects the *measurement*. A train question that
    overlaps a dev or test question turns the margin into a memorization
    score. We compare content-word sets and drop the evaluation row on any
    material overlap — dropping from the eval side, never the train side, so
    the training set is never quietly shaped by the test set.
    """


@code
def _hygiene():
    def question_key(text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    def dedup(rows: list[dict]) -> list[dict]:
        seen, kept = set(), []
        for row in rows:
            key = question_key(row["question"])
            if key not in seen:
                seen.add(key)
                kept.append(row)
        return kept

    def jaccard(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if (a or b) else 0.0

    def decontaminate(
        eval_rows: list[dict], train_rows: list[dict], *, threshold: float = 0.6
    ) -> tuple[list[dict], int]:
        train_sets = [content_words(row["question"]) for row in train_rows]
        kept, dropped = [], 0
        for row in eval_rows:
            words = content_words(row["question"])
            if any(jaccard(words, other) >= threshold for other in train_sets):
                dropped += 1
                continue
            kept.append(row)
        return kept, dropped

    train_items = dedup(raw_train)
    dev_items, dev_dropped = decontaminate(dedup(raw_dev), train_items)
    test_items, test_dropped = decontaminate(dedup(raw_test), train_items)

    print(f"train: {len(train_items)} questions")
    print(f"dev:   {len(dev_items)} questions ({dev_dropped} dropped as contaminated)")
    print(f"test:  {len(test_items)} questions ({test_dropped} dropped) — LOCKED")


@notebook_only
@code
def _peek_exam():
    item = dev_items[0]
    print(item["question"])
    print(f"\ngold: {item['gold_answer'][:200]}")
    for claim in item["rubric"]:
        print(f"  [w={claim['weight']}] {claim['statement']}")


# ── 3. The two scorers ───────────────────────────────────────────────────


@markdown
def _scorers_intro():
    """
    ## Two scorers: one for the reward, one for the judgment

    **Claim coverage (reward).** For each rubric claim, take its content words
    and measure what fraction appear in the answer; count the claim as covered
    past a threshold, and average with the claim weights. It is crude, but it
    is free, deterministic, and returns in microseconds — which is what a
    reward called thousands of times per step has to be.

    One detail worth the two lines it costs: the tokenizer keeps dots so
    `dspy.Predict` survives as one token, which means it would also keep the
    period ending a sentence — turning `module.` into a token that can never
    match the `module` in an answer. Stripping edge dots keeps the identifiers
    and drops the punctuation.

    **Rubric judge (dev/test signal).** The teacher sees the question, the
    gold answer, and the claims, and votes on each claim at temperature 0,
    `n_votes` times, with a majority vote per claim. Slower and much better;
    used only on the few dozen evaluation questions. A claim the judge fails
    to answer for is recorded as failed, never silently scored zero.
    """


@code
def _claim_coverage():
    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
        "when", "which", "with", "you", "your",
    }

    def content_words(text: str) -> set[str]:
        # Dots are kept so `dspy.Predict` stays one token, then stripped from
        # the edges so a sentence-final period can't make a token unmatchable.
        words = (word.strip(".") for word in re.findall(r"[a-z0-9_.]{3,}", text.lower()))
        return {word for word in words if word and word not in _STOPWORDS}

    def claim_coverage(answer: str, rubric: list[dict], *, threshold: float = 0.6) -> float:
        """Weighted fraction of rubric claims whose content words appear."""
        answer_words = content_words(answer)
        total_weight, hit_weight = 0.0, 0.0
        for claim in rubric:
            claim_words = content_words(claim["statement"])
            if not claim_words:
                continue
            weight = float(claim.get("weight", 1))
            overlap = len(claim_words & answer_words) / len(claim_words)
            total_weight += weight
            if overlap >= threshold:
                hit_weight += weight
        return hit_weight / total_weight if total_weight else 0.0

    def length_penalty(answer: str, *, budget_words: int = 220) -> float:
        """Discourage answering by reciting the corpus back."""
        overflow = max(0, len(answer.split()) - budget_words)
        return min(0.3, 0.002 * overflow)


@code
def _judge():
    JUDGE_VOTES = 3

    JUDGE_PROMPT = (
        "You are grading an answer against a rubric. For each claim, decide "
        "whether the answer actually contains it. Ignore style, wording, and "
        "extra detail; judge only whether the claim is present and correct.\n\n"
        "Return ONLY a JSON object, no prose:\n"
        '{{"verdicts": [{{"claim_id": "c1", "present": true}}]}}\n\n'
        "Question: {question}\n\n"
        "Reference answer: {gold_answer}\n\n"
        "Claims:\n{claims}\n\n"
        "Answer to grade:\n{answer}"
    )

    def _one_vote(deployment: ModelDeployment, item: dict, answer: str) -> dict:
        claims = "\n".join(
            f"- {claim['claim_id']} (weight {claim['weight']}): {claim['statement']}"
            for claim in item["rubric"]
        )
        response = deployment.generate(
            JUDGE_PROMPT.format(
                question=item["question"],
                gold_answer=item["gold_answer"],
                claims=claims,
                answer=answer or "(empty)",
            ),
            ensure_ready=False,
            temperature=0.0,
            max_tokens=512,
            chat_template_kwargs={"enable_thinking": False},
        )
        verdicts = (parse_json_object(response) or {}).get("verdicts") or []
        return {
            str(verdict.get("claim_id")): bool(verdict.get("present"))
            for verdict in verdicts
            if isinstance(verdict, dict)
        }

    def judge_claims(deployment: ModelDeployment, item: dict, answer: str) -> dict:
        """Majority vote per claim over `JUDGE_VOTES` independent judgments."""
        ballots = [_one_vote(deployment, item, answer) for _ in range(JUDGE_VOTES)]

        total_weight, hit_weight, per_claim, failed = 0.0, 0.0, {}, 0
        for claim in item["rubric"]:
            votes = [b[claim["claim_id"]] for b in ballots if claim["claim_id"] in b]
            weight = float(claim["weight"])
            total_weight += weight
            if not votes:
                # No verdict at all: record the failure, never a silent zero.
                failed += 1
                continue
            present = Counter(votes).most_common(1)[0][0]
            per_claim[claim["claim_id"]] = bool(present)
            if present:
                hit_weight += weight

        graded_weight = total_weight - sum(
            float(c["weight"]) for c in item["rubric"] if c["claim_id"] not in per_claim
        )
        return {
            "claim_score": hit_weight / graded_weight if graded_weight else None,
            "per_claim": per_claim,
            "failed_claims": failed,
        }

    def bootstrap_ci95(values: list[float], *, resamples: int = 10000, seed: int = 0):
        """Seeded percentile bootstrap — reproduces bit-identically."""
        if not values:
            return (0.0, 0.0)
        rng = random.Random(seed)
        n = len(values)
        means = sorted(
            sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples)
        )
        return (means[int(0.025 * resamples)], means[int(0.975 * resamples)])


# ── 4. The search harness ────────────────────────────────────────────────


@markdown
def _harness_intro():
    """
    ## The harness: the student answers by searching, not remembering

    LAB's QA tasks are not closed-book. The student gets `grep` and `read`
    over the corpus and answers from what it finds — so what we are training
    is not recall of facts but the *behavior* of finding them: what to search
    for, when to read more, when to stop and answer.

    This matters for the margin too. A closed-book 4B model can only improve
    by memorizing the corpus, which is a small and fragile win. A searching
    model can improve at search, which transfers to passages it never saw.

    The protocol is deliberately tiny, because every token of format
    instruction is a token the student can fail on:

    - `<search>query</search>` → the top matching passages, by content-word
      overlap
    - `<open>doc#i</open>` → that passage in full
    - `<final>answer</final>` → done

    The same loop runs in two places, and that is the point: `search_answer()`
    below drives a served deployment for evaluation, and the rollout function
    in the next section drives the training engine. If those two ever diverge,
    you are training one policy and scoring another.
    """


@code
def _harness():
    HARNESS_SYSTEM = (
        "You are an expert on the DSPy framework answering from its "
        "documentation.\n"
        "You may search the docs before answering, using exactly these "
        "commands, one per message:\n"
        "  <search>keywords</search>  - list passages matching the keywords\n"
        "  <open>doc#i</open>         - read one passage in full\n"
        "  <final>answer</final>      - give your final answer and stop\n"
        "Search first when you are unsure. Answer in at most 200 words, "
        "naming the specific classes, arguments, and behaviors involved. "
        "Do not speculate: if the docs do not say, say so."
    )

    MAX_TURNS = 6
    SEARCH_HITS = 4

    def build_index(chunks: list[dict]) -> dict[str, str]:
        return {chunk["source"]: chunk["text"] for chunk in chunks}

    def tool_search(index: dict[str, str], query: str) -> str:
        """Rank passages by content-word overlap with the query."""
        wanted = content_words(query)
        if not wanted:
            return "(empty query)"
        scored = sorted(
            (
                (len(wanted & content_words(text)) / len(wanted), source, text)
                for source, text in index.items()
            ),
            reverse=True,
        )
        hits = [(s, src, txt) for s, src, txt in scored[:SEARCH_HITS] if s > 0]
        if not hits:
            return "(no matches)"
        return "\n".join(
            f"{src} (score {score:.2f}): {text[:200].strip()}…"
            for score, src, text in hits
        )

    def tool_open(index: dict[str, str], doc_id: str) -> str:
        text = index.get(doc_id.strip())
        return text[:1600] if text else f"(no such passage: {doc_id})"

    def handle_tool_call(index: dict[str, str], text: str) -> tuple[str, str] | None:
        """Return (tool_name, result) for the first command in `text`."""
        search = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
        if search:
            return ("search", tool_search(index, search.group(1)))
        open_doc = re.search(r"<open>(.*?)</open>", text, re.DOTALL)
        if open_doc:
            return ("open", tool_open(index, open_doc.group(1)))
        return None

    def extract_final(text: str) -> str | None:
        match = re.search(r"<final>(.*?)</final>", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def direct_answer(deployment: ModelDeployment, question: str) -> dict:
        """Closed-book fallback for the tracks where the harness is given."""
        reply = deployment.generate(
            question,
            ensure_ready=False,
            temperature=0.0,
            max_tokens=768,
            chat_template_kwargs={"enable_thinking": False},
        )
        return {
            "answer": reply.strip(),
            "messages": [{"role": "assistant", "content": reply}],
            "turns": 1,
            "searches": 0,
            "opens": 0,
            "gave_up": False,
        }

    def search_answer(
        deployment: ModelDeployment, index: dict[str, str], question: str
    ) -> dict:
        """Run the harness against a served model; return answer + transcript."""
        if not RULES["build_harness"]:
            return direct_answer(deployment, question)

        messages = [
            {"role": "system", "content": HARNESS_SYSTEM},
            {"role": "user", "content": question},
        ]
        searches, opens = 0, 0

        for turn in range(MAX_TURNS):
            reply = deployment.generate(
                question,
                ensure_ready=False,
                messages=messages,
                temperature=0.0,
                max_tokens=768,
                chat_template_kwargs={"enable_thinking": False},
            )
            messages.append({"role": "assistant", "content": reply})

            answer = extract_final(reply)
            if answer is not None:
                return {
                    "answer": answer,
                    "messages": messages,
                    "turns": turn + 1,
                    "searches": searches,
                    "opens": opens,
                    "gave_up": False,
                }

            call = handle_tool_call(index, reply)
            if call is None:
                # No command and no final answer: take the raw text as the
                # answer rather than burning turns on a model that has
                # stopped following the protocol.
                return {
                    "answer": reply.strip(),
                    "messages": messages,
                    "turns": turn + 1,
                    "searches": searches,
                    "opens": opens,
                    "gave_up": True,
                }

            tool, result = call
            searches += tool == "search"
            opens += tool == "open"
            messages.append({"role": "user", "content": f"<result>\n{result}\n</result>"})

        return {
            "answer": "",
            "messages": messages,
            "turns": MAX_TURNS,
            "searches": searches,
            "opens": opens,
            "gave_up": True,
        }


# ── 5. Dataset ───────────────────────────────────────────────────────────


@markdown
def _dataset_intro():
    """
    ## Turn the exam into a training set

    Each row carries the prompt the student sees and a `label` holding the
    gold answer and rubric. The label never enters the prompt — it reaches the
    reward function as `sample.label` and nothing else.

    **The exam does not travel in the closure.** Modal cloudpickles every
    function with whatever its closure reaches and rejects the result past
    64 KiB — and a hundred questions with gold answers and rubrics is bigger
    than that. So the exam is parked on a Volume and the config carries two
    filenames; the container reads them back with `Volume.read_file`, which
    needs no mount. The rule of thumb: *code* rides in the closure, *data*
    rides on a volume.
    """


@code
def _dataset():
    EXAM_VOLUME = "learning-agent-exam"

    def ship(key: str, payload: list[dict]) -> str:
        """Park data on the volume under `key` and hand back the key."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
        volume = modal.Volume.from_name(EXAM_VOLUME, create_if_missing=True)
        with volume.batch_upload(force=True) as batch:
            batch.put_file(handle.name, f"/{key}")
        return key

    def fetch(key: str) -> list[dict]:
        """Read a shipment back — here, or inside a training container."""
        volume = modal.Volume.from_name(EXAM_VOLUME, create_if_missing=True)
        return json.loads(b"".join(volume.read_file(key)).decode())

    class ExamDataset(DatasetConfig):
        input_key = "messages"
        label_key = "label"
        apply_chat_template = True
        always_prepare = True
        # Keys on the volume, not the rows themselves: this instance is
        # cloudpickled into the container that materializes the dataset.
        train_key: str = ""
        eval_key: str = ""

        def load(self, split="all"):
            rows = fetch(self.train_key) if self.train_key else []
            eval_rows = fetch(self.eval_key) if self.eval_key else []
            if split == "eval":
                return eval_rows
            if split == "train":
                return rows
            return rows + eval_rows

        def prepare(self, path: str, eval_paths: dict[str, str] | None = None):
            import os

            from datasets import Dataset

            def to_example(item: dict) -> dict:
                return {
                    "messages": [
                        {"role": "system", "content": HARNESS_SYSTEM},
                        {"role": "user", "content": item["question"]},
                    ],
                    "label": json.dumps(
                        {"gold_answer": item["gold_answer"], "rubric": item["rubric"]}
                    ),
                }

            os.makedirs(os.path.dirname(path), exist_ok=True)
            Dataset.from_list(
                [to_example(i) for i in self.load(split="train")]
            ).to_parquet(path)
            eval_rows = self.load(split="eval")
            for eval_path in (eval_paths or {}).values():
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                Dataset.from_list(
                    [to_example(i) for i in eval_rows]
                ).to_parquet(eval_path)


# ── 6. Rollouts and reward ───────────────────────────────────────────────


@markdown
def _rollout_intro():
    """
    ## Rollouts: the same harness, inside the training engine

    `custom_generate_function` replaces slime's default single-shot
    generation, so a training rollout is a whole search episode rather than
    one completion. Two things matter here:

    **Loss masking.** The tool results we paste back are not the model's
    output; training on them teaches the model to imitate our search engine.
    Every tool result gets `loss_mask=0` and only the model's own text is
    trained.

    **The corpus rides along.** The searchable index is captured in the
    closure. This function is not subject to the 64 KiB budget the dataset
    hit: the gym cloudpickles `custom_generate_function` into a module baked
    into the training image rather than into a Modal function, so a few
    hundred KB of markdown is cheaper to ship than to coordinate.

    Everything the episode learns about itself is stashed on
    `sample.metadata`. Numeric keys there are picked up automatically by the
    dashboard and charted per rollout, so `searches`, `turns`, and
    `claim_coverage` become curves next to the reward without any extra
    plumbing; `trajectory_messages` renders as the conversation itself in the
    rollout inspector.
    """


@code
def _rollout():
    def make_search_generate(index: dict[str, str]):
        """Bind the corpus into the rollout function; the closure is cloudpickled
        into the training container along with it."""

        async def search_generate(args, sample, sampling_params):
            from slime.rollout.sglang_rollout import GenerateState
            from slime.utils.http_utils import post
            from slime.utils.types import Sample

            state = GenerateState(args)
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

            prompt_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
            transcript = ""
            segments: list[tuple[str, int]] = []
            searches, opens, answer = 0, 0, ""
            status = Sample.Status.COMPLETED

            for turn in range(MAX_TURNS):
                output = await post(
                    url,
                    {
                        "text": f"{sample.prompt}\n{transcript}".strip(),
                        "sampling_params": sampling_params,
                    },
                )
                finish_type = output["meta_info"]["finish_reason"]["type"]
                if finish_type == "abort":
                    sample.status = Sample.Status.ABORTED
                    return sample

                model_text = output["text"]
                transcript += model_text
                segments.append((model_text, 1))  # trainable: the model's own tokens

                final = extract_final(model_text)
                if final is not None:
                    answer = final
                    break

                call = handle_tool_call(index, model_text)
                if call is None:
                    answer = model_text.strip()
                    break

                tool, result = call
                searches += tool == "search"
                opens += tool == "open"
                observation = f"\n<result>\n{result}\n</result>\n"
                transcript += observation
                segments.append((observation, 0))  # masked: our text, not the model's

                if finish_type == "length":
                    status = Sample.Status.TRUNCATED
                    break

            response_token_ids: list[int] = []
            loss_masks: list[int] = []
            for segment_text, trainable in segments:
                token_ids = state.tokenizer(segment_text, add_special_tokens=False)["input_ids"]
                response_token_ids += token_ids
                loss_masks += [trainable] * len(token_ids)

            sample.tokens = prompt_ids + response_token_ids
            sample.response_length = len(response_token_ids)
            sample.response = transcript
            sample.loss_mask = loss_masks
            sample.status = status

            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["answer"] = answer
            metadata["searches"] = searches
            metadata["opens"] = opens
            metadata["turns"] = len(segments) - opens - searches
            metadata["answered"] = int(bool(answer))
            # Rendered as a conversation by the dashboard's rollout inspector.
            metadata["trajectory_messages"] = [
                {"role": "assistant" if trainable else "user", "content": text}
                for text, trainable in segments
            ]
            sample.metadata = metadata
            return sample

        return search_generate


@markdown
def _reward_intro():
    """
    ## The reward

    Claim coverage of the *final answer*, with two shaping terms on top of it,
    never inside it: a length penalty, and a small format bonus for using the
    protocol at all. LAB is explicit about that ordering — shaping that can
    substitute for correctness is how you train a model to look right.

    Note what the reward does *not* do: call the teacher. A judge in the
    reward path costs seconds per sample, and at
    `rollout_batch_size × n_samples_per_prompt` samples per step that is the
    entire step budget. And the student never grades itself, which would be
    reward hacking by construction.
    """


@code
def _reward():
    student_model = Qwen3_4B()

    async def claim_coverage_rm(args, sample, **kwargs) -> float:
        label = getattr(sample, "label", None)
        if isinstance(label, str):
            label = json.loads(label)
        rubric = label.get("rubric", []) if isinstance(label, dict) else []

        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}

        answer = metadata.get("answer") or ""
        coverage = claim_coverage(answer, rubric)
        score = coverage - length_penalty(answer)
        if metadata.get("answered") and metadata.get("searches"):
            score += 0.05  # used the protocol as designed

        metadata["claim_coverage"] = coverage
        metadata["reward"] = score
        sample.metadata = metadata
        return float(score)


# ── 7. Evaluate the base student ─────────────────────────────────────────


@markdown
def _base_eval_intro():
    """
    ## Measure the base student

    `EvalConfig.evaluate()` runs the dev questions through the *whole
    harness*, scores each row, and writes the result to the metadata volume —
    so the run appears on the dashboard's *Evals* tab, live, with each row's
    prompt, the student's search transcript, and its score.

    Every row carries more than its score: claim coverage (what training
    optimizes), the number of searches and turns (what the harness did), the
    per-claim verdicts (what the judge thought), and the corpus pin (which
    corpus produced it). That is LAB's run record, one row at a time.
    """


@code
def _base_eval():
    def make_eval_fn(
        index: dict[str, str], items_by_question: dict[str, dict], corpus_pin: str
    ):
        def exam_eval_fn(deployment: ModelDeployment, example: dict) -> EvalRowResult:
            item = items_by_question[example["question"]]
            episode = search_answer(deployment, index, item["question"])
            verdict = judge_claims(ENDPOINTS["teacher"], item, episode["answer"])
            score = verdict["claim_score"]
            return EvalRowResult(
                score=0.0 if score is None else score,
                prompt=item["question"],
                response=episode["answer"],
                metadata={
                    "claim_coverage": claim_coverage(episode["answer"], item["rubric"]),
                    "searches": episode["searches"],
                    "opens": episode["opens"],
                    "turns": episode["turns"],
                    "judge_failed_claims": verdict["failed_claims"],
                    "per_claim": verdict["per_claim"],
                    "trajectory_messages": episode["messages"],
                    "source": item["source"],
                    "corpus_pin": corpus_pin[:16],
                    "track": TRACK,
                },
            )

        return exam_eval_fn

    def mean_of(result, key: str) -> float:
        values = [row.metadata.get(key, 0.0) for row in result.rows]
        return sum(values) / len(values) if values else float("nan")

    train_dataset = ExamDataset(
        train_key=ship("exam-train.json", train_items),
        eval_key=ship("exam-dev.json", dev_items),
    )
    dev_dataset = ExamDataset(eval_key="exam-dev.json")
    test_dataset = ExamDataset(eval_key=ship("exam-test.json", test_items))

    dev_eval = EvalConfig(
        dataset=dev_dataset,
        eval_fn=make_eval_fn(
            build_index(dev_chunks),
            {i["question"]: i for i in dev_items},
            corpus_pin,
        ),
        prompt_column="question",
    )
    test_eval = EvalConfig(
        dataset=test_dataset,
        eval_fn=make_eval_fn(
            build_index(test_chunks),
            {i["question"]: i for i in test_items},
            corpus_pin,
        ),
        prompt_column="question",
    )

    ENDPOINTS["base"] = DeploymentConfig(
        model=student_model,
        app_name="learning-agent-student-base",
        served_model_name="qwen3-4b-base",
        unauthenticated=True,
    ).serve()

    base_dev = dev_eval.evaluate(ENDPOINTS["base"], max_concurrency=4)
    print(f"Base student — judge {base_dev.mean:.1%}, "
          f"coverage {mean_of(base_dev, 'claim_coverage'):.1%}, "
          f"{mean_of(base_dev, 'searches'):.1f} searches/question")


# ── 8. Train ─────────────────────────────────────────────────────────────


@markdown
def _train_intro():
    """
    ## Train the student

    GRPO with slime: for each question the student runs
    `n_samples_per_prompt` independent search episodes, the reward ranks them
    against each other, and the advantage pushes toward the episodes that
    found and stated the right claims. No gold answer is ever shown to the
    model — it only ever sees its own attempts, ranked.

    Open the dashboard with `training-gym open` and watch it live.
    """


@code
def _train():
    search_generate = (
        make_search_generate(build_index(train_chunks))
        if RULES["build_harness"]
        else None
    )

    training_run = TrainConfig(
        model=student_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            custom_generate_function=search_generate,
            custom_rm_function=claim_coverage_rm,

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=24,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=2048,
            rollout_temperature=1.0,

            lr=5e-7,
            save_interval=8,
            apply_chat_template_kwargs='{"enable_thinking": false}',
        ),
    )

    print("——— Training ———")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


# ── 9. Score, and only then open the test split ──────────────────────────


@markdown
def _trained_eval_intro():
    """
    ## Measure the trained student — dev first, then test once

    Same questions, same harness, same judge; only the weights changed.

    Then, exactly once, the locked `test` split: questions written from
    passages that were never searched during training and never looked at
    while iterating. The dev margin tells you whether the loop worked; the
    test margin tells you whether the dev margin was real.
    """


@code
def _trained_eval():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    ENDPOINTS["trained"] = DeploymentConfig(
        model=Qwen3_4B(),
        checkpoint=checkpoint,
        app_name="learning-agent-student-trained",
        served_model_name="qwen3-4b-learned",
        unauthenticated=True,
    ).serve()

    trained_dev = dev_eval.evaluate(ENDPOINTS["trained"], max_concurrency=4)
    base_test = test_eval.evaluate(ENDPOINTS["base"], max_concurrency=4)
    trained_test = test_eval.evaluate(ENDPOINTS["trained"], max_concurrency=4)


@code
def _margin():
    def report(name: str, base_result, trained_result) -> None:
        base_scores = [row.score for row in base_result.rows]
        trained_scores = [row.score for row in trained_result.rows]
        base_lo, base_hi = bootstrap_ci95(base_scores)
        trained_lo, trained_hi = bootstrap_ci95(trained_scores)
        margin = trained_result.mean - base_result.mean
        print(f"[{name}] base    {base_result.mean:6.1%}  "
              f"95% CI [{base_lo:.1%}, {base_hi:.1%}]")
        print(f"[{name}] trained {trained_result.mean:6.1%}  "
              f"95% CI [{trained_lo:.1%}, {trained_hi:.1%}]")
        print(f"[{name}] margin  {margin:+.1%}")

    report("dev ", base_dev, trained_dev)
    report("test", base_test, trained_test)
    print(f"\ntrack={TRACK}  corpus=sha256:{corpus_pin[:16]}  "
          f"judge_votes={JUDGE_VOTES}")


# ── 10. Observability ────────────────────────────────────────────────────


@markdown
def _observability():
    """
    ## Reading the run on the dashboard

    A learning loop fails quietly. The reward can climb while the model learns
    nothing, the exam can be trivial, the rubric can be unhittable, the
    student can stop searching — none of that shows up in a final number,
    which is why LAB's own runs ship with an observatory. Here the gym
    dashboard plays that role, and every artifact above already reports to it.
    What to look at, in order:

    **Reward chart (Summary tab).** Is there a signal at all? Flat at zero
    means the rubric is unhittable; flat near the top means the exam is
    trivial. You want a curve that starts low and moves.

    **Score distribution and advantage charts.** GRPO learns from *spread
    within a group*. If all eight episodes for a question earn the same
    reward, the advantage is zero and the step teaches nothing however good
    the mean looks. Reward collapse is visible here several steps before the
    mean flatlines.

    **Custom metric charts.** Every numeric key the rollout and reward
    functions put on `sample.metadata` — `searches`, `opens`, `turns`,
    `answered`, `claim_coverage` — is charted per rollout automatically. This
    is the specific instrumentation that catches the failure mode of this
    tutorial: if `searches` decays toward zero while reward rises, the model
    has learned to skip the corpus and guess, and the dev margin will not
    survive the test split.

    **Rollouts tab.** `trajectory_messages` means each sample opens as the
    actual search conversation — the queries it tried, what came back, what it
    concluded. Read the episodes that scored highest. Keyword-spraying answers
    with high claim coverage and no content give themselves away here, and
    nowhere else.

    **Step & substep timeline.** Where the wall clock goes. A search rollout
    is many small generations, so *Generate rollouts* dominates; a growing
    share usually means episodes are running to `MAX_TURNS` instead of
    answering.

    **Evals tab.** Four evals, side by side: base and trained on dev, base and
    trained on test. Each row carries per-claim verdicts, search counts, the
    transcript, and the corpus pin, so a number can always be traced back to
    the passages and the corpus that produced it.

    **Deployments tab.** Teacher, base student, trained student — with URLs
    and status. The LAB run-record equivalent of "what was serving when this
    number was produced".
    """


@markdown
def _lab_map():
    """
    ## How this maps back to LAB

    | LAB | this tutorial |
    |---|---|
    | corpus at a pinned hash (`bench/pins.json`) | DSPy docs at a pinned tag + content tree-hash on every eval row |
    | `easy` / `medium` / `hard` tracks | the `TRACK` knob |
    | agent authors its own dev set | teacher writes questions + weighted rubrics |
    | `data_tool/pool/dedup_decontam` | `dedup()` + `decontaminate()` before any training |
    | held-out `test.json`, opened once | `test_items`, from disjoint passages, evaluated last |
    | `harness_tool/react_loop` (grep/read) | `search_answer()` + `search_generate()` |
    | reward function the agent writes | `claim_coverage_rm` |
    | n-vote rubric judge, majority per claim | `judge_claims(..., JUDGE_VOTES)` |
    | seeded bootstrap 95% CI | `bootstrap_ci95(..., seed=0)` |
    | margin over the untrained base | dev and test margins |
    | `runs/LEARNING_LOG.jsonl` | the dashboard's runs, evals, and deployments |
    | observatory viewer | the gym dashboard |
    | `toolbox/training_tool/slime` | `SlimeRecipe` + `TrainConfig.train()` |

    What LAB has that the gym does not, and this tutorial therefore drops:
    **GPU/CPU telemetry** (`system_monitor` samples), the **contestant
    agent's own trace** (the thinking/tool-use timeline of the agent driving
    the loop), the **workspace snapshot**, and the **contamination audit** of
    that agent's session. Those describe the agent running the experiment
    rather than the experiment, which is why LAB's observatory keeps them and
    the gym does not have a surface for them.
    """


@markdown
def _next_steps():
    """
    ## Next steps

    1. **Train the harness, not just the weights.** The protocol, the number
       of search hits, and the turn budget are all part of the policy. Sweep
       them with `TrainingGroup` and the dashboard will filter the runs
       together.
    2. **Iterate the exam.** Regenerate questions from the passages the
       trained student still fails, and train a second round from the
       checkpoint — the agent-driven version of curriculum design.
    3. **Judge the reward.** Sample a handful of rollouts per step and score
       them with the teacher offline, then plot judge-vs-coverage over
       training. The gap widening is the earliest reliable reward-hacking
       signal.
    4. **Swap the domain.** Anything the base model has not memorized works:
       internal docs, a private codebase, a legal corpus. The loop is
       unchanged; only `CORPUS_REPO` moves.
    5. **Run the agentic tasks.** LAB's `tau2_*` and `alfworld` tasks replace
       the judge with an environment verifier — a deterministic reward with no
       rubric at all. Start from
       [`002_multiturn`](/tutorials/rl/002_multiturn/) and
       [`000_agent_sandbox`](/tutorials/agent/000_agent_sandbox/).
    """
