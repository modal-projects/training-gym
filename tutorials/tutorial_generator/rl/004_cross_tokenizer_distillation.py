# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `004_cross_tokenizer_distillation` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 2×H200",
    "summary": "Cross-tokenizer distillation on function calling — DeepSeek V4 Flash teacher, Qwen3-4B student",
    "difficulty": "Advanced",
    "order": 40,
    "api_classes": [
        "Qwen3_4B",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "SlimeRecipe",
        "TrainConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Cross-tokenizer distillation: DeepSeek V4 Flash teaches Qwen3-4B

    In the previous tutorial, we learned how simple on-policy distllation (OPD) works between
    two models, Qwen3-8B and Qwen-4B, of the same base family, Qwen3. A larger teacher model, Qwen3-8B, distills
    it's belief for each token generated from a rollout response to the student model. We compared forward KL divergence versus
    reverse KL in the context of OPD and formulated a reward function for the student model that combines reverse KL with a math eval score (scalar).

    This tutorial shows how to do on-policy distillation between two different model families, DeepSeek-V4 and Qwen3. Since the tokenizers will not match between 
    model famililes, we will employ a cross-tokenization alignment algorithm called SimCT (https://arxiv.org/abs/2605.07711).
    This algorithm groups a sequence of mismatching tokens (span) and uses the mean of logprobs (geometric mean) as a single value covering the span (minmally aligned unit, or MAU).
    We will use the Top-K logprobs, including any MAUs, to increase the density of the reverse KL divergence signal by distilling high probability tokens from the teacher to student. 

    The data recipe uses Berkeley Function Calling Leaderboard (BFCL) toolcall data from the Gorilla (https://gorilla.cs.berkeley.edu/leaderboard.html) project at UC Berkeley.
    We evaluate our student model's rollout response by parsing the toolcall input as an Abstract Syntax Tree and assigning a non-binary reward based on the correct function definition. 
    This reward signal is incorporated into our student's reward function alongside reverse KL divergence.

    ### Steps

    1. Deploy DeepSeek V4 Flash as teacher on SGLang
    2. Load the BFCL dataset (1,000 Python function-calling examples)
    4. Evaluate the base student on function calling
    5. Define SimCT alignment helpers and cross-tokenizer reward function
    6. Train with GRPO + cross-tokenizer OPD + AST reward
    7. Evaluate the trained student and compare
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/rl/004_cross_tokenizer_distillation/004_cross_tokenizer_distillation.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import ast
    import json
    import re

    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        ModelDeployment,
        Qwen3_4B,
        SlimeRecipe,
        TrainConfig,
        list_checkpoints,
    )
    from modal_training_gym.common.models.base import HFModelConfiguration
    from modal_training_gym.deploy_recipes.sglang_recipe import (
        DeepSeek_V4_Flash_SglangRecipe,
    )


# ── Deploy teacher ───────────────────────────────────────────────────────


@markdown
def _deploy_intro():
    """
    ## Deploying DeepSeek-V4 Flash as the Teacher Model
    We can use Modal Training Gym's `DeepSeek_V4_Flash_SglangRecipe` to deploy our teacher model. 
    DeepSeek-V4 Flash is a 284B MoE model with 13B active parameters. We are using a 4:B200 GPU configuration on Modal.
    The SGLang engine serving this model will return top-k logprobs from next-token prediction of every token in the input sequence. 
    """


@code
def _deploy_teacher():
    teacher_deployment = DeploymentConfig(
        model=HFModelConfiguration(model_name="deepseek-ai/DeepSeek-V4-Flash"),
        recipe=DeepSeek_V4_Flash_SglangRecipe(),
        app_name="smoke-dsv4-flash",
        served_model_name="deepseek-v4-flash",
    ).serve()
    print(f"Teacher URL: {teacher_deployment.url}")

    TEACHER_GENERATE_URL = f"{teacher_deployment.url}/generate"


# ── Student model ────────────────────────────────────────────────────────


@code
def _student_model():
    base_model = Qwen3_4B()


# ── Dataset ──────────────────────────────────────────────────────────────


@markdown
def _dataset_intro():
    """
    ## Berkeley Function-Calling Leaderboard (BFCL) Dataset

    We borrow the BFCL dataset from UC Berkeley's Gorilla project (https://gorilla.cs.berkeley.edu/leaderboard.html).
    Our model receives the `prompt` column from the dataset, which contains a function definition and user prompt.
    The dataset's `label` column contains the "ground_truth" output for comparison with our student's output. 
    """


@code
def _dataset():
    from modal_training_gym.common.dataset import DatasetConfig

    BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    BFCL_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]

    def _build_prompt(question: str, funcs: list[dict]) -> str:
        sigs = []
        for f in funcs:
            props = f.get("parameters", {}).get("properties", {})
            req = set(f.get("parameters", {}).get("required", []))
            params = ", ".join(f"{p}: {v.get('type','any')}" + ("" if p in req else "?") for p, v in props.items())
            sigs.append(f"  {f['name']}({params}) — {f.get('description','')}")
        return f"Functions:\n" + "\n".join(sigs) + f"\n\nUser: {question}\n\nCall the function(s): [func(param=value)]. Output ONLY the call list."

    class BFCLDataset(DatasetConfig):
        input_column = "prompt"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = False

        def __init__(self, split="train", eval_per_cat=25):
            self.categories = BFCL_CATEGORIES
            self._split = split
            self._eval_per_cat = eval_per_cat
            self.input_key = "prompt"
            self.label_key = "label"

        def _load_rows(self):
            from huggingface_hub import hf_hub_download as dl
            rows = []
            for cat in self.categories:
                pf = dl(repo_id=BFCL_REPO, filename=f"BFCL_v3_{cat}.json", repo_type="dataset")
                af = dl(repo_id=BFCL_REPO, filename=f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")
                cat_rows = []
                with open(pf) as fp, open(af) as fa:
                    for pl, al in zip(fp, fa):
                        pd, ad = json.loads(pl), json.loads(al)
                        cat_rows.append({
                            "prompt": _build_prompt(pd["question"][0][0]["content"], pd["function"]),
                            "label": json.dumps({"ground_truth": ad["ground_truth"], "function": pd["function"], "test_category": cat}),
                        })
                if self._split == "eval":
                    rows.extend(cat_rows[-self._eval_per_cat:])
                else:
                    rows.extend(cat_rows[:-self._eval_per_cat])
            return rows

        def load(self):
            return self._load_rows()

        def prepare(self, path: str, eval_paths: dict = None):
            import os
            rows = self._load_rows()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)

    train_dataset = BFCLDataset(split="train", eval_per_cat=25)
    eval_dataset = BFCLDataset(split="eval", eval_per_cat=25)


# ── AST evaluation ──────────────────────────────────────────────────────


@markdown
def _ast_intro():
    """
    ## AST evaluation with partial scoring

    We parse each model response into an AST and compare it against
    the ground truth. Rather than binary pass/fail, the score is
    additive across three components so that "almost right" responses
    still give GRPO useful gradient signal:

    | Component               | Weight | Criterion                             |
    |-------------------------|--------|---------------------------------------|
    | Function name correct   |  +0.3  | Called the right function              |
    | Parameter names correct |  +0.3  | All required params present, no extras |
    | Parameter values correct|  +0.4  | Values match accepted answers          |

    A perfect call scores 1.0. Getting the function name right but
    nothing else scores 0.3. For parallel categories, per-call scores
    are averaged.
    """


@code
def _ast_eval():
    _NORM = re.compile(r"[ ,./\-_*^]")

    def _val_match(val, possible, type_str):
        """Check if val matches any entry in possible (BFCL ground truth list)."""
        for pv in possible:
            if pv == "":
                continue
            if isinstance(val, str) and isinstance(pv, str):
                if _NORM.sub("", val).lower() == _NORM.sub("", pv).lower():
                    return True
            elif val == pv:
                return True
        return False

    def _score_call(func_descs, call, gt):
        """Score a single call: +0.3 name, +0.3 params, +0.4 values."""
        gt_name, gt_params = next(iter(gt.items()))
        call_name, call_params = next(iter(call.items()))

        if call_name != gt_name:
            return 0.0

        score = 0.3  # function name correct

        desc = next((f for f in func_descs if f["name"] == gt_name), {})
        props = desc.get("parameters", {}).get("properties", {})
        required = set(desc.get("parameters", {}).get("required", []))

        params_ok = True
        for p in required:
            if p not in call_params and "" not in gt_params.get(p, []):
                params_ok = False
                break
        if params_ok:
            for p in call_params:
                if p not in gt_params and p not in props:
                    params_ok = False
                    break

        if not params_ok:
            return score

        score += 0.3  # parameter names correct

        for p, v in call_params.items():
            if p in gt_params and not _val_match(v, gt_params[p], props.get(p, {}).get("type", "any")):
                return score

        score += 0.4  # parameter values correct
        return score

    def bfcl_partial_score(func_descs, model_output, ground_truth, category):
        """Additive score: +0.3 name, +0.3 params, +0.4 values per call."""
        if not model_output:
            return 0.0
        if "parallel" in category:
            if not ground_truth:
                return 0.0
            used = set()
            call_scores = []
            for gt_entry in ground_truth:
                best_score, best_idx = 0.0, None
                for i in range(len(model_output)):
                    if i in used:
                        continue
                    s = _score_call(func_descs, model_output[i], gt_entry)
                    if s > best_score:
                        best_score, best_idx = s, i
                    if s == 1.0:
                        break
                if best_idx is not None:
                    used.add(best_idx)
                call_scores.append(best_score)
            return sum(call_scores) / len(call_scores)
        if len(model_output) != 1:
            return 0.0
        return _score_call(func_descs, model_output[0], ground_truth[0])

    def parse_function_calls(text: str) -> list[dict]:
        """Parse `[func(a=1, b=2)]` → [{func: {a: 1, b: 2}}]."""
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = text.strip().strip("`\n ")
        if not text.startswith("["):
            text = "[" + text
        if not text.endswith("]"):
            text = text + "]"
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError:
            return []
        elems = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body] if isinstance(tree.body, ast.Call) else []
        results = []
        for elem in elems:
            if not isinstance(elem, ast.Call):
                continue
            parts, node = [], elem.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            def _safe_eval(n):
                try:
                    return ast.literal_eval(n)
                except (ValueError, SyntaxError):
                    return ast.unparse(n)
            results.append({".".join(reversed(parts)): {kw.arg: _safe_eval(kw.value) for kw in elem.keywords}})
        return results

    def bfcl_eval_fn(deployment: ModelDeployment, example: dict) -> EvalRowResult:
        label = json.loads(example.get("label", "{}"))
        response = deployment.generate(example.get("prompt", ""), ensure_ready=False)
        parsed = parse_function_calls(response)
        score = bfcl_partial_score(
            label.get("function", []), parsed,
            label.get("ground_truth", []),
            label.get("test_category", "simple"),
        )
        return EvalRowResult(
            score=score, response=response,
            metadata={
                "correct": score == 1.0,
                "category": label.get("test_category", ""),
                "parsed_calls": len(parsed),
            },
        )


# ── Evaluate base student ───────────────────────────────────────────────


@markdown
def _eval_base_intro():
    """
    ## Baseline eval

    Let's see how the base model performs on basic function calling. Both the training split and evaluation split
    draw from 1000 examples in the "simple", "multiple", "parallel", and "parallel_multiple" categories of BFCL. Eval
    uses 100 examples total while training uses 900. 
    """


@code
def _eval_base():
    base_deployment = DeploymentConfig(model=base_model).serve()
    print(f"Student URL: {base_deployment.url}")

    eval_config = EvalConfig(dataset=eval_dataset, eval_fn=bfcl_eval_fn)
    print("--- Evaluating base student... ---")
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    n_correct = sum(1 for r in base_eval.rows if r.metadata.get("correct"))
    print(f"Base accuracy: {n_correct}/{len(base_eval.rows)} "
          f"({base_eval.mean:.1%})")


@notebook_only
@code
def _base_examples():
    for r in base_eval.rows[:5]:
        status = "PASS" if r.metadata["correct"] else "FAIL"
        cat = r.metadata["category"]
        print(f"[{status}] ({cat}) calls={r.metadata['parsed_calls']} "
              f"err={r.metadata['error']}")
        print(f"  ...{r.response[-120:]}")
        print()


# ── SimCT cross-tokenizer alignment ─────────────────────────────────────


@markdown
def _simct_explainer():
    """
    ## Cross-tokenizer alignment via SimCT Minimally Aligned Units (MTUs)

    To solve the problem of cross-tokenizer misalignment, where two tokenizers may not share the same token vocabularies or
    may merge tokens differently, we employ an algorithm called SimCT (https://arxiv.org/abs/2605.07711) that averages the logprobs
    of subtokens in a misaligned sequence (span) and includes this single scalar in the KL divergence calculation. 

    Here's an example where SimCT would create a Minimally Aligned Unit (MAU) for normalizing both model's different vocabularies:
    For the sentence "I am happy", the student model tokenizes it as ["I", "am", "ha", "pp", "y"] and the teacher model tokenizes it as 
    ["I", "am", "hap", "py"]. The "I" and "am" tokens match up perfectly, but the word "happy" is split across multiple tokens.
    
    SimCT loops through the tokens. When it comes across two tokens in a sequence that are not the same (in this case "hap" and "ha"),
    it marks the start of a span. The greedy algorithm loops through until it finds tokens that match up and form the culmulative preceding string ("happy"). 
    In this case, the MAU would be "happy" and span the tokens ["ha", "pp", "y"] for model 1 and ["hap", "py"] for model 2. The algorithm
    computes the mean of the logprobs (geometric mean) and uses that as the logprob for the MAU. This MAU logprob is included in the KL divergence
    calculation over the top-k tokens at each position.
    """


@code
def _mau_helpers():
    def align_cross_tokenizer(
        teacher_token_texts: list[str],
        teacher_logprobs: list[float],
        teacher_top_logprobs: list,
        student_token_texts: list[str],
    ):
        """SimCT alignment with top-k entropy weighting.

        For each MAU, uses the teacher's top-k distribution to weight the
        logprob contribution: confident positions (peaked distribution)
        contribute more to the KL signal, uncertain positions less.
        """
        import torch
        import math

        def _offsets(texts):
            out, pos = [], 0
            for t in texts:
                out.append((pos, pos + len(t)))
                pos += len(t)
            return out

        def _confidence(top_k_entry):
            """1 - normalized_entropy: high when teacher is confident."""
            if not top_k_entry:
                return 1.0
            lps = [e[0] for e in top_k_entry if e and e[0] is not None]
            if not lps:
                return 1.0
            max_lp = max(lps)
            probs = [math.exp(lp - max_lp) for lp in lps]
            total = sum(probs)
            probs = [p / total for p in probs]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_ent = math.log(len(probs) + 1e-10)
            return 1.0 - (entropy / max_ent) if max_ent > 0 else 1.0

        t_off = _offsets(teacher_token_texts)
        s_off = _offsets(student_token_texts)
        t_bounds = {s for s, e in t_off} | {e for s, e in t_off}
        s_bounds = {s for s, e in s_off} | {e for s, e in s_off}
        shared = sorted(t_bounds & s_bounds)

        result = torch.zeros(len(student_token_texts), dtype=torch.float32)
        for i in range(len(shared) - 1):
            lo, hi = shared[i], shared[i + 1]
            t_idx = [j for j, (s, e) in enumerate(t_off) if s >= lo and e <= hi]
            s_idx = [j for j, (s, e) in enumerate(s_off) if s >= lo and e <= hi]
            if t_idx and s_idx:
                weights = [_confidence(teacher_top_logprobs[j] if j < len(teacher_top_logprobs) else None) for j in t_idx]
                total_w = sum(weights)
                if total_w > 0:
                    mau_lp = sum(weights[k] * teacher_logprobs[j] for k, j in enumerate(t_idx)) / total_w
                else:
                    mau_lp = sum(teacher_logprobs[j] for j in t_idx) / len(t_idx)
                for j in s_idx:
                    result[j] = mau_lp
        return result


# ── Cross-tokenizer reward function ─────────────────────────────────────


@markdown
def _rm_intro():
    """
    ## Reward function

    The reward function factors in the reverse KL divergence from the top-k logprobs and the AST function call result score.
    By including a reward aligned with our target task, we can help "reward-tilt" the student model 
    towards the reward-weighted verison of the teacher's distribution rather than the raw distribution. The following blog shows a helpful 
    visualization of reward-tilting a student model: https://emilianopp.github.io/Privileged-Information-Distillation-and-Self-Distillation/.
    """


@code
def _rm():
    _student_tokenizer_cache = {}

    def _get_student_tokenizer():
        if "tok" not in _student_tokenizer_cache:
            from transformers import AutoTokenizer
            _student_tokenizer_cache["tok"] = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-4B", trust_remote_code=True
            )
        return _student_tokenizer_cache["tok"]

    async def cross_tokenizer_reward(args, sample, **kwargs):
        """Collect teacher log-probs by sending decoded text to the teacher."""
        import aiohttp
        import asyncio

        tokenizer = _get_student_tokenizer()
        full_text = tokenizer.decode(sample.tokens, skip_special_tokens=False)

        payload = {
            "text": full_text,
            "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 5,
            "return_text_in_logprobs": True,
        }
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(args.rm_url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)

    def cross_tokenizer_post_process(args, samples, **kwargs):
        """Build MAU-aligned teacher_log_probs and compute AST rewards."""
        tokenizer = _get_student_tokenizer()
        raw_rewards = [s.get_reward_value(args) for s in samples]

        ast_rewards = []
        for sample in samples:
            label = json.loads(getattr(sample, "label", "{}") or "{}")
            parsed = parse_function_calls(sample.response)
            ast_rewards.append(bfcl_partial_score(
                label.get("function", []), parsed,
                label.get("ground_truth", []),
                label.get("test_category", "simple"),
            ))

        for sample, reward in zip(samples, raw_rewards):
            meta = reward.get("meta_info", {})
            entries = meta.get("input_token_logprobs", [])
            top_entries = meta.get("input_top_logprobs", [])

            t_lps = [e[0] if e[0] is not None else 0.0 for e in entries if e is not None]
            t_texts = [e[2] if len(e) > 2 else "" for e in entries if e is not None]
            t_top = [top_entries[i] if i < len(top_entries) else None for i, e in enumerate(entries) if e is not None]

            s_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in sample.tokens]
            aligned = align_cross_tokenizer(t_texts, t_lps, t_top, s_texts)
            sample.teacher_log_probs = aligned[-sample.response_length:]

        return ast_rewards, ast_rewards


# ── Training ─────────────────────────────────────────────────────────────


@markdown
def _train_intro():
    """
    ## Training

    We train with GRPO + cross-tokenizer OPD on 2×H200 GPUs with
    tensor parallelism (TP=2). The actor and rollout engine are
    colocated, where slime swaps optimizer states to CPU during rollout
    and model weights during training.

    Again, we train on 900 examples from the afore-mentioned training/eval BFCL split. 
    """


@code
def _train():
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            custom_rm_function=cross_tokenizer_reward,
            image_overlay=lambda img: img.pip_install("modal"),

            gpu_type="H200",
            colocate=True,
            actor_num_gpus_per_node=2,
            rollout_num_gpus=2,
            tensor_model_parallel_size=2,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=2,

            num_rollout=20,
            rollout_batch_size=16,
            n_samples_per_prompt=4,
            rollout_max_response_len=512,
            rollout_temperature=0.7,

            global_batch_size=16,
            lr=1e-6,
            save_interval=5,

            environment={
                "PYTHONPATH": "/root/Megatron-LM/:/root",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
            },

            extra_config={
                "use_opd": True,
                "opd_type": "sglang",
                "opd_kl_coef": 1.0,
                "custom_reward_post_process_path": (
                    "004_cross_tokenizer_distillation"
                    ".cross_tokenizer_post_process"
                ),
                "rm_url": TEACHER_GENERATE_URL,
            },
        ),
    )

    print("--- Starting cross-tokenizer OPD training... ---")
    print(f"  Teacher: DeepSeek V4 Flash")
    print(f"  Student: Qwen3-4B")
    print(f"  Dataset: BFCL (800 function-calling examples)")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")
    print("--- Training complete ---")


# ── Evaluate trained ─────────────────────────────────────────────────────


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained student

    Deploy the last checkpoint and run the BFCL evaluation split of 100 examples. 
    """


@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=Qwen3_4B(),
        checkpoint=checkpoint,
        app_name="qwen3-4b-cross-tok-trained",
        served_model_name="qwen3-4b-fc-trained",
    ).serve()
    print(f"Trained student URL: {trained_deployment.url}")

    print("--- Evaluating trained student... ---")
    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    n_correct = sum(1 for r in trained_eval.rows if r.metadata.get("correct"))
    print(f"Trained accuracy: {n_correct}/{len(trained_eval.rows)} "
          f"({trained_eval.mean:.1%})")


@notebook_only
@code
def _trained_examples():
    for base_r, trained_r in zip(base_eval.rows[:5], trained_eval.rows[:5]):
        b_status = "PASS" if base_r.metadata["correct"] else "FAIL"
        t_status = "PASS" if trained_r.metadata["correct"] else "FAIL"
        cat = base_r.metadata["category"]
        print(f"({cat}) Base=[{b_status}] Trained=[{t_status}]")
        if not base_r.metadata["correct"] and trained_r.metadata["correct"]:
            print(f"  Fixed! {trained_r.response[-100:]}")
        print()



@code
def _compare():
    from collections import Counter

    base_by_cat = Counter()
    trained_by_cat = Counter()
    total_by_cat = Counter()

    for b_r, t_r in zip(base_eval.rows, trained_eval.rows):
        cat = b_r.metadata["category"]
        total_by_cat[cat] += 1
        if b_r.metadata["correct"]:
            base_by_cat[cat] += 1
        if t_r.metadata["correct"]:
            trained_by_cat[cat] += 1

    print(f"{'Category':<25} {'Base':>8} {'Trained':>8} {'Delta':>8}")
    print("-" * 51)
    for cat in sorted(total_by_cat):
        total = total_by_cat[cat]
        b = base_by_cat[cat] / total
        t = trained_by_cat[cat] / total
        print(f"{cat:<25} {b:>7.1%} {t:>7.1%} {t - b:>+7.1%}")
    print("-" * 51)
    total = len(base_eval.rows)
    print(f"{'Overall':<25} {base_eval.mean:>7.1%} "
          f"{trained_eval.mean:>7.1%} "
          f"{trained_eval.mean - base_eval.mean:>+7.1%}")


@markdown
def _next_steps():
    """
    ## Next steps

    Ways to improve this tutorial:

    1. **LLM-as-a-judge**: The Gorilla project supports LLM-as-a-judge evaluation for multi-turn, multi-tool call scenarios.
    You could stress-test the model by having it produce multiple-tool calls across an agentic trajectory and evaluate it
    with the Gorilla OpenFunctions-V2 model.
    2. **Different student/teacher model configuration**: Test alternative teacher models, such as the bigger DeepSeek-V4 Pro model with 1.6T parameters.
    Try out bigger student models from the Qwen3 lineup or use one of Google's Gemma 4 open-weight models. 
    3. **Top-k Logprob Ablation Test**: Try ablating the top-k returned logprobs to just one for the student's output token, observing the difference
    in eval score from our default of `5`. By including more token probabilities, you may push the student model towards high-probability teacher modes
    instead of solely penalizing models valued by the student.
    """
