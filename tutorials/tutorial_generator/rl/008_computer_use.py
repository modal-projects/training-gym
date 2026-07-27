# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `008_computer_use` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "GUI grounding with Qwen3-VL-8B — predict click coordinates from screenshots",
    "difficulty": "Advanced",
    "order": 45,
    "api_classes": [
        "Qwen3_VL_8B",
        "Qwen3_VL_8b_Recipe",
        "MultimodalDataset",
        "DeploymentConfig",
        "EvalConfig",
        "ModelDeployment",
        "TrainConfig",
        "WandbConfig",
        "list_checkpoints",
    ],
    "required_modal_secrets": [
        {"name": "wandb-secret", "key": "WANDB_API_KEY"},
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # GUI Grounding with Qwen3-VL-8B

    This tutorial trains **Qwen3-VL-8B-Instruct** via GRPO to predict click
    coordinates given a screenshot and a natural-language instruction like
    "click the Submit button".

    The task is simple: given an image of a GUI and an instruction identifying
    a UI element, output the normalized `(x, y)` center coordinate of that
    element. This is a foundational capability for computer-use agents.

    We use the [ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot)
    benchmark — a standard GUI grounding evaluation set covering iOS, Android,
    macOS, Windows, and Web screenshots with annotated bounding boxes.

    The reward is bbox-aware: a click that lands anywhere inside the target
    element scores +1 (a real click succeeds), and predictions that miss decay
    toward −1 over a margin scaled to the element's own size.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run locally (your machine drives the Modal GPU workers):

    ```
    cd training-gym
    uv sync
    uv run python tutorials/rl/008_computer_use/008_computer_use.py
    ```

    To detach and watch it from the Modal dashboard instead:

    ```
    uv run modal run -d tutorials/rl/008_computer_use/008_computer_use.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import re

    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        ImageEvalRowResult,
        ModelDeployment,
        MultimodalDataset,
        Qwen3_VL_8B,
        Qwen3_VL_8b_Recipe,
        TrainConfig,
        WandbConfig,
        list_checkpoints,
    )


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use [rootsautomation/ScreenSpot](https://huggingface.co/datasets/rootsautomation/ScreenSpot)
    — ~1,200 GUI screenshots annotated with natural-language instructions and
    bounding boxes. Each row has:

    - `image` — a screenshot from iOS/Android/macOS/Windows/Web
    - `instruction` — e.g. "click the Submit button"
    - `bbox` — `[left, top, right, bottom]` in normalized [0, 1] coordinates

    We keep the full bounding box as the training target (a click anywhere
    inside it counts as a hit) and ask the model to output a single `(x, y)`
    click point.

    For this tutorial we train on 800 samples and hold out 200 for evaluation.
    """


@code
def _dataset():
    # The "<image>" marker is where slime interleaves the screenshot column.
    GROUNDING_PROMPT = (
        "<image>\n"
        "You are a GUI agent. Given the screenshot, click on the element "
        "described below.\n\n"
        "Instruction: {instruction}\n\n"
        "Respond with ONLY the normalized (x, y) coordinates of the click "
        "target, formatted as: (x, y)\n"
        "where x and y are decimals between 0 and 1 representing the "
        "horizontal and vertical position on the screen."
    )

    class ScreenSpotDataset(MultimodalDataset):
        """GUI grounding dataset from ScreenSpot."""

        modality = "image"
        hf_repo = "rootsautomation/ScreenSpot"
        hf_split = "test"
        n_rows = 800
        row_offset = 0
        always_prepare = True
        # Collapse to one chat-templated string; the VL processor crashes on raw
        # message lists.
        apply_chat_template = True

        def __init__(self, **kwargs):
            super().__init__(rows=[], **kwargs)

        def _build_rows(self) -> list[dict]:
            import base64
            import io

            from datasets import load_dataset

            ds = load_dataset(self.hf_repo, split=self.hf_split)
            start = min(self.row_offset, len(ds))
            stop = min(start + self.n_rows, len(ds))
            # Demo-scale: inline base64 rows in memory; stream large corpora.
            rows = []
            for row in ds.select(range(start, stop)):
                left, top, right, bottom = row["bbox"]
                instruction = row["instruction"]

                buf = io.BytesIO()
                row["image"].save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                data_uri = f"data:image/png;base64,{img_b64}"

                rows.append(
                    {
                        self.input_key: GROUNDING_PROMPT.format(
                            instruction=instruction
                        ),
                        self.media_column: [data_uri],
                        self.label_key: (
                            f"{left:.4f},{top:.4f},{right:.4f},{bottom:.4f}"
                        ),
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


@code
def _make_datasets():
    train_dataset = ScreenSpotDataset(n_rows=800)
    eval_dataset = ScreenSpotDataset(n_rows=200, row_offset=800)


@notebook_only
@code
def _dataset_peek():
    rows = eval_dataset.load()
    for row in rows[:2]:
        print(f"prompt: {row['prompt'][:100]}...")
        print(f"  label (left, top, right, bottom): {row['label']}")
        print()


@markdown
def _reward_intro():
    """
    ## Reward function

    Rather than measuring distance to the box's *center*, we reward whether the
    click would actually land on the element. Let `outside` be the Euclidean
    distance from the predicted point to the bounding box (`0` when the point is
    inside it), and `margin = max(diagonal_of_box, 0.05)`:

    ```text
    R = +1.0                          if outside == 0   (click inside element)
      = 1.0 - 2.0 * outside / margin  if 0 < outside < margin
      = -1.0                          if outside >= margin
    ```

    This fixes two problems with a center-distance reward:

    - **Click success is rewarded directly.** Any point inside the element gets
      the full +1, even if it's far from the geometric center — exactly like a
      real click.
    - **Tolerance scales with element size.** A fixed 5%-of-screen threshold is
      too lenient on tiny icons (you can miss and still score) and too harsh on
      big buttons (a valid click gets penalized). Scaling the falloff to the
      element's own diagonal (with a 5% floor for tiny targets) avoids both.

    The model also gets −1 if it fails to output parseable coordinates.
    """


@code
def _reward():
    def _parse_coordinates(text: str) -> tuple[float, float] | None:
        """Extract (x, y) from model output like '(0.45, 0.32)' or '0.45, 0.32'."""
        nums = re.findall(r"([\d.]+)", text)
        if len(nums) < 2:
            return None
        try:
            x, y = float(nums[0]), float(nums[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                return (x, y)
        except (ValueError, IndexError):
            pass
        return None

    def _parse_bbox(label: str) -> tuple[float, float, float, float]:
        """Parse a 'left,top,right,bottom' label into floats."""
        left, top, right, bottom = (float(v) for v in label.split(","))
        return left, top, right, bottom

    def _distance_outside_box(
        x: float, y: float, box: tuple[float, float, float, float]
    ) -> float:
        """Euclidean distance from (x, y) to the bbox; 0.0 when inside it."""
        left, top, right, bottom = box
        dx = max(left - x, 0.0, x - right)
        dy = max(top - y, 0.0, y - bottom)
        return (dx * dx + dy * dy) ** 0.5

    async def grounding_reward(args, sample, **kwargs) -> float:
        response = getattr(sample, "response", "") or ""
        label = getattr(sample, "label", "") or ""

        pred = _parse_coordinates(response)
        if pred is None:
            return -1.0

        box = _parse_bbox(label)
        left, top, right, bottom = box

        # Any click inside the element succeeds → full reward.
        outside = _distance_outside_box(pred[0], pred[1], box)
        if outside == 0.0:
            return 1.0

        # Outside: decay from +1 at the edge to −1 a full diagonal away, with a
        # floor so tiny targets keep a usable gradient.
        diag = ((right - left) ** 2 + (bottom - top) ** 2) ** 0.5
        margin = max(diag, 0.05)
        if outside >= margin:
            return -1.0
        return 1.0 - 2.0 * outside / margin


@markdown
def _eval_base_intro():
    """
    ## Baseline Eval

    Let's evaluate the base Qwen3-VL-8B model on our held-out set before
    training to see how well it grounds UI elements out of the box.

    Returning an `ImageEvalRowResult` folds a thumbnail of the screenshot into the row.
    """


@code
def _eval_helpers():
    def _thumbnail(data_uri: str, max_dim: int = 512) -> str:
        # Downscale the screenshot to a dashboard-sized thumbnail so the eval
        # summary stays small (we score on the full-res image, below).
        import base64
        import io

        from PIL import Image

        _, _, b64 = data_uri.partition(",")
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def grounding_eval_fn(
        deployment: ModelDeployment, example: dict
    ) -> ImageEvalRowResult:
        # Eval sends the screenshot as a separate image_url, so drop the marker.
        prompt = example.get("prompt", "").replace("<image>", "").strip()
        label = example.get("label", "")
        images = example.get("images", [])

        # Build the OpenAI chat content: text + one image_url part per screenshot.
        content = [
            {"type": "text", "text": prompt},
            *({"type": "image_url", "image_url": {"url": img}} for img in images),
        ]
        response = deployment.generate(content, ensure_ready=False)

        pred = _parse_coordinates(response)
        box = _parse_bbox(label)
        if pred is None:
            inside = False
            outside = 1.0
        else:
            outside = _distance_outside_box(pred[0], pred[1], box)
            inside = outside == 0.0

        return ImageEvalRowResult(
            score=1.0 if inside else 0.0,
            response=response,
            prompt=prompt,
            image=_thumbnail(images[0]) if images else None,
            metadata={
                "inside_box": inside,
                "dist_outside": round(outside, 4),
                "pred": f"{pred[0]:.4f},{pred[1]:.4f}" if pred else "PARSE_FAIL",
                "label": label,
            },
        )


@code
def _eval_base():
    base_model = Qwen3_VL_8B()
    base_deployment = DeploymentConfig(
        model=base_model,
        unauthenticated=True,
    ).serve()
    print(f"Base model URL: {base_deployment.url}")

    eval_config = EvalConfig(dataset=eval_dataset, eval_fn=grounding_eval_fn)
    print("--- Evaluating base model... ---")
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    n_hits = sum(1 for r in base_eval.rows if r.metadata.get("inside_box"))
    print(
        f"Base accuracy (clicks inside element): "
        f"{n_hits}/{len(base_eval.rows)} ({base_eval.mean:.1%})"
    )


@notebook_only
@code
def _base_examples():
    for r in base_eval.rows[:3]:
        status = "HIT" if r.metadata["inside_box"] else "MISS"
        print(f"[{status}] label={r.metadata['label']}, pred={r.metadata['pred']}")
        print(f"  dist_outside={r.metadata['dist_outside']:.4f}")
        print(f"  ...{r.response[-100:]}")
        print()


@markdown
def _train_intro():
    """
    ## Training

    We use `Qwen3_VL_8b_Recipe` which carries VL-specific defaults:
    - **Frozen vision tower** (`freeze_params_name_list=["vision_model"]`) — RL
      only updates the language backbone. This is the standard recipe for VLM RL:
      a single sparse reward is too noisy to safely fine-tune a pretrained visual
      encoder (you'd risk collapsing its features), and grounding is really about
      teaching the decoder to *read out* coordinates from features the ViT already
      provides. It's also cheaper — no optimizer state or backward pass for the ViT.
    - Padded (bshd) batches for the vision encoder
    - TP=4 for the 8B model across 8 H100s
    - Short response cap (64 tokens — coordinates are brief)
    - A high SGLang KV-cache fraction (0.75) for fast colocated rollouts

    The recipe overrides below are tuned to speed up training (~30m → ~19m on
    one 8×H100 node) while preserving the reward curve, uniquely fitted to this
    tutorial's short coordinate outputs. Short outputs let us increase rollout
    concurrency, which sets the memory budget, which sets the shard count.

    This tutorial runs 15 rollouts as a quick demo. For a more meaningful
    accuracy gain, increase `num_rollout`.

    We pass `wandb=WandbConfig(project="…")` so reward/KL/length curves stream to
    Weights & Biases — the key comes from the `wandb-secret` Modal secret. The
    Training Gym dashboard picks up the run's project/entity/id and wires up the
    **Open in W&B** button on the run. Drop `wandb=` to disable logging.
    """


@code
def _train():
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=Qwen3_VL_8b_Recipe(
            # TP=4 shards the 8B weights across 4 GPUs, freeing enough VRAM per
            # GPU for the large 0.75 KV pool below. (TP=2 OOMs at mem=0.75.)
            tensor_model_parallel_size=4,
            custom_rm_function=grounding_reward,
            num_rollout=15,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
            # We only need 64 tokens here because the model just outputs coordinates.
            rollout_max_response_len=64,
            # Give SGLang 75% of VRAM for its KV cache so it can run more
            # rollouts concurrently, which is feasible because TP=4 frees the VRAM.
            sglang_mem_fraction_static=0.75,
            global_batch_size=16,
            lr=1e-6,
            save_interval=15,
            # Skip writing optimizer state to the checkpoint since we only serve the
            # final weights for eval (not resuming training).
            no_save_optim=True,
            wandb=WandbConfig(project="computer-use-grounding"),
        ),
    )
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _eval_trained_intro():
    """
    ## Evaluate the trained model

    Let's run the same eval on the trained checkpoint and compare accuracy.
    """


@code
def _eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(f"Checkpoint: {checkpoint.path}")

    trained_deployment = DeploymentConfig(
        model=Qwen3_VL_8B(),
        checkpoint=checkpoint,
        app_name="qwen3-vl-8b-grounding-serve",
        served_model_name="qwen3-vl-8b-grounding",
        unauthenticated=True,
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")

    print("--- Evaluating trained model... ---")
    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    n_hits = sum(1 for r in trained_eval.rows if r.metadata.get("inside_box"))
    print(
        f"Trained accuracy (clicks inside element): "
        f"{n_hits}/{len(trained_eval.rows)} ({trained_eval.mean:.1%})"
    )


@notebook_only
@code
def _trained_examples():
    for base_r, trained_r in zip(base_eval.rows[:3], trained_eval.rows[:3]):
        label = base_r.metadata["label"]
        b_status = "HIT" if base_r.metadata["inside_box"] else "MISS"
        t_status = "HIT" if trained_r.metadata["inside_box"] else "MISS"
        print(f"label={label}")
        print(
            f"  Base:    [{b_status}] pred={base_r.metadata['pred']} dist_outside={base_r.metadata['dist_outside']:.4f}"
        )
        print(
            f"  Trained: [{t_status}] pred={trained_r.metadata['pred']} dist_outside={trained_r.metadata['dist_outside']:.4f}"
        )
        print()


@markdown
def _compare_intro():
    """
    ## Results

    Let's compare base vs trained accuracy.
    """


@code
def _compare():
    base_hits = sum(1 for r in base_eval.rows if r.metadata.get("inside_box"))
    trained_hits = sum(1 for r in trained_eval.rows if r.metadata.get("inside_box"))
    total = len(base_eval.rows)
    print(f"Base model:    {base_hits}/{total} ({base_eval.mean:.1%})")
    print(f"Trained model: {trained_hits}/{total} ({trained_eval.mean:.1%})")
    print(f"Delta:         {trained_eval.mean - base_eval.mean:+.1%}")
