# ---
# order: 11
# deps: pillow
# ---
#
# # Painting flowers with code
#
# As [this tweet](https://x.com/kickingkeys/status/2091570990048276897) and
# [blog post](https://surya.website/rling-qwen-to-paint-with-code) show,
# you can easily train [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to write
# [p5.brush](https://p5brush.org) watercolour sketches of flowers.
# For each rollout, sketches are rendered to PNGs in a
# [Modal Sandbox](https://modal.com/docs/guide/sandboxes), and
# [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) compares each sketch
# to reference images.
#
# In the original blog post, they hand-curate the reference pool. Here, we
# generate them using:
# 
# `modal run tutorials/paint_flowers/build_pool.py`

import asyncio
import itertools
import random

import helpers
from modal_training_gym import (
    DatasetConfig,
    Endpoint,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    Qwen3_6_27B,
    TrainConfig,
)
from modal_training_gym.common.sample_extraction import IMAGE_SAMPLE_LIMIT_ENV

base_model = Qwen3_5_4B()

# ## Get the dataset
#
# To create our dataset, we seed one using the present species and palettes in the reference pool.

SPECIES = [
    "hibiscus",
    "poppy",
    "cosmos",
    "peony",
    "tulip",
    "magnolia",
    "dahlia",
    "iris",
]
PALETTES = [
    "peach",
    "crimson",
    "butter",
    "lilac",
    "coral",
    "indigo",
    "blush",
    "amber",
]

SYSTEM_PROMPT = """\
You write p5.js sketches that paint a single watercolour flower using p5.brush.

Rules:
- Reply with one ```javascript code fence containing a complete sketch, and nothing else.
- Define exactly one function: `function setup() { ... }`. Never define draw().
- Start setup() with: createCanvas(512, 512, WEBGL); angleMode(DEGREES); brush.load();
- End setup() with: noLoop();
- (0,0) is the CENTRE of the canvas; x and y run from -256 to 256. Compose around (0,0).
- The only brush calls that exist are: brush.fill(colour, alpha), brush.noFill(), brush.stroke(colour), brush.noStroke(), brush.strokeWeight(w), brush.bleed(amount), brush.pick(name), brush.polygon([[x,y],...]), brush.circle(x,y,radius), brush.line(x1,y1,x2,y2). Any other brush.* call is a hallucination and will be dropped.
- brush.pick names: "pen", "2B", "HB", "cpencil", "charcoal", "marker". Never "spray".
- Every colour is a quoted hex string, e.g. brush.fill("#e2725b", 100). A bare number like brush.fill(30, 100) is GREYSCALE and scores zero. Pick 4-6 hex colours from the requested palette before you paint and use only those.
- The third argument of brush.circle is a RADIUS, not a diameter.
- brush.strokeWeight is 1-4, always. A weight above 6 paints a black mass over the flower and scores zero. Outlines are thin; volume comes from fills, not from fat strokes.
- The paper is painted with p5's background("#hex") as the very first call after brush.load(). brush.fill() does not paint a background.
- p5's own background(), color(), lerpColor(), random(), sin(), cos() and for-loops are all available. p5 transforms (translate/rotate) do NOT reach the brush layer: compute every vertex in absolute canvas coordinates.
- Watercolour is built by repetition: paint each shape 4-8 times in a loop with brush.bleed(0.1-0.3) and brush.fill(colour, 80-120), jittering position, angle and colour slightly each pass. Alpha under 40 is invisible however many passes you stack.
- Paint, in order: a coloured paper background, a stem and pointed leaves, the petals, dry petal outlines with brush.pick("cpencil"), then the flower centre and stamens.
- Petals are SEPARATE shapes: 5-8 of them, each its own convex brush.polygon of 4-8 vertices, placed around (0,0) at evenly spaced angles computed with cos()/sin(). One big many-armed star polygon is not a flower.
- Draw BIG: the bloom spans about 300 of the 512 pixels. A flower in the middle 120 pixels scores zero.
- Scores zero: a blank page, a single blob, scribbled lines, a grey flower, a flower painted off the canvas edge, speckle noise, fewer than three petals.
- No loadImage, no fetch, no DOM access, no external assets, no comments over one line.
"""

USER_TEMPLATE = (
    "Paint a {palette} {species} in watercolour: one bloom, seen from the "
    "front, with a stem and leaves, on coloured paper."
)


def build_prompts(combos: list[tuple[str, str]], n: int) -> list[dict[str, str]]:
    rows = []
    for species, palette in itertools.islice(itertools.cycle(combos), n):
        rows.append(
            {
                "prompt": USER_TEMPLATE.format(species=species, palette=palette),
                "species": species,
                "palette": palette,
            }
        )
    return rows


class FlowerPromptDataset(DatasetConfig):
    def __init__(self, prompts: list[dict[str, str]]):
        self.prompts = prompts

    def input_key(self) -> str:
        return "messages"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        return [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ],
                "label": f"{r['species']}::{r['prompt']}",
            }
            for r in self.prompts
        ]


N_TRAIN = 224
N_EVAL = 8
combos = list(itertools.product(SPECIES, PALETTES))
random.Random(7).shuffle(combos)
train_dataset = FlowerPromptDataset(build_prompts(combos[N_EVAL:], N_TRAIN))
eval_dataset = FlowerPromptDataset(build_prompts(combos[:N_EVAL], N_EVAL))


# ## Creating a reward function
#
# The reward function renders each sketch in a
# [Modal Sandbox](https://modal.com/docs/guide/sandboxes).
# We serve the judge as an
# [Endpoint](https://modal.com/docs/guide/endpoints).

judge = Endpoint.launch(
    Qwen3_6_27B(),
    endpoint_name="flower-judge-27b",
    unauthenticated=True,
    recreate_if_existing=True,
)
judge.wait_until_ready(timeout=15 * 60)


async def flower_rm(args, sample, **kwargs) -> float:
    label = getattr(sample, "label", None) or ""
    code = helpers.extract_sketch(sample.response, base_model.parse_response)
    if code is None:
        reward, meta, png = 0.0, {"gate": "no valid sketch"}, None
    else:
        png, render_meta = await asyncio.to_thread(helpers.render_in_sandbox, code)
        reward, meta, png = await asyncio.to_thread(
            helpers.score_png, png, label, code, judge, render_meta
        )
    metadata = {**(getattr(sample, "metadata", None) or {}), **meta}
    if png is not None:
        metadata["image"] = png
    sample.metadata = metadata
    return reward


# ## Training
#
# After that, it's simple to start training!


config = TrainConfig(
    model=base_model,
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    recipe=Qwen3_5_4B_Recipe(
        custom_rm_function=flower_rm,
        num_rollout=100,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=8,
        max_tokens_per_gpu=16384,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        image_overlay=lambda image: (
            image.uv_pip_install("modal~=1.5.2", "httpx~=0.28.1", "pillow~=11.1")
            .run_function(helpers.download_clip, env={"HF_HOME": "/tmp/hf"})
            .add_local_file(helpers.__file__, remote_path="/root/helpers.py", copy=True)
            .add_local_dir(helpers.ASSETS_DIR, helpers.REMOTE_ASSETS_DIR, copy=True)
            .env({IMAGE_SAMPLE_LIMIT_ENV: str(8 * 8)})
        ),
    ),
)

run = config.launch()
print(f"run id: {run.training_run_id}")
