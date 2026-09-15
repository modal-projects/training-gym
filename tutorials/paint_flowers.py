# ---
# order: 11
# deps: pillow
# ---
#
# # Painting flowers with code
#
# This tutorial trains **Qwen3.5-4B** with GRPO to write [p5.brush](https://p5brush.org)
# sketches that paint watercolour flowers. The prompt is a brief —
# *"paint a peach hibiscus in watercolour"* — and the response is a complete
# JavaScript sketch. There is no reference answer to diff against: the sketch is
# rendered to a PNG in a Modal Sandbox and the *picture* is what gets scored.
#
# It follows [Surya Narreddi's "RL-ing Qwen to paint with
# code"](https://surya.website/rling-qwen-to-paint-with-code), and in particular
# its second attempt at a reward. The first attempt — nine weighted terms,
# several of them VLM judges scoring 0-10 — plateaued at 0.65 with every output
# collapsed to the same flat five-petal clip-art flower, because the judges were
# 0.85-0.95 correlated with each other and a third of the reward came from a
# code-length term that saturated in the first few steps. What replaced it was
# four terms, only one of which is really doing the work:
#
# | term | weight | what it is |
# | --- | --- | --- |
# | gate | 0.05 | the sketch parsed, rendered, and used p5.brush |
# | length | 0.05 | the sketch is long enough to be a painting |
# | preference | 0.30 | a CLIP probe fitted on hand-rated renders |
# | pairwise | 0.60 | a VLM judge, candidate against the reference pool |
#
# The important move is the last row: instead of asking a judge *"rate this
# 0-10"*, it shows the judge the render **and an image from a pool of pictures we
# already decided we like**, and asks which one is better. Absolute scores
# compress — a mediocre flower and a good one both get a 7 — while a pairwise
# win rate keeps its spread all the way up.

import base64
import hashlib
import io
import itertools
import json
import os
import random
import re
import threading

import modal

from modal_training_gym import Qwen3_5_4B
from modal_training_gym.common.dataset import DatasetConfig

base_model = Qwen3_5_4B()

# ## The brief
#
# Prompts are generated from a small grammar: eight species crossed with eight
# palettes, 64 briefs in total. Eight combinations are held out for eval, so the
# eval set asks for pairings the policy never trained on (an *indigo tulip*
# it has painted neither as a tulip in indigo nor in that pairing).
#
# The grammar is deliberately narrow. The reward compares renders against a
# fixed pool of reference pictures, and that comparison is only meaningful if
# the pool covers what the prompts ask for.

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

# ## The system prompt
#
# The blog's sharpest prompt finding: a ~400-line p5.brush API reference made
# things *worse*, because the model treated the documentation as a menu and
# invented neighbouring methods that do not exist. A short allowlist of the eight
# calls the task actually needs beat it. Everything below is constraint, not
# documentation — the canvas setup, the coordinate frame, the allowed calls, and
# the failure modes that score zero.
#
# One rule is there because of a trace, not a guess. Ten smoke steps in, most
# sketches compiled but rendered *grey*: the model had settled on p5's
# single-argument grayscale form, `brush.fill(30, 100)`, which is legal p5 and
# ignores the palette in the brief entirely. Reward alone would have unlearned
# it eventually; one line demanding quoted hex strings does it for free.

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
    """Prompt-only dataset generated from the flower grammar."""

    input_key = "messages"
    label_key = "label"
    n_train: int = 224
    n_eval: int = 8

    def load(self, split: str = "all"):
        combos = list(itertools.product(SPECIES, PALETTES))
        random.Random(7).shuffle(combos)
        # Held out first: the grammar is smaller than n_train, so cycling would
        # otherwise pull the eval pairings into the train split as well.
        eval_rows = build_prompts(combos[: self.n_eval], self.n_eval)
        train_rows = build_prompts(combos[self.n_eval :], self.n_train)
        if split == "train":
            return train_rows
        if split == "eval":
            return eval_rows
        return train_rows + eval_rows

    def _rows_to_records(self, rows: list[dict[str, str]]) -> list[dict]:
        return [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ],
                # The reward needs the species to pick like-for-like references
                # out of the pool, so it rides along in the label.
                "label": f"{r['species']}::{r['prompt']}",
            }
            for r in rows
        ]

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        def write(p: str, rows: list[dict[str, str]]) -> None:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            records = self._rows_to_records(rows)
            if p.endswith(".parquet"):
                from datasets import Dataset

                Dataset.from_list(records).to_parquet(p)
            else:
                with open(p, "w") as f:
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")

        write(path, self.load("train"))
        for p in (eval_paths or {}).values():
            write(p, self.load("eval"))


# ## Extracting the sketch
#
# The first gate is textual and free: one JavaScript fence, a `setup()`, no
# network or DOM access, and a length ceiling so a runaway generation cannot
# stall a render sandbox for three minutes.

_JS_FENCE = re.compile(r"```(?:javascript|js)\s*\n(.*?)```", re.DOTALL)
_BANNED = re.compile(
    r"\b(loadImage|fetch|XMLHttpRequest|WebSocket|document\.|window\.|eval|import|require)\b"
)


def extract_sketch(response: str) -> str | None:
    """The sketch inside the response, or None if it is not usable."""
    parsed = base_model.parse_response(response)
    m = _JS_FENCE.search(parsed.content or "")
    if not m:
        return None
    code = m.group(1).strip()
    if not code or "function setup" not in code or len(code) > 8000:
        return None
    if _BANNED.search(code):
        return None
    return code


# ## The renderer
#
# Rendering is a Modal Sandbox running headless Chromium with p5 and p5.brush,
# driven by puppeteer: write the sketch to disk, screenshot the canvas, hand back
# PNG bytes. Two details in the page harness matter more than they look.
#
# **The brush facade.** Every `brush.*` access goes through a Proxy: unknown
# methods become no-ops and unknown brush names fall back to real ones, so one
# hallucinated call costs a single stroke instead of the whole picture. It also
# restores p5's modelview matrix around each call, because flushing a watercolour
# mask leaves the full-canvas quad transform in place and everything painted
# afterwards lands half a canvas away.
#
# **Partial renders count.** A sketch that throws in WEBGL draws nothing at all,
# so on an error the harness drops the lines naming the blamed identifier and
# re-runs, up to three times, and captures whatever reached the canvas either
# way. Early in training most rollouts are partial renders; scoring them instead
# of zeroing them is the difference between a reward that has a gradient and one
# that is flat at 0.05 for the first hundred steps.

RENDER_APP_NAME = "training-gym-flower-render"

RENDER_JS = r"""
const fs = require('fs');
const puppeteer = require('puppeteer-core');

(async () => {
  const sketch = fs.readFileSync(process.argv[2], 'utf8');
  const p5js = fs.readFileSync('/render/node_modules/p5/lib/p5.min.js', 'utf8');
  const brushjs = fs.readFileSync('/render/node_modules/p5.brush/dist/p5.brush.js', 'utf8');
  const buildHtml = (sketch) => `<!DOCTYPE html><html><body>
<script>window.__err=null;window.onerror=(m)=>{window.__err=String(m)};window.onunhandledrejection=(e)=>{window.__err=String(e.reason)};</script>
<script>${p5js}</script>
<script>${brushjs}</script>
<script>
(function(){
  const real = window.brush;
  // "spray" is deliberately absent: speckle storms read to a judge as texture.
  const names = ["pen","rotring","2B","HB","2H","cpencil","charcoal",
                 "hatch_brush","marker","marker2"];
  const pick = real.pick, setHatch = real.setHatch;
  const noop = () => {};
  const patched = {
    pick: (n) => pick(names.includes(n) ? n : "HB"),
    setHatch: (n, c, w) => setHatch(names.includes(n) ? n : "hatch_brush", c, w),
    // brush.rect misreads p5's CORNER mode here and floods the canvas.
    rect: noop,
  };
  const guard = (fn) => function () {
    const r = window._renderer;
    const m = r && r.uMVMatrix ? Array.from(r.uMVMatrix.mat4) : null;
    try { return fn.apply(real, arguments); }
    finally { if (m) r.uMVMatrix.mat4.set(m); }
  };
  const facade = new Proxy(real, {
    get(target, key) {
      const value = key in patched ? patched[key] : target[key];
      if (value === undefined) return () => {};
      if (typeof value === "function") return guard(value);
      return value;
    },
  });
  window.brush = facade;
  // Sketches call lerpColor on hex strings; coercing keeps a whole petal pass.
  const realLerp = window.lerpColor;
  window.lerpColor = (a, b, t) => {
    const c = (v) => (typeof v === "string" || typeof v === "number")
      ? window.color(v) : v;
    return realLerp(c(a), c(b), t);
  };
  for (const name of ["pick", "spline", "flowLine", "polygon", "hatch",
                      "noHatch", "setHatch", "bleed", "field", "noField"]) {
    if (window[name] === undefined) window[name] = (...a) => facade[name](...a);
  }
})();
</script>
<script>try{${sketch}}catch(e){window.__err=String(e)}</script>
</body></html>`;
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium',
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu',
           '--enable-unsafe-swiftshader', '--use-angle=swiftshader'],
  });
  const attempt = async (src) => {
    const page = await browser.newPage();
    try {
      await page.setContent(buildHtml(src), { waitUntil: 'load', timeout: 20000 });
      await page.waitForFunction(
        'window.__err !== null || document.querySelector("canvas") !== null',
        { timeout: 15000 }).catch(() => {});
      await new Promise(r => setTimeout(r, 2000));
      const err = await page.evaluate('window.__err');
      const canvas = await page.$('canvas');
      if (!canvas) return { err: err || 'no canvas', buf: null };
      // p5.brush buffers strokes; flush them before capturing.
      await page.evaluate(`
        if (window.brush) {
          try { brush.reDraw(); } catch (e) {}
          try { brush.reBlend(); } catch (e) {}
        }
      `);
      await new Promise(r => setTimeout(r, 1000));
      // Watercolour fills settle across frames: poll until the canvas is still.
      let buf = await canvas.screenshot({ type: 'png' });
      let prev = '';
      for (let i = 0; i < 8; i++) {
        await new Promise(r => setTimeout(r, 1000));
        const shot = await canvas.screenshot({ type: 'png' });
        const sig = Buffer.from(shot).toString('base64');
        buf = shot;
        if (sig === prev) break;
        prev = sig;
      }
      return { err, buf };
    } finally {
      await page.close();
    }
  };

  // The identifier a sketch error blames, so its lines can be dropped.
  const blamed = (err) => {
    const s = String(err);
    const pats = [/(\w+) is not defined/, /calling (\w+)\(\)/,
                  /\w+\.(\w+) is not a function/, /(\w+) is not a function/];
    const keep = ["setup", "createCanvas", "function", "draw", "background"];
    for (const p of pats) {
      const m = p.exec(s);
      if (m) return keep.includes(m[1]) ? null : m[1];
    }
    return null;
  };

  try {
    let src = sketch;
    let res = await attempt(src);
    for (let i = 0; i < 3 && res.err; i++) {
      const name = blamed(res.err);
      if (!name) break;
      const stripped = src.split('\n').filter(l => !l.includes(name)).join('\n');
      if (stripped === src || !/\S/.test(stripped)) break;
      src = stripped;
      res = await attempt(src);
    }
    if (!res.buf) {
      console.error('SKETCH_ERROR: ' + (res.err || 'no canvas'));
      process.exit(2);
    }
    if (res.err) console.error('SKETCH_PARTIAL: ' + res.err);
    process.stdout.write('PNGB64:' + Buffer.from(res.buf).toString('base64'));
    process.exit(0);
  } finally {
    await browser.close();
  }
})().catch(e => { console.error('RENDER_ERROR: ' + e); process.exit(3); });
"""


def render_image() -> modal.Image:
    """Chromium, node, and the two JS libraries the sketches are written against."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("chromium", "nodejs", "npm", "fonts-liberation")
        .run_commands(
            "mkdir -p /render",
            "cd /render && npm install --no-audit --no-fund"
            " p5@1.11.3 p5.brush@1.1.2 puppeteer-core@23.11.1",
        )
    )


def render_sketch(code: str) -> tuple[bytes | None, dict]:
    """Render sketch code to PNG bytes in a Modal Sandbox."""
    app = modal.App.lookup(RENDER_APP_NAME, create_if_missing=True)
    sandbox = modal.Sandbox._experimental_create(
        "sleep",
        "infinity",
        app=app,
        image=render_image(),
        workdir="/render",
        timeout=300,
        cpu=1.0,
        memory=2048,
    )
    try:
        sandbox.filesystem.write_text(RENDER_JS, "/render/render.js")
        sandbox.filesystem.write_text(code, "/render/sketch.js")
        proc = sandbox.exec(
            "node", "/render/render.js", "/render/sketch.js", timeout=180
        )
        proc.wait()
        out, err = proc.stdout.read(), proc.stderr.read()
        if "PNGB64:" in out:
            png = base64.b64decode(out.split("PNGB64:", 1)[1].strip())
            if "SKETCH_PARTIAL:" in (err or ""):
                return png, {"render": "partial", "stderr": err[-200:]}
            return png, {"render": "ok"}
        return None, {"render": "fail", "stderr": (err or "")[-400:]}
    except Exception as e:
        return None, {"render": "fail", "stderr": f"{type(e).__name__}: {e}"[-400:]}
    finally:
        sandbox.terminate()
        sandbox.detach()


# ## The reference pool
#
# The blog's pool is 581 images: 1,664 generations from earlier runs, hand-rated
# into *love* (117), *okay* (266), and the rest discarded. That pool is a
# by-product of having already trained the model once. Starting cold, there is
# nothing to rate — a base model that has never seen p5.brush emits sketches that
# render as blank paper — so this task bootstraps the pool from hand-written
# code instead.
#
# `paint_flowers_assets/reference_sketches.py` is a parametric flower painter:
# eight species geometries, eight palettes, randomised petal size, spin, bleed,
# leaf placement and pass count. 192 of those were rendered, alongside 84
# deliberate failure modes from `negative_sketches.py` (blank washes, single
# blobs, scribble storms, invisible-alpha flowers, grey flowers, off-canvas
# flowers, two-petal flowers), and the 192 were rated on a contact sheet:
# 122 *love*, 70 *okay*. `ratings.py` records the verdicts.
#
# Twenty *love* renders, spread across species and palettes, are committed in
# `paint_flowers_assets/refs/` — that is the pool the pairwise judge compares
# against. Writing the pool by hand rather than mining it from rollouts also
# avoids the trap the blog's own pool has: if the references are the policy's
# own past output, the ceiling is the policy's past output.

ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "paint_flowers_assets"
)
# On a rollout worker the module arrives by cloudpickle without its directory, so
# the assets are baked into the image at these paths instead.
REMOTE_ASSETS_DIR = "/root/flower_assets"
REMOTE_CLIP_DIR = "/root/flower_assets/clip"

_CACHE: dict[str, object] = {}
# Renders are scored concurrently, and importing transformers from several
# threads at once yields a half-initialised module, so loading is serialised.
# The lock is created on first use: a live lock among the module globals makes
# it unpicklable, which silently strips the reward's helpers on the worker.
_LOCKS: dict[str, threading.Lock] = {}


def assets_dir() -> str:
    return REMOTE_ASSETS_DIR if os.path.isdir(REMOTE_ASSETS_DIR) else ASSETS_DIR


def reference_pool() -> list[tuple[str, bytes]]:
    """The pool as `(species, png_bytes)`, decoded once per worker."""
    with _LOCKS.setdefault("refs", threading.Lock()):
        if "refs" not in _CACHE:
            from PIL import Image

            refs = []
            root = os.path.join(assets_dir(), "refs")
            for name in sorted(os.listdir(root)):
                buf = io.BytesIO()
                Image.open(os.path.join(root, name)).convert("RGB").save(
                    buf, format="PNG"
                )
                refs.append((name.split("_")[0], buf.getvalue()))
            _CACHE["refs"] = refs
    return _CACHE["refs"]  # type: ignore[return-value]


def pick_references(species: str, key: str, k: int = 4) -> list[bytes]:
    """`k` references for one comparison: half same-species, half anything.

    Same-species references make the comparison about *this* flower rather than
    about watercolour in general, and the rest keep the judge from rewarding a
    single memorised silhouette.
    """
    pool = reference_pool()
    rng = random.Random(hashlib.sha1(key.encode()).hexdigest())
    same = [png for sp, png in pool if sp == species]
    rest = [png for sp, png in pool if sp != species]
    chosen = rng.sample(same, min(len(same), k // 2))
    chosen += rng.sample(rest, k - len(chosen))
    return chosen


# ## Term 3: the preference probe
#
# The blog's 0.30 term is HPSv3, a human-preference model. HPSv3's weights are a
# gated multi-billion-parameter download, and running it beside the policy costs
# a GPU, so this task uses the cheap equivalent it already has the data for: a
# logistic head on frozen CLIP ViT-B/32 embeddings, fitted on the *love* renders
# against the deliberate failure modes.
#
# On a held-out split it orders 99.7% of (love, failure) pairs correctly, so it
# is a reliable *"this is not a blank page, a blob, or a scribble"* detector. It
# is much weaker as a taste model: the *okay* tier scores 1.36 against love's
# 1.56 and failure's -1.03, i.e. it sees the difference between a real flower and
# a mess far more clearly than between a good flower and a dull one. That is
# exactly the division of labour intended — the probe is 40ms of CPU that catches
# collapse, and the taste judgement is the pairwise term's job. Being a different
# model family from the judge also means its mistakes are not the judge's
# mistakes, which was the failure of the blog's first rubric.

PROBE_PATH_NAME = "flower_taste.npz"


def _load_probe() -> None:
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    data = np.load(os.path.join(assets_dir(), PROBE_PATH_NAME), allow_pickle=False)
    clip = REMOTE_CLIP_DIR if os.path.isdir(REMOTE_CLIP_DIR) else "openai/clip-vit-base-patch32"
    _CACHE.update(
        model=CLIPModel.from_pretrained(clip).eval(),
        proc=CLIPProcessor.from_pretrained(clip),
        w=torch.tensor(data["w"]),
        b=float(data["b"][0]),
        lo=float(data["lo"]),
        hi=float(data["hi"]),
        torch=torch,
    )


def probe_score(png: bytes) -> float:
    """The probe's preference for this render, squashed to roughly [0, 1].

    `lo` and `hi` are the probe's mean logit on the failure modes and on the
    loved renders; a logistic through them puts those anchors near 0.15 and 0.85.
    Hard clipping was worse: early rollouts land below the failure mean, and a
    clipped term gives that whole bottom of the batch identical reward and
    nothing to climb.
    """
    import math

    from PIL import Image

    with _LOCKS.setdefault("probe", threading.Lock()):
        if "model" not in _CACHE:
            _load_probe()
    torch = _CACHE["torch"]
    img = Image.open(io.BytesIO(png)).convert("RGB")
    with torch.no_grad():
        f = _CACHE["model"].get_image_features(
            **_CACHE["proc"](images=[img], return_tensors="pt")
        )
        f = getattr(f, "pooler_output", f)
        f = f / f.norm(dim=-1, keepdim=True)
        logit = float(f[0] @ _CACHE["w"]) + _CACHE["b"]
    lo, hi = _CACHE["lo"], _CACHE["hi"]
    mid, span = (lo + hi) / 2, max(hi - lo, 1e-6)
    return 1.0 / (1.0 + math.exp(-3.5 * (logit - mid) / span))


# ## Term 4: the pairwise judge
#
# A 27B dense multimodal critic, deployed with `CustomDeployment.launch()`, sees
# the candidate render and one reference and answers with the winner as JSON.
# Each candidate is compared against four references with the sides alternated,
# because a VLM shown two images has a real preference for the first one; the
# reward is the win rate over the votes that parsed.
#
# Measured on held-out renders it separates the tiers cleanly and, unlike a 0-10
# score, keeps its spread: *love* renders win 0.94 of comparisons, *okay* renders
# 0.72, deliberate failure modes 0.10. A policy has somewhere to go from every
# point on that scale.

JUDGE_URL_ENV = "FLOWER_JUDGE_URL"
JUDGE_MODEL = "flower-judge"
JUDGE_REFS = 4

PAIRWISE_PROMPT = """You are judging two watercolour illustrations painted in code.

The brief was: "{prompt}"

Image A and Image B are two attempts. Pick the one that is the better
illustration for that brief. Weigh, in order:
1. Is it recognisably the flower asked for, with petals, a centre, and foliage?
2. Does it follow the brief's colour and species?
3. Is it a pleasing watercolour painting rather than a blob, a scribble, or an
   empty page?

Answer with strict JSON only: {{"winner": "A" or "B", "why": "<8 words>"}}"""


def judge_pair(candidate: bytes, reference: bytes, prompt: str, flip: bool):
    """1.0 if the candidate wins, 0.0 if it loses, None if the judge failed."""
    import httpx

    def uri(png: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png).decode()

    a, b = (reference, candidate) if flip else (candidate, reference)
    body = {
        "model": JUDGE_MODEL,
        "temperature": 0.3,
        "max_tokens": 96,
        # Qwen3.8 answers from its reasoning channel by default, which leaves
        # the content field empty and every vote unparseable.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PAIRWISE_PROMPT.format(prompt=prompt)},
                    {"type": "text", "text": "Image A:"},
                    {"type": "image_url", "image_url": {"url": uri(a)}},
                    {"type": "text", "text": "Image B:"},
                    {"type": "image_url", "image_url": {"url": uri(b)}},
                ],
            }
        ],
    }
    url = os.environ.get(JUDGE_URL_ENV, "")
    headers = {
        "Modal-Key": os.environ.get("MODAL_KEY", ""),
        "Modal-Secret": os.environ.get("MODAL_SECRET", ""),
    }
    text = ""
    for attempt in range(4):
        try:
            resp = httpx.post(
                f"{url}/v1/chat/completions", json=body, headers=headers, timeout=180
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            break
        except Exception:
            if attempt == 3:
                return None
    winner = None
    try:
        winner = json.loads(text[text.index("{") : text.rindex("}") + 1]).get("winner")
    except Exception:
        for token in ('"A"', '"B"'):
            if token in text:
                winner = token.strip('"')
                break
    if winner not in ("A", "B"):
        return None
    return float((winner == "A") != flip)


def judge_win_rate(png: bytes, prompt: str, species: str) -> tuple[float, dict]:
    """Win rate against the reference pool, sides alternated to cancel bias."""
    from concurrent.futures import ThreadPoolExecutor

    refs = pick_references(species, prompt + str(len(png)), JUDGE_REFS)
    with ThreadPoolExecutor(max_workers=len(refs)) as pool:
        votes = list(
            pool.map(
                lambda ir: judge_pair(png, ir[1], prompt, flip=bool(ir[0] % 2)),
                enumerate(refs),
            )
        )
    got = [v for v in votes if v is not None]
    return (sum(got) / len(got) if got else 0.0), {
        "judge_votes": len(got),
        "judge_wins": round(sum(got), 2) if got else 0.0,
    }


# ## The pixel gates
#
# Three cheap statistics run before the judge is ever called. They exist because
# every one of them was, at some point, a way to score well without painting a
# flower: an empty canvas that the probe had no opinion about, a speckle storm
# that read as texture, and a single flood fill that covered every cell of the
# canvas. Catching those on the CPU also keeps them off the judge's queue.


def ink_fraction(png: bytes) -> float:
    """Fraction of pixels that differ from the dominant background colour."""
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((64, 64))
    hist = img.histogram()
    bg = max(range(256), key=lambda i: hist[i])
    px = list(img.getdata())
    return sum(1 for v in px if abs(v - bg) > 16) / len(px)


def coverage_fraction(png: bytes) -> float:
    """Fraction of an 8x8 grid of the canvas that is substantially inked."""
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((64, 64))
    hist = img.histogram()
    bg = max(range(256), key=lambda i: hist[i])
    ink = [abs(v - bg) > 16 for v in img.getdata()]
    cells = 0
    for by in range(8):
        for bx in range(8):
            lit = sum(
                ink[(by * 8 + y) * 64 + bx * 8 + x] for y in range(8) for x in range(8)
            )
            cells += lit / 64.0 > 0.15
    return cells / 64.0


def speckle_fraction(png: bytes) -> float:
    """Fraction of pixels that differ sharply from their local neighbourhood."""
    from PIL import Image, ImageFilter

    img = Image.open(io.BytesIO(png)).convert("L").resize((128, 128))
    blurred = img.filter(ImageFilter.BoxBlur(2))
    px, bpx = list(img.getdata()), list(blurred.getdata())
    return sum(1 for v, b in zip(px, bpx) if abs(v - b) > 40) / len(px)


def dark_fraction(png: bytes) -> float:
    """Fraction of pixels that are nearly black."""
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((128, 128))
    px = list(img.getdata())
    return sum(1 for v in px if v < 48) / len(px)


# ## The reward
#
# The four terms, assembled. The weights are the blog's collapsed rubric, and the
# shape of the failure it is collapsing away from is worth restating: reward that
# is mostly a length ramp and a committee of correlated judges will climb without
# the pictures improving. Here 0.60 of the reward is a single comparison against
# pictures we chose, 0.30 is a model from a different family, and the two cheap
# terms are gates rather than gradients — a sketch that renders and is long
# enough banks 0.10 and then has to actually paint something.
#
# The length term is a ramp to 1,200 characters and flat after it. It is
# deliberately small and deliberately saturating early: it exists only to get a
# cold policy past the "emit three lines and stop" attractor.

GATE_WEIGHT = 0.05
LENGTH_WEIGHT = 0.05
PROBE_WEIGHT = 0.30
JUDGE_WEIGHT = 0.60
LENGTH_TARGET = 1200


def score_response(response: str, label: str) -> tuple[float, dict, bytes | None]:
    """Reward, metadata, and the render, for one rollout."""
    species, _, prompt = label.partition("::")
    if not prompt:
        species, prompt = SPECIES[0], label
    code = extract_sketch(response)
    if code is None:
        return 0.0, {"gate": "no valid sketch"}, None
    png, meta = render_sketch(code)
    if png is None:
        return 0.0, meta, None
    meta["species"] = species
    reward = GATE_WEIGHT if "brush." in code else 0.0
    reward += LENGTH_WEIGHT * min(1.0, len(code) / LENGTH_TARGET)

    ink = ink_fraction(png)
    speckle = speckle_fraction(png)
    coverage = coverage_fraction(png)
    dark = dark_fraction(png)
    meta.update(
        ink=round(ink, 3),
        speckle=round(speckle, 3),
        coverage=round(coverage, 3),
        dark=round(dark, 3),
    )
    # An empty canvas, a noise storm, a flood fill, or an ink slick: nothing to
    # judge. The darkness gate is the one the policy found on its own — a fat
    # brush.strokeWeight paints a black mass that still renders and still has
    # ink, so without it the cheap terms pay out for a silhouette.
    if ink < 0.02 or speckle > 0.12 or coverage > 0.97 or dark > 0.3:
        return round(reward, 4), meta, png

    probe = probe_score(png)
    wins, judge_meta = judge_win_rate(png, prompt, species)
    meta.update(probe=round(probe, 3), wins=round(wins, 3), **judge_meta)
    reward += PROBE_WEIGHT * probe + JUDGE_WEIGHT * wins
    return round(reward, 4), meta, png


async def flower_rm(args, sample, **kwargs) -> float:
    """Training Gym reward hook.

    The render is attached to `sample.metadata["image"]`, which is what puts the
    painting on the run's page in the dashboard — for a task whose output is a
    picture, being able to scroll a step's renders is most of the debugging.
    """
    import asyncio

    label = getattr(sample, "label", None) or ""
    reward, meta, png = await asyncio.to_thread(score_response, sample.response, label)
    metadata = {**(getattr(sample, "metadata", None) or {}), **meta}
    if png is not None:
        metadata["image"] = png
    sample.metadata = metadata
    return reward


# ## Training
#
# GRPO with eight samples per prompt: the whole signal here is *which of these
# eight paintings of the same brief came out best*, so the group has to be wide
# enough for a win rate to differentiate it.
#
# The rollout worker needs three things baked into its image — the reference
# pool, the probe weights, and a CLIP snapshot — because the reward runs on the
# worker and the module reaches it by cloudpickle, without its directory.
#
# The judge is a separate deployment. Launch it once with
# `CustomDeployment.launch(Qwen3_8_27B(), app_name=..., unauthenticated=True)`
# and pass its URL in through `FLOWER_JUDGE_URL`; keeping it out of the training
# app means the critic survives a restart of the trainer, and one judge serves
# several concurrent runs.


def launch(num_rollout: int, judge_url: str, n_train: int = 224, load: str = "") -> None:
    """Train the flower task. `load` continues from a checkpoint directory."""
    from modal_training_gym import SlimeRecipe, TrainConfig
    from modal_training_gym.common.sample_extraction import IMAGE_SAMPLE_LIMIT_ENV

    def overlay(image: modal.Image) -> modal.Image:
        return (
            image.run_commands(
                "uv pip install --system 'modal~=1.5.2' 'httpx~=0.28.1' 'pillow~=11.1'",
                # The HF cache path is a volume mount at runtime, so the build
                # downloads through a scratch cache and leaves it empty.
                f'HF_HOME=/tmp/hf python -c "from huggingface_hub import '
                f"snapshot_download as d; d('openai/clip-vit-base-patch32', "
                f"local_dir='{REMOTE_CLIP_DIR}')\"",
                "rm -rf /tmp/hf /root/.cache/huggingface",
            )
            .add_local_dir(ASSETS_DIR, REMOTE_ASSETS_DIR, copy=True)
            # Capture every sample's render: rollout_batch_size * n_samples.
            .env({IMAGE_SAMPLE_LIMIT_ENV: str(8 * 8), JUDGE_URL_ENV: judge_url})
        )

    config = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=FlowerPromptDataset(n_train=n_train, always_prepare=True),
        recipe=SlimeRecipe(
            custom_rm_function=flower_rm,
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=num_rollout,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            global_batch_size=8,
            max_tokens_per_gpu=16384,
            save_interval=10,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=overlay,
            load=load,
            # The saved scheduler stops at the previous horizon, so continuing
            # past it needs the new schedule to win over the checkpoint's.
            extra_config={"override_opt_param_scheduler": True} if load else {},
        ),
    )
    result = config.train()
    print(f"Training run id: {result.training_run_id}")


# ## What 100 steps buys
#
# Run `lazy-cache-56eb24da74da`, Qwen3.5-4B, 100 rollouts of 8 prompts x 8
# samples on one H100, about 2.5 hours:
#
# | | step 1 | step 40 | step 100 |
# | --- | --- | --- | --- |
# | mean reward | 0.070 | 0.288 | 0.498 |
# | renders that produce an image | 21/64 | 64/64 | 62/64 |
# | win rate against the reference pool | 0.03 | 0.44 | 0.66 |
#
# The order in which things are learned is the interesting part, and it is the
# order the reward is stacked in. Syntax first: the cheap gates are the only
# terms a cold policy can reach, so the first ten steps are spent learning that
# a sketch has to parse and run. Colour second, once the probe starts paying.
# Composition last, and only because the pairwise judge does not care about
# anything else.
#
# Two honest caveats. The final samples converge hard on one composition — five
# or six round, overlapping petal washes on a stem, seen face-on — which is the
# blog's clip-art collapse arriving by a different route: a pairwise judge with
# twenty references teaches the policy the *average* of that pool, and variance
# is not rewarded. And leaves mostly disappear by step 100; they cost tokens and
# the judge rarely punishes their absence. Both are reference-pool problems
# rather than reward-weight problems, and the fix is a wider, deliberately
# varied pool rather than another term.


if __name__ == "__main__":
    import sys

    launch(
        num_rollout=int(sys.argv[1]) if len(sys.argv) > 1 else 1,
        judge_url=sys.argv[2] if len(sys.argv) > 2 else os.environ.get(JUDGE_URL_ENV, ""),
        load=sys.argv[3] if len(sys.argv) > 3 else "",
    )
