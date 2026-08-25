"""Eye-illustration RL task: dataset, renderer sandbox, judge, reward.

The model writes a p5.js + p5.brush sketch that draws an eye. The sketch is
rendered to a PNG in a Modal Sandbox (node + headless chromium), the PNG is
judged by a VLM endpoint, and the rendered image is attached to
``sample.metadata["image"]`` so it shows up in the Training Gym dashboard.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import random
import re

import modal

from modal_training_gym import Qwen3_5_4B
from modal_training_gym.common.dataset import DatasetConfig

base_model = Qwen3_5_4B()

# ── Prompt grammar ───────────────────────────────────────────────────────

SUBJECTS = [
    "an anime heroine's eye",
    "a shonen protagonist's determined eye",
    "a magical girl's sparkling eye",
    "a cool anime rival's narrowed eye",
    "a cheerful anime schoolgirl's wide eye",
    "an anime idol's glossy eye",
    "a mysterious anime villain's eye",
    "a shy anime character's downcast eye",
]
IRIS_COLORS = ["amber", "emerald green", "ice blue", "crimson", "violet", "golden"]
STYLES = [
    "clean cel-shaded anime style",
    "90s retro anime style",
    "modern glossy anime key-visual style",
    "shoujo manga style with heavy sparkle",
    "bold shonen anime style with thick line art",
]

SYSTEM_PROMPT = """\
You write p5.js sketches that draw anime-style illustrations, using p5's solid fills for flat colour and the p5.brush library for line work.

Rules:
- Reply with a single ```javascript code fence containing a complete sketch and nothing else.
- Define exactly one function: `function setup() { ... }`. Do not define draw().
- Start setup() with: createCanvas(512, 512, WEBGL); angleMode(DEGREES); brush.load();
- The coordinate origin (0,0) is the CENTER of the canvas; x and y range from -256 to 256. Compose around (0,0).
- End setup() with: noLoop();
- Draw with the p5.brush API plus p5's own solid-fill drawing: background, noStroke(), fill(r,g,b), ellipse(x,y,w,h), triangle(...), and beginShape()/vertex(x,y)/bezierVertex(cx1,cy1,cx2,cy2,x,y)/endShape(CLOSE) for custom filled shapes. push/pop, translate, rotate are fine.
- Anime eyes are flat colour areas with crisp edges, so build them from p5 fill() shapes; use p5.brush only for the thin lower-lid and crease lines. Do NOT hatch or scribble: hatching makes it look like a pencil sketch, which scores zero here.
- Allowed brush calls: brush.pick(name), brush.stroke(color), brush.strokeWeight(w), brush.noStroke(), brush.line(x1,y1,x2,y2), brush.circle(x,y,r), brush.polygon([[x,y],...]), brush.spline([[x,y],...], curvature), brush.flowLine(x,y,length,dirAngle), brush.setHatch(brushName,color,weight), brush.hatch(dist,angle,{rand:0.1,continuous:true}), brush.noHatch(), brush.field("seabed"), brush.noField().
- NEVER use brush.fill/brush.noFill/brush.bleed (watercolor fills erase every stroke on this renderer), and never use brush.beginShape/vertex/endShape or brush.rect.
- Brush names available to brush.pick and brush.setHatch: "pen", "rotring", "2B", "HB", "2H", "cpencil", "charcoal", "hatch_brush", "marker", "marker2". Never use "spray" — speckle textures are rejected.
- Draw crisp shapes on a mostly empty white canvas: ink storms and speckle score zero.
- No loadImage, no fetch/XHR, no DOM access, no external assets, no comments longer than one line.
- Compose the whole illustration inside the 512x512 canvas; large bold shapes, few of them, crisply layered.

An anime eye is built from these parts, in this order. Vary the numbers, colours,
lash shapes and proportions to suit the requested subject and style, but keep the
anatomy and the anime look:
1. Sclera: a near-white almond, wide and low, filled with p5:
   fill(252,250,252); beginShape(); vertex(-180,4);
   bezierVertex(-120,-125, 110,-135, 180,-14); bezierVertex(110,58, -110,76, -180,4); endShape(CLOSE);
2. Iris: a LARGE oval, taller than it is wide, filling most of the opening and
   tucked under the upper lash line. Build it from 3 concentric ovals in the
   requested colour, dark at the top to light at the bottom, e.g.
   fill(74,40,120); ellipse(-4,-4,140,176); fill(126,78,196); ellipse(-4,10,120,148);
   fill(186,150,240); ellipse(-4,34,96,92);
3. Pupil: a tall near-black oval in the iris centre, e.g. fill(18,14,22); ellipse(-4,-2,56,86);
4. Highlights: two pure-white ovals, a big one high on one side of the pupil and a
   small one low on the other: fill(255); ellipse(-34,-42,44,34); ellipse(28,38,20,16);
   This glossy specular pair is what makes the eye look alive rather than dead.
5. Upper lash line: a THICK solid black wedge arcing over the top of the iris,
   thickest in the middle, plus 2-3 sharp black lash spikes flaring off the outer
   corner as filled triangles. This is the boldest shape in the drawing.
6. Lower lid: one thin dark p5.brush spline under the iris, and a thin crease line
   arcing above the lash line. Keep them light — anime lower lids are understated.
Optionally add small white sparkle shapes for a shoujo style.
No hatching, no sketchy scribbles, no grey pencil shading: flat colour, crisp
edges, thick black lash line, glossy highlights.
"""

USER_TEMPLATE = (
    "Illustrate {subject} with an iris in {color}, in {style}. "
    "Make it lively and expressive: a big glossy iris with colour banding and a "
    "dark pupil, bright white specular highlights, a thick black upper lash line "
    "with sharp lash spikes, and a light lower lid — flat anime colour, no "
    "pencil hatching."
)


def shuffled_combos(seed: int = 7) -> list[tuple[str, str, str]]:
    combos = list(itertools.product(SUBJECTS, IRIS_COLORS, STYLES))
    random.Random(seed).shuffle(combos)
    return combos


def build_prompts(n: int, combos: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    rows = []
    for subject, color, style in itertools.islice(itertools.cycle(combos), n):
        rows.append(
            {"prompt": USER_TEMPLATE.format(subject=subject, color=color, style=style)}
        )
    return rows


class EyePromptDataset(DatasetConfig):
    """Prompt-only dataset generated from the eye grammar."""

    input_key = "messages"
    label_key = "label"
    n_train: int = 256
    n_eval: int = 16

    def load(self, split="all"):
        # The grammar has fewer combinations than n_train, so the eval combos are
        # held out first — otherwise cycling would put them in the train split too.
        combos = shuffled_combos()
        eval_rows = build_prompts(self.n_eval, combos[: self.n_eval])
        train_rows = build_prompts(self.n_train, combos[self.n_eval :])
        if split == "train":
            return train_rows
        if split == "eval":
            return eval_rows
        return train_rows + eval_rows

    def _rows_to_records(self, rows):
        return [
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ],
                "label": r["prompt"],
            }
            for r in rows
        ]

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        def write(p, rows):
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


# ── Code extraction / static gates ───────────────────────────────────────

_JS_FENCE = re.compile(r"```(?:javascript|js)\s*\n(.*?)```", re.DOTALL)
_BANNED = re.compile(
    r"\b(loadImage|fetch|XMLHttpRequest|WebSocket|document\.|window\.|eval|import|require)\b"
)


def extract_sketch(response: str) -> str | None:
    parsed = base_model.parse_response(response)
    content = parsed.content or ""
    m = _JS_FENCE.search(content)
    if not m:
        return None
    code = m.group(1).strip()
    if not code or "function setup" not in code:
        return None
    if _BANNED.search(code):
        return None
    if len(code) > 8000:
        return None
    return code


# ── Renderer: Modal Sandbox with node + chromium ─────────────────────────

RENDER_APP_NAME = "training-gym-eye-render"

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
// Forgiving brush facade: unknown brush names fall back to real ones and
// hallucinated brush.* methods become no-ops, so one bad call does not throw
// away the whole drawing.
(function(){
  const real = window.brush;
  // "spray" is deliberately absent: speckle storms fool the judge, so any
  // request for it falls back to a line brush.
  const names = ["pen","rotring","2B","HB","2H","cpencil","charcoal",
                 "hatch_brush","marker","marker2"];
  const pick = real.pick, setHatch = real.setHatch;
  const noop = () => {};
  const patched = {
    pick: (n) => pick(names.includes(n) ? n : "HB"),
    setHatch: (n, c, w) => setHatch(names.includes(n) ? n : "hatch_brush", c, w),
    // Watercolor fills composite a full-canvas rect that erases every stroke
    // in headless WEBGL, so they are dropped instead of ruining the drawing.
    fill: noop, noFill: noop, bleed: noop, rect: noop,
  };
  const facade = new Proxy(real, {
    get(target, key) {
      if (key in patched) return patched[key];
      const value = target[key];
      if (value === undefined) return () => {};
      return value;
    },
  });
  window.brush = facade;
  // Brush-only names are also exposed bare, since sketches often drop the
  // "brush." prefix. p5's own globals (line, circle, fill, stroke) are untouched.
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
  // One render attempt: returns whatever the canvas holds plus any error, so a
  // sketch that throws half way through still yields its partial drawing.
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
      // Fills settle across frames, so poll until the canvas stops changing.
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

  // The name a sketch error blames, so its lines can be dropped: hallucinated
  // globals, misused p5 calls and bad method calls all name one identifier.
  const blamed = (err) => {
    const s = String(err);
    const pats = [/(\w+) is not defined/, /calling (\w+)\(\)/,
                  /\w+\.(\w+) is not a function/, /(\w+) is not a function/];
    // Dropping these would delete the sketch itself rather than a bad call.
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
    // A sketch that throws draws nothing at all in WEBGL, so instead of losing
    // it, the blamed lines are dropped and the rest of the sketch re-runs.
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


def _render_image() -> modal.Image:
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
        image=_render_image(),
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
        out = proc.stdout.read()
        err = proc.stderr.read()
        if "PNGB64:" in out:
            png = base64.b64decode(out.split("PNGB64:", 1)[1].strip())
            # A partial render still gets judged on whatever made it to canvas.
            if "SKETCH_PARTIAL:" in (err or ""):
                return png, {"render": "partial", "stderr": err[-200:]}
            return png, {"render": "ok"}
        return None, {"render": "fail", "stderr": (err or "")[-400:]}
    except Exception as e:
        return None, {"render": "fail", "stderr": f"{type(e).__name__}: {e}"[-400:]}
    finally:
        sandbox.terminate()
        sandbox.detach()


# ── VLM judge ────────────────────────────────────────────────────────────

JUDGE_URL = "https://modal-labs--ep-eye-judge-server.us-west.modal.direct"
JUDGE_MODEL = "google/gemma-4-E4B-it"

# Reference eyes the candidate is compared against, shipped into the training
# image by launch(). Absolute yes/no questions do not work here: the judge calls
# a circle inside a box an eye, so it is asked to grade against a real drawing.
LOCAL_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs")
REF_DIRS = ("/root/eye_refs", LOCAL_REF_DIR)


def reference_pngs() -> list[bytes]:
    import glob

    for d in REF_DIRS:
        paths = sorted(glob.glob(os.path.join(d, "ref_*.png")))
        if paths:
            return [open(p, "rb").read() for p in paths]
    return []


RATING_SCALE = (
    "0 = not an eye at all (random lines, blobs, boxes, lone circles)\n"
    "1 = only vaguely eye-suggestive, or a sketchy pencil/hatched eye with no "
    "flat colour\n"
    "2 = readable eye with a big filled iris and pupil, but missing the thick "
    "black lash line or the white highlights\n"
    "3 = clear anime eye: big filled iris with dark pupil, white highlight, and a "
    "bold dark lash line above it\n"
    "4 = polished anime eye as good as A: flat crisp colour, banded glossy iris, "
    "thick black lash line with spikes, bright highlights"
)
# Anatomy is worth far more than ink: a 1 must not be a comfortable place to sit.
RATING_REWARD = {0: 0.0, 1: 0.08, 2: 0.35, 3: 0.7, 4: 1.0}


def side_by_side(ref: bytes, cand: bytes) -> bytes:
    """Reference on the left, candidate on the right, labelled A and B."""
    import io

    from PIL import Image, ImageDraw

    a = Image.open(io.BytesIO(ref)).convert("RGB").resize((384, 384))
    b = Image.open(io.BytesIO(cand)).convert("RGB").resize((384, 384))
    sheet = Image.new("RGB", (784, 410), "white")
    sheet.paste(a, (0, 26))
    sheet.paste(b, (400, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((150, 8), "A (reference eye)", fill="black")
    draw.text((550, 8), "B (candidate)", fill="black")
    draw.line([(392, 0), (392, 410)], fill="black")
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def judge_once(png: bytes, prompt: str, ref: bytes) -> tuple[float, dict]:
    """Grade the candidate against a reference eye on a 0-4 anatomy scale."""
    import httpx

    b64 = base64.b64encode(side_by_side(ref, png)).decode()
    body = {
        "model": JUDGE_MODEL,
        "max_tokens": 192,
        # Non-zero so the votes averaged in judge_image actually decorrelate.
        "temperature": 0.7,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Image A is a reference illustration of an eye. "
                            "Image B is a candidate drawing, made for the "
                            f'brief: "{prompt}"\n\nFirst line, starting with '
                            "B_IS:, describe in under 12 words what shapes B "
                            "actually contains.\nThen a line starting with "
                            "RATING:, a single digit 0-4 for how much B looks "
                            "like an ANIME eye drawn in the same style as A "
                            "(flat crisp colour, one big glossy iris with a "
                            "dark pupil, white specular highlights, a thick "
                            "black upper lash line with spikes):\n"
                            f"{RATING_SCALE}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }
    headers = {
        "Modal-Key": os.environ.get("MODAL_KEY", ""),
        "Modal-Secret": os.environ.get("MODAL_SECRET", ""),
    }
    import time

    last_err = ""
    for attempt in range(10):
        if attempt:
            time.sleep(30)
        try:
            resp = httpx.post(
                f"{JUDGE_URL}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"RATING:\s*([0-4])", text.upper())
            if m:
                rating = int(m.group(1))
                desc = text.split("RATING:")[0].replace("B_IS:", "").strip()
                return RATING_REWARD[rating], {
                    "judge": str(rating),
                    "judge_desc": desc[:120],
                }
            last_err = f"unparseable: {text[:80]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:200]
    return 0.0, {"judge_error": last_err}


JUDGE_VOTES = 3


def judge_image(png: bytes, prompt: str) -> tuple[float, dict]:
    """Average votes against several references — one Gemma vote is too noisy."""
    from concurrent.futures import ThreadPoolExecutor

    refs = reference_pngs()
    if not refs:
        return 0.0, {"judge_error": "no reference eyes found"}
    chosen = [refs[i % len(refs)] for i in range(JUDGE_VOTES)]
    with ThreadPoolExecutor(max_workers=JUDGE_VOTES) as pool:
        votes = list(pool.map(lambda r: judge_once(png, prompt, r), chosen))
    scores = [s for s, _ in votes]
    return sum(scores) / len(scores), {
        "judge": " ".join(m.get("judge", "?") for _, m in votes),
        "judge_desc": votes[0][1].get("judge_desc", ""),
    }


# ── Reward ───────────────────────────────────────────────────────────────


def ink_fraction(png: bytes) -> float:
    """Fraction of pixels that differ from the dominant background color."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((64, 64))
    hist = img.histogram()
    bg = max(range(256), key=lambda i: hist[i])
    px = list(img.getdata())
    return sum(1 for v in px if abs(v - bg) > 16) / len(px)


def coverage_fraction(png: bytes) -> float:
    """Fraction of an 8x8 grid of the canvas that is substantially inked.

    Line work leaves most cells empty; speckle storms and ink floods cover the
    whole canvas, which is what fools the judge into scoring them as drawings.
    """
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((64, 64))
    hist = img.histogram()
    bg = max(range(256), key=lambda i: hist[i])
    ink = [abs(v - bg) > 16 for v in img.getdata()]
    cells = 0
    for by in range(8):
        for bx in range(8):
            cell = sum(
                ink[(by * 8 + y) * 64 + bx * 8 + x] for y in range(8) for x in range(8)
            )
            cells += cell / 64.0 > 0.15
    return cells / 64.0


def score_response(response: str, prompt: str) -> tuple[float, dict, bytes | None]:
    code = extract_sketch(response)
    if code is None:
        return 0.0, {"gate": "no valid sketch"}, None
    png, meta = render_sketch(code)
    if png is None:
        return 0.0, meta, None
    ink = ink_fraction(png)
    meta["ink"] = round(ink, 3)
    if ink < 0.02:
        return 0.02, meta, png
    coverage = coverage_fraction(png)
    meta["coverage"] = round(coverage, 3)
    # A legible drawing is line work on a mostly empty canvas; speckle storms and
    # ink-flooded canvases fool the judge, so gate them out before judging.
    if coverage > 0.40 or ink > 0.35:
        return 0.02, meta, png
    judge_score, judge_meta = judge_image(png, prompt)
    meta.update(judge_meta)
    return 0.05 + 0.95 * judge_score, meta, png


async def eye_rm(args, sample, **kwargs) -> float:
    import asyncio

    prompt = getattr(sample, "label", None) or ""
    reward, meta, png = await asyncio.to_thread(score_response, sample.response, prompt)
    md = {**(getattr(sample, "metadata", None) or {}), **meta}
    if png is not None:
        md["image"] = png
    sample.metadata = md
    return reward


# ── Training entrypoint ──────────────────────────────────────────────────


def launch(num_rollout: int, n_train: int = 256, model: str = "4b") -> None:
    from modal_training_gym import (
        Qwen3_8_27B,
        Qwen3_8_27b_Recipe,
        SlimeRecipe,
        TrainConfig,
    )
    from modal_training_gym.common.sample_extraction import IMAGE_SAMPLE_LIMIT_ENV

    dataset = EyePromptDataset(n_train=n_train, always_prepare=True)

    def overlay(image):
        # The reward compares each render against these reference eyes.
        return image.run_commands(
            "uv pip install --system 'modal~=1.5.2' 'httpx~=0.28.1' 'pillow~=11.1'",
        ).add_local_dir(LOCAL_REF_DIR, "/root/eye_refs", copy=True)

    # Capture every sample's render: rollout_batch_size * n_samples_per_prompt.
    all_images = str(8 * 8)

    def overlay_all_images(image):
        return overlay(image).env({IMAGE_SAMPLE_LIMIT_ENV: all_images})

    if model == "27b":
        config = TrainConfig(
            model=Qwen3_8_27B(),
            dataset=dataset,
            recipe=Qwen3_8_27b_Recipe(
                custom_rm_function=eye_rm,
                num_rollout=num_rollout,
                rollout_batch_size=8,
                n_samples_per_prompt=8,
                rollout_max_response_len=3072,
                rollout_temperature=1.0,
                save_interval=10,
                apply_chat_template_kwargs='{"enable_thinking": false}',
                image_overlay=overlay_all_images,
            ),
        )
        result = config.train()
        print(f"Training run id: {result.training_run_id}")
        return
    config = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=dataset,
        recipe=SlimeRecipe(
            custom_rm_function=eye_rm,
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=num_rollout,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=6144,
            rollout_temperature=1.0,
            global_batch_size=8,
            max_tokens_per_gpu=16384,
            save_interval=10,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=overlay_all_images,
        ),
    )
    result = config.train()
    print(f"Training run id: {result.training_run_id}")


if __name__ == "__main__":
    import sys

    launch(
        num_rollout=int(sys.argv[1]) if len(sys.argv) > 1 else 1,
        model=sys.argv[2] if len(sys.argv) > 2 else "4b",
    )
