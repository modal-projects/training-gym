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
import threading

import modal

from modal_training_gym import Qwen3_5_4B
from modal_training_gym.common.dataset import DatasetConfig

base_model = Qwen3_5_4B()

# ── Prompt grammar ───────────────────────────────────────────────────────

SUBJECTS = [
    "a young woman's eye in soft daylight",
    "a calm half-lidded eye looking to the side",
    "a wide open eye seen close up",
    "a gently downcast eye with long lashes",
    "an eye looking straight at the viewer",
    "a tired eye with a soft lower lid",
    "a delicate eye with a thin double eyelid crease",
    "an eye caught mid-glance",
]
IRIS_COLORS = ["warm brown", "mossy green", "grey blue", "hazel", "deep amber", "slate"]

# The style axis is the user's hand-rated taste, not a guess: each family is a
# cluster of the images they marked "love", and the reward compares a render
# against the loved images of the same family, so prompt and reference agree.
STYLE_FAMILIES = {
    "glitch": (
        "glitched CRT style: horizontal colour-shifted scanline bands and offset "
        "channels laid over a painted eye"
    ),
    "anime": (
        "glossy anime illustration: a large banded iris, bright speculars and a "
        "bold dark lash line"
    ),
    "nouveau": (
        "art-nouveau painting: flowing ornamental curves and pale washed grounds "
        "around the eye"
    ),
    "collage": (
        "painted paper collage: torn translucent colour planes and printed dot "
        "texture layered over the eye"
    ),
    "psych": (
        "psychedelic painting: concentric radiating colour waves rippling out "
        "from the iris"
    ),
    "woodcut": (
        "woodcut-inspired painting: carved dark planes, a strong vertical grain "
        "and a lit iris"
    ),
    "graphic": (
        "graphic painted study: strong flat colour shapes with visible pigment "
        "grain and ink marks"
    ),
}
STYLES = list(STYLE_FAMILIES)

SYSTEM_PROMPT = """\
You write p5.js sketches that paint stylised eye illustrations, using p5.brush watercolour washes for painted grounds, skin, sclera and iris and p5.brush strokes for lashes and hairs.

Rules:
- Reply with a single ```javascript code fence containing a complete sketch and nothing else.
- Define exactly one function: `function setup() { ... }`. Do not define draw().
- Start setup() with: createCanvas(512, 512, WEBGL); angleMode(DEGREES); brush.load();
- The coordinate origin (0,0) is the CENTER of the canvas; x and y range from -256 to 256. Compose around (0,0).
- End setup() with: noLoop();
- Draw with the p5.brush API plus p5's own drawing: background, noStroke(), fill(r,g,b) and fill(r,g,b,alpha), ellipse(x,y,w,h), triangle(...), and beginShape()/vertex(x,y)/bezierVertex(cx1,cy1,cx2,cy2,x,y)/endShape(CLOSE) for custom filled shapes. push/pop, translate, rotate, lerpColor(color(...),color(...),t), for-loops, sin/cos/random are all fine.
- The medium is always PAINT, whatever the requested style: build every area by repetition, drawing the same shape 20-40 times in a for-loop with low alpha (8-25), shifting its colour and position slightly each pass so edges break up and pigment varies. Hard geometric shapes are allowed when the style asks for them, but they must be painted this way rather than filled flat.
- Give the whole canvas a painted ground first, in colours that suit the requested style, then paint the eye over it.
- Do NOT hatch or scribble with the hatch brushes: use p5.brush only for individual lash strands, brow hairs and stray hair strands.
- Allowed brush calls: brush.pick(name), brush.stroke(color), brush.strokeWeight(w), brush.noStroke(), brush.fill(color,alpha), brush.bleed(amount), brush.noFill(), brush.line(x1,y1,x2,y2), brush.circle(x,y,r), brush.polygon([[x,y],...]), brush.spline([[x,y],...], curvature), brush.flowLine(x,y,length,dirAngle), brush.setHatch(brushName,color,weight), brush.hatch(dist,angle,{rand:0.1,continuous:true}), brush.noHatch(), brush.field("seabed"), brush.noField().
- brush.fill(colour, alpha) + brush.bleed(0.05-0.4) paint watercolour washes with soft blooming edges: this is the main way to build skin, sclera and iris. Call brush.noStroke() before a filled shape so it has no outline, brush.noFill() when you are done with washes, and brush.stroke(colour)/brush.strokeWeight(w) before drawing hairs.
- The third argument of brush.circle(x,y,r) is a RADIUS, not a diameter, so an iris is brush.circle(0,0,78) and a pupil brush.circle(0,0,28).
- Draw BIG: the eye opening spans about 380 of the 512 canvas and the socket shading spans about 460. A drawing whose features sit in the middle 150 pixels scores zero.
- Never use brush.beginShape/vertex/endShape or brush.rect.
- Brush names available to brush.pick and brush.setHatch: "pen", "rotring", "2B", "HB", "2H", "cpencil", "charcoal", "hatch_brush", "marker", "marker2". Never use "spray" — speckle textures are rejected.
- The canvas is fully painted, never white paper: start with a background colour that suits the style. Speckle and noise textures score zero.
- No loadImage, no fetch/XHR, no DOM access, no external assets, no comments longer than one line.
- Compose one eye, filling most of the 512x512 canvas, as if photographed close up.

Paint the eye in this order. Vary the colours, proportions, gaze direction and
lash length to suit the brief, but keep the anatomy and the soft painted look:
0. An almond helper, used for both the socket and the eye opening:
   function lid(w,up,lo,cy){ let p=[];
   for(let i=0;i<=26;i++){let t=i/26,x=lerp(-w/2,w/2,t); p.push([x,cy-up*sin(180*t)]);}
   for(let i=26;i>=0;i--){let t=i/26,x=lerp(-w/2,w/2,t); p.push([x,cy+lo*sin(180*t)]);}
   return p; }
1. Skin: background(238,214,203) or another skin tone, then a wide socket wash,
   e.g. brush.noStroke(); brush.bleed(0.4); brush.fill("#c08d74",50);
   brush.polygon(lid(460,150,120,-10));
2. Eye opening: an almond of near-white sclera washed over the socket, its
   corners coming to points rather than a plain ellipse, e.g.
   brush.bleed(0.13); brush.fill("#f5f0ea",95); brush.polygon(lid(380,105,70,0));
3. Iris: radius about 78, its top edge tucked under the lid shadow so the lid
   slightly overlaps it. Paint it as nested washes from a DARK limbal ring at the
   outside to a lighter centre in the requested colour, e.g.
   brush.bleed(0.1); brush.fill("#6f93ab",100); brush.circle(-10,0,78);
   brush.bleed(0.08); brush.fill("#2c4b66",92); brush.circle(-10,0,54);
   then a few thin brush.line fibres radiating from the pupil.
4. Pupil: a soft near-black wash of radius about 28
   (brush.bleed(0.04); brush.fill("#120d10",100); brush.circle(-10,4,28)), then
   brush.noFill() before drawing any hairs.
5. Specular: ONE small white highlight near the top of the iris, plus a faint
   larger glow — never two big cartoon ovals.
6. Upper lash line and lashes: a soft dark band hugging the top of the eye
   opening (loop low-alpha shapes, not one solid black wedge), then 25-40
   INDIVIDUAL lash strands as brush.spline curves with brush.pick("2B") and a
   thin brush.strokeWeight, fanning outward and curling up, longest at the outer
   corner and thinning to hair-fine tips. Every strand STARTS on the upper lid
   line and curves away from it, so the lashes read as a dense fan along the lid
   rather than hairs floating over the eye; walk the lid with a for-loop over the
   lid curve rather than placing them by hand, e.g.
   brush.stroke("#3b241f"); brush.strokeWeight(1.4); brush.pick("2B");
   for (let i=0;i<36;i++){ let t=i/35, x=lerp(-185,185,t), y=-8-102*sin(180*t),
   len=28+52*t, cur=10+30*t;
   brush.spline([[x,y],[x+cur*0.4,y-len*0.5],[x+cur,y-len]], 0.7); }
7. Lower lid: a soft warm-pink crease line under the eye, a light catchlight on
   the lid rim, 10-20 short fine lower lashes, and a pinkish tear duct.
8. Brow: an arc of 40+ short fine brush strokes above the eye, dark at the inner
   end, fading and sweeping outward. Add 2-3 stray hair strands across the skin.
9. Style: on top of that anatomy, commit hard to the style the brief asks for —
   scanline bands, ornamental curves, torn collage planes, radiating waves,
   carved planes — painted in the same layered low-alpha way, and never so
   opaque that the iris, pupil, highlight and lashes stop being readable.
No black wedges, no white paper background, no speckle noise: a painted ground,
a readable eye, and fine individual lash hairs along the lid.
"""

USER_TEMPLATE = (
    "Paint {subject} with an iris in {color}. Style: {style}. "
    "Paint the whole canvas, keep the eye anatomy readable — sclera, gradient "
    "iris with a pupil and a small specular highlight, a dense fan of fine lash "
    "hairs along the upper lid — and push the style hard."
)


def shuffled_combos(seed: int = 7) -> list[tuple[str, str, str]]:
    combos = list(itertools.product(SUBJECTS, IRIS_COLORS, STYLES))
    random.Random(seed).shuffle(combos)
    return combos


def build_prompts(n: int, combos: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    rows = []
    for subject, color, family in itertools.islice(itertools.cycle(combos), n):
        rows.append(
            {
                "prompt": USER_TEMPLATE.format(
                    subject=subject, color=color, style=STYLE_FAMILIES[family]
                ),
                "family": family,
            }
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
                # The reward needs the style family as well as the brief, so it
                # can compare the render against the loved images of that family.
                "label": f"{r['family']}::{r['prompt']}",
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
    // brush.rect misreads p5's CORNER mode here and floods the canvas.
    rect: noop,
  };
  // Flushing a watercolour mask leaves p5's modelview matrix holding the
  // full-canvas quad transform, so every shape after a fill colour change lands
  // half a canvas off. Restoring the matrix around each brush call keeps the
  // sketch's own transforms and undoes the leak.
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
  // Gradients are built with lerpColor, which sketches often call on hex
  // strings; coercing them keeps a whole iris from being lost to one call.
  const realLerp = window.lerpColor;
  window.lerpColor = (a, b, t) => {
    const c = (v) => (typeof v === "string" || typeof v === "number")
      ? window.color(v) : v;
    return realLerp(c(a), c(b), t);
  };
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

# Judge size is the binding constraint on this task. A 4B critic rated
# skin-coloured blobs 3/4, so reward climbed while the images lost their sclera
# and highlight; a 26B-A4B MoE critic stopped that but, with only ~4B active
# parameters, capped honest eyes at 2/4 without discriminating lash quality.
# This is a 27B dense multimodal critic: every parameter looks at every image.
JUDGE_URL = "https://modal-labs-joy-dev--ep-eye-judge-27b-server.us-west.modal.direct"
JUDGE_MODEL = "Qwen/Qwen3.8-27B"
# Qwen3.8 answers from its reasoning channel by default, which leaves the content
# field empty and every vote unparseable, so thinking is turned off explicitly.
JUDGE_TEMPLATE_KWARGS = {"enable_thinking": False}

# The user hand-rated 220 eye illustrations into love / okay / nope. Asking the
# VLM which of two images is better does NOT recover that taste: on held-out
# rated images its verdicts ranked rejected images above loved ones about as
# often as not (AUC ~0.5). So taste is learned from the ratings directly, by a
# ridge probe over CLIP embeddings of all 220 images (held-out love/nope
# AUC 0.78, 0.71 when whole style families are held out), and the VLM is left to
# do only what a probe cannot: check that the drawing is still an eye.
LOCAL_PROBE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "taste_probe.npz"
)
# On a rollout worker the module arrives by cloudpickle without its directory, so
# the probe weights and CLIP snapshot are baked into the image at these paths.
REMOTE_PROBE_PATH = "/root/eye_taste/taste_probe.npz"
REMOTE_CLIP_DIR = "/root/eye_taste/clip"
_PROBE: dict[str, object] = {}
# Renders are scored concurrently, and importing transformers from several
# threads at once yields a half-initialised module, so loading is serialised.
# The lock is created on first use rather than at import: this module is
# cloudpickled to the rollout workers, and a live lock among its globals makes
# it unpicklable, which silently strips the reward's helpers.
_LOCKS: dict[str, threading.Lock] = {}


def _probe():
    """CLIP encoder plus the fitted probe weights, loaded once per worker."""
    with _LOCKS.setdefault("probe", threading.Lock()):
        if "model" not in _PROBE:
            _load_probe()
    return _PROBE


def _load_probe() -> None:
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    path = REMOTE_PROBE_PATH if os.path.exists(REMOTE_PROBE_PATH) else LOCAL_PROBE_PATH
    data = np.load(path, allow_pickle=False)
    name = REMOTE_CLIP_DIR if os.path.isdir(REMOTE_CLIP_DIR) else str(data["model"])
    _PROBE.update(
        model=CLIPModel.from_pretrained(name).eval(),
        proc=CLIPProcessor.from_pretrained(name),
        w=data["w"],
        lo=float(data["lo"]),
        hi=float(data["hi"]),
        torch=torch,
    )


def taste_score(png: bytes) -> float:
    """The user's rated preference for this image, mapped to roughly [0, 1].

    The anchors are the probe's mean score on the images they rejected and on the
    ones they loved, mapped to 0.15 and 0.85 by a logistic curve. Hard clipping
    was worse: half the renders from earlier runs landed below the rejected mean,
    so a clipped term gave the whole bottom of the batch identical reward and no
    gradient to climb.
    """
    import io
    import math

    from PIL import Image

    p = _probe()
    img = Image.open(io.BytesIO(png)).convert("RGB")
    with p["torch"].no_grad():
        f = p["model"].get_image_features(
            **p["proc"](images=[img], return_tensors="pt")
        )
        # Newer transformers wrap the projected embedding in an output object.
        if hasattr(f, "pooler_output"):
            f = f.pooler_output
        v = (f / f.norm(dim=-1, keepdim=True)).numpy()[0]
    raw = float(v @ p["w"][:-1] + p["w"][-1])
    t = (raw - p["lo"]) / (p["hi"] - p["lo"])
    return 1.0 / (1.0 + math.exp(-3.47 * (t - 0.5)))


# CLIP reads short style phrases far better than the long grammar briefs: on the
# rated pool these recover the family for 41% of images against 14% chance,
# where the brief text managed 29%.
STYLE_TEXT = {
    "glitch": "a glitch art eye with colour-shifted scanlines",
    "anime": "a glossy anime eye",
    "nouveau": "an art nouveau eye with ornamental curves",
    "collage": "a torn paper collage eye",
    "psych": "a psychedelic eye with radiating colour waves",
    "woodcut": "a woodcut print eye with carved lines",
    "graphic": "a flat graphic poster eye",
}
# Chance level for picking one family out of seven, and the confidence at which
# a render counts as fully on-brief (the median rated image of a family sits
# near 0.18, so 0.35 is reachable without being free).
STYLE_FLOOR = 1.0 / len(STYLE_TEXT)
STYLE_CEIL = 0.35


def style_score(png: bytes, family: str) -> float:
    """How well the render reads as its prompted style family, not another one.

    Taste alone collapses the batch: one pale watercolour eye scores acceptably
    for every prompt, so the policy stopped varying with the brief. CLIP's
    text tower ranks the render against all seven family descriptions, and this
    term is the probability mass on the family that was actually asked for.
    """
    import io

    from PIL import Image

    p = _probe()
    names = list(STYLE_TEXT)
    if family not in STYLE_TEXT:
        return 0.0
    torch = p["torch"]
    with _LOCKS.setdefault("style", threading.Lock()):
        if "style_text" not in _PROBE:
            with torch.no_grad():
                t = p["model"].get_text_features(
                    **p["proc"](
                        text=[STYLE_TEXT[n] for n in names],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                )
            if hasattr(t, "pooler_output"):
                t = t.pooler_output
            _PROBE["style_text"] = t / t.norm(dim=-1, keepdim=True)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    with torch.no_grad():
        f = p["model"].get_image_features(
            **p["proc"](images=[img], return_tensors="pt")
        )
        if hasattr(f, "pooler_output"):
            f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100.0 * (f @ _PROBE["style_text"].T)
        prob = float(torch.softmax(logits, dim=-1)[0, names.index(family)])
    return max(0.0, min(1.0, (prob - STYLE_FLOOR) / (STYLE_CEIL - STYLE_FLOOR)))


# Each check answered NO caps the whole reward, so an image the probe likes for
# its colour and texture cannot score highly if it stopped being an eye.
CHECKS = (
    (
        "SCLERA",
        "is a pale almond-shaped eye white visible on BOTH sides of the iris, "
        "rather than only a dark smudge",
        0.30,
    ),
    (
        "IRIS",
        "is the iris a graded disc with a distinctly darker pupil inside it",
        0.45,
    ),
    (
        "LASHES",
        "do the lashes run along the upper lid line as fine hairs, rather than "
        "spraying outward from the middle of the eye",
        0.65,
    ),
)
CHECK_BLOCK = "\n".join(
    f"{name}: YES or NO - {question}?" for name, question, _ in CHECKS
)

# Lashes are the part the policy skips, because no other term moves much when
# they are missing, so they get their own graded question and slice of reward.
LASH_QUESTION = (
    "LASHFAN: 0, 1 or 2 for the lashes - 0 = none, or only stray hairs floating "
    "away from the lid; 1 = a few short hairs on the lid line; 2 = a dense fan "
    "of many fine separate lash hairs following the upper lid and curling up"
)
TASTE_WEIGHT = 0.40
STYLE_WEIGHT = 0.20
ANATOMY_WEIGHT = 0.20
LASH_WEIGHT = 0.20
JUDGE_VOTES = 2


def judge_vote(png: bytes, prompt: str) -> tuple[float, dict]:
    """One VLM vote on eye anatomy: which parts are present, how the lashes read."""
    import time

    import httpx

    b64 = base64.b64encode(png).decode()
    body = {
        "model": JUDGE_MODEL,
        "max_tokens": 320,
        "chat_template_kwargs": JUDGE_TEMPLATE_KWARGS,
        # Non-zero so the votes averaged in judge_image actually decorrelate.
        "temperature": 0.7,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This illustration was drawn for the brief:\n"
                            f'"{prompt}"\n\n'
                            "First line, starting with IMAGE_IS:, describe in "
                            "under 12 words what shapes the image actually "
                            "contains. Then answer these, one per line, exactly "
                            f"as named:\n{CHECK_BLOCK}\n{LASH_QUESTION}"
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
            upper = text.upper()
            answers = {
                name: re.search(rf"{name}:\s*\**\s*(YES|NO)", upper)
                for name, _, _ in CHECKS
            }
            if any(answers.values()):
                missing = "".join(
                    name[0]
                    for name, _, _ in CHECKS
                    if answers[name] is None or answers[name].group(1) == "NO"
                )
                present = 1.0 - len(missing) / len(CHECKS)
                fan = re.search(r"LASHFAN:\s*\**\s*([0-2])", upper)
                desc = text.split("IMAGE_IS:")[-1].splitlines()[0].strip()
                return present, {
                    "missing": missing,
                    "lash": int(fan.group(1)) / 2 if fan else 0.0,
                    "desc": desc[:120],
                }
            last_err = f"unparseable: {text[:80]}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"[:200]
    return 0.0, {"judge_error": last_err, "lash": 0.0, "missing": ""}


def judge_image(png: bytes, prompt: str, family: str) -> tuple[float, dict]:
    """Score a render on the user's taste, gated by VLM eye-anatomy checks."""
    from concurrent.futures import ThreadPoolExecutor

    taste = taste_score(png)
    style = style_score(png, family)
    with ThreadPoolExecutor(max_workers=JUDGE_VOTES) as pool:
        votes = list(pool.map(lambda _: judge_vote(png, prompt), range(JUDGE_VOTES)))
    anatomy = sum(s for s, _ in votes) / len(votes)
    lash = sum(m.get("lash", 0.0) for _, m in votes) / len(votes)
    score = (
        TASTE_WEIGHT * taste
        + STYLE_WEIGHT * style
        + ANATOMY_WEIGHT * anatomy
        + LASH_WEIGHT * lash
    )
    # A missing sclera, iris or lash line caps the whole reward: the probe can be
    # charmed by a striking image that is not an eye, and this is the backstop.
    for name, _, cap in CHECKS:
        if sum(1 for _, m in votes if name[0] in m.get("missing", "")) > len(votes) / 2:
            score = min(score, cap)
    return score, {
        "judge_family": family,
        "taste": round(taste, 3),
        "style": round(style, 3),
        "judge_anatomy": round(anatomy, 3),
        "judge_lash": round(lash, 2),
        "judge_missing": " ".join(m.get("missing", "") or "-" for _, m in votes),
        "judge_desc": votes[0][1].get("desc", ""),
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


def speckle_fraction(png: bytes) -> float:
    """Fraction of pixels that differ sharply from their local neighbourhood.

    A painted eye covers the whole canvas but varies smoothly, so coverage alone
    cannot separate it from the speckle and noise storms that fool the judge.
    High-frequency detail can, since noise disagrees with its own neighbours.
    """
    import io

    from PIL import Image, ImageFilter

    img = Image.open(io.BytesIO(png)).convert("L").resize((128, 128))
    blurred = img.filter(ImageFilter.BoxBlur(2))
    px, bpx = list(img.getdata()), list(blurred.getdata())
    return sum(1 for v, b in zip(px, bpx) if abs(v - b) > 40) / len(px)


def score_response(response: str, label: str) -> tuple[float, dict, bytes | None]:
    family, _, prompt = label.partition("::")
    if not prompt:
        family, prompt = next(iter(STYLE_FAMILIES)), label
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
    meta["coverage"] = round(coverage_fraction(png), 3)
    speckle = speckle_fraction(png)
    meta["speckle"] = round(speckle, 3)
    # This style paints the whole canvas, so only noise is gated: speckle storms
    # otherwise read to the judge as texture and score like real rendering.
    if speckle > 0.12:
        return 0.02, meta, png
    judge_score, judge_meta = judge_image(png, prompt, family)
    meta.update(judge_meta)
    return 0.05 + 0.95 * judge_score, meta, png


async def eye_rm(args, sample, **kwargs) -> float:
    import asyncio

    label = getattr(sample, "label", None) or ""
    reward, meta, png = await asyncio.to_thread(score_response, sample.response, label)
    md = {**(getattr(sample, "metadata", None) or {}), **meta}
    if png is not None:
        md["image"] = png
    sample.metadata = md
    return reward


# ── Training entrypoint ──────────────────────────────────────────────────


def launch(
    num_rollout: int,
    n_train: int = 256,
    model: str = "4b",
    load: str = "",
) -> None:
    """Train the eye task. ``load`` continues from a training checkpoint dir."""
    from modal_training_gym import (
        Qwen3_8_27B,
        Qwen3_8_27b_Recipe,
        SlimeRecipe,
        TrainConfig,
    )
    from modal_training_gym.common.sample_extraction import IMAGE_SAMPLE_LIMIT_ENV

    dataset = EyePromptDataset(n_train=n_train, always_prepare=True)

    def overlay(image):
        # The reward scores each render with a CLIP probe fitted on the user's
        # ratings, so the worker needs CLIP weights and the probe file locally.
        return image.run_commands(
            "uv pip install --system 'modal~=1.5.2' 'httpx~=0.28.1' 'pillow~=11.1'",
            # The HF cache path is a volume mount at runtime, so the build
            # has to download through a scratch cache and leave it empty.
            f'HF_HOME=/tmp/hf python -c "from huggingface_hub import '
            f"snapshot_download as d; d('openai/clip-vit-base-patch32', "
            f"local_dir='{REMOTE_CLIP_DIR}')\"",
            "rm -rf /tmp/hf /root/.cache/huggingface",
        ).add_local_file(LOCAL_PROBE_PATH, REMOTE_PROBE_PATH, copy=True)

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
                load=load,
                extra_config={"override_opt_param_scheduler": True} if load else {},
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
            load=load,
            # The saved scheduler stops at the previous horizon, so continuing
            # past it needs the new schedule to win over the checkpoint's.
            extra_config={"override_opt_param_scheduler": True} if load else {},
        ),
    )
    result = config.train()
    print(f"Training run id: {result.training_run_id}")


if __name__ == "__main__":
    import sys

    launch(
        num_rollout=int(sys.argv[1]) if len(sys.argv) > 1 else 1,
        model=sys.argv[2] if len(sys.argv) > 2 else "4b",
        load=sys.argv[3] if len(sys.argv) > 3 else "",
    )
