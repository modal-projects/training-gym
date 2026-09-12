import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
import threading

import modal


_JS_FENCE = re.compile(r"```(?:javascript|js)\s*\n(.*?)```", re.DOTALL)
_BANNED = re.compile(
    r"\b(loadImage|fetch|XMLHttpRequest|WebSocket|document\.|window\.|eval|import|require)\b"
)


def extract_sketch(response: str, parse_response) -> str | None:
    parsed = parse_response(response)
    m = _JS_FENCE.search(parsed.content or "")
    if not m:
        return None
    code = m.group(1).strip()
    if not code or "function setup" not in code or len(code) > 8000:
        return None
    if _BANNED.search(code):
        return None
    return code


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
<script>try{(0,eval)(${JSON.stringify(sketch).replace(/</g, '\\u003c')})}catch(e){window.__err=String(e)}</script>
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
      // Watercolour fills settle across frames.
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


RENDER_APP_NAME = "training-gym-flower-render"


def render_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("chromium", "nodejs", "npm", "fonts-liberation")
        .run_commands(
            "mkdir -p /render",
            "cd /render && npm install --no-audit --no-fund"
            " p5@1.11.3 p5.brush@1.1.2 puppeteer-core@23.11.1",
        )
    )


def render_in_sandbox(code: str) -> tuple[bytes | None, dict]:
    app = modal.App.lookup(RENDER_APP_NAME, create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=app,
        image=render_image(),
        workdir="/render",
        timeout=300,
        cpu=1.0,
        memory=2048,
        block_network=True,
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


ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
REMOTE_ASSETS_DIR = "/root/flower_assets"
REMOTE_CLIP_DIR = "/root/flower_assets/clip"


def download_clip() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download("openai/clip-vit-base-patch32", local_dir=REMOTE_CLIP_DIR)
    shutil.rmtree("/tmp/hf", ignore_errors=True)
    shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)


_CACHE: dict[str, object] = {}
_LOCKS: dict[str, threading.Lock] = {}


def assets_dir() -> str:
    return REMOTE_ASSETS_DIR if os.path.isdir(REMOTE_ASSETS_DIR) else ASSETS_DIR


def reference_pool() -> list[tuple[str, bytes]]:
    with _LOCKS.setdefault("refs", threading.Lock()):
        if "refs" not in _CACHE:
            from PIL import Image

            refs = []
            root = assets_dir()
            for name in sorted(os.listdir(root)):
                if not name.endswith(".webp"):
                    continue
                buf = io.BytesIO()
                Image.open(os.path.join(root, name)).convert("RGB").save(
                    buf, format="PNG"
                )
                refs.append((name.split("_")[0], buf.getvalue()))
            _CACHE["refs"] = refs
    return _CACHE["refs"]


def pick_references(species: str, key: str, k: int = 4) -> list[bytes]:
    pool = reference_pool()
    rng = random.Random(hashlib.sha1(key.encode()).hexdigest())
    same = [png for sp, png in pool if sp == species]
    rest = [png for sp, png in pool if sp != species]
    chosen = rng.sample(same, min(len(same), k // 2))
    chosen += rng.sample(rest, k - len(chosen))
    return chosen


PROBE_PATH_NAME = "flower_taste.npz"


def _load_probe() -> None:
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    data = np.load(os.path.join(assets_dir(), PROBE_PATH_NAME), allow_pickle=False)
    clip = (
        REMOTE_CLIP_DIR
        if os.path.isdir(REMOTE_CLIP_DIR)
        else "openai/clip-vit-base-patch32"
    )
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


def judge_pair(candidate: bytes, reference: bytes, prompt: str, flip: bool, judge):
    def uri(png: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png).decode()

    a, b = (reference, candidate) if flip else (candidate, reference)
    try:
        msg = judge.chat(
            [
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
            timeout=180,
            max_tokens=96,
            temperature=0.3,
        )
        text = msg.get("content") or ""
    except Exception:
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


def judge_win_rate(png: bytes, prompt: str, species: str, judge) -> tuple[float, dict]:
    from concurrent.futures import ThreadPoolExecutor

    refs = pick_references(species, prompt + str(len(png)), JUDGE_REFS)
    with ThreadPoolExecutor(max_workers=len(refs)) as pool:
        votes = list(
            pool.map(
                lambda ir: judge_pair(
                    png, ir[1], prompt, flip=bool(ir[0] % 2), judge=judge
                ),
                enumerate(refs),
            )
        )
    got = [v for v in votes if v is not None]
    return (sum(got) / len(got) if got else 0.0), {
        "judge_votes": len(got),
        "judge_wins": round(sum(got), 2) if got else 0.0,
    }


def ink_fraction(png: bytes) -> float:
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((64, 64))
    hist = img.histogram()
    bg = max(range(256), key=lambda i: hist[i])
    px = list(img.getdata())
    return sum(1 for v in px if abs(v - bg) > 16) / len(px)


def coverage_fraction(png: bytes) -> float:
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
    from PIL import Image, ImageFilter

    img = Image.open(io.BytesIO(png)).convert("L").resize((128, 128))
    blurred = img.filter(ImageFilter.BoxBlur(2))
    px, bpx = list(img.getdata()), list(blurred.getdata())
    return sum(1 for v, b in zip(px, bpx) if abs(v - b) > 40) / len(px)


def dark_fraction(png: bytes) -> float:
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("L").resize((128, 128))
    px = list(img.getdata())
    return sum(1 for v in px if v < 48) / len(px)


GATE_WEIGHT = 0.05
LENGTH_WEIGHT = 0.05
PROBE_WEIGHT = 0.30
JUDGE_WEIGHT = 0.60
LENGTH_TARGET = 1200


def score_png(
    png: bytes | None, label: str, code: str, judge, render_meta: dict
) -> tuple[float, dict, bytes | None]:
    species, _, prompt = label.partition("::")
    if not prompt:
        species, prompt = "hibiscus", label
    if png is None:
        return 0.0, render_meta, None
    meta = {**render_meta, "species": species}
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
    # A fat brush.strokeWeight still renders and still has ink, so the
    # darkness gate is what stops a silhouette from collecting cheap reward.
    if ink < 0.02 or speckle > 0.12 or coverage > 0.97 or dark > 0.3:
        return round(reward, 4), meta, png

    probe = probe_score(png)
    wins, judge_meta = judge_win_rate(png, prompt, species, judge)
    meta.update(probe=round(probe, 3), wins=round(wins, 3), **judge_meta)
    reward += PROBE_WEIGHT * probe + JUDGE_WEIGHT * wins
    return round(reward, 4), meta, png
