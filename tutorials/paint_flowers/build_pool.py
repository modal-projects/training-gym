"""Rebuild the reference pool and the taste probe from scratch.

modal run tutorials/paint_flowers/build_pool.py
"""

from __future__ import annotations

import base64
import collections
import concurrent.futures as cf
import io
import itertools
import pathlib
import random
import re
import sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from helpers import RENDER_APP_NAME, RENDER_JS, render_image


LOVE = [
    1,
    2,
    3,
    5,
    6,
    12,
    13,
    14,
    17,
    18,
    19,
    21,
    22,
    25,
    26,
    27,
    28,
    29,
    31,
    35,
    36,
    39,
    41,
    42,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    54,
    57,
    58,
    59,
    60,
    61,
    62,
    65,
    66,
    67,
    68,
    69,
    70,
    72,
    73,
    75,
    76,
    77,
    78,
    81,
    82,
    83,
    84,
    85,
    86,
    89,
    90,
    91,
    92,
    93,
    94,
    97,
    98,
    100,
    101,
    102,
    105,
    106,
    109,
    110,
    113,
    114,
    117,
    118,
    120,
    121,
    122,
    123,
    124,
    125,
    127,
    129,
    130,
    132,
    133,
    134,
    137,
    138,
    140,
    141,
    142,
    144,
    145,
    147,
    149,
    150,
    153,
    154,
    155,
    156,
    157,
    158,
    161,
    162,
    163,
    165,
    166,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    177,
    180,
    181,
    182,
    185,
    186,
    190,
]

PALETTES = {
    "peach": ("#f7c9a8", "#ee9d72", "#d9713f", "#8a4a2b", "#f6efe6"),
    "crimson": ("#e0798a", "#c94a5e", "#8f2038", "#4a101f", "#f3e9e4"),
    "butter": ("#f6e2a0", "#e9c257", "#c9922a", "#7a5410", "#f5f1e2"),
    "lilac": ("#dccbec", "#b195d4", "#7f5aab", "#42295c", "#efeaf3"),
    "coral": ("#f7b9a4", "#e97f66", "#c9503a", "#6f2519", "#f7ece6"),
    "indigo": ("#bcc6e8", "#7a8bc8", "#4a5896", "#232a54", "#eceef6"),
    "blush": ("#f8d8de", "#eba6b6", "#d2778c", "#7f3346", "#f8f0f0"),
    "amber": ("#f4cf95", "#e5a145", "#c2761c", "#6d3b0c", "#f6efe3"),
}

SPECIES = {
    "hibiscus": dict(petals=5, length=150, width=112, notch=0.26, curl=14, rows=1),
    "poppy": dict(petals=4, length=142, width=136, notch=0.12, curl=24, rows=1),
    "cosmos": dict(petals=8, length=146, width=64, notch=0.34, curl=6, rows=1),
    "peony": dict(petals=9, length=124, width=86, notch=0.08, curl=20, rows=2),
    "tulip": dict(petals=6, length=138, width=76, notch=0.04, curl=28, rows=1),
    "magnolia": dict(petals=7, length=152, width=70, notch=0.06, curl=16, rows=1),
    "dahlia": dict(petals=12, length=132, width=54, notch=0.18, curl=10, rows=2),
    "iris": dict(petals=6, length=148, width=92, notch=0.22, curl=32, rows=1),
}

N_NEGATIVES = 84


def flower_sketch(species: str, palette: str, *, seed: int = 0) -> str:
    rng = random.Random(seed)
    spec = SPECIES[species]
    light, mid, dark, centre, ground = PALETTES[palette]
    n = spec["petals"]
    length = spec["length"] + rng.randint(-12, 12)
    width = spec["width"] + rng.randint(-10, 10)
    notch = spec["notch"]
    curl = spec["curl"] + rng.randint(-6, 6)
    rows = spec["rows"]
    spin = rng.randint(0, 60)
    cx, cy = rng.randint(-16, 16), rng.randint(-24, 10)
    passes = rng.choice([5, 6, 7])
    bleed = round(rng.uniform(0.12, 0.26), 3)
    stem = rng.random() < 0.8
    zoom = round(rng.uniform(0.85, 1.25), 3)
    length = int(length * zoom)
    width = int(width * zoom)

    return f"""\
function petalPts(ox, oy, a, len, wid, notch, curl) {{
  let pts = [];
  const put = (u, v) => {{
    let x = ox + u * cos(a - 90) - v * sin(a - 90);
    let y = oy + u * sin(a - 90) + v * cos(a - 90);
    pts.push([x, y]);
  }};
  for (let i = 0; i <= 16; i++) {{
    let t = i / 16;
    put(-wid / 2 * sin(180 * t) * (1 - 0.3 * t), len * t + curl * sin(180 * t));
  }}
  for (let i = 16; i >= 0; i--) {{
    let t = i / 16;
    let dent = t > 0.85 ? notch * len * (t - 0.85) * 6 : 0;
    put(wid / 2 * sin(180 * t) * (1 - 0.3 * t), len * t + curl * sin(180 * t) - dent);
  }}
  return pts;
}}

function setup() {{
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  randomSeed({seed});
  background("{ground}");
  const cx = {cx}, cy = {cy};

  brush.noStroke();
  brush.bleed(0.5);
  for (let i = 0; i < 3; i++) {{
    brush.fill("{light}", 12);
    brush.circle(cx + random(-50, 50), cy + random(-40, 60), random(200, 300));
  }}
{_stem_js(rng, cx, cy) if stem else ""}
  const cols = ["{light}", "{mid}", "{dark}"];
  for (let row = {rows - 1}; row >= 0; row--) {{
    let scale = 1 - 0.3 * row;
    for (let k = 0; k < {n}; k++) {{
      let a = {spin} + row * {180 / n:.1f} + k * {360 / n:.2f};
      for (let i = 0; i < {passes}; i++) {{
        let t = i / {passes};
        brush.bleed({bleed} + random(-0.04, 0.06));
        brush.fill(lerpColor(color(cols[0]), color(cols[2]), 0.25 + 0.55 * t + 0.2 * row),
                   78 + 34 * t);
        brush.polygon(petalPts(cx + random(-3, 3), cy + random(-3, 3), a + random(-4, 4),
                               {length} * scale * (0.85 + 0.18 * t),
                               {width} * scale * (1.06 - 0.22 * t), {notch}, {curl}));
      }}
    }}
  }}

  brush.noFill();
  brush.pick("cpencil");
  brush.stroke("{dark}");
  brush.strokeWeight(0.8);
  for (let k = 0; k < {n}; k++) {{
    let a = {spin} + k * {360 / n:.2f};
    brush.polygon(petalPts(cx, cy, a, {length} * 0.95, {width} * 0.98, {notch}, {curl}));
  }}

  brush.noStroke();
  for (let i = 0; i < 5; i++) {{
    brush.bleed(0.1);
    brush.fill("{mid}", 115);
    brush.circle(cx + random(-4, 4), cy + random(-4, 4), 48 - i * 4);
  }}
  brush.fill("{centre}", 150);
  brush.circle(cx, cy, 24);
  brush.noFill();
  brush.pick("2B");
  brush.stroke("{centre}");
  brush.strokeWeight(1.4);
  for (let i = 0; i < 20; i++) {{
    let a = random(360), r = random(22, 52);
    brush.line(cx, cy, cx + r * cos(a), cy + r * sin(a));
    brush.circle(cx + r * cos(a), cy + r * sin(a), 3);
  }}
  noLoop();
}}
"""


GREENS = [
    ("#5c7247", "#6d8a4f", "#3f5730"),
    ("#4f6b4a", "#6b8a63", "#35492f"),
    ("#6b7a3c", "#87994d", "#4a5527"),
]


def _stem_js(rng: random.Random, cx: int, cy: int) -> str:
    stalk, leaf, edge = rng.choice(GREENS)
    lean = rng.randint(-34, 34)
    top = cy + 40
    sides = [-1, 1] if rng.random() < 0.75 else [rng.choice([-1, 1])]
    leaves = ""
    for s in sides:
        h = rng.randint(60, 190)
        span = rng.randint(62, 116)
        tilt = rng.randint(-30, 25)
        leaves += f"""
  {{
    let by = {cy + 40} + {h}, bx = {cx} + ({lean} - {cx}) * ({h} / 216);
    let pts = leafPts(bx, by, {s * 1.0:.1f}, {span}, {tilt});
    for (let i = 0; i < 2; i++) {{
      brush.bleed(0.1);
      brush.fill(i ? "{leaf}" : "{stalk}", 120);
      brush.polygon(pts);
    }}
    brush.noFill();
    brush.pick("cpencil");
    brush.stroke("{edge}");
    brush.strokeWeight(1);
    brush.polygon(pts);
    brush.line(bx, by, bx + {s} * {span} * cos({tilt}), by + {span} * sin({tilt}));
    brush.noStroke();
  }}"""
    return f"""
  function leafPts(bx, by, s, span, tilt) {{
    let pts = [];
    const put = (u, v) => pts.push([bx + s * (u * cos(tilt) - v * sin(tilt)),
                                    by + (u * sin(tilt) + v * cos(tilt))]);
    for (let i = 0; i <= 14; i++) {{ let t = i / 14; put(span * t, -0.3 * span * sin(180 * t)); }}
    for (let i = 14; i >= 0; i--) {{ let t = i / 14; put(span * t, 0.26 * span * sin(180 * t)); }}
    return pts;
  }}

  brush.noStroke();
  brush.bleed(0.08);
  for (let i = 0; i < 2; i++) {{
    brush.fill("{stalk}", 140);
    brush.polygon([[{cx} - 4 + random(-1, 1), {top}], [{cx} + 4 + random(-1, 1), {top}],
                   [{lean} + 5, 256], [{lean} - 5, 256]]);
  }}{leaves}
"""


HEAD = """function setup() {
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  randomSeed(%d);
  background("%s");
"""
TAIL = "  noLoop();\n}\n"


def _wrap(ground: str, body: str, seed: int) -> str:
    return HEAD % (seed, ground) + body + TAIL


def blank(palette: str, seed: int) -> str:
    light, _, _, _, ground = PALETTES[palette]
    return _wrap(
        ground,
        f"""  brush.noStroke(); brush.bleed(0.5);
  for (let i = 0; i < 3; i++) {{ brush.fill("{light}", 10); brush.circle(random(-60,60), random(-60,60), 260); }}
""",
        seed,
    )


def blob(palette: str, seed: int) -> str:
    _, mid, dark, _, ground = PALETTES[palette]
    rng = random.Random(seed)
    return _wrap(
        ground,
        f"""  brush.noStroke(); brush.bleed({rng.uniform(0.2, 0.6):.2f});
  brush.fill("{mid}", 120); brush.circle(0, 0, {rng.randint(150, 340)});
  brush.fill("{dark}", 90); brush.circle({rng.randint(-40, 40)}, {rng.randint(-40, 40)}, {rng.randint(60, 160)});
""",
        seed,
    )


def scribble(palette: str, seed: int) -> str:
    _, mid, dark, _, ground = PALETTES[palette]
    return _wrap(
        ground,
        f"""  brush.noFill(); brush.pick("2B"); brush.strokeWeight(2);
  for (let i = 0; i < 90; i++) {{
    brush.stroke(i % 2 ? "{mid}" : "{dark}");
    brush.line(random(-240,240), random(-240,240), random(-240,240), random(-240,240));
  }}
""",
        seed,
    )


def faint(species: str, palette: str, seed: int) -> str:
    return flower_sketch(species, palette, seed=seed).replace(
        "78 + 34 * t", "6 + 3 * t"
    )


def grey(species: str, palette: str, seed: int) -> str:
    code = flower_sketch(species, palette, seed=seed)
    for hexcol in [c for tup in PALETTES.values() for c in tup[:4]]:
        code = code.replace(hexcol, "#8b8b8b")
    return code


def offcanvas(species: str, palette: str, seed: int) -> str:
    code = flower_sketch(species, palette, seed=seed)
    return code.replace("const cx = ", "const cx = 430 + 0 * ", 1)


def two_petals(species: str, palette: str, seed: int) -> str:
    code = flower_sketch(species, palette, seed=seed)
    return re.sub(
        r"for \(let k = 0; k < \d+; k\+\+\)", "for (let k = 0; k < 2; k++)", code
    )


def make_negatives(n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    kinds = ["blank", "blob", "scribble", "faint", "grey", "offcanvas", "two_petals"]
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        sp, pa = rng.choice(list(SPECIES)), rng.choice(list(PALETTES))
        s = rng.randint(0, 10**6)
        fn = globals()[kind]
        code = fn(pa, s) if kind in ("blank", "blob", "scribble") else fn(sp, pa, s)
        out.append((f"neg{i:03d}_{kind}", code))
    return out


CLIP = "openai/clip-vit-base-patch32"

app = modal.App("training-gym-flower-pool")
POOL_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "pillow", "torch", "transformers")
    .add_local_file(
        HERE / "build_pool.py", remote_path="/root/build_pool.py", copy=True
    )
    .add_local_file(HERE / "helpers.py", remote_path="/root/helpers.py", copy=True)
)


def open_sandbox(render_app: modal.App, timeout: int = 5400) -> modal.Sandbox:
    sb = modal.Sandbox.create(
        "sleep",
        "infinity",
        app=render_app,
        image=render_image(),
        workdir="/render",
        timeout=timeout,
        cpu=4.0,
        memory=8192,
    )
    sb.filesystem.write_text(RENDER_JS, "/render/render.js")
    return sb


def render_in(sb: modal.Sandbox, name: str, code: str) -> bytes | None:
    path = f"/render/{name}.js"
    sb.filesystem.write_text(code, path)
    proc = sb.exec("node", "/render/render.js", path, timeout=180)
    proc.wait()
    out = proc.stdout.read()
    if "PNGB64:" not in out:
        return None
    return base64.b64decode(out.split("PNGB64:", 1)[1].strip())


def render_corpus(out: pathlib.Path, sandboxes: int = 64, workers: int = 8) -> None:
    rng = random.Random(11)
    items = []
    for species, palette in itertools.product(SPECIES, PALETTES):
        for _ in range(3):
            seed = rng.randint(0, 10**6)
            items.append(
                (
                    f"pos{len(items):03d}_{species}_{palette}_{seed}",
                    flower_sketch(species, palette, seed=seed),
                )
            )
    items += make_negatives(N_NEGATIVES, seed=3)
    out.mkdir(parents=True, exist_ok=True)
    render_app = modal.App.lookup(RENDER_APP_NAME, create_if_missing=True)
    with cf.ThreadPoolExecutor(sandboxes) as created:
        sbs = list(created.map(lambda _: open_sandbox(render_app), range(sandboxes)))
    chunk = (len(items) + sandboxes - 1) // sandboxes
    try:

        def run(sb, batch):
            with cf.ThreadPoolExecutor(workers) as ex:
                return list(ex.map(lambda it: (it[0], render_in(sb, *it)), batch))

        with cf.ThreadPoolExecutor(sandboxes) as ex:
            results = ex.map(
                lambda i: run(sbs[i], items[i * chunk : (i + 1) * chunk]),
                range(sandboxes),
            )
            written = [item for batch in results for item in batch]
            failed = 0
            for name, png in sorted(written, key=lambda item: item[0]):
                if png is None:
                    failed += 1
                else:
                    (out / f"{name}.png").write_bytes(png)
        print(f"rendered {len(items) - failed}/{len(items)}")
    finally:
        for sb in sbs:
            sb.terminate()


def embed(paths: list[pathlib.Path]):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    torch.manual_seed(0)
    model, proc = (
        CLIPModel.from_pretrained(CLIP).eval(),
        CLIPProcessor.from_pretrained(CLIP),
    )
    vecs = []
    for i in range(0, len(paths), 32):
        ims = [Image.open(p).convert("RGB") for p in paths[i : i + 32]]
        with torch.no_grad():
            f = model.get_image_features(**proc(images=ims, return_tensors="pt"))
        f = getattr(f, "pooler_output", f)
        vecs.append((f / f.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(vecs).astype(np.float32)


def fit_probe(corpus: pathlib.Path):
    import numpy as np
    import torch

    pos = sorted(corpus.glob("pos*.png"))
    neg = sorted(corpus.glob("neg*.png"))
    paths = pos + neg
    X = embed(paths)
    index = {p.stem: i for i, p in enumerate(paths)}
    loved = set(LOVE)
    love = [index[p.stem] for p in pos if int(p.stem[3:6]) in loved]
    bad = [index[p.stem] for p in neg]

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    holdout = set(rng.choice(love, 24, replace=False)) | set(
        rng.choice(bad, 16, replace=False)
    )
    train = [i for i in love + bad if i not in holdout]
    y = torch.tensor([1.0 if i in set(love) else 0.0 for i in train])
    Xt = torch.tensor(X[train])
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, y)
        (loss + 1e-3 * w.pow(2).sum()).backward()
        opt.step()
    wn, bn = w.detach().numpy(), float(b.detach().numpy()[0])

    def score(ix):
        return X[np.array(ix)] @ wn + bn

    hl = sorted(i for i in holdout if i in set(love))
    hb = sorted(i for i in holdout if i in set(bad))
    pairwise = float(np.mean([[a > c for c in score(hb)] for a in score(hl)]))
    print("held-out (love, failure) pairs ordered correctly:", round(pairwise, 3))
    probe = io.BytesIO()
    np.savez(
        probe,
        w=wn,
        b=np.array([bn], dtype=np.float32),
        lo=float(score(bad).mean()),
        hi=float(score(love).mean()),
    )
    return probe.getvalue(), X, [p.stem for p in paths], wn, bn


def pick_refs(corpus: pathlib.Path, X, names, w, b, k: int = 20):
    from PIL import Image

    score = {n: float(X[i] @ w + b) for i, n in enumerate(names)}
    loved = set(LOVE)
    best: dict[tuple[str, str], str] = {}
    for name in names:
        if not name.startswith("pos") or int(name[3:6]) not in loved:
            continue
        _, species, palette, _ = name.split("_")
        key = (species, palette)
        if key not in best or score[name] > score[best[key]]:
            best[key] = name
    species_seen: collections.Counter = collections.Counter()
    palette_seen: collections.Counter = collections.Counter()
    chosen = []
    for (species, palette), name in sorted(
        best.items(), key=lambda kv: (-score[kv[1]], kv[1])
    ):
        if species_seen[species] >= 3 or palette_seen[palette] >= 3:
            continue
        chosen.append(name)
        species_seen[species] += 1
        palette_seen[palette] += 1
        if len(chosen) == k:
            break
    refs: dict[str, bytes] = {}
    for name in sorted(chosen):
        img = Image.open(corpus / f"{name}.png").convert("RGB")
        buf = io.BytesIO()
        img.resize((448, 448), Image.LANCZOS).save(
            buf, format="WEBP", quality=80, method=6
        )
        refs[f"{name.split('_', 1)[1]}.webp"] = buf.getvalue()
    return refs, sorted(chosen)


@app.function(image=POOL_IMAGE, timeout=7200, cpu=4.0, memory=8192, serialized=True)
def build_pool_remote(corpus: str = "/tmp/flower-corpus") -> dict:
    dest = pathlib.Path(corpus)
    n_pos = len(SPECIES) * len(PALETTES) * 3
    if (
        len(list(dest.glob("pos*.png"))) < n_pos
        or len(list(dest.glob("neg*.png"))) < N_NEGATIVES
    ):
        render_corpus(dest)
    probe, X, names, w, b = fit_probe(dest)
    refs, _ = pick_refs(dest, X, names, w, b)
    return {"refs": refs, "probe": probe}


@app.local_entrypoint()
def main(corpus: str = "/tmp/flower-corpus") -> None:
    payload = build_pool_remote.remote(corpus)
    dest = HERE / "artifacts"
    dest.mkdir(exist_ok=True)
    for old in dest.glob("*.webp"):
        old.unlink()
    for name, data in payload["refs"].items():
        (dest / name).write_bytes(data)
    (dest / "flower_taste.npz").write_bytes(payload["probe"])
    print(f"wrote {len(payload['refs'])} references to {dest}")
