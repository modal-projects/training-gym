"""Parametric hand-authored p5.brush watercolour flower sketches.

Seeds the reference pool: the blog's pool is hand-rated model output, but a cold
base model cannot paint anything worth rating in a niche library, so the pool
starts from sketches written by hand, varied over a parameter grid, and then
hand-rated the same way.

Authoring constraints discovered against the renderer:
  * p5 transforms do not reach the brush layer reliably (`translate` is dropped),
    so every vertex is computed in absolute canvas coordinates.
  * watercolour fill alpha below ~40 is invisible however many passes you stack.
"""

from __future__ import annotations

import random

# palette -> (light, mid, dark, centre, ground)
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

# species -> petal geometry
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


def flower_sketch(species: str, palette: str, *, seed: int = 0) -> str:
    """A complete p5.js sketch painting one watercolour flower."""
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
// petal outline in absolute canvas coords: tip at angle `a` from (ox, oy)
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
  background("{ground}");
  const cx = {cx}, cy = {cy};

  // faint ground wash so the flower sits in paper rather than on it
  brush.noStroke();
  brush.bleed(0.5);
  for (let i = 0; i < 3; i++) {{
    brush.fill("{light}", 12);
    brush.circle(cx + random(-50, 50), cy + random(-40, 60), random(200, 300));
  }}
{_stem_js(rng, cx, cy) if stem else ""}
  // petals, back row first, each built from a few offset watercolour passes
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

  // petal edges, drawn dry so the shapes keep an outline
  brush.noFill();
  brush.pick("cpencil");
  brush.stroke("{dark}");
  brush.strokeWeight(0.8);
  for (let k = 0; k < {n}; k++) {{
    let a = {spin} + k * {360 / n:.2f};
    brush.polygon(petalPts(cx, cy, a, {length} * 0.95, {width} * 0.98, {notch}, {curl}));
  }}

  // centre disc and stamens
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
    # leaves at different heights and sizes on each side, sometimes only one
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
  // pointed leaf outline, absolute coords, tip `span` away at angle `tilt`
  function leafPts(bx, by, s, span, tilt) {{
    let pts = [];
    const put = (u, v) => pts.push([bx + s * (u * cos(tilt) - v * sin(tilt)),
                                    by + (u * sin(tilt) + v * cos(tilt))]);
    for (let i = 0; i <= 14; i++) {{ let t = i / 14; put(span * t, -0.3 * span * sin(180 * t)); }}
    for (let i = 14; i >= 0; i--) {{ let t = i / 14; put(span * t, 0.26 * span * sin(180 * t)); }}
    return pts;
  }}

  // stem and leaves, painted before the petals so they read as behind
  brush.noStroke();
  brush.bleed(0.08);
  for (let i = 0; i < 2; i++) {{
    brush.fill("{stalk}", 140);
    brush.polygon([[{cx} - 4 + random(-1, 1), {top}], [{cx} + 4 + random(-1, 1), {top}],
                   [{lean} + 5, 256], [{lean} - 5, 256]]);
  }}{leaves}
"""
