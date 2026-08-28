"""Deliberate failure modes, as negatives for the taste probe.

These are the shapes bad rollouts actually take: a blank wash, invisible pigment,
a single blob, scribble storms, a grey flower, a flower painted off-canvas.
Generating them by hand means the probe can separate them before the policy ever
produces one.
"""

from __future__ import annotations

import random

from reference_sketches import PALETTES, flower_sketch

HEAD = """function setup() {
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  background("%s");
"""
TAIL = "  noLoop();\n}\n"


def _wrap(ground: str, body: str) -> str:
    return HEAD % ground + body + TAIL


def blank(palette: str, seed: int) -> str:
    light, _, _, _, ground = PALETTES[palette]
    return _wrap(
        ground,
        f"""  brush.noStroke(); brush.bleed(0.5);
  for (let i = 0; i < 3; i++) {{ brush.fill("{light}", 10); brush.circle(random(-60,60), random(-60,60), 260); }}
""",
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
    )


def faint(species: str, palette: str, seed: int) -> str:
    """A real flower painted at alpha the watercolour layer cannot show."""
    return flower_sketch(species, palette, seed=seed).replace("78 + 34 * t", "6 + 3 * t")


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
    import re

    return re.sub(r"for \(let k = 0; k < \d+; k\+\+\)", "for (let k = 0; k < 2; k++)", code)


def make_negatives(n: int, seed: int = 0) -> list[tuple[str, str]]:
    from reference_sketches import SPECIES

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
