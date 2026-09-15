"""Local smoke test of the eye reward pipeline (renderer + judge + gates)."""

import sys

sys.path.insert(0, "eye_rl")
from train_eyes import (  # noqa: E402
    EyePromptDataset,
    extract_sketch,
    judge_image,
    render_sketch,
    score_response,
    speckle_fraction,
    taste_score,
)

GOOD_SKETCH = """function setup() {
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  background(238, 214, 203);
  noStroke();
  // soft socket shading
  for (let i = 0; i < 28; i++) {
    fill(212, 172, 162, 10);
    ellipse(0, -70 - i * 2, 440 - i * 8, 210 - i * 4);
    fill(226, 190, 180, 8);
    ellipse(-10, 110 - i, 380 - i * 8, 150 - i * 3);
  }
  // eye opening, soft edged
  for (let i = 0; i < 22; i++) {
    fill(246, 238, 236, 22);
    beginShape();
    vertex(-176 + i, 4);
    bezierVertex(-118 + i, -116 + i * 1.4, 108 - i, -126 + i * 1.4, 176 - i, -12);
    bezierVertex(106 - i, 52 - i, -108 + i, 68 - i, -176 + i, 4);
    endShape(CLOSE);
  }
  // iris: dark limbal ring blending to a lighter centre
  for (let i = 0; i < 24; i++) {
    let t = i / 23;
    fill(lerpColor(color(26, 48, 30), color(126, 168, 108), t));
    ellipse(-4, -2, 124 - i * 3.8, 124 - i * 3.8);
  }
  // pupil with blurred edge
  for (let i = 0; i < 10; i++) {
    fill(16, 16, 18, 60);
    ellipse(-4, -2, 44 + i * 2.4, 44 + i * 2.4);
  }
  // lid shadow over the top of the iris
  for (let i = 0; i < 18; i++) {
    fill(120, 78, 66, 12);
    ellipse(-4, -104 + i * 2, 300 - i * 6, 120 - i * 4);
  }
  // one small specular plus a faint glow
  fill(255, 255, 252, 40);
  ellipse(-22, -34, 46, 34);
  fill(255);
  ellipse(-24, -38, 14, 11);
  // lashes: individual strands
  brush.pick("2B");
  brush.stroke("#241a18");
  for (let i = 0; i < 32; i++) {
    let t = i / 31;
    let x = -170 + t * 350;
    let y = -26 - 44 * sin(t * 180);
    let dx = (t - 0.5) * 60;
    brush.strokeWeight(1.6 - t * 0.6);
    brush.spline([[x, y], [x + dx * 0.5, y - 20], [x + dx, y - 36]], 0.6);
  }
  // lower lashes
  for (let i = 0; i < 14; i++) {
    let t = i / 13;
    let x = -130 + t * 250;
    let y = 46 + 26 * sin(t * 180);
    let dx = (t - 0.5) * 30;
    brush.strokeWeight(1);
    brush.spline([[x, y], [x + dx * 0.5, y + 10], [x + dx, y + 20]], 0.6);
  }
  // brow
  brush.pick("cpencil");
  brush.stroke("#4a3227");
  for (let i = 0; i < 44; i++) {
    let t = i / 43;
    let x = -196 + t * 380;
    let y = -164 - 40 * sin(t * 180);
    brush.strokeWeight(1.4 - t * 0.6);
    brush.line(x, y, x + 18, y + 6 - 24 * cos(t * 180));
  }
  // lower crease and tear duct
  brush.pick("pen");
  brush.stroke("#b98a7e");
  brush.strokeWeight(2);
  brush.spline([[-158, 44], [-70, 88], [30, 94], [120, 72], [174, 16]], 0.5);
  fill(206, 140, 132, 90);
  ellipse(-166, 6, 30, 22);
  noLoop();
}"""

# The flat cel-shaded eye the previous task trained: it must now score clearly
# below the painted sketch above.
ANIME_SKETCH = """function setup() {
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  background(255);
  noStroke();
  // sclera
  fill(252, 250, 252);
  beginShape();
  vertex(-180, 4);
  bezierVertex(-120, -125, 110, -135, 180, -14);
  bezierVertex(110, 58, -110, 76, -180, 4);
  endShape(CLOSE);
  // banded iris
  fill(20, 74, 52);
  ellipse(-4, -4, 140, 176);
  fill(46, 150, 104);
  ellipse(-4, 10, 120, 148);
  fill(140, 226, 178);
  ellipse(-4, 34, 96, 92);
  // pupil
  fill(18, 14, 22);
  ellipse(-4, -2, 56, 86);
  // specular highlights
  fill(255);
  ellipse(-34, -42, 44, 34);
  ellipse(28, 38, 20, 16);
  // thick upper lash line + spikes
  fill(20, 16, 24);
  beginShape();
  vertex(-190, 8);
  bezierVertex(-128, -146, 116, -156, 192, -18);
  bezierVertex(120, -66, 90, -94, 0, -96);
  bezierVertex(-96, -94, -142, -18, -190, 8);
  endShape(CLOSE);
  triangle(150, -56, 218, -110, 184, -28);
  triangle(96, -96, 146, -144, 140, -84);
  triangle(-190, 8, -226, -26, -168, -16);
  // lower lid + crease
  brush.pick("pen");
  brush.stroke("#2a2230");
  brush.strokeWeight(4);
  brush.spline([[-174,24],[-80,66],[20,74],[110,56],[176,2]], 0.5);
  brush.strokeWeight(2);
  brush.spline([[-150,-34],[-60,-112],[40,-122],[130,-92]], 0.5);
  noLoop();
}"""

PENCIL_SKETCH = """function setup() {
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  background(245, 240, 230);
  // iris + pupil, hatched
  brush.pick("cpencil");
  brush.stroke("#2f7d43");
  brush.strokeWeight(1);
  brush.setHatch("cpencil", "#2f7d43", 1);
  brush.hatch(4, 45, {rand: 0.1, continuous: true});
  brush.circle(0, -6, 60);
  brush.noHatch();
  brush.pick("charcoal");
  brush.stroke("#101010");
  brush.strokeWeight(2);
  brush.setHatch("charcoal", "#101010", 1);
  brush.hatch(2, 0, {rand: 0.05});
  brush.circle(0, -6, 24);
  brush.noHatch();
  // lids
  brush.pick("2B");
  brush.stroke("#242424");
  brush.strokeWeight(4);
  brush.spline([[-170,0],[-96,-64],[0,-84],[98,-60],[170,0]], 0.5);
  brush.strokeWeight(3);
  brush.spline([[-170,0],[-92,52],[0,68],[96,50],[170,0]], 0.5);
  // crease
  brush.pick("cpencil");
  brush.stroke("#4a4a4a");
  brush.strokeWeight(2);
  brush.spline([[-150,-26],[-70,-96],[10,-114],[104,-88],[156,-30]], 0.5);
  // lashes
  brush.pick("pen");
  brush.stroke("#1a1a1a");
  brush.strokeWeight(2);
  for (let i = 0; i < 9; i++) {
    let x = -140 + i * 34;
    brush.line(x, -30 - (i%3)*8, x * 1.14, -70 - (i%3)*10);
  }
  noLoop();
}"""

BAD_SKETCH_RUNTIME = """function setup() {
  createCanvas(512, 512, WEBGL);
  undefinedFunctionCall();
  noLoop();
}"""

GOOD_RESPONSE = f"```javascript\n{GOOD_SKETCH}\n```"
PROMPT = (
    "Paint a young woman's eye in soft daylight with an iris in mossy green, "
    "in soft digital painting, semi-realistic."
)
FAMILY = "nouveau"
# The reward reads the style family off the front of the dataset label.
LABEL = f"{FAMILY}::{PROMPT}"


def main():
    # gates
    assert extract_sketch("no code here") is None
    assert extract_sketch("```javascript\nfetch('x')\nfunction setup(){}\n```") is None
    assert extract_sketch("```python\nprint(1)\n```") is None
    assert extract_sketch(GOOD_RESPONSE) is not None
    print("gates: OK")

    # a sketch that draws nothing: recovery may salvage the canvas, but the
    # blank-ink gate has to keep its reward at the floor
    bad_reward, bad_meta, _ = score_response(
        f"```javascript\n{BAD_SKETCH_RUNTIME}\n```", LABEL
    )
    print("bad sketch reward:", bad_reward, bad_meta)
    assert bad_reward <= 0.02

    # render success path
    png, meta = render_sketch(GOOD_SKETCH)
    print("good sketch render:", meta, "bytes:", len(png) if png else 0)
    assert png is not None
    open("/tmp/eye_render.png", "wb").write(png)

    # taste probe: monotone in [0, 1], and never saturated flat, so every render
    # in a batch gets a distinguishable score
    painted_taste = taste_score(png)
    print("taste painted:", round(painted_taste, 3))
    assert 0.0 < painted_taste < 1.0

    # judge: taste probe plus anatomy checks
    score, jmeta = judge_image(png, PROMPT, FAMILY)
    print("judge painted:", score, jmeta)
    print("painted speckle:", round(speckle_fraction(png), 3))
    # The bar is the user's ratings, not a fixed style: they loved flat anime
    # eyes and rejected sparse pencil line work, so anime is allowed to outscore
    # the painterly sketch, while pencil hatching has to come last.
    scores = {"painted": score}
    for name, sketch in (("anime", ANIME_SKETCH), ("pencil", PENCIL_SKETCH)):
        other_png, other_meta = render_sketch(sketch)
        assert other_png is not None, other_meta
        scores[name], other_jmeta = judge_image(other_png, PROMPT, FAMILY)
        print(f"judge {name}:", scores[name], other_jmeta)
    assert scores["painted"] > scores["pencil"], scores
    assert scores["anime"] > scores["pencil"], scores

    # full path
    reward, smeta, _ = score_response(GOOD_RESPONSE, LABEL)
    print("full reward:", reward, smeta)
    reward0, smeta0, _ = score_response("I cannot draw that.", LABEL)
    assert reward0 == 0.0
    print("malformed reward:", reward0, smeta0)

    # dataset formatting
    ds = EyePromptDataset(n_train=4, n_eval=2)
    ds.prepare("/tmp/eye_train.jsonl", {"eval": "/tmp/eye_eval.jsonl"})
    ds.validate_prepared("/tmp/eye_train.jsonl")
    import json

    row = json.loads(open("/tmp/eye_train.jsonl").readline())
    assert row["messages"][0]["role"] == "system" and row["label"]
    print("dataset: OK —", row["messages"][1]["content"][:70])


if __name__ == "__main__":
    main()
