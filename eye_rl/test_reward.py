"""Local smoke test of the eye reward pipeline (renderer + judge + gates)."""

import sys

sys.path.insert(0, "eye_rl")
from train_eyes import (  # noqa: E402
    EyePromptDataset,
    extract_sketch,
    judge_image,
    render_sketch,
    score_response,
)

GOOD_SKETCH = """function setup() {
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
PROMPT = "Illustrate a human eye with a emerald green iris, rendered as a loose ink sketch. Make it detailed and expressive."


def main():
    # gates
    assert extract_sketch("no code here") is None
    assert extract_sketch("```javascript\nfetch('x')\nfunction setup(){}\n```") is None
    assert extract_sketch("```python\nprint(1)\n```") is None
    assert extract_sketch(GOOD_RESPONSE) is not None
    print("gates: OK")

    # render failure path
    png, meta = render_sketch(BAD_SKETCH_RUNTIME)
    print("bad sketch render:", meta)
    assert png is None

    # render success path
    png, meta = render_sketch(GOOD_SKETCH)
    print("good sketch render:", meta, "bytes:", len(png) if png else 0)
    assert png is not None
    open("/tmp/eye_render.png", "wb").write(png)

    # judge
    score, jmeta = judge_image(png, PROMPT)
    print("judge:", score, jmeta)

    # full path
    reward, smeta, _ = score_response(GOOD_RESPONSE, PROMPT)
    print("full reward:", reward, smeta)
    reward0, smeta0, _ = score_response("I cannot draw that.", PROMPT)
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
