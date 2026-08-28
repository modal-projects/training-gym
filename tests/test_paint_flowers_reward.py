"""Reward plumbing for the flower painting tutorial.

The renderer, the probe and the judge all need remote resources, so they are
stubbed here: what these tests pin down is the part that runs on every rollout
regardless of what came back from the sandbox — sketch extraction, the static
gates, the pixel gates, the term weights, and the render ending up on
``sample.metadata["image"]`` for the dashboard.
"""

import asyncio
import importlib.util
import io
import pathlib
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TUTORIAL = REPO_ROOT / "tutorials" / "paint_flowers.py"


@pytest.fixture(scope="module")
def flowers():
    spec = importlib.util.spec_from_file_location("paint_flowers", TUTORIAL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["paint_flowers"] = module
    spec.loader.exec_module(module)
    return module


def _png(color, size=(512, 512), noise=False):
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", size, color)
    if noise:
        import random

        rng = random.Random(0)
        draw = pytest.importorskip("PIL.ImageDraw").Draw(img)
        # Blocks, not single pixels: the statistic downsamples to 128px first,
        # where pixel-level noise averages away exactly as intended.
        for _ in range(size[0] * size[1] // 90):
            x, y = rng.randrange(size[0]), rng.randrange(size[1])
            draw.rectangle((x, y, x + 5, y + 5), fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _flower_png():
    """A blank ground with an off-centre disc: inked, smooth, not full-canvas."""
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    img = Image.new("RGB", (512, 512), (246, 239, 230))
    ImageDraw.Draw(img).ellipse((120, 120, 392, 392), fill=(238, 157, 114))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SKETCH = """```javascript
function setup() {
  createCanvas(512, 512, WEBGL);
  angleMode(DEGREES);
  brush.load();
  background("#f6efe6");
  brush.noStroke();
  brush.bleed(0.2);
  brush.fill("#ee9d72", 100);
  brush.circle(0, 0, 120);
  noLoop();
}
```"""


# ── extraction ───────────────────────────────────────────────────────────


def test_extracts_fenced_sketch(flowers):
    code = flowers.extract_sketch(SKETCH)
    assert code is not None and code.startswith("function setup")


@pytest.mark.parametrize(
    "response",
    [
        "",
        "here is how you would do it, in prose",
        "```javascript\nlet x = 1;\n```",  # no setup()
        "```javascript\nfunction setup(){ fetch('http://x'); }\n```",  # banned
        "```javascript\nfunction setup(){ " + "//x\n" * 4000 + "}\n```",  # too long
    ],
)
def test_rejects_unusable_responses(flowers, response):
    assert flowers.extract_sketch(response) is None


def test_missing_sketch_scores_zero_without_rendering(flowers, monkeypatch):
    def boom(code):
        raise AssertionError("must not render an unusable response")

    monkeypatch.setattr(flowers, "render_sketch", boom)
    reward, meta, png = flowers.score_response("no code here", "iris::paint an iris")
    assert (reward, png) == (0.0, None)
    assert meta["gate"] == "no valid sketch"


def test_render_failure_scores_zero(flowers, monkeypatch):
    monkeypatch.setattr(
        flowers, "render_sketch", lambda code: (None, {"render": "fail"})
    )
    reward, meta, png = flowers.score_response(SKETCH, "iris::paint an iris")
    assert (reward, png, meta["render"]) == (0.0, None, "fail")


# ── gates and weights ────────────────────────────────────────────────────


def _stub_scoring(flowers, monkeypatch, png, probe=1.0, wins=1.0):
    monkeypatch.setattr(flowers, "render_sketch", lambda code: (png, {"render": "ok"}))
    monkeypatch.setattr(flowers, "probe_score", lambda png: probe)
    monkeypatch.setattr(
        flowers, "judge_win_rate", lambda png, prompt, species: (wins, {})
    )


def test_blank_render_keeps_only_the_static_terms(flowers, monkeypatch):
    _stub_scoring(flowers, monkeypatch, _png((255, 255, 255)))
    monkeypatch.setattr(
        flowers, "probe_score", lambda png: pytest.fail("blank render was judged")
    )
    reward, meta, png = flowers.score_response(SKETCH, "iris::paint an iris")
    assert png is not None
    assert meta["ink"] < 0.02
    # gate 0.05 + a partial length ramp, and nothing from probe or judge.
    assert 0.05 < reward < 0.11


def test_speckle_storm_is_gated(flowers, monkeypatch):
    _stub_scoring(flowers, monkeypatch, _png((255, 255, 255), noise=True))
    reward, meta, _ = flowers.score_response(SKETCH, "iris::paint an iris")
    assert meta["speckle"] > 0.12
    assert reward < 0.11


def test_full_marks_sum_to_one(flowers, monkeypatch):
    long_sketch = SKETCH.replace("noLoop();", "// pad\n" * 400 + "  noLoop();")
    _stub_scoring(flowers, monkeypatch, _flower_png(), probe=1.0, wins=1.0)
    reward, meta, _ = flowers.score_response(long_sketch, "iris::paint an iris")
    assert reward == pytest.approx(1.0)
    assert meta["probe"] == 1.0 and meta["wins"] == 1.0


def test_terms_are_weighted_as_documented(flowers, monkeypatch):
    long_sketch = SKETCH.replace("noLoop();", "// pad\n" * 400 + "  noLoop();")
    _stub_scoring(flowers, monkeypatch, _flower_png(), probe=0.5, wins=0.25)
    reward, _, _ = flowers.score_response(long_sketch, "iris::paint an iris")
    expected = 0.05 + 0.05 + 0.30 * 0.5 + 0.60 * 0.25
    assert reward == pytest.approx(expected)


def test_sketch_without_brush_loses_the_gate(flowers, monkeypatch):
    plain = "```javascript\nfunction setup(){ createCanvas(512,512); noLoop(); }\n```"
    _stub_scoring(flowers, monkeypatch, _flower_png(), probe=0.0, wins=0.0)
    reward, _, _ = flowers.score_response(plain, "iris::paint an iris")
    assert reward == pytest.approx(
        0.05 * len(flowers.extract_sketch(plain)) / 1200, abs=1e-4
    )


def test_label_without_species_still_scores(flowers, monkeypatch):
    _stub_scoring(flowers, monkeypatch, _flower_png(), probe=1.0, wins=1.0)
    reward, meta, _ = flowers.score_response(SKETCH, "paint something floral")
    assert reward > 0.8 and meta["species"] in flowers.SPECIES


# ── reward hook ──────────────────────────────────────────────────────────


def test_rm_attaches_the_render_for_the_dashboard(flowers, monkeypatch):
    png = _flower_png()
    _stub_scoring(flowers, monkeypatch, png, probe=1.0, wins=1.0)
    sample = types.SimpleNamespace(
        response=SKETCH, label="iris::paint an iris", metadata={"step": 3}
    )
    reward = asyncio.run(flowers.flower_rm(None, sample))
    assert reward > 0.8
    assert sample.metadata["image"] == png
    assert sample.metadata["step"] == 3


# ── dataset ──────────────────────────────────────────────────────────────


def test_eval_pairings_are_held_out(flowers):
    dataset = flowers.FlowerPromptDataset()
    train = {r["prompt"] for r in dataset.load("train")}
    evals = {r["prompt"] for r in dataset.load("eval")}
    assert len(evals) == dataset.n_eval
    assert not (train & evals)


def test_records_carry_the_system_prompt_and_species_label(flowers):
    dataset = flowers.FlowerPromptDataset()
    record = dataset._rows_to_records(dataset.load("eval"))[0]
    assert record["messages"][0]["role"] == "system"
    species, _, prompt = record["label"].partition("::")
    assert species in flowers.SPECIES and species in prompt


# ── reference pool ───────────────────────────────────────────────────────


def test_reference_pool_covers_every_species(flowers):
    pytest.importorskip("PIL.Image")
    pool = flowers.reference_pool()
    assert len(pool) >= 16
    assert len({species for species, _ in pool}) >= 6
    assert all(png.startswith(b"\x89PNG") for _, png in pool)


def test_pick_references_is_deterministic_and_species_aware(flowers):
    pytest.importorskip("PIL.Image")
    first = flowers.pick_references("iris", "key", 4)
    assert len(first) == 4
    assert first == flowers.pick_references("iris", "key", 4)
    iris = {png for species, png in flowers.reference_pool() if species == "iris"}
    assert sum(1 for png in first if png in iris) >= 1
