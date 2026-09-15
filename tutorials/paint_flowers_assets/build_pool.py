"""Rebuild the reference pool and the taste probe from scratch.

    uv run --with torch --with transformers --with pillow \
        tutorials/paint_flowers_assets/build_pool.py corpus

Three stages, in order:

1. Render 192 hand-written flower sketches (`reference_sketches.py`) and 84
   deliberate failure modes (`negative_sketches.py`) in a Modal Sandbox.
2. Fit the taste probe: a logistic head on frozen CLIP ViT-B/32 embeddings,
   trained on the renders rated *love* in `ratings.py` against the failure
   modes, and written to `flower_taste.npz`.
3. Pick 20 loved renders spread over species and palettes, downscale them to
   448px webp, and write them to `refs/` as the pairwise judge's pool.

Stage 1 costs a few minutes of sandbox CPU; the ratings in `ratings.py` were
made by looking at contact sheets of stage 1's output, so re-running stages 1-3
with different seeds means re-rating.
"""

from __future__ import annotations

import base64
import collections
import concurrent.futures as cf
import itertools
import json
import pathlib
import random
import sys

import modal

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from negative_sketches import make_negatives  # noqa: E402
from paint_flowers import RENDER_APP_NAME, RENDER_JS, render_image  # noqa: E402
from ratings import LOVE  # noqa: E402
from reference_sketches import PALETTES, SPECIES, flower_sketch  # noqa: E402

HERE = pathlib.Path(__file__).parent
CLIP = "openai/clip-vit-base-patch32"


def open_sandbox(timeout: int = 5400) -> modal.Sandbox:
    app = modal.App.lookup(RENDER_APP_NAME, create_if_missing=True)
    sb = modal.Sandbox._experimental_create(
        "sleep",
        "infinity",
        app=app,
        image=render_image(),
        workdir="/render",
        timeout=timeout,
        cpu=4.0,
        memory=8192,
    )
    sb.filesystem.write_text(RENDER_JS, "/render/render.js")
    return sb


def render_in(sb: modal.Sandbox, name: str, code: str) -> bytes | None:
    """One render inside an already-warm sandbox, so chromium starts once."""
    path = f"/render/{name}.js"
    sb.filesystem.write_text(code, path)
    proc = sb.exec("node", "/render/render.js", path, timeout=180)
    proc.wait()
    out = proc.stdout.read()
    if "PNGB64:" not in out:
        return None
    return base64.b64decode(out.split("PNGB64:", 1)[1].strip())


def render_corpus(out: pathlib.Path, sandboxes: int = 3, workers: int = 8) -> None:
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
    items += make_negatives(84, seed=3)
    out.mkdir(parents=True, exist_ok=True)
    sbs = [open_sandbox() for _ in range(sandboxes)]
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
            failed = 0
            for batch in results:
                for name, png in batch:
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

    model, proc = CLIPModel.from_pretrained(CLIP).eval(), CLIPProcessor.from_pretrained(CLIP)
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

    hl = [i for i in holdout if i in set(love)]
    hb = [i for i in holdout if i in set(bad)]
    pairwise = float(np.mean([[a > c for c in score(hb)] for a in score(hl)]))
    print("held-out (love, failure) pairs ordered correctly:", round(pairwise, 3))
    np.savez(
        HERE / "flower_taste.npz",
        w=wn,
        b=np.array([bn], dtype=np.float32),
        lo=float(score(bad).mean()),
        hi=float(score(love).mean()),
    )
    return X, [p.stem for p in paths], wn, bn


def pick_refs(corpus: pathlib.Path, X, names, w, b, k: int = 20) -> None:
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
    # At most three per species and per palette, so the pool cannot be dominated
    # by whichever combination the probe happens to like most.
    species_seen: collections.Counter = collections.Counter()
    palette_seen: collections.Counter = collections.Counter()
    chosen = []
    for (species, palette), name in sorted(best.items(), key=lambda kv: -score[kv[1]]):
        if species_seen[species] >= 3 or palette_seen[palette] >= 3:
            continue
        chosen.append(name)
        species_seen[species] += 1
        palette_seen[palette] += 1
        if len(chosen) == k:
            break
    refs = HERE / "refs"
    refs.mkdir(exist_ok=True)
    for old in refs.glob("*.webp"):
        old.unlink()
    for name in sorted(chosen):
        img = Image.open(corpus / f"{name}.png").convert("RGB")
        img.resize((448, 448), Image.LANCZOS).save(
            refs / f"{name.split('_', 1)[1]}.webp", quality=80, method=6
        )
    print(f"wrote {len(chosen)} references to {refs}")
    (HERE / "pool.json").write_text(json.dumps(sorted(chosen), indent=1))


def main() -> None:
    corpus = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    if not any(corpus.glob("pos*.png")):
        render_corpus(corpus)
    X, names, w, b = fit_probe(corpus)
    pick_refs(corpus, X, names, w, b)


if __name__ == "__main__":
    main()
