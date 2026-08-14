"""Learning Agent eval harness — ONE eval, parameterized by --task.

Serves the student (Qwen3.5-9B) via sglang and runs it as a ReAct agent
(grep / glob / read_file over the TASK corpus) across search budgets:
  - 0   : direct, no tools  (closed-book)
  - 5   : up to 5 tool iterations
  - 20  : up to 20 tool iterations
then writes the answers. Judging is left to the workflow (an LLM-judge agent reads
candidates + dev/test gold+rubric directly) — this script does NOT grade.


  modal run harness/eval.py::expertise --task openclaw --model Qwen/Qwen3.5-9B \
      --budgets 0,5,20 --split dev
  modal run harness/eval.py::expertise --task fav2 \
      --model /out/models/<tag>/merged --budgets 0,5,20 --split test --tag <tag>

Per task:
  - corpus = tasks/<TASK>/corpus, baked into the image at /corpus
  - SYS    = tasks/<TASK>/sys.txt (verbatim)
  - questions: split=dev -> tasks/<TASK>/dev.json ; split=test -> tasks/<TASK>/test.json
Writes runs/<tag>/budget_<b>/candidates.json (+ eval_meta.json).

GOTCHAS kept: Qwen3.5-9B is a thinking model — chat template is applied with
enable_thinking=False and <think>...</think> is stripped before parse/grade.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("openclaw", "fav2", "maud")
# Per-task default grep/glob extension (TypeScript vs Python vs filings/legal corpus).
GLOB_EXT = {"openclaw": "**/*.ts", "fav2": "**/*.txt", "maud": "**/*.txt"}
# Per-task FINAL-line hint (code tasks answer with a single fenced block; fav2/maud are prose).
FINAL_HINT = {"openclaw": "<your full answer with the single code block>",
              "fav2": "<your full answer>",
              "maud": "<your full answer>"}

app = modal.App("lab-eval")
HF_CACHE = modal.Volume.from_name("lab-hf-cache", create_if_missing=True)
OUT = modal.Volume.from_name("lab-out", create_if_missing=True)
HF_CACHE_DIR = "/hf-cache"
CORPUS_ROOT = "/corpus"


def _config_base_model() -> str:
    """global.base_model from bench/config.yaml — the ONE source of truth for the
    student. Fallback literal is used only where the config file is absent (inside
    the Modal container the repo is not mounted; callers there pass the model
    explicitly), so a config change never silently diverges on the operator side."""
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "bench" / "config.yaml").read_text())
        return cfg["global"]["base_model"]
    except Exception:  # noqa: BLE001  (container / partial tree)
        return "Qwen/Qwen3.5-9B"


BASE_MODEL = _config_base_model()

SLIME_IMAGE = "slimerl/slime@sha256:087a57732cf4fb271729df47530b01a9530144f4339247efc422f03e2b6988e1"


def _base_image():
    return (
        modal.Image.from_registry(SLIME_IMAGE)
        .pip_install("datasets", "hf_transfer")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_DIR})
    )


# Bake EACH task's corpus into its own image at /corpus (so grep/glob/read_file see it).
# A task whose corpus is not on this machine gets the bare image instead of
# blocking every other task's eval at mount time; the entrypoint refuses to
# score a task whose corpus is missing.
def _task_image(t: str):
    corpus = ROOT / "tasks" / t / "corpus"
    if not corpus.is_dir():
        return _base_image()
    return _base_image().add_local_dir(str(corpus), CORPUS_ROOT)


IMAGES = {t: _task_image(t) for t in TASKS}


def _sys_direct(task: str) -> str:
    return (ROOT / "tasks" / task / "sys.txt").read_text()


def _sys_react(sys_direct: str, glob_ext: str,
               final_hint: str = "<your full answer with the single code block>") -> str:
    return sys_direct + f"""

You may SEARCH the source + docs (rooted at /corpus) before answering. Each turn output \
EXACTLY ONE of:

  ACTION: {{"tool": "grep",      "args": {{"pattern": "<regex>", "glob": "{glob_ext}"}}}}
  ACTION: {{"tool": "glob",      "args": {{"pattern": "<glob pattern>"}}}}
  ACTION: {{"tool": "read_file", "args": {{"path": "<path under /corpus>", "start_line": 1, "end_line": 120}}}}
  FINAL: {final_hint}

Use searches to ground your answer in the real API and avoid hallucinating. When ready, \
emit FINAL. Be efficient — search only what you need."""


# ---------- corpus tools (run in-container over /corpus) ----------
# Output caps (grep 40 hits / glob 80 files / read 200 lines) bound each OBSERVATION so a
# large file cannot blow the model's context window; the toolbox ReAct loop uses the same caps.

def _safe(path: str):
    p = os.path.normpath(os.path.join(CORPUS_ROOT, path.lstrip("/")))
    return p if p.startswith(CORPUS_ROOT) and os.path.exists(p) else None


def _grep(pattern, glob="**/*", max_results=40):
    import glob as G
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"[grep error: bad regex: {e}]"
    hits = []
    for fp in G.glob(os.path.join(CORPUS_ROOT, glob), recursive=True):
        if not os.path.isfile(fp):
            continue
        try:
            for i, line in enumerate(open(fp, encoding="utf-8", errors="replace"), 1):
                if rx.search(line):
                    hits.append(f"{os.path.relpath(fp, CORPUS_ROOT)}:{i}: {line.rstrip()[:200]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits) + f"\n[truncated at {max_results}]"
        except OSError:
            continue
    return "\n".join(hits) if hits else "[no matches]"


def _glob(pattern, max_results=80):
    import glob as G
    fs = [os.path.relpath(p, CORPUS_ROOT)
          for p in G.glob(os.path.join(CORPUS_ROOT, pattern), recursive=True) if os.path.isfile(p)]
    return "\n".join(sorted(fs)[:max_results]) if fs else "[no files]"


def _read_file(path, start_line=1, end_line=0, max_lines=200):
    p = _safe(path)
    if not p or not os.path.isfile(p):
        return f"[not found: {path}]"
    lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
    s = max(1, int(start_line)); e = int(end_line) if end_line else s + max_lines - 1
    e = min(e, len(lines), s + max_lines - 1)
    return "\n".join(f"{i:>5}: {lines[i-1]}" for i in range(s, e + 1)) + (
        f"\n[file has {len(lines)} lines]" if e < len(lines) else "")


def _extract_json(s):
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def _run_tool(obj, default_glob="**/*"):
    tool = (obj or {}).get("tool"); args = (obj or {}).get("args", {}) or {}
    try:
        if tool == "grep":
            return _grep(args.get("pattern", ""), args.get("glob", default_glob))
        if tool == "glob":
            return _glob(args.get("pattern", "**/*"))
        if tool == "read_file":
            return _read_file(args.get("path", ""), args.get("start_line", 1), args.get("end_line", 0))
        return f"[unknown tool: {tool}]"
    except Exception as e:  # noqa: BLE001
        return f"[tool error: {e}]"


def _strip_think(text):
    """Qwen3.5 is a thinking model: drop any <think>...</think> block (and a dangling
    open <think> that closed but lost its opener) so FINAL/ACTION parsing and the graded
    answer see only the post-reasoning content."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


def _apply_ct(tok, msgs):
    """Apply chat template with thinking DISABLED for Qwen3.5 (enable_thinking=False).
    Falls back gracefully for tokenizers that don't accept the kwarg."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _react_rollout(engine, tok, question, max_iters, sys_direct, sys_react,
                   default_glob, note=""):
    """Returns (answer, n_tool_calls, total_completion_tokens). `note` (if set) is the
    open-book cheatsheet prepended to the system prompt."""
    pre = f"Reference notes on the current API (study these):\n{note}\n\n" if note else ""
    # Decode greedily (temperature 0.0) so the eval is deterministic and reproduces bit-for-bit;
    # 4096 new tokens is ample for one FINAL answer or a single ACTION line per turn.
    if max_iters == 0:
        msgs = [{"role": "system", "content": pre + sys_direct},
                {"role": "user", "content": question}]
        out = engine.generate(_apply_ct(tok, msgs), {"temperature": 0.0, "max_new_tokens": 4096})
        return _strip_think(out["text"]), 0, (out.get("meta_info", {}) or {}).get("completion_tokens", 0)

    msgs = [{"role": "system", "content": pre + sys_react}, {"role": "user", "content": question}]
    tot = 0
    for it in range(max_iters + 1):
        force = it == max_iters
        if force:
            msgs.append({"role": "user",
                         "content": "Search budget exhausted. Output FINAL with your answer now."})
        out = engine.generate(_apply_ct(tok, msgs), {"temperature": 0.0, "max_new_tokens": 4096})
        text = _strip_think(out["text"])
        tot += (out.get("meta_info", {}) or {}).get("completion_tokens", 0)
        fm = re.search(r"FINAL\s*:?", text); am = re.search(r"ACTION\s*:?", text)
        if force or (fm and (not am or fm.start() < am.start())):
            return (text[fm.end():].strip() if fm else text.strip()), it, tot
        if am:
            msgs.append({"role": "assistant", "content": text.strip()})
            msgs.append({"role": "user",
                         "content": f"OBSERVATION:\n{_run_tool(_extract_json(text[am.end():]), default_glob)}"})
            continue
        # no parseable directive -> treat as the answer
        return text.strip(), it, tot
    return "", max_iters, tot


def _react_eval(questions, budgets, model, task, sys_direct, default_glob,
                adapter="", tag="eval", note="", tp_size=1):
    import sglang as sgl
    from transformers import AutoTokenizer
    sys_react = _sys_react(sys_direct, default_glob,
                           FINAL_HINT.get(task, "<your full answer with the single code block>"))
    tok = AutoTokenizer.from_pretrained(model)
    # mem_fraction_static=0.85: leave ~15% VRAM headroom for the KV cache and activation
    # spikes; context_length=131072 = the student's full 128K window (fits large corpus reads).
    kw = dict(model_path=model, tp_size=tp_size, mem_fraction_static=0.85,
              context_length=131072, trust_remote_code=True)
    if adapter:
        kw["lora_paths"] = [adapter]
    engine = sgl.Engine(**kw)
    per_budget = {}
    for b in budgets:
        answers, meta = {}, {}
        for q in questions:
            ans, nc, nt = _react_rollout(engine, tok, q["question"], b, sys_direct, sys_react,
                                         default_glob, note=note)
            answers[q["id"]] = ans
            meta[q["id"]] = {"tool_calls": nc, "completion_tokens": nt}
        per_budget[str(b)] = {"answers": answers, "meta": meta}
        print(f"[{task} budget {b}] {len(answers)} answers, "
              f"avg_calls={sum(m['tool_calls'] for m in meta.values())/len(meta):.1f}, "
              f"avg_tok={sum(m['completion_tokens'] for m in meta.values())/len(meta):.0f}")
    d = f"/out/{tag}"; os.makedirs(d, exist_ok=True)
    json.dump({"task": task, "model": model, "adapter": adapter, "per_budget": per_budget},
              open(f"{d}/eval.json", "w"), indent=2)
    OUT.commit()
    return per_budget


# One Modal function per task so each gets its own corpus-baked image. `gpu` can scale
# up to 8x H200 (tensor-parallel serving) via --tp.
@app.function(image=IMAGES["openclaw"], gpu="H200", timeout=120 * 60,
              volumes={HF_CACHE_DIR: HF_CACHE, "/out": OUT},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def react_eval_openclaw(questions, budgets, model, sys_direct, adapter="", tag="eval",
                        note="", tp_size=1):
    return _react_eval(questions, budgets, model, "openclaw", sys_direct,
                       GLOB_EXT["openclaw"], adapter=adapter, tag=tag, note=note, tp_size=tp_size)


@app.function(image=IMAGES["fav2"], gpu="H200", timeout=120 * 60,
              volumes={HF_CACHE_DIR: HF_CACHE, "/out": OUT},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def react_eval_fav2(questions, budgets, model, sys_direct, adapter="", tag="eval",
                    note="", tp_size=1):
    return _react_eval(questions, budgets, model, "fav2", sys_direct,
                       GLOB_EXT["fav2"], adapter=adapter, tag=tag, note=note, tp_size=tp_size)


@app.function(image=IMAGES["maud"], gpu="H200", timeout=120 * 60,
              volumes={HF_CACHE_DIR: HF_CACHE, "/out": OUT},
              secrets=[modal.Secret.from_name("huggingface-secret")])
def react_eval_maud(questions, budgets, model, sys_direct, adapter="", tag="eval",
                    note="", tp_size=1):
    return _react_eval(questions, budgets, model, "maud", sys_direct,
                       GLOB_EXT["maud"], adapter=adapter, tag=tag, note=note, tp_size=tp_size)


REACT_FN = {"openclaw": react_eval_openclaw,
            "fav2": react_eval_fav2, "maud": react_eval_maud}


@app.local_entrypoint()
def expertise(task: str, model: str = BASE_MODEL, budgets: str = "0,5,20",
              split: str = "dev", tag: str = "", adapter: str = "", note_file: str = "",
              tp: int = 1):
    """--task <openclaw|fav2> --model <hf-id|/out/models/<tag>/merged> --budgets 0,5,20
    --split dev|test [--note-file].

    Questions come from tasks/<TASK>/dev.json (dev) or tasks/<TASK>/test.json (test).
    test.json is harness-only and OFF LIMITS to the agent — this entrypoint is the harness.
    """
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; expected one of {TASKS}")
    if not (ROOT / "tasks" / task / "corpus").is_dir():
        raise SystemExit(f"tasks/{task}/corpus is not present on this machine; "
                         "its image was built without a corpus mount")
    if split not in ("dev", "test"):
        raise SystemExit(f"unknown split {split!r}; expected dev|test")
    if not tag:
        tag = f"{task}_{split}"

    qfile = ROOT / "tasks" / task / (f"{split}.json")
    rows = json.loads(qfile.read_text())
    questions = [{"id": r["id"], "question": r["question"]} for r in rows]

    sys_direct = _sys_direct(task)
    bl = [int(x) for x in budgets.split(",")]
    note = (ROOT / note_file).read_text() if note_file else ""
    print(f"[{task}] eval: {len(questions)} {split} Qs, budgets={bl}, model={model} "
          f"adapter={adapter or '-'} note={'yes' if note else 'no'} tp={tp}")

    pb = REACT_FN[task].remote(questions, bl, model, sys_direct, adapter=adapter,
                               tag=tag, note=note, tp_size=tp)

    run = ROOT / "runs" / tag
    for b, data in pb.items():
        bdir = run / f"budget_{b}"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "candidates.json").write_text(json.dumps(data["answers"], indent=2))
        (bdir / "eval_meta.json").write_text(json.dumps(data["meta"], indent=2))
    print(f"saved per-budget answers under {run}/budget_*/candidates.json")
    print("NEXT: workflow LLM-judge grades candidates.json against "
          f"tasks/{task}/{split}.json (gold + rubric).")
