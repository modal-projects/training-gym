"""Rubric grader: weighted rubric claims + deterministic code checks.

A row = {id, topic, question, gold_answer, rubric:[{claim_id,claim_type,weight,statement}], evidence}.
An answer = the model's response string (may contain python code blocks).

Scoring:
  - claim score: each rubric claim judged satisfied in [0,1] by `judge_fn`; the
    weighted sum (weights sum to 100) gives `claim_score` in [0,1].
  - deterministic: extract code, check it parses (compiles) and uses only real
    DSPy APIs (no hallucinated `dspy.<X>`); reported as flags.
  - final score combination is a HARNESS KNOB (see `combine`). Default:
    claim_score, soft-penalized for non-compiling code and hallucinated APIs.

`judge_fn(question:str, answer:str, claims:list[dict]) -> list[float]` is injected
(Claude / a Modal-served judge). A keyword-overlap stub is provided ONLY for
plumbing tests — it is not a real judge.
"""
from __future__ import annotations
import ast, json, re
from pathlib import Path

_API_PATH = Path(__file__).resolve().parents[1] / "data" / "api_surface.json"


def load_api_surface(path: Path = _API_PATH) -> dict:
    api = json.loads(Path(path).read_text())
    submods = {m.split(".")[1] for m in api["modules"] if m.count(".") >= 1}
    api["_valid_dspy_attrs"] = set(api["public_namespace"]) | set(api["all_symbols"]) | submods
    return api


# ---------- code extraction & deterministic checks ----------

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _FENCE.finditer(text or "")]


def check_compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def hallucinated_apis(code: str, api: dict) -> list[str]:
    """`dspy.<attr>` references whose <attr> is not a real DSPy symbol."""
    valid = api["_valid_dspy_attrs"]
    bad: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bad
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "dspy":
            if node.attr not in valid and not node.attr.startswith("_"):
                bad.append(f"dspy.{node.attr}")
    return sorted(set(bad))


def deterministic_report(answer: str, api: dict) -> dict:
    blocks = extract_code_blocks(answer)
    has_code = len(blocks) > 0
    compiles = all(check_compiles(b) for b in blocks) if has_code else None
    halluc: list[str] = []
    for b in blocks:
        halluc += hallucinated_apis(b, api)
    return {
        "has_code": has_code,
        "n_blocks": len(blocks),
        "compiles": compiles,
        "hallucinated_apis": sorted(set(halluc)),
    }


# ---------- claim scoring ----------

def score_claims(row: dict, answer: str, judge_fn) -> dict:
    claims = row["rubric"]
    scores = judge_fn(row, answer)  # judge_fn(row, answer) -> [score per claim]
    assert len(scores) == len(claims), "judge must return one score per claim"
    total_w = sum(c["weight"] for c in claims) or 1
    weighted = sum(s * c["weight"] for s, c in zip(scores, claims)) / total_w
    per = [
        {"claim_id": c["claim_id"], "claim_type": c["claim_type"], "weight": c["weight"], "score": round(float(s), 3)}
        for c, s in zip(claims, scores)
    ]
    return {"claim_score": round(weighted, 4), "per_claim": per}


def combine(claim_score: float, det: dict, mode: str = "strict",
            penalty_noncompile: float = 0.5, penalty_halluc: float = 0.15) -> float:
    """Fold deterministic checks into the final score.

      strict  : non-compile OR hallucinated API -> automatic 0 (default)
      lenient : ignore deterministic checks, score against the rubric alone (small models)
      soft    : multiplicative penalties instead of auto-zero
    """
    if mode == "lenient":
        return round(claim_score, 4)
    if mode == "strict":
        if det["has_code"] and det["compiles"] is False:
            return 0.0
        if det["hallucinated_apis"]:
            return 0.0
        return round(claim_score, 4)
    # soft
    score = claim_score
    if det["has_code"] and det["compiles"] is False:
        score *= (1 - penalty_noncompile)
    if det["hallucinated_apis"]:
        score *= (1 - penalty_halluc)
    return round(max(0.0, min(1.0, score)), 4)


def grade(row: dict, answer: str, judge_fn, api: dict, mode: str = "strict", **knobs) -> dict:
    det = deterministic_report(answer, api)
    cl = score_claims(row, answer, judge_fn)
    final = combine(cl["claim_score"], det, mode=mode, **knobs)
    return {"id": row["id"], "topic": row["topic"], "final_score": final, "mode": mode,
            "claim_score": cl["claim_score"], "deterministic": det, "per_claim": cl["per_claim"]}


# ---------- stub judge (plumbing tests only) ----------

def keyword_judge(row: dict, answer: str) -> list[float]:
    """NOT a real judge. Token-overlap between claim statement and answer."""
    ans_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", (answer or "").lower()))
    out = []
    for c in row["rubric"]:
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_\.]+", c["statement"].lower()))
        toks = {t for t in toks if len(t) > 3}
        out.append(len(toks & ans_tokens) / max(1, len(toks)))
    return out
