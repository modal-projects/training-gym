"""Claude-sub-agent judge for Learning Agent rubric claims.

The judge is realized as a Claude sub-agent (spawned via the Agent tool), not an
in-process API call. Flow:

  1. `build_judge_prompt(row, answer)` -> a prompt asking the sub-agent to score
     each rubric claim 0/1 (satisfied) with a one-line reason.
  2. The sub-agent returns JSON conforming to JUDGE_SCHEMA.
  3. Verdicts are cached to `verdicts.json` as {row_id: {claim_id: score}}.
  4. `judge_fn_from_verdicts(verdicts)` -> a callable(row, answer) the grader uses.

Per-claim scoring is BINARY (0 or 1) for "satisfied / not";
0.5 is allowed for genuinely-partial claims. The gold_answer is shown to the
judge as a reference (it carries the "decoy" / what-NOT-to-credit cues).
"""
from __future__ import annotations
import json, re

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "score": {"type": "number", "description": "1 satisfied, 0 not, 0.5 partial"},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id", "score"],
            },
        }
    },
    "required": ["verdicts"],
}

JUDGE_SYS = """You are a strict grader for DSPy coding answers. You are given a question, a
reference (gold) answer, the candidate answer to grade, and a weighted rubric of claims.

For EACH claim, decide whether the CANDIDATE answer satisfies it:
  - score 1   : the candidate clearly satisfies the claim
  - score 0   : it does not (including if it falls for a stated "decoy" / "does NOT satisfy" case)
  - score 0.5 : genuinely partial — only when the claim has separable parts and some are met

Judge ONLY what the candidate answer actually does. The gold answer is a reference for what
"correct" looks like; do not credit the candidate for things only the gold answer does. Many
claims name a specific correct approach AND an incorrect "decoy" — credit only the correct one.
Return strict JSON: {"verdicts":[{"claim_id","score","reason"}...]} with one entry per claim."""

JUDGE_SYS_FIN = """You are a strict grader for expert financial-analysis answers grounded in SEC
filings. You are given a question, a reference (gold) answer, the candidate answer to grade,
and a weighted rubric of claims.

For EACH claim, decide whether the CANDIDATE answer satisfies it:
  - score 1   : the candidate clearly and explicitly satisfies the claim
  - score 0   : it does not (vague references and hedged non-answers earn no credit)
  - score 0.5 : genuinely partial — only when the claim has separable parts and some are met

Numeric claims: credit requires the candidate's figure to match within roughly 1% relative
tolerance (or the tolerance the claim itself states). If the candidate contradicts a claim
elsewhere in its answer, score that claim 0. Judge ONLY what the candidate answer actually
says. The gold answer is a reference for what "correct" looks like; do not credit the
candidate for things only the gold answer does.
Return strict JSON: {"verdicts":[{"claim_id","score","reason"}...]} with one entry per claim."""

# Per-task judge persona; tasks not listed use the default JUDGE_SYS (code tasks).
JUDGE_SYS_BY_TASK = {"fav2": JUDGE_SYS_FIN}


def build_judge_prompt(row: dict, answer: str, sys_prompt: str | None = None) -> str:
    claims = "\n".join(
        f"- {c['claim_id']} (weight {c['weight']}, {c['claim_type']}): {c['statement']}"
        for c in row["rubric"]
    )
    return f"""{sys_prompt or JUDGE_SYS}

## QUESTION
{row['question']}

## GOLD (reference) ANSWER
{row['gold_answer']}

## CANDIDATE ANSWER (grade this)
{answer if answer.strip() else '[EMPTY ANSWER]'}

## RUBRIC CLAIMS
{claims}

Output JSON only."""


def parse_verdict(text: str, row: dict) -> dict:
    """Sub-agent text -> {claim_id: score}. Tolerant of fences / surrounding prose."""
    claim_ids = [c["claim_id"] for c in row["rubric"]]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    scores: dict[str, float] = {}
    if m:
        try:
            obj = json.loads(m.group(0))
            for v in obj.get("verdicts", []):
                cid = v.get("claim_id")
                if cid in claim_ids:
                    scores[cid] = max(0.0, min(1.0, float(v.get("score", 0))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # default missing claims to 0 (unsatisfied)
    return {cid: scores.get(cid, 0.0) for cid in claim_ids}


def judge_fn_from_verdicts(verdicts: dict):
    """verdicts = {row_id: {claim_id: score}} -> callable(row, answer) for the grader."""
    def judge_fn(row: dict, answer: str) -> list[float]:
        per = verdicts.get(row["id"], {})
        return [float(per.get(c["claim_id"], 0.0)) for c in row["rubric"]]
    return judge_fn
