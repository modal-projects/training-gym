"""Tutorial-local reward model for `slime_haiku`.

Kept here (not in `modal_training_gym.common`) because the haiku-specific
rubric and syllable scoring aren't general training-gym infrastructure —
they're part of this tutorial. SLIME loads the reward function at training
time via `custom_rm_path="haiku.haiku_rm"`; the slime launcher ships every
sibling `.py` file from the tutorial directory into the training image via
`add_local_python_source`, so `import haiku` resolves there as well.

Two reusable parts:
    - `score_haiku_structure`  — 5-7-5 syllable scorer, via CMUdict.
    - `HaikuStyleJudge`        — `LlmJudge` subclass with a haiku rubric.

Composed in `haiku_rm`, the async SLIME reward entrypoint. Style scoring
is gated on `LLM_JUDGE_URL` being set in the env; when absent, the reward
falls back to structure-only so the tutorial is runnable without a judge
deployment.
"""

from __future__ import annotations

import os
import re
from typing import Any

from modal_training_gym.common.llm_judge import LlmJudge

_CMUDICT: dict = {}


def _get_cmudict() -> dict:
    """Lazy-load NLTK's CMUdict; cached on first call per process."""
    import nltk
    from nltk.corpus import cmudict as nltk_cmudict

    if not _CMUDICT:
        nltk.download("cmudict", quiet=True)
        _CMUDICT.update(nltk_cmudict.dict())
    return _CMUDICT


def _count_syllables_for_word(word: str, cmudict: dict) -> int:
    original = word
    word = word.lower().strip()
    phones = cmudict.get(word)
    if phones:
        return len([p for p in phones[0] if p[-1].isdigit()])
    if original.isupper() and 2 <= len(original) <= 6 and original.isalpha():
        # Letters-as-name heuristic for unknown acronyms; 'w' = "double-u" = 3 syllables.
        return sum(3 if c == "w" else 1 for c in original.lower())
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _diff_syllables(text: str, target: int, cmudict: dict) -> int:
    words = re.findall(r"[a-zA-Z]+", text)
    total = sum(_count_syllables_for_word(w, cmudict) for w in words)
    return abs(total - target)


def _segment_haiku_lines(response: str) -> list[str]:
    if "/" in response:
        lines = [line.strip() for line in response.split("/")]
    elif ". " in response:
        lines = [line.strip() for line in response.split(". ")]
    else:
        lines = [line.strip() for line in response.split("\n")]
    return [line for line in lines if line]


def score_haiku_structure(response: str, cmudict: dict) -> float:
    """Score in [0, 1]: 1/4 for 3 lines, 1/4 per line with correct syllables."""
    lines = _segment_haiku_lines(response)
    score = 0.25 if len(lines) == 3 else 0.0
    for i, target in enumerate([5, 7, 5]):
        if i < len(lines):
            diff = _diff_syllables(lines[i], target, cmudict)
            if diff == 0:
                score += 0.25
            elif diff == 1:
                score += 0.125
    return score


class HaikuStyleJudge(LlmJudge):
    """`LlmJudge` with a haiku rubric: relevance + poetic quality + vocab."""

    def build_prompt(self, prompt: str, response: str, **kwargs: Any) -> str:
        return (
            "You are evaluating a haiku poem.\n\n"
            f"Topic: {prompt}\n"
            f"Response:\n{response}\n\n"
            "Score 0-10 on relevance to the topic and poetic quality. "
            "single number (0-10), nothing else."
        )


def _extract_topic(sample) -> str:
    """Recover the haiku topic ("cat", "modal", …) from a SLIME sample.

    SLIME's rollout `sample` objects expose the decoded generation under
    `.response` but don't guarantee the original dataset columns. The
    dataset prep in the tutorial stashes the question string on the
    `prompt` field and the keyword under `question` via the chat-
    template transform; SLIME typically surfaces those back as a `data`
    mapping or as direct attributes, depending on version. Try each
    known location in turn and fall back to the empty string — the
    judge degrades to topic-agnostic scoring rather than erroring.
    """
    for attr in ("question", "prompt"):
        val = getattr(sample, attr, None)
        if isinstance(val, str) and val:
            return val
    data = getattr(sample, "data", None) or {}
    for key in ("question", "prompt"):
        val = data.get(key) if hasattr(data, "get") else None
        if isinstance(val, str) and val:
            return val
    return ""


async def haiku_rm(args, sample, **kwargs) -> float:
    """Async SLIME reward — structure + (optional) LLM style score.

    Referenced from the tutorial as `custom_rm_path="haiku.haiku_rm"`. The
    slime launcher adds this file to the training image via
    `add_local_python_source`, so this plain `haiku` module is importable
    both locally (from the tutorial dir on `sys.path`) and on the remote
    container.

    Reads the judge endpoint from `LLM_JUDGE_URL` (and judge model from
    `LLM_JUDGE_MODEL`, default `qwen3-4b`); when the URL is unset, skips
    style scoring and returns structure only.
    """
    import aiohttp

    cmudict = _get_cmudict()
    structure = score_haiku_structure(sample.response, cmudict)

    judge_url = os.environ.get("LLM_JUDGE_URL", "")
    if not judge_url or structure < 1.0:
        return structure

    judge = HaikuStyleJudge(
        model_name=os.environ.get("LLM_JUDGE_MODEL", "qwen3-4b"),
        base_url=judge_url,
        max_score=10,
    )
    async with aiohttp.ClientSession() as session:
        style = await judge.score(session, _extract_topic(sample), sample.response)
    return structure + style
