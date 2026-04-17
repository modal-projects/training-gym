"""Default GSM8K reward function for verl.

verl reads a custom reward function via `custom_reward_function.path=<file>`
+ `custom_reward_function.name=<fn>`. The launcher resolves the absolute
path of *this* module inside the image and passes it to verl, so the
framework ships a ready-made GSM8K reward out of the box. Users can point
verl at a different file via `VerlConfig.reward_function_path`.
"""

import re
from typing import Literal, Optional


def extract_solution(
    solution_str: str, method: Literal["strict", "flexible"] = "strict"
) -> Optional[str]:
    """Extract numerical answer from model output.

    `strict` requires the GSM8K `#### N` format (also validates formatting);
    `flexible` picks the last valid number as a fallback. Returns None if
    nothing matches.
    """
    if method == "strict":
        match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
        if match is None:
            return None
        return match.group(0).split("#### ")[1].replace(",", "").replace("$", "")

    answers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
    if not answers:
        return None
    invalid = {"", "."}
    for candidate in reversed(answers):
        if candidate not in invalid:
            return candidate
    return None


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> float:
    """Binary reward: 1.0 if the extracted answer matches ground truth exactly, else 0.0.

    Signature matches verl's `custom_reward_function` interface.
    """
    answer = extract_solution(solution_str=solution_str, method="strict")
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.0
