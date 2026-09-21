"""
image: radixark/miles:dev-202608041247
commit: https://github.com/radixark/miles/commit/5c517599fb55c2528a55c749e6eda5f99f51f0ad
file: miles/miles/rollout/generate_hub/single_turn.py::generate
"""

from __future__ import annotations

import pathlib

MARKER = "PATCHED_GEMMA4_VL_SINGLE_TURN_IDS"

TARGET = pathlib.Path("/root/miles/miles/rollout/generate_hub/single_turn.py")

_IMAGE_TOKEN = "<|image|>"


def gemma4_collapse_image_token_ids(
    prompt_ids: list[int], image_token_id: int
) -> list[int]:
    """Keep one image-token id per consecutive run."""
    token_id = int(image_token_id)
    out: list[int] = []
    in_run = False
    for raw in prompt_ids:
        tid = int(raw)
        if tid == token_id:
            if not in_run:
                out.append(tid)
                in_run = True
            continue
        out.append(tid)
        in_run = False
    return out


OLD = """    payload, halt_status = compute_request_payload(
        args, input_ids=input_ids, sampling_params=sampling_params, multimodal_inputs=sample.multimodal_inputs
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(url, payload, headers=compute_routing_headers(args, sample))
"""

NEW = f"""    payload, halt_status = compute_request_payload(
        args, input_ids=input_ids, sampling_params=sampling_params, multimodal_inputs=sample.multimodal_inputs
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    if payload.get("image_data") and len(sample.response) == 0:
        # {MARKER}: SGLang counts <|image|> ids against image_data, one per
        # image. Local processor ids are one token per patch.
        if not sample.tokens:
            sample.tokens = list(payload["input_ids"])
        _tok_id = getattr(getattr(input.state, "processor", None), "image_token_id", None)
        if _tok_id is None:
            _tok_id = input.state.tokenizer.convert_tokens_to_ids("{_IMAGE_TOKEN}")
        _out, _run = [], False
        for _raw in payload["input_ids"]:
            _t = int(_raw)
            if _t == int(_tok_id):
                if not _run:
                    _out.append(_t)
                    _run = True
            else:
                _out.append(_t)
                _run = False
        payload["input_ids"] = _out

    output = await post(url, payload, headers=compute_routing_headers(args, sample))
"""


def _patch_file(path: pathlib.Path) -> None:
    if not path.exists():
        print(f"{path} not found; skipping Gemma-4 VL single-turn ids patch")
        return
    src = path.read_text()
    if MARKER in src:
        print("Gemma-4 VL single-turn ids patch already applied")
        return
    if OLD not in src:
        raise SystemExit(
            "Gemma-4 VL single-turn ids patch did not match; miles' "
            "generate_hub/single_turn.py payload construction has changed. "
            "Re-check it before shipping."
        )
    path.write_text(src.replace(OLD, NEW, 1))
    print(
        "Patched Gemma-4 VL single-turn generate to POST one image-token id per image"
    )


if __name__ == "__main__":
    _patch_file(TARGET)
