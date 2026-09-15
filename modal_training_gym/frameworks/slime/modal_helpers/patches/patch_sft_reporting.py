import ast
from pathlib import Path

MARKER = "# TRAINING_GYM_SFT_REPORTING"


def patch_model(source: str) -> str:
    if MARKER in source:
        return source
    tree = ast.parse(source)
    train = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    anchors = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "logger.info"
        and "log_dict" in ast.unparse(node)
    ]
    if len(anchors) != 1:
        raise RuntimeError(
            "SFT reporting requires one reduced train-metrics log anchor"
        )
    anchor = anchors[0]
    lines = source.splitlines(keepends=True)
    indent = " " * anchor.col_offset
    lines.insert(
        anchor.lineno - 1,
        f"{indent}{MARKER}\n{indent}from modal_training_gym.frameworks.slime.sft_reporting import report_train_metrics\n{indent}report_train_metrics(args, log_dict)\n",
    )
    return "".join(lines)


def patch_data(source: str) -> str:
    if MARKER in source:
        return source
    tree = ast.parse(source)
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "log_rollout_data"
    )
    lines = source.splitlines(keepends=True)
    first = func.body[0]
    lines.insert(
        first.lineno - 1,
        f"    {MARKER}\n    if getattr(args, 'loss_type', None) == 'sft_loss':\n        return\n",
    )
    return "".join(lines)


if __name__ == "__main__":
    root = Path("/root/slime/slime/backends/megatron_utils")
    for filename, patch in (("model.py", patch_model), ("data.py", patch_data)):
        path = root / filename
        path.write_text(patch(path.read_text()))
