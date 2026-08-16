"""The stitch trainer's call sites, checked against the pinned cookbook checkout.

``build_stitch_app`` runs on ``cookbook.common.{hooks,launch,process,ray_cluster}``
and ``cookbook.miles_disagg.prep`` out of a clone pinned only by
``STITCH_REPO_REF``, so nothing in this repo notices when one of those contracts
moves — a renamed namespace field or a changed signature lands as a failure
minutes into a GPU run. These tests fetch the pin and check the call sites
against it: the built train command, and the attribute names the two namespaces
the launcher hand-builds (``prep.prepare_checkpoints``'s experiment,
``hooks.claim_pool``'s args) are read by.

They need the pin on disk: point ``STITCH_CHECKOUT`` at a clone, or let the
fixture fetch it (cached under ``~/.cache/training-gym``). Skipped when neither
is available, so an offline run stays green.
"""

from __future__ import annotations

import ast
import os
import subprocess
import textwrap
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models.qwen3_30b import Qwen3_30B
from modal_training_gym.train_recipes.stitch_recipe import Qwen3_30B_A3B_Stitch_Recipe
from modal_training_gym.train_recipes.stitch_recipe.recipe import (
    HOOK_CONFIG_FIELDS,
    YAML_CONFIG_FIELDS,
)
from modal_training_gym.train_recipes.stitch_recipe.pins import (
    MILES_ROOT,
    STITCH_REPO_REF,
    STITCH_REPO_URL,
)

LAUNCHER = (
    Path(__file__).parents[1]
    / "modal_training_gym"
    / "frameworks"
    / "stitch"
    / "launcher.py"
)


# ── the pinned checkout ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def cookbook() -> Path:
    """``<stitch>/cookbook`` at ``STITCH_REPO_REF``."""
    if local := os.environ.get("STITCH_CHECKOUT"):
        root = Path(local)
        head = _git(root, "rev-parse", "HEAD")
        if head != STITCH_REPO_REF:
            pytest.skip(f"STITCH_CHECKOUT is at {head}, not the pin {STITCH_REPO_REF}")
        return root / "cookbook"

    root = Path.home() / ".cache" / "training-gym" / "stitch" / STITCH_REPO_REF
    if not (root / "cookbook").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        try:
            _git(root, "init", "-q")
            _git(root, "remote", "add", "origin", STITCH_REPO_URL)
            _git(root, "fetch", "-q", "--depth", "1", "origin", STITCH_REPO_REF)
            _git(root, "checkout", "-q", "--detach", "FETCH_HEAD")
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"cannot fetch stitch at {STITCH_REPO_REF}: {exc}")
    return root / "cookbook"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()


def _load(path: Path) -> ModuleType:
    """Import a cookbook module that stands alone (no ``stitch``/``modal`` needed)."""
    spec = spec_from_file_location(f"stitch_cookbook_{path.stem}", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── launch.resolve_config + launch.build_train_cmd ─────────────────────────────
def _trainer_cfg(tmp_path: Path):
    """The carrier the trainer builds, with the per-run fields it injects."""
    from modal_training_gym.frameworks.stitch.launcher import _MilesArgs

    recipe = Qwen3_30B_A3B_Stitch_Recipe()
    payload = recipe.to_payload(model=Qwen3_30B(), dataset=_Dataset())
    cfg = _MilesArgs(
        payload.fields,
        async_mode=payload.async_mode,
        miles_model_script=payload.miles_model_script,
    )
    cfg.rollout_endpoint_url = "https://pool.modal.run"
    cfg.update_weight_disk_dir = str(tmp_path / "run" / "updates")
    cfg.custom_config_path = {
        **{field: getattr(recipe.train, field) for field in sorted(HOOK_CONFIG_FIELDS)},
        "experiment_volume_name": "bulletin",
        "rollout_modal_flash_app_name": "stitch-app",
        "rollout_modal_flash_server_cls_name": "Server",
        "run_id": "run-1",
    }
    return recipe, cfg


class _Dataset(DatasetConfig):
    hf_repo = "zhuzilin/dapo-math-17k"
    input_key = "prompt"
    label_key = "label"


def test_resolve_config_leaves_prepared_paths_alone(cookbook, tmp_path) -> None:
    """The trainer's checkpoints are already local (``prepare_checkpoints`` built
    them), and the resolve runs on a GPU node with no HF network fetch to fall
    back on: a resolve that stopped treating an absolute path as materialized
    would fail the run at launch."""
    launch = _load(cookbook / "common" / "launch.py")
    recipe, cfg = _trainer_cfg(tmp_path)

    launch.resolve_config(
        cfg, str(tmp_path), (*YAML_CONFIG_FIELDS, "custom_config_path")
    )

    assert cfg.hf_checkpoint == recipe.served_checkpoint_path
    assert str(cfg.hf_checkpoint).startswith("/")
    # ``custom_config_path`` is a path flag by the time it reaches miles: the
    # hooks read this dict off args, and a dict repr as a filename is a crash.
    assert cfg.custom_config_path == str(tmp_path / "custom_config_path.yaml")
    written = yaml.safe_load(Path(cfg.custom_config_path).read_text())
    assert written["run_id"] == "run-1"
    assert set(HOOK_CONFIG_FIELDS) <= written.keys()


def test_build_train_cmd_is_a_shell_string_the_trainer_can_run(
    cookbook, tmp_path
) -> None:
    """The launcher hands the result to ``bash -lc`` inside a ``tee`` pipeline, so
    a list-returning (or arch-script-dropping) build_train_cmd would either
    stringify as a Python repr or run miles without its MODEL_ARGS."""
    launch = _load(cookbook / "common" / "launch.py")
    recipe, cfg = _trainer_cfg(tmp_path)
    launch.resolve_config(
        cfg, str(tmp_path), (*YAML_CONFIG_FIELDS, "custom_config_path")
    )

    cmd = launch.build_train_cmd(cfg, MILES_ROOT, "miles_model_script")

    assert isinstance(cmd, str)
    script = "train_async.py" if recipe.train.async_mode else "train.py"
    assert f"source {MILES_ROOT}/{recipe.train.miles_model_script}" in cmd
    assert f"python3 {MILES_ROOT}/{script}" in cmd
    assert "${MODEL_ARGS[@]}" in cmd
    assert "--rollout-endpoint-url https://pool.modal.run" in cmd
    assert f"--custom-config-path {cfg.custom_config_path}" in cmd
    # Never a flag: miles' parser is Megatron's parse_known_args, which drops
    # what it doesn't know, so a leaked control field is silent.
    assert "--async-mode" not in cmd and "--miles-model-script" not in cmd


# ── the namespaces the launcher hand-builds ────────────────────────────────────
def _reads(module: Path, entrypoint: str, param: str) -> tuple[set[str], set[str]]:
    """(required, optional) attribute names read off ``param``, transitively.

    ``prep`` and ``hooks`` take a namespace the launcher fills in by hand, so the
    names are the contract. A plain ``ns.x`` is required; a ``getattr(ns, "x",
    default)`` is optional.
    """
    tree = ast.parse(module.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required: set[str] = set()
    optional: set[str] = set()
    seen: set[str] = set()

    def visit(name: str, param_name: str) -> None:
        if name in seen or name not in functions:
            return
        seen.add(name)
        node = functions[name]
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "getattr"
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id == param_name
                and isinstance(child.args[1], ast.Constant)
            ):
                target = optional if len(child.args) > 2 else required
                target.add(str(child.args[1].value))
            elif (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == param_name
            ):
                required.add(child.attr)
            # A helper the entrypoint forwards the namespace to.
            elif (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and any(
                    isinstance(arg, ast.Name) and arg.id == param_name
                    for arg in child.args
                )
            ):
                callee = functions.get(child.func.id)
                if callee is not None:
                    index = next(
                        i
                        for i, arg in enumerate(child.args)
                        if isinstance(arg, ast.Name) and arg.id == param_name
                    )
                    visit(child.func.id, callee.args.args[index].arg)

    visit(entrypoint, param)
    return required, optional - required


def test_prepare_checkpoints_reads_the_experiment_the_launcher_builds(
    cookbook,
) -> None:
    """``prep`` reads its constants off an experiment *module*; the launcher fakes
    one from the recipe, so an upstream rename silently drops (e.g.) the bf16
    masters path and the pool boots on the wrong baseline."""
    recipe = Qwen3_30B_A3B_Stitch_Recipe()
    exp = SimpleNamespace(
        SOURCE_MODEL=recipe.train.source_hf_checkpoint
        or recipe.served_baseline(recipe.model_config_class()),
        BF16_CHECKPOINT_PATH=recipe.bf16_checkpoint_path,
        SERVED_CHECKPOINT_FORMAT=recipe.served_checkpoint_format,
        MATERIALIZE_BF16_MASTERS=bool(recipe.bf16_checkpoint_path),
        PREP_ENV=dict(recipe.prep_env),
        miles=recipe.train,
    )
    required, _ = _reads(
        cookbook / "miles_disagg" / "prep.py", "prepare_checkpoints", "exp"
    )

    missing = {name for name in required if not hasattr(exp, name)}
    assert not missing, f"prep.prepare_checkpoints reads unsupplied fields: {missing}"


def test_claim_pool_reads_the_args_the_launcher_builds(cookbook) -> None:
    """The publish hooks take their run coordinates off the miles args namespace;
    the launcher builds the same one directly for the launch-time claim, which
    is what resets the pool to base before the first publish."""
    recipe = Qwen3_30B_A3B_Stitch_Recipe()
    args = SimpleNamespace(
        update_weight_disk_dir="/bulletin/run-1/updates",
        **{field: getattr(recipe.train, field) for field in sorted(HOOK_CONFIG_FIELDS)},
        experiment_volume_name="bulletin",
        rollout_modal_flash_app_name="stitch-app",
        rollout_modal_flash_server_cls_name="Server",
        run_id="run-1",
    )
    required, optional = _reads(cookbook / "common" / "hooks.py", "claim_pool", "args")

    missing = {name for name in required | optional if not hasattr(args, name)}
    assert not missing, f"hooks.claim_pool reads unsupplied fields: {missing}"


# ── the entry points the launcher and recipe name ──────────────────────────────
def _defines(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_named_cookbook_entry_points_exist(cookbook) -> None:
    """Half of these are reached by dotted-path strings miles imports at runtime
    (the hook fields), which no import in this repo would catch."""
    from modal_training_gym.train_recipes.stitch_recipe.train import StitchTrainConfig

    train = StitchTrainConfig
    hook_paths = [
        train.custom_update_weight_post_write_path,
        train.custom_rollout_request_hook_path,
    ]
    for dotted in hook_paths:
        module, _, attr = dotted.rpartition(".")
        path = cookbook.parent / (module.replace(".", "/") + ".py")
        assert path.is_file(), f"{dotted}: no {path}"
        assert attr in _defines(path), f"{dotted}: not defined in {path.name}"

    assert {"claim_pool", "commit_and_wake"} <= _defines(cookbook / "common/hooks.py")
    assert {"resolve_config", "build_train_cmd", "materialize_node_local_yaml"} <= (
        _defines(cookbook / "common/launch.py")
    )
    assert "apply_git_patches" in _defines(cookbook / "common/process.py")
    assert "get_modal_cluster_context" in _defines(cookbook / "common/ray_cluster.py")


def test_serve_startup_takes_the_arguments_the_replica_passes(cookbook) -> None:
    """The pool's ``@modal.enter`` is the one call site a failure hits latest: a
    stale keyword only shows up when a replica boots."""
    passed = _call_keywords(LAUNCHER, "serve_startup") | _call_keywords(
        LAUNCHER, "serve_stop"
    )
    accepted: set[str] = set()
    for node in ast.parse((cookbook / "common" / "server.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "serve_startup",
            "serve_stop",
        }:
            accepted |= {arg.arg for arg in node.args.args + node.args.kwonlyargs}

    assert passed <= accepted, f"serve_startup no longer takes {passed - accepted}"


def _call_keywords(source: Path, func_name: str) -> set[str]:
    return {
        keyword.arg
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == func_name
        for keyword in node.keywords
        if keyword.arg is not None
    }


def test_the_pin_is_an_exact_commit() -> None:
    """A branch tip would move under a cached image layer, and these checks would
    then verify something the image never built."""
    assert len(STITCH_REPO_REF) == 40 and all(
        char in "0123456789abcdef" for char in STITCH_REPO_REF
    ), textwrap.dedent(
        f"""STITCH_REPO_REF must be a full commit sha, got {STITCH_REPO_REF!r}"""
    )
