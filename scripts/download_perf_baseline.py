"""Download baseline validation results from merged PRs for perf comparison.

Runs on a GitHub Actions runner between the download-results and summarize
steps: for each model this run produced a result for, fetch the most recent
`validate-result-<model>` artifact that (a) came from a workflow run on a PR
that was later merged and (b) recorded a successful validation, and unzip it
into the baseline directory consumed by
`validate_model_configs.py summarize --baseline-dir`.

Also writes a `{artifact}.meta.json` sidecar with the commit SHA/URL the
baseline ran on, so the CI comment can link to that commit.

Missing or failing artifacts are skipped with a warning so a stale baseline
never blocks the comment; the baseline directory is created regardless.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

# ``requests`` and PyGithub are only in the ``ci`` dependency group, so they are
# imported where they are used rather than at module scope: the test job syncs
# ``dev`` alone and still has to import this module.
if TYPE_CHECKING:
    from github.Artifact import Artifact
    from github.Repository import Repository

ARTIFACT_PREFIX = "validate-result-"
# Artifacts are inspected newest-first; stop after this many per model so a
# long streak of unmerged/failed candidates can't stall the whole job.
MAX_CANDIDATES_PER_MODEL = 25


def artifact_name_for_model(model_name: str) -> str:
    """Mirror the artifact naming in validate-models.yml ('/' -> '-')."""
    return ARTIFACT_PREFIX + model_name.replace("/", "-")


def _is_from_merged_pr(
    repo: Repository, artifact: Artifact, merged_sha_cache: dict[str, bool]
) -> bool:
    """Whether the artifact's workflow run commit belongs to a merged PR.

    Cached per head SHA: one workflow run uploads an artifact per model, so
    every model's candidate list revisits the same commits.
    """
    run = artifact.workflow_run
    sha = run.head_sha if run is not None else None
    if not sha:
        return False
    if sha not in merged_sha_cache:
        pulls = repo.get_commit(sha).get_pulls()
        merged_sha_cache[sha] = any(pr.merged_at is not None for pr in pulls)
    return merged_sha_cache[sha]


def _fetch_artifact_zip(artifact: Artifact, token: str) -> zipfile.ZipFile:
    """Download the artifact archive.

    ``requests`` follows the API's redirect to blob storage and drops the
    Authorization header across hosts, which the signed URL requires.
    """
    import requests

    response = requests.get(
        artifact.archive_download_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


# TODO(anthony): Use synmon model runs as a baseline instead of searching CI
#                once synmon is set up
def download_baseline_for_model(
    repo: Repository,
    token: str,
    model_name: str,
    baseline_dir: Path,
    merged_sha_cache: dict[str, bool],
) -> bool:
    """Write the latest merged-PR successful result for a model into baseline_dir.

    Walks the model's artifacts newest-first (``get_artifacts`` paginates
    transparently) and keeps the first candidate whose validation actually
    succeeded, so a bad run merged to main never poisons the baseline.

    Alongside the result JSON, writes ``{artifact_name}.meta.json`` with the
    commit SHA/URL the baseline ran on, so summarize can link to it.
    """
    artifact_name = artifact_name_for_model(model_name)
    inspected = 0
    for artifact in repo.get_artifacts(name=artifact_name):
        if inspected >= MAX_CANDIDATES_PER_MODEL:
            print(
                f"warning: gave up after inspecting {inspected} "
                f"{artifact_name!r} candidates"
            )
            break
        inspected += 1
        if artifact.expired:
            continue
        if not _is_from_merged_pr(repo, artifact, merged_sha_cache):
            continue
        with _fetch_artifact_zip(artifact, token) as archive:
            result_name = f"{artifact_name}.json"
            result_bytes = archive.read(result_name)
            result = json.loads(result_bytes)
            if not result.get("succeeded"):
                continue
            (baseline_dir / result_name).write_bytes(result_bytes)
            _write_baseline_meta(
                repo, artifact, baseline_dir / f"{artifact_name}.meta.json"
            )
        print(f"downloaded baseline {artifact_name!r} (artifact id {artifact.id})")
        return True
    print(
        f"warning: no successful {artifact_name!r} artifact found from a "
        f"merged PR; skipping baseline for {model_name}"
    )
    return False


def _write_baseline_meta(repo: Repository, artifact: Artifact, meta_path: Path) -> None:
    """Persist commit provenance next to the extracted baseline result JSON."""
    run = artifact.workflow_run
    sha = run.head_sha if run is not None else None
    if not sha:
        print(f"warning: no head SHA on artifact {artifact.id}; skipping meta")
        return
    meta = {
        "commit_sha": sha,
        "commit_url": f"{repo.html_url}/commit/{sha}",
        "artifact_id": artifact.id,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def models_with_results(results_dir: Path) -> list[str]:
    """Model names this run actually produced a result for.

    A baseline is only ever compared against a result from the same run, so
    the results directory — not the registry — is the right list. Asking the
    registry would scan artifacts for models the run never validated: every
    model the diff didn't select, plus dispatch-only ones like Kimi, which can
    have no PR artifact at all.
    """
    names = set()
    for path in sorted(results_dir.glob(f"{ARTIFACT_PREFIX}*.json")):
        try:
            name = json.loads(path.read_text()).get("base_model_name")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: unreadable result {path.name}: {exc}")
            continue
        if name:
            names.add(name)
    return sorted(names)


def download_baselines(baseline_dir: Path, results_dir: Path) -> None:
    from github import Auth, Github

    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPOSITORY"]

    baseline_dir.mkdir(parents=True, exist_ok=True)
    repo = Github(auth=Auth.Token(token)).get_repo(repo_name)

    downloaded = 0
    merged_sha_cache: dict[str, bool] = {}
    models = models_with_results(results_dir)
    if not models:
        print(f"no validation results in {results_dir}; no baselines to fetch")
        return
    for model_name in models:
        try:
            downloaded += download_baseline_for_model(
                repo, token, model_name, baseline_dir, merged_sha_cache
            )
        except Exception as exc:
            print(f"warning: failed to download baseline for {model_name}: {exc}")

    print(f"downloaded {downloaded}/{len(models)} baselines into {baseline_dir}")


def __main__():
    parser = argparse.ArgumentParser(
        description=(
            "Download baseline validation artifacts from merged PRs for each model."
        )
    )
    parser.add_argument(
        "-d",
        "--baseline-dir",
        default="baseline",
        help="Directory to unzip baseline result JSON files into. Defaults to 'baseline'.",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        default="results",
        help="Directory of this run's result JSON files. Baselines are fetched "
        "only for the models found here. Defaults to 'results'.",
    )
    args = parser.parse_args()
    download_baselines(Path(args.baseline_dir), Path(args.results_dir))


if __name__ == "__main__":
    __main__()
