#!/usr/bin/env python3
"""clone_repos — materialize the pinned cloned packages from toolbox/repos.yaml.

The cloned reference packages are gitignored —
~200MB of third-party trees don't belong in the benchmark repo. This script
makes a checkout complete: for every manifest entry whose dest is missing or
empty, it either copies the tree from another checkout (--copy-from, what
workspace seeding uses: instant, offline) or clones it from upstream at the
EXACT pinned commit (git fetch <sha>, checkout, strip .git — a vendor is a
plain pinned tree, not a live repo).

    python3 toolbox/clone_repos.py                  # clone whatever is missing
    python3 toolbox/clone_repos.py --copy-from /path/to/seed/repo
    python3 toolbox/clone_repos.py --check          # report, change nothing (rc 1 if missing)

In the source repo this file lives at toolbox_bank/clone_repos.py and
materializes the clones inside the bank; in a seeded workspace it lands at
toolbox/clone_repos.py next to the generated repos.yaml.

Stdlib only; the manifest is parsed line-wise (name:, repo:, commit:, dest:) so
no yaml dependency. Idempotent: an existing non-empty dest is never touched —
edit the pin in repos.yaml and delete the dest to re-materialize.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "repos.yaml"


def resolve_dest(dest: str) -> Path:
    """Where a manifest dest ("toolbox/training_tool/<pkg>") lands on disk.

    In a seeded workspace this file is toolbox/clone_repos.py and the dest
    resolves from the workspace root. In the source repo it is
    toolbox_bank/clone_repos.py — the bank IS the toolbox there, so the same
    dest resolves inside the bank with the "toolbox/" prefix dropped."""
    if HERE.name == "toolbox_bank":
        return HERE / Path(dest).relative_to("toolbox")
    return HERE.parent / dest


def copy_src(copy_from: Path, dest: str) -> Path | None:
    """The source tree for --copy-from: workspace layout first, bank layout
    as the fallback (so --copy-from may point at either kind of checkout)."""
    for cand in (copy_from / dest, copy_from / Path(dest).relative_to("toolbox")):
        if present(cand):
            return cand
    return None


def read_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    """{name: {repo, commit, dest}} from the flat two-level manifest."""
    vendors: dict[str, dict] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            current = line.rstrip(":").strip()
            vendors[current] = {}
        elif current is not None and ":" in line:
            key, _, value = line.strip().partition(":")
            vendors[current][key.strip()] = value.strip()
    bad = [n for n, v in vendors.items()
           if not all(v.get(k) for k in ("repo", "commit", "dest"))]
    if bad:
        raise SystemExit(f"[repos] manifest entries missing repo/commit/dest: {bad}")
    return vendors


def present(dest: Path) -> bool:
    return dest.is_dir() and any(dest.iterdir())


def _write_git_guard(dest: Path) -> None:
    """A clone has no .git, so git commands run INSIDE it resolve upward to
    this repo — a package's own tooling (e.g. a clone's release scripts,
    which headers every `git ls-files` .py) would then rewrite this repo's files.
    Observed 2026-08-06. A deliberately broken gitfile makes git fail loudly
    inside the clone instead."""
    (dest / ".git").write_text("gitdir: this-is-a-pinned-clone-not-a-git-repo\n")


def clone_at_pin(repo: str, commit: str, dest: Path) -> None:
    """Fetch exactly the pinned commit (depth 1), check it out, strip .git."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".cloning")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        git("init", "-q")
        git("remote", "add", "origin", repo)
        git("fetch", "-q", "--depth", "1", "origin", commit)
        git("checkout", "-q", "FETCH_HEAD")
        shutil.rmtree(tmp / ".git")
        _write_git_guard(tmp)
        tmp.rename(dest)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"[repos] clone failed for {repo} @ {commit[:7]}: {e}") from e


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize pinned cloned packages.")
    ap.add_argument("--copy-from", default=None,
                    help="another checkout to copy present vendors from (offline path; "
                         "falls back to cloning anything it too is missing)")
    ap.add_argument("--check", action="store_true",
                    help="report presence only; rc 1 if anything is missing")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--only", default="",
                    help="comma/space-separated package names to materialize "
                         "(others are skipped); default: everything in the manifest")
    args = ap.parse_args()

    vendors = read_manifest(Path(args.manifest))
    if args.only:
        wanted = set(args.only.replace(",", " ").split())
        unknown = wanted - set(vendors)
        if unknown:
            raise SystemExit(f"[repos] --only names unknown packages: {sorted(unknown)}")
        vendors = {k: v for k, v in vendors.items() if k in wanted}
    missing = 0
    for name, v in vendors.items():
        dest = resolve_dest(v["dest"])
        if present(dest):
            if not (dest / ".git").exists():
                _write_git_guard(dest)
            print(f"[repos] {name}: present")
            continue
        if args.check:
            print(f"[repos] {name}: MISSING ({v['dest']})")
            missing += 1
            continue
        src = copy_src(Path(args.copy_from), v["dest"]) if args.copy_from else None
        if src is not None and present(src):
            print(f"[repos] {name}: copying from {src}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            # APFS copy-on-write when available (instant), plain copy elsewhere.
            rc = subprocess.run(["cp", "-Rc", str(src), str(dest)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            if rc != 0:
                shutil.copytree(src, dest)
        else:
            print(f"[repos] {name}: cloning {v['repo']} @ {v['commit'][:7]}")
            clone_at_pin(v["repo"], v["commit"], dest)
    if args.check and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
