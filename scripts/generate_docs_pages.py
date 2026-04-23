from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import textwrap
from pathlib import Path, PurePosixPath


REPO_URL = "https://github.com/modal-projects/training-gym"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STARLIGHT_DIR = ROOT / "docs-next" / "src" / "content" / "docs"

CALLOUT_VARIANTS = {
    "CAUTION": "caution",
    "IMPORTANT": "caution",
    "NOTE": "note",
    "TIP": "tip",
    "WARNING": "caution",
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")


def branch_exists_on_origin(branch: str) -> bool:
    if not branch:
        return False
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def current_ref() -> str:
    for env_var in ("GITHUB_REF_NAME", "VERCEL_GIT_COMMIT_REF"):
        value = os.getenv(env_var)
        if value:
            return value

    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    if branch_exists_on_origin(branch):
        return branch
    return "main"


REF = current_ref()
BLOB_BASE = f"{REPO_URL}/blob/{REF}"
TREE_BASE = f"{REPO_URL}/tree/{REF}"
EDIT_BASE = f"{REPO_URL}/edit/{REF}"


def convert_github_callouts(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0

    while index < len(lines):
        match = re.match(r"> \[!(\w+)\]\s*$", lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue

        kind = match.group(1).upper()
        variant = CALLOUT_VARIANTS.get(kind, "note")
        title = kind.title()
        body: list[str] = []
        index += 1

        while index < len(lines) and lines[index].startswith(">"):
            body.append(lines[index][1:].lstrip())
            index += 1

        output.append(f":::{variant}[{title}]")
        output.extend(body)
        output.append(":::")
        output.append("")

    return "\n".join(output).strip() + "\n"


def rewrite_links(
    markdown: str,
    *,
    source_dir: PurePosixPath,
    home_link: str,
    tutorials_link: str,
) -> str:
    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)

        path_part, hash_part = (target.split("#", 1) + [""])[:2]
        normalized = posixpath.normpath(PurePosixPath(source_dir, path_part).as_posix())

        if normalized in {".", "README.md"}:
            rewritten = home_link
        elif normalized == "tutorials/README.md":
            rewritten = tutorials_link
        elif normalized.startswith("tutorials/") and not PurePosixPath(normalized).suffix:
            rewritten = f"{TREE_BASE}/{normalized}"
        elif normalized.endswith(".md") or normalized == "LICENSE":
            rewritten = f"{BLOB_BASE}/{normalized}"
        else:
            rewritten = f"{BLOB_BASE}/{normalized}"

        if hash_part:
            rewritten = f"{rewritten}#{hash_part}"
        return f"[{label}]({rewritten})"

    return MARKDOWN_LINK.sub(replace, markdown)


def strip_first_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def transform_markdown(
    source: Path,
    *,
    source_dir: PurePosixPath,
    home_link: str,
    tutorials_link: str,
) -> str:
    page = source.read_text()
    page = convert_github_callouts(page)
    page = rewrite_links(
        page,
        source_dir=source_dir,
        home_link=home_link,
        tutorials_link=tutorials_link,
    )
    return strip_first_heading(page)


def starlight_frontmatter(destination: str) -> str:
    if destination == "index.md":
        return textwrap.dedent(
            f"""\
            ---
            title: Training Gym SDK
            description: Reusable building blocks and runnable examples for distributed training on Modal.
            editUrl: {EDIT_BASE}/README.md
            ---
            """
        )

    return textwrap.dedent(
        f"""\
        ---
        title: Tutorials
        description: Runnable Modal training examples across intro, RL, SFT, and infrastructure-focused walkthroughs.
        editUrl: {EDIT_BASE}/tutorials/README.md
        ---
        """
    )


def generate_starlight(output_dir: Path) -> None:
    pages = (
        (
            "index.md",
            ROOT / "README.md",
            PurePosixPath("."),
            "/",
            "/tutorials/",
        ),
        (
            "tutorials/index.md",
            ROOT / "tutorials" / "README.md",
            PurePosixPath("tutorials"),
            "/",
            "/tutorials/",
        ),
    )

    for destination, source, source_dir, home_link, tutorials_link in pages:
        content = transform_markdown(
            source,
            source_dir=source_dir,
            home_link=home_link,
            tutorials_link=tutorials_link,
        )
        output_path = output_dir / destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(starlight_frontmatter(destination) + "\n" + content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Starlight docs pages from the repo READMEs."
    )
    parser.add_argument(
        "--target",
        choices=["starlight"],
        default="starlight",
        help="Docs surface to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STARLIGHT_DIR,
        help="Directory where generated pages should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target == "starlight":
        generate_starlight(args.output_dir)


if __name__ == "__main__":
    main()
