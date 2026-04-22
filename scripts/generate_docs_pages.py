from __future__ import annotations

import os
import posixpath
import re
import subprocess
from pathlib import Path, PurePosixPath

import mkdocs_gen_files


REPO_URL = "https://github.com/modal-projects/training-gym"
ROOT = Path(__file__).resolve().parents[1]

CALL_OUT_KIND = {
    "CAUTION": "warning",
    "IMPORTANT": "warning",
    "NOTE": "note",
    "TIP": "tip",
    "WARNING": "warning",
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")


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
    return branch or "main"


REF = current_ref()
BLOB_BASE = f"{REPO_URL}/blob/{REF}"
TREE_BASE = f"{REPO_URL}/tree/{REF}"


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
        admonition = CALL_OUT_KIND.get(kind, "note")
        title = kind.title()
        body: list[str] = []
        index += 1

        while index < len(lines) and lines[index].startswith(">"):
            body.append(lines[index][1:].lstrip())
            index += 1

        output.append(f'!!! {admonition} "{title}"')
        if body:
            output.extend(f"    {line}" if line else "" for line in body)
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


def write_page(
    destination: str,
    source: Path,
    *,
    source_dir: PurePosixPath,
    home_link: str,
    tutorials_link: str,
) -> None:
    page = source.read_text()
    page = convert_github_callouts(page)
    page = rewrite_links(
        page,
        source_dir=source_dir,
        home_link=home_link,
        tutorials_link=tutorials_link,
    )

    if destination == "index.md":
        page = page.replace("# Training Gym", "# Training Gym SDK", 1)

    with mkdocs_gen_files.open(destination, "w") as generated:
        generated.write(page)


write_page(
    "index.md",
    ROOT / "README.md",
    source_dir=PurePosixPath("."),
    home_link="index.md",
    tutorials_link="tutorials/index.md",
)
write_page(
    "tutorials/index.md",
    ROOT / "tutorials" / "README.md",
    source_dir=PurePosixPath("tutorials"),
    home_link="../index.md",
    tutorials_link="index.md",
)
