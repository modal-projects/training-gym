from __future__ import annotations

import argparse
import os
import posixpath
import re
import subprocess
import textwrap
from dataclasses import dataclass
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
MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_SRC = re.compile(
    r"(<img\b[^>]*?\bsrc=)([\"'])([^\"']+)(\2)",
    re.IGNORECASE,
)


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
        if value and branch_exists_on_origin(value):
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
RAW_BASE = f"https://raw.githubusercontent.com/modal-projects/training-gym/{REF}"


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
        elif (
            normalized.startswith("tutorials/") and not PurePosixPath(normalized).suffix
        ):
            rewritten = f"{TREE_BASE}/{normalized}"
        elif normalized.endswith(".md") or normalized == "LICENSE":
            rewritten = f"{BLOB_BASE}/{normalized}"
        else:
            rewritten = f"{BLOB_BASE}/{normalized}"

        if hash_part:
            rewritten = f"{rewritten}#{hash_part}"
        return f"[{label}]({rewritten})"

    return MARKDOWN_LINK.sub(replace, markdown)


_RAW_ASSET_PREFIX = (
    "https://raw.githubusercontent.com/modal-projects/training-gym/main/assets/"
)


def rewrite_image_target(target: str, *, source_dir: PurePosixPath) -> str:
    if target.startswith(_RAW_ASSET_PREFIX):
        filename = target[len(_RAW_ASSET_PREFIX) :]
        return f"/{filename}"

    if target.startswith(("http://", "https://", "data:", "#", "/")):
        return target

    path_part, hash_part = (target.split("#", 1) + [""])[:2]
    normalized = posixpath.normpath(PurePosixPath(source_dir, path_part).as_posix())

    if normalized.startswith("assets/"):
        filename = normalized[len("assets/") :]
        rewritten = f"/{filename}"
    else:
        rewritten = f"{RAW_BASE}/{normalized}"

    if hash_part:
        rewritten = f"{rewritten}#{hash_part}"
    return rewritten


def rewrite_images(markdown: str, *, source_dir: PurePosixPath) -> str:
    def replace(match: re.Match[str]) -> str:
        alt_text, target = match.groups()
        return f"![{alt_text}]({rewrite_image_target(target, source_dir=source_dir)})"

    return MARKDOWN_IMAGE.sub(replace, markdown)


def rewrite_html_image_sources(markdown: str, *, source_dir: PurePosixPath) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, quote, target, _closing_quote = match.groups()
        rewritten = rewrite_image_target(target, source_dir=source_dir)
        return f"{prefix}{quote}{rewritten}{quote}"

    return HTML_IMAGE_SRC.sub(replace, markdown)


@dataclass(frozen=True)
class Catalog:
    begin: str
    end: str
    css_class: str
    column_tracks: tuple[str, ...]

    @property
    def grid_template(self) -> str:
        return " ".join(f"minmax(0, {track})" for track in self.column_tracks)


CATALOGS: tuple[Catalog, ...] = (
    Catalog(
        begin="<!-- BEGIN MODELS TABLE -->",
        end="<!-- END MODELS TABLE -->",
        css_class="models-catalog",
        column_tracks=("1.7fr", "0.7fr", "1.1fr", "1.6fr"),
    ),
)


def wrap_catalogs(markdown: str) -> str:
    for catalog in CATALOGS:
        has_begin = catalog.begin in markdown
        has_end = catalog.end in markdown
        if not has_begin and not has_end:
            continue
        if has_begin != has_end:
            raise ValueError(f"catalog {catalog.css_class} has a one-sided marker pair")
        markdown = markdown.replace(
            catalog.begin,
            (
                f'<div class="catalog {catalog.css_class}" '
                f'style="--catalog-columns: {catalog.grid_template}">\n\n'
                f"{catalog.begin}"
            ),
            1,
        )
        markdown = markdown.replace(
            catalog.end,
            f"{catalog.end}\n\n</div>",
            1,
        )
    return markdown


def strip_first_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def strip_developer_guide(markdown: str) -> str:
    """Drop the trailing `# Developer Guide` section (and its preceding `---`).

    The Developer Guide lives in README.md for GitHub readers but is intentionally
    omitted from the published docs site.
    """
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "# Developer Guide":
            cutoff = index
            while cutoff > 0 and not lines[cutoff - 1].strip():
                cutoff -= 1
            if cutoff > 0 and lines[cutoff - 1].strip() == "---":
                cutoff -= 1
            return "\n".join(lines[:cutoff]).rstrip() + "\n"
    return markdown


def transform_markdown(
    source: Path,
    *,
    source_dir: PurePosixPath,
    home_link: str,
    tutorials_link: str,
) -> str:
    page = source.read_text()
    page = strip_developer_guide(page)
    page = wrap_catalogs(page)
    page = convert_github_callouts(page)
    page = rewrite_images(page, source_dir=source_dir)
    page = rewrite_html_image_sources(page, source_dir=source_dir)
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
            """\
            ---
            title: Training Gym SDK
            description: Open-source Python SDK for GRPO and RL post-training of LLMs on Modal.
            next: false
            pagefind: false
            tableOfContents:
              minHeadingLevel: 2
              maxHeadingLevel: 2
            ---
            """
        )

    return textwrap.dedent(
        """\
        ---
        title: Tutorials
        description: Examples for using the Training Gym SDK.
        prev: false
        next: false
        pagefind: false
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
