from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = ROOT / "README.md"
DEFAULT_SOURCE = ROOT / "scripts" / "quickstart.py"
BEGIN_MARKER = "<!-- BEGIN QUICKSTART -->"
END_MARKER = "<!-- END QUICKSTART -->"


def render_section(source: Path) -> str:
    code = source.read_text()
    if not code.endswith("\n"):
        code += "\n"
    return "\n".join(
        [
            BEGIN_MARKER,
            "```python",
            code.rstrip("\n"),
            "```",
            END_MARKER,
        ]
    )


def rendered_readme(readme_path: Path, source: Path) -> tuple[str, str]:
    content = readme_path.read_text()
    if BEGIN_MARKER not in content or END_MARKER not in content:
        raise SystemExit(
            f"Markers {BEGIN_MARKER} / {END_MARKER} not found in {readme_path}"
        )
    before, rest = content.split(BEGIN_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    return content, f"{before}{render_section(source)}{after}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the README quickstart snippet from scripts/quickstart.py."
    )
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the README snippet is out of date.",
    )
    args = parser.parse_args()
    current, updated = rendered_readme(args.readme, args.source)
    if args.check:
        if current != updated:
            raise SystemExit(
                f"{args.readme} quickstart snippet is out of date. Run "
                "uv run scripts/generate_quickstart.py"
            )
        print("Quickstart snippet is up to date")
        return
    if current != updated:
        args.readme.write_text(updated)
    print(f"Wrote quickstart snippet from {args.source}")


if __name__ == "__main__":
    main()
