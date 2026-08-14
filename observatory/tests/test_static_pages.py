"""The static pages and their routes stay in sync across both servers.

FastAPI isn't installed in the default test env (it lives in the Modal image /
observatory venv), so these tests pin the static contract at the source level:
every page a server routes to must exist, and both servers must know the same
page set. Route behavior itself is exercised by running either server.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

OBS = Path(__file__).resolve().parents[1]
STATIC = OBS / "static"
DOCS_SLUGS = ("workflow", "tasks", "agent", "toolbox", "training", "eval", "integrity", "runs")
PAGES = (["index.html", "run.html", "docs.html", "tools.html"]
         + [f"docs-{s}.html" for s in DOCS_SLUGS])


class StaticPagesExist(unittest.TestCase):
    def test_every_page_file_exists(self):
        for name in PAGES:
            with self.subTest(page=name):
                self.assertTrue((STATIC / name).is_file(), f"missing static/{name}")

    def test_app_routes_every_page(self):
        src = (OBS / "app.py").read_text()
        for name in ("index.html", "run.html", "docs.html", "tools.html"):
            with self.subTest(page=name):
                self.assertIn(f'_page("{name}")', src)
        # docs subpages route through the slug whitelist — it must cover every file
        m = re.search(r"DOCS_PAGES = \(([^)]*)\)", src)
        self.assertIsNotNone(m, "app.py lost its DOCS_PAGES whitelist")
        routed = set(re.findall(r'"(\w+)"', m.group(1)))
        self.assertEqual(routed, set(DOCS_SLUGS))
        self.assertIn('_page(f"docs-{page}.html")', src)
        # the old single-page walkthrough redirects into the docs
        self.assertIn('RedirectResponse("/docs"', src)

    def test_dev_server_rewrites_every_page(self):
        src = (OBS / "static_dev_server.py").read_text()
        for marker in ('"/index.html"', '"/run.html"', '"/docs.html"',
                       '"/tools.html"', 'f"/docs-'):
            with self.subTest(marker=marker):
                self.assertIn(marker, src)


class DocsContract(unittest.TestCase):
    """The docs must keep covering the whole pipeline, and every page must
    carry the shared shell (nav, docs.js, theme tokens only)."""

    COVERAGE = {
        "docs.html": ["margin", "submission/eval.py", "autonomy ladder"],
        "docs-workflow.html": ["run_sandbox_modal", "LEARNING_LOG.jsonl",
                               "candidates.json", "margin", "GRPO"],
        "docs-tasks.html": ["task.yaml", "easy", "medium", "hard", "brief.md"],
        "docs-agent.html": ["scaffold", "responses_shim", "timer.sh", "Re-prompt"],
        "docs-toolbox.html": ["docstring", "TOOLS.md", "invented", "repos.yaml"],
        "docs-training.html": ["training_tool/", "gpu_launcher", "/out/models/",
                               "LEARNING_LOG.jsonl"],
        "docs-eval.html": ["n_votes", "bootstrap", "canonical", "base_floor",
                           "read_file", "FINAL"],
        "docs-integrity.html": ["pins.json", "freeze", ".learning_agent_sandbox",
                                "audit", "git archive HEAD"],
        "docs-runs.html": ["run_sandbox", "lab-observatory", "nth_use",
                           "sync-scores"],
    }

    def test_pages_cover_their_topics(self):
        for page, markers in self.COVERAGE.items():
            text = (STATIC / page).read_text()
            for marker in markers:
                with self.subTest(page=page, marker=marker):
                    self.assertIn(marker, text)

    def test_docs_pages_share_the_shell(self):
        for page in ["docs.html"] + [f"docs-{s}.html" for s in DOCS_SLUGS]:
            text = (STATIC / page).read_text()
            with self.subTest(page=page):
                self.assertIn('id="docs-nav"', text)
                self.assertIn("/assets/docs.js", text)
                self.assertIn("/assets/docs.css", text)
                self.assertIn('id="theme-toggle"', text)

    def test_theme_tokens_only(self):
        # colors come from styles.css custom properties so both themes work;
        # the only literal hex allowed is the favicon data URI
        for page in PAGES:
            text = (STATIC / page).read_text()
            head = text.split("</head>")[0]
            style = re.search(r"<style>(.*?)</style>", head, re.S)
            if style is None:
                continue
            hexes = re.findall(r"#[0-9a-fA-F]{3,8}\b", style.group(1))
            with self.subTest(page=page):
                self.assertEqual(hexes, [], f"hardcoded colors in {page}: {hexes}")


if __name__ == "__main__":
    unittest.main()
