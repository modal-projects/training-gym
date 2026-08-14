"""ingest --data-dir stages the viewer's local layout; --archive-workspace
packs the submission folder (minus corpus/.git/venvs) into raw/.

Both run offline against the demo fixture (--no-upload), so no modal dep.
"""
from __future__ import annotations

import tarfile
import tempfile
import unittest
from pathlib import Path

from observatory import cli, schema

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "demo" / \
    "ws_claude_dspy_20260717T090000"


class DataDirStaging(unittest.TestCase):
    def test_stages_into_viewer_layout(self):
        with tempfile.TemporaryDirectory() as td:
            rc = cli.main(["ingest", str(FIXTURE), "--no-upload", "--data-dir", td])
            self.assertEqual(rc, 0)
            runs = list((Path(td) / schema.RUNS_PREFIX).iterdir())
            self.assertEqual(len(runs), 1)
            run_dir = runs[0]
            for name in (schema.RECORD_FILE, schema.WORKSPACE_FILE, schema.STATUS_FILE):
                self.assertTrue((run_dir / name).is_file(), f"missing {name}")
            self.assertTrue((run_dir / schema.RAW_DIR / "trace.jsonl").is_file())


class WorkspaceArchive(unittest.TestCase):
    def test_archive_contains_submission_excludes_bulk(self):
        with tempfile.TemporaryDirectory() as td:
            rc = cli.main(["ingest", str(FIXTURE), "--no-upload", "--data-dir", td,
                           "--archive-workspace"])
            self.assertEqual(rc, 0)
            run_dir = next((Path(td) / schema.RUNS_PREFIX).iterdir())
            tar_path = run_dir / schema.RAW_DIR / "workspace.tar.gz"
            self.assertTrue(tar_path.is_file())
            with tarfile.open(tar_path) as tar:
                names = [n.lstrip("./") for n in tar.getnames()]
            self.assertIn("submission/eval.py", names)
            for kept_out in names:
                self.assertNotIn(".git/", kept_out + "/")
                self.assertFalse(kept_out.startswith("agents/_runs"),
                                 f"run dir recursed into archive: {kept_out}")
                parts = Path(kept_out).parts
                if len(parts) >= 3 and parts[0] == "tasks":
                    self.assertNotEqual(parts[2], "corpus",
                                        f"corpus leaked into archive: {kept_out}")

    def test_archive_filter_prunes_nested_and_prefixed(self):
        keep = cli._archive_filter(Path("/tmp/x"))
        for pruned in ("./.git/config", "./tasks/dspy/corpus/a.py",
                       "./.venv-rl/bin/python", "./agents/_runs/r1/trace.jsonl",
                       "./sub/__pycache__/m.pyc"):
            info = tarfile.TarInfo(pruned)
            self.assertIsNone(keep(info), f"should prune {pruned}")
        for kept in ("./submission/eval.py", "./tasks/dspy/task.md", "./bench.py"):
            info = tarfile.TarInfo(kept)
            self.assertIsNotNone(keep(info), f"should keep {kept}")


if __name__ == "__main__":
    unittest.main()
