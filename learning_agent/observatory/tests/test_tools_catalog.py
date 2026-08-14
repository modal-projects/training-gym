"""Every tool conforms to the TOOLS.md contract, and the catalog the
dashboard serves (static/assets/tools.json) is exactly what the validator
emits — a stale committed catalog fails here, not in production."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "observatory" / "static" / "assets" / "tools.json"


class TestToolsCatalog(unittest.TestCase):
    def _emit(self, path: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "observatory/validate_tools.py", "--emit-json", str(path)],
            cwd=REPO, capture_output=True, text=True)

    def test_all_tools_conform_to_spec(self):
        cp = self._emit(Path(tempfile.mkdtemp()) / "tools.json")
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)

    def test_committed_catalog_is_fresh(self):
        self.assertTrue(CATALOG.is_file(),
                        "run: python3 observatory/validate_tools.py --emit-json "
                        "observatory/static/assets/tools.json")
        tmp = Path(tempfile.mkdtemp()) / "tools.json"
        cp = self._emit(tmp)
        self.assertEqual(cp.returncode, 0, cp.stdout + cp.stderr)
        fresh = json.loads(tmp.read_text())
        committed = json.loads(CATALOG.read_text())
        self.assertEqual(committed, fresh,
                         "stale tools.json — re-run the --emit-json command above")

    def test_bank_tools_md_carries_no_catalog(self):
        """The bank TOOLS.md is the source superset of PROSE only: the catalog
        is generated per workspace at seeding (--catalog-for), so nothing may
        be committed below the marker."""
        tools_md = REPO / "workspace_setup" / "toolbox_bank" / "TOOLS.md"
        self.assertTrue(tools_md.is_file())
        sys.path.insert(0, str(REPO / "observatory"))
        try:
            import validate_tools as vt
        finally:
            sys.path.pop(0)
        head, tail = tools_md.read_text().split(vt.MARKER)
        self.assertEqual(tail.strip(), "",
                         "committed catalog below the marker — remove it; "
                         "seeding generates the catalog per workspace")

    def test_catalog_for_generates_from_directory(self):
        """catalog_for derives rows from the files present in ONE composed
        toolbox dir — the bank itself is a valid such dir."""
        sys.path.insert(0, str(REPO / "observatory"))
        try:
            import validate_tools as vt
        finally:
            sys.path.pop(0)
        md = vt.catalog_for(REPO / "workspace_setup" / "toolbox_bank")
        self.assertIn("## harness_tool", md)
        self.assertIn("react_loop.py", md)
        self.assertIn("## cloned packages", md)
        self.assertIn("training_gym", md)

    def test_catalog_has_multiple_categories(self):
        if not CATALOG.is_file():
            self.skipTest("catalog not yet emitted")
        cats = json.loads(CATALOG.read_text())["categories"]
        self.assertGreaterEqual(len(cats), 3, f"suspiciously few categories: {list(cats)}")


if __name__ == "__main__":
    unittest.main()
