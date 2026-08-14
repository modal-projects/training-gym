"""Dev + test score columns: the checkpoint-ledger dev fallback and the
operator-leaderboard test overlay stay wired end to end (source-level for the
app/frontend, like test_static_pages; pure unit for the collector)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

OBS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OBS.parent))

from observatory.normalize.collect import _best_dev  # noqa: E402


class TestBestDevFallback(unittest.TestCase):
    def test_results_win_when_present(self):
        results = [{"split": "dev", "mean": 0.3, "bootstrap_ci95": [0.2, 0.4], "tag": "a"}]
        cps = [{"tag": "b", "dev_score": 0.9}]
        self.assertEqual(_best_dev(results, cps), (0.3, [0.2, 0.4], "a"))

    def test_checkpoint_fallback_when_no_results(self):
        cps = [{"tag": "v1", "dev_score": 0.19}, {"tag": "v11", "dev_score": 0.3717},
               {"tag": "junk"}]  # rows without dev_score are ignored
        self.assertEqual(_best_dev([], cps), (0.3717, None, "v11"))

    def test_nothing_available(self):
        self.assertEqual(_best_dev([], []), (None, None, None))


class TestOverlayContract(unittest.TestCase):
    def test_app_overlays_leaderboard(self):
        src = (OBS / "app.py").read_text()
        self.assertIn("leaderboard.jsonl", src)
        self.assertIn("_test_overlays", src)
        self.assertIn("submission_mean", src)

    def test_cli_has_sync_scores(self):
        src = (OBS / "cli.py").read_text()
        self.assertIn("sync-scores", src)
        self.assertIn("push_file", src)

    def test_index_table_has_both_score_columns(self):
        html = (OBS / "static" / "index.html").read_text()
        self.assertIn("dev score", html)
        self.assertIn("test score", html)
        js = (OBS / "static" / "assets" / "index.js").read_text()
        self.assertIn("test_score", js)
        self.assertIn("test_margin", js)


if __name__ == "__main__":
    unittest.main()
