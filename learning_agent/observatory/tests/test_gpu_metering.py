"""Attribution logic tests for gpu_metering — pure, no modal/network."""

from __future__ import annotations

import unittest

from observatory.gpu_metering import attribute

NOW = 1_000_000.0
H = 3600.0


def sb(sid, start, end, gpu="H200", n=1, tags=None, app="ap-x"):
    return {"sandbox_id": sid, "app_id": app, "tags": tags or {},
            "created_at": start, "started_at": start, "finished_at": end,
            "gpu_type": gpu, "n_gpus": n}


class TestAttribute(unittest.TestCase):
    def test_window_clipping_and_totals(self):
        windows = {"run_a": (NOW - 10 * H, NOW - 5 * H)}
        boxes = [
            sb("sb1", NOW - 9 * H, NOW - 7 * H),            # 2h inside
            sb("sb2", NOW - 6 * H, NOW - 4 * H, n=2),       # 1h inside x2 gpus
            sb("sb3", NOW - 3 * H, NOW - 1 * H),            # fully outside
        ]
        out = attribute(boxes, windows, now=NOW)
        run = out["runs"]["run_a"]
        self.assertAlmostEqual(run["gpu_hours"], 4.0)        # 2 + 1*2
        self.assertAlmostEqual(run["by_gpu"]["H200"], 4.0)
        self.assertEqual(run["n_gpu_sandboxes"], 2)
        self.assertFalse(run["shared_window"])
        # sb2's 1h outside the window + sb3's 2h are unattributed
        self.assertAlmostEqual(out["unattributed_gpu_hours"], 4.0)

    def test_running_run_and_running_sandbox_use_now(self):
        windows = {"live": (NOW - 2 * H, None)}              # still running
        boxes = [sb("sb1", NOW - 1 * H, None)]               # still running
        out = attribute(boxes, windows, now=NOW)
        self.assertAlmostEqual(out["runs"]["live"]["gpu_hours"], 1.0)

    def test_shared_window_flagged_on_both(self):
        windows = {"a": (NOW - 4 * H, NOW - 2 * H),
                   "b": (NOW - 3 * H, NOW - 1 * H)}          # overlapping runs
        boxes = [sb("sb1", NOW - 4 * H, NOW - 1 * H)]
        out = attribute(boxes, windows, now=NOW)
        self.assertTrue(out["runs"]["a"]["shared_window"])
        self.assertTrue(out["runs"]["b"]["shared_window"])
        self.assertAlmostEqual(out["runs"]["a"]["gpu_hours"], 2.0)  # clipped
        self.assertAlmostEqual(out["runs"]["b"]["gpu_hours"], 2.0)

    def test_run_id_tag_beats_window(self):
        windows = {"a": (NOW - 4 * H, NOW - 2 * H),
                   "b": (NOW - 3 * H, NOW - 1 * H)}
        boxes = [sb("sb1", NOW - 3 * H, NOW - 2 * H,
                    tags={"learning_agent_run_id": "b"})]
        out = attribute(boxes, windows, now=NOW)
        self.assertNotIn("a", out["runs"])                   # tag wins, no split
        self.assertAlmostEqual(out["runs"]["b"]["gpu_hours"], 1.0)
        self.assertFalse(out["runs"]["b"]["shared_window"])

    def test_cpu_sandboxes_ignored(self):
        windows = {"a": (NOW - 2 * H, NOW)}
        boxes = [{"sandbox_id": "sb1", "app_id": "ap", "tags": {},
                  "created_at": NOW - H, "started_at": NOW - H,
                  "finished_at": NOW, "gpu_type": None, "n_gpus": 0}]
        out = attribute(boxes, windows, now=NOW)
        self.assertEqual(out["runs"], {})
        self.assertEqual(out["unattributed_gpu_hours"], 0.0)


if __name__ == "__main__":
    unittest.main()
