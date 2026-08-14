"""The toolbox grows without infrastructure edits: scaffolder produces valid
tools, unknown categories still classify, invented folder tools get honest
provenance and card-refined kinds, and invented cards surface in the record."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

OBS = Path(__file__).resolve().parents[1]
REPO = OBS.parent
sys.path.insert(0, str(REPO))

from observatory.normalize import learning                     # noqa: E402
from observatory.normalize.collect import _collect_learning    # noqa: E402


class TestUnknownCategoryClassification(unittest.TestCase):
    def test_catchall_classifies_new_category_as_tool(self):
        actions = learning.classify_command(
            "python3 toolbox/agent_toolbox/memory_bank/run.py --out o")
        self.assertEqual([(a["kind"], a["tool"]) for a in actions],
                         [("tool", "memory_bank")])

    def test_known_categories_not_shadowed_by_catchall(self):
        for cmd, want in (
            ("python3 toolbox/data_tool/synthetic/paraphrase.py --n 5", ("data", "paraphrase")),
            ("python3 toolbox/data_tool/pool/mix.py --out p", ("data", "mix")),
            ("python3 toolbox/harness_tool/completion_qa.py --out o", ("harness", "completion_qa")),
            ("python3 toolbox/inference_tool/vllm_serve.py --model m", ("infra", "vllm_serve")),
            # legacy paths classify forever
            ("python3 toolbox/data_toolbox/gen/paraphrase/run.py --n 5", ("data", "paraphrase")),
        ):
            actions = learning.classify_command(cmd)
            self.assertEqual([(a["kind"], a["tool"]) for a in actions], [want], cmd)


class TestCollectorRefinement(unittest.TestCase):
    """_collect_learning is pure over its inputs — feed it synthetic events,
    a manifest, and a snapshot, and check kind/provenance refinement."""

    def _run(self, cmd, manifest_paths, snapshot_files):
        events = [{"i": 0, "ts": None, "blocks": [
            {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}]
        run_dir = REPO / "observatory"  # any dir without seed_manifest.txt
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "seed_manifest.txt").write_text(
            "".join(f"100644 blob 0\t{p}\n" for p in manifest_paths))
        snapshot = {"files": [{"path": p, "content": c, "inline": True}
                              for p, c in snapshot_files.items()]}
        return _collect_learning(events, "claude-stream-json", d, None, snapshot)

    def test_invented_folder_tool_gets_invented_provenance_and_card_kind(self):
        cmd = "python3 toolbox/agent_toolbox/memory_bank/run.py --out o"
        card = "name: memory_bank\nkind: data\nsummary: s\n"
        actions, counts, cards = self._run(
            cmd, manifest_paths=["bench.py"],
            snapshot_files={"toolbox/agent_toolbox/memory_bank/tool.yaml": card,
                            "toolbox/agent_toolbox/memory_bank/run.py": "x"})
        self.assertEqual(actions[0]["kind"], "data")          # refined from card
        self.assertEqual(actions[0]["provenance"], "invented")
        self.assertEqual(counts["invented_tools"], 1)
        self.assertEqual(cards[0]["path"], "toolbox/agent_toolbox/memory_bank")

    def test_invented_tool_md_card_is_collected(self):
        cmd = "python3 toolbox/agent_toolbox/memory_bank/run.py --out o"
        doc = "# memory_bank — store and retrieve notes\n\ncost: free   provenance: invented (run: r1)\n"
        actions, counts, cards = self._run(
            cmd, manifest_paths=["bench.py"],
            snapshot_files={"toolbox/agent_toolbox/memory_bank/tool.md": doc,
                            "toolbox/agent_toolbox/memory_bank/run.py": "x"})
        self.assertEqual(actions[0]["provenance"], "invented")
        self.assertEqual(counts["invented_tools"], 1)
        self.assertEqual(cards[0]["path"], "toolbox/agent_toolbox/memory_bank")
        self.assertIn("memory_bank", cards[0]["tool_md"])

    def test_seed_folder_tool_stays_seed(self):
        cmd = "python3 toolbox/data_toolbox/gen/paraphrase/run.py --n 5"
        actions, counts, cards = self._run(
            cmd, manifest_paths=["toolbox/data_toolbox/gen/paraphrase/run.py"],
            snapshot_files={})
        self.assertEqual(actions[0]["provenance"], "seed")
        self.assertEqual(counts["invented_tools"], 0)
        self.assertEqual(cards, [])


if __name__ == "__main__":
    unittest.main()
