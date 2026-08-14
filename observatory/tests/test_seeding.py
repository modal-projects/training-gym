"""The seeded workspace is exactly the learning_agent_workspace/ surface plus
the injected run machinery, prompt material, and the ONE task — and nothing
else from the operator repo. Runs the real seeding routine
(workspace_setup/prepare_workspace.sh) against HEAD, so it sees the last COMMITTED
state: commit before running.
"""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TRACK, SCAFFOLD, TASK = "easy", "codex_kimi3", "openclaw"


class TestSeeding(unittest.TestCase):
    ws: Path
    parent: Path

    @classmethod
    def setUpClass(cls):
        cls.parent = Path(tempfile.mkdtemp(prefix="lab-seed-test-"))
        script = (
            f'source "{REPO}/workspace_setup/prepare_workspace.sh" && '
            f'prepare_workspace "{REPO}" "{cls.parent}" {TRACK} {SCAFFOLD} {TASK} 6'
        )
        cp = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        if cp.returncode != 0:
            raise AssertionError(f"prepare_workspace failed:\n{cp.stderr[-2000:]}")
        cls.ws = cls.parent / "workspace"

    def test_agent_surface_present(self):
        for rel in ("AGENTS.md", "toolbox/TOOLS.md",
                    "toolbox/gpu_tools/gpu_launcher.py", "submission/eval.py",
                    "runs/LEARNING_LOG.jsonl", ".learning_agent_sandbox"):
            self.assertTrue((self.ws / rel).exists(), f"missing {rel}")

    def test_ledgers_start_empty(self):
        self.assertEqual((self.ws / "runs/LEARNING_LOG.jsonl").read_text(), "",
                         "learning log not empty")

    def test_run_machinery_injected(self):
        for rel in ("agents/run.sh", "agents/lib/make_prompt.py",
                    "agents/lib/audit_trace.py", f"agents/{SCAFFOLD}/solve.sh",
                    "bench/config.yaml"):
            self.assertTrue((self.ws / rel).exists(), f"missing {rel}")
        # the setup surface never enters a workspace: the agent gets the
        # stitched AGENTS.md, not the template and every track's blocks
        self.assertFalse((self.ws / "instructions").exists())
        self.assertTrue((self.ws / "AGENTS.md").is_file())

    def test_scaffold_dependencies_resolve(self):
        """A scaffold may depend only on its own dir and agents/lib/ — the
        seeded set. Anything it sources/executes from another scaffold's dir
        breaks in the workspace (the codex_glm52 -> modal_glm52/config.env
        incident, 2026-08-11)."""
        import re
        solve = (self.ws / "agents" / SCAFFOLD / "solve.sh").read_text()
        for m in re.finditer(r'\$SCAFFOLD_DIR/\.\./([\w./-]+)', solve):
            dep = m.group(1)
            self.assertTrue(dep.startswith("lib/"),
                            f"scaffold depends on non-lib sibling: ../{dep}")
            self.assertTrue((self.ws / "agents" / dep).exists(),
                            f"scaffold dependency missing from workspace: agents/{dep}")

    def test_operator_material_absent(self):
        for rel in ("bench.py", "harness", "observatory", "dev", "launch",
                    "bench/pins.json", "agents/run_sandbox.sh",
                    "agents/modal_runner.py", "learning_agent_workspace"):
            self.assertFalse((self.ws / rel).exists(), f"{rel} leaked into workspace")

    def test_unused_scaffolds_absent(self):
        scaffolds = [d.name for d in (self.ws / "agents").iterdir()
                     if d.is_dir() and (d / "solve.sh").is_file()]
        self.assertEqual(scaffolds, [SCAFFOLD])

    def test_one_task_at_task_dir_and_no_heldout(self):
        # the task CONFIG never enters a workspace — the agent gets assets only
        self.assertFalse((self.ws / "task" / "task.yaml").exists())
        self.assertTrue((self.ws / "task" / "task.md").is_file())
        self.assertTrue((self.ws / "task" / "corpus").is_dir(), "corpus not seeded")
        self.assertFalse((self.ws / "task" / "test.json").exists())
        self.assertFalse((self.ws / "tasks").exists(), "operator tasks/ leaked")
        self.assertTrue((self.ws / "model").is_dir(), "model/ placeholder missing")

    def test_spec_is_filled(self):
        spec = (self.ws / "AGENTS.md").read_text()
        for placeholder in ("<TASK>", "<TASK_MODEL>", "<MEASURING_YOURSELF>"):
            self.assertNotIn(placeholder, spec)
        self.assertIn(TASK, spec)

    def test_manifest_matches_workspace(self):
        """Every manifest path is workspace-relative and (held-out files
        aside) exists in the workspace — the invariant the observatory's
        seed-vs-invented provenance depends on."""
        manifest = (self.parent / "seed_manifest.txt").read_text().splitlines()
        self.assertGreater(len(manifest), 20)
        allowed_missing = {"task/test.json", "task/dev.json"}
        missing = []
        for line in manifest:
            path = line.split("\t", 1)[1]
            self.assertFalse(path.startswith("learning_agent_workspace/"),
                             f"manifest path not workspace-relative: {path}")
            if not (self.ws / path).exists() and path not in allowed_missing:
                missing.append(path)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
