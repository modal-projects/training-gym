# Agent Guide: Validating Example Drift

This document defines a durable validation pipeline for agents that need to check whether the examples in this repository still run as documented.

Use this runbook together with [agent-modal-training.md](agent-modal-training.md), which covers Modal launch and debugging workflow.

## Goal

- Detect when example code, README instructions, pinned dependencies, or Modal runtime behavior have drifted apart.
- Give every example at least one cheap validation path.
- Reserve expensive multi-node training for scheduled checks instead of treating it as per-PR gating.

## Principles

- Run commands from the repo root so mounts and relative paths match the documented workflow.
- Keep the default pipeline cheap enough to run automatically after example-facing changes.
- Treat dataset prep, model download, checkpoint conversion, and training as separate checkpoints.
- Prefer the smallest runnable path that still exercises the real entrypoint.
- Use detached Modal runs for long-lived training and follow the detached-app workflow in [agent-modal-training.md](agent-modal-training.md).

## Discovering What To Validate

Do not hardcode assumptions about which examples exist.

- Start from the changed README, launcher module, or agent-facing doc.
- Identify the documented user path first. In practice this usually means looking for `modal run ...` commands in README files.
- Inspect the corresponding launcher module to understand the available stages and cheaper checkpoints.
- Treat helper stages such as download, dataset prep, conversion, benchmark, training, and evaluation as separate validation targets when the code exposes them separately.
- If a launcher supports discovery-oriented flags such as `--help`, `--list`, or `--dry-run`, use those before spending remote capacity.
- If the README and launcher disagree, treat that as drift even if one of them still works in isolation.

## Validation Tiers

### Tier 0: Local Sanity

Run this before any remote work:

- Confirm the changed launcher and related Python modules still parse.
- Re-read the relevant README and make sure the documented command names still exist.
- Fail immediately on import-time or syntax-time breakage before spending GPU time.

Suggested commands:

```bash
python -m compileall path/to/example_or_module
```

### Tier 1: Cheap Drift Checks

Run on every PR that changes an example directory, shared training utility, shared docs, or agent guidance.

- Prefer list, dry-run, dataset-prep, conversion, or single-node smoke commands.
- Do not require full multi-node convergence.
- A Tier 1 failure blocks merge for the touched workflow.

### Tier 2: Scheduled Smoke

Run nightly or before cutting a release branch.

- Exercise one real remote path per maintained flow.
- Prefer single-node or smallest-cluster commands when the example supports them.
- Confirm the app launches, user code starts, and the first meaningful artifact or log marker appears.

### Tier 3: Full Example Validation

Run weekly, before major releases, or after dependency/runtime updates.

- Execute the canonical multi-node command from the example README.
- Include long downloads, checkpoint conversion, and full training paths where the example documents them.
- These runs are for drift detection and confidence, not for per-PR gating.

## Choosing Commands For Each Tier

Use the launcher and README to derive the command for each tier instead of relying on a fixed inventory.

- Tier 1 should prefer the cheapest command that still proves the docs and code agree.
  This is often a help command, recipe listing, dry run, dataset prep step, conversion step, benchmark, or single-node smoke launch.
- Tier 2 should pick one real remote path that exercises user code and produces the first expected artifact or progress signal.
- Tier 3 should follow the canonical staged workflow from the README, including prerequisite stages that are required before training.

## Default Pipeline

For normal agent validation, use this order:

1. Run Tier 0 locally.
2. Identify the affected user-facing workflows by reading the changed README files and launcher modules.
3. Derive the cheapest meaningful Tier 1 command for each affected workflow.
4. If shared Modal or training infrastructure changed, broaden Tier 1 to every workflow that depends on that shared surface.
5. Escalate to Tier 2 only when Tier 1 cannot prove the path, or when recent failures suggest runtime drift instead of local breakage.

## Scheduled Pipeline

Use a separate scheduled workflow for repo-wide confidence:

1. Nightly: Tier 2 for each maintained user-facing workflow.
2. Weekly: Tier 3 for workflows with canonical multi-stage or multi-node training paths.
3. After major dependency, Modal runtime, or CUDA image changes: rerun Tier 3 for every workflow that shares the changed dependency surface.

## What Counts As Passing

A validation step passes only when all of the following are true:

- The documented command still resolves and starts the intended entrypoint.
- The Modal app reaches user code instead of failing during import, image setup, or argument parsing.
- The expected first artifact appears:
  - dataset files for prep steps,
  - downloaded model files for cache/bootstrap steps,
  - converted checkpoints for conversion steps,
  - rank/startup logs and at least one training or benchmark progress signal for training steps.
- The command exits successfully, or the detached app stays healthy long enough to prove the example started correctly.

## What To Record On Failure

When an example fails validation, record:

- the exact command,
- whether it failed in Tier 0, 1, 2, or 3,
- the Modal app ID if one exists,
- the first actionable error,
- whether the failure looks like README drift, dependency drift, secret/config drift, or infrastructure drift.

This makes it possible to decide whether the fix belongs in example code, docs, or the shared Modal workflow.

## Updating This Runbook

- Keep this document focused on the decision process an agent should apply, not a snapshot of the current example inventory.
- If a workflow grows a better smoke path, update the discovery guidance and tier definitions rather than adding a new hardcoded matrix.
