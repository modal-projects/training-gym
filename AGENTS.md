# AGENTS.md

Repo-wide notes for coding agents working in this repository.

## Working Rules

- Never commit raw secrets, API keys, or token values to the repository or its docs.
- Keep repository docs generic to contributors using this repo, not tailored to one local machine or one user's workflow.
- Use `uv` for Python dependency management and command execution in this repo. Prefer `uv run`, `uv lock`, and `uv sync`, and do not install Python packages at the system level.
- For an overview of the training-gym setup — core abstractions, framework
  catalog, how to add a model, and how to author a tutorial — start at
  [skills/agent-training-gym-overview.md](skills/agent-training-gym-overview.md).
- For launching and debugging Modal training jobs, see [skills/agent-modal-training.md](skills/agent-modal-training.md).
- For repo-wide example drift validation, see [skills/agent-example-validation.md](skills/agent-example-validation.md).
- Keep agent-facing docs focused on durable repo workflow. Move one-off incident notes into commit messages, PRs, or issue threads instead of preserving them as permanent guidance.
