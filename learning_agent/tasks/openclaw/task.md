# Task: openclaw

## Corpus
`tasks/openclaw/corpus/` is the **OpenClaw** source tree — a TypeScript, self-hosted
personal-AI-assistant framework. It includes the application source (`src/`), the
plugin/extension packages (`extensions/`, `packages/`), and the docs (`docs/`). The
corpus is read-only; ground every answer in the real symbols you find here (do not
invent APIs).

## Topic areas (what the questions probe)
- **model_fallback_and_failover_logic** — the model-fallback / failover control flow
  (e.g. `src/agents/model-fallback.ts`): which provider is tried, when it throws vs
  returns, how status is surfaced.
- **memory_core_dreaming_and_promotion_pipeline** — the memory-core "dreaming" and
  promotion pipeline (e.g. `extensions/memory-core/src/dreaming.ts`).
- **new_plugin_provider_and_channel_integration_requests** — adding a new
  plugin/provider or channel: the real registration / integration surface.
- **cross_session_channel_context_and_session_behavior_requests** — channel context
  across sessions and session-behavior config.

## Answer format
Give correct, runnable **TypeScript** using real OpenClaw APIs in a **single
` ```typescript ` code block**, plus a **brief explanation** of the key decisions
(which API, why, and any pitfall avoided). The system prompt is `sys.txt`.

## Data
- `dev.json` — 8 questions, each with `question`, `gold_answer`, `rubric`, `evidence`.
  Steer ONLY by these.
- `test.json` — the hidden 12-question test (gold + rubric). **OFF LIMITS**;
  harness-only.
