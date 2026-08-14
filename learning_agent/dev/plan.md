# Plan — codex harness · recipe cookbook toolbox · dspy removal · big-teaches-small

> For review (rev 2, 2026-07-31). Scope pivot per Leon: **no self-training** —
> a big frozen learning agent trains a small student; the big model may evolve
> the *learning harness*, the small model gets trained in *weights and its
> answering harness*. Orchestration harness = **Codex CLI (open-source)**.
> ☐ = decision needed.

---

## 0. Scouting facts this plan stands on

- `ep-glm-5-2-fp8` is a **managed Modal Flash endpoint** (SGLang 0.5.13.post1,
  not ours to upgrade) speaking OpenAI `/v1` natively; its Anthropic emulation
  rejects Claude Code (ecosystem-wide CC ≥ 2.1.154 system-role issue, open on
  SGLang and vLLM). **Codex CLI speaks OpenAI wire natively → the whole
  Anthropic-bridge problem disappears** with the harness choice. Codex CLI
  0.145.0 installed; open source (openai/codex), so the orchestration harness
  itself is inspectable and, later, forkable — consistent with "evolve the
  learning harness".
- GLM-5.2 run autopsy: healthy harness mechanics (2% tool-error over 23.6 h,
  reprompt loop carried it); failure cases = (1) self-judge depended on the
  local `claude` CLI (absent in containers), (2) dev→test collapse
  0.372 → 0.069 (judge mismatch vs overfit — diagnostic D1 below), (3) run-1
  tool-format leak on `<think>` file content (SGLang tool-parser quirk).
- Sibling repo `continual-learning-synthetic-data` (ascl): LongHealth corpus
  (6 patients ≈ 70k tokens, 120 MCQs), 10 generators, finalize/hygiene
  pipeline, **data-quality gates that refuse to train**, knowledge-map
  coverage generation, closed-book eval + cloze probes. ⚠ ~10k lines
  uncommitted in its working tree — commit that repo first.
- Models: `Qwen/Qwen3.6-27B` (dense) and `Qwen/Qwen3.6-35B-A3B` (MoE) on HF —
  candidates for a second big model we serve ourselves (SGLang latest, our
  version). GLM-5.2 endpoint is the big model already live.
- [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
  recipe taxonomy: sl_basic/rl_basic, chat_sl, math_rl, code_rl, preference
  (RLHF), search_tool, prompt_distillation, distillation, **rubric** (LLM-judge
  rewards), sdft, multiplayer_rl… Our toolbox already half-implements several
  (SFT=chat_sl, GRPO=rl_basic, self_distill=sdft, rubric_eval=rubric) but as
  scattered scripts, not a uniform recipe surface.

## The thesis (restated)

> The learning agent is an agent that can (a) call training recipes
> appropriate to the scenario, and (b) invent new tools/recipes.

So the toolbox's job is to *be a cookbook*: one uniform recipe contract the
big model can select from, compose, and mutate — and the benchmark measures
how well it wields + extends that cookbook to make the small model expert.

---

## P0-A — Codex harness scaffolds (replaces the CC-bridge plan)

1. `agents/codex_glm52/` — Codex CLI against the GLM endpoint via an isolated
   `CODEX_HOME` (house pattern from `codex_non_api*`) with
   `model_providers.modal_glm = { base_url = "$MODAL_GLM52_BASE_URL/v1",
   wire_api = "chat" }`; `codex exec --json`; reprompt loop
   (`codex exec resume --last`, house pattern from `codex_*_reprompt`).
2. Same scaffold parameterized for any OpenAI-compatible endpoint
   (`agents/codex_endpoint/` reading a `config.env`) so a future
   `codex_qwen36` is a config file, not new code.
3. Contract tests + preflight (endpoint live, codex on PATH).
4. Smoke battery: text, bash tool call, **file-write containing `<think>`**
   (run-1's killer — likely fixed by codex's different tool wire), trace
   parses (`codex/human_readable_trace.py` reused), `--prepare-only` sandbox.
5. Parked: Claude-Code-via-litellm bridge (documented in this plan's rev 1;
   revisit only if codex harness underperforms OpenCode baseline).

## P0-B — Remove dspy (+ RL generalization)

- Deregister from `bench/config.yaml`; `git rm -r tasks/dspy` (+ 211 MB local
  corpus); drop `data/api_surface.json` pin + `harness/api_surface.py` dspy
  branch; retarget tests/docs/examples that used dspy as fixture → `openclaw`;
  runner case-lists lose the dspy arm.
- **RL stays live, generalized** (per Leon: RL is a pipeline; the agent brings
  its own data + reward): `pipeline/rl.py` gains explicit `--prompts <file>` +
  `--reward <module>` parameterization + per-task corpus bake; dspy reward/
  prompts leave with dspy; a *generic judge-backed reward* ships in the
  toolbox (built on P1's endpoint LLM client). `learning_agent.md` drops
  "(currently dspy-only)" → same re-freeze batch.
- Observatory demo fixture keeps its dspy-named sample data (synthetic; churn
  for zero value). ☐ agree?

## P1-A — Toolbox v2: skill-style tools (NOT a recipe framework)

Tools stay tools. The tinker influence is ergonomic, not architectural — we
are not doing Tinker, and no new abstraction layer gets built. Per Leon:
*each tool needs a README like a skill, plus simple scripts that help the
model use the tool.*

1. **A SKILL.md-style README per tool** (data_toolbox generators, train
   levers, eval_toolbox, evolve, harness_toolbox): when to reach for it,
   inputs/outputs, 2-3 copy-paste invocations, cost/GPU expectations, known
   failure modes (e.g. "rows at exactly the decode cap are truncated garbage
   — drop them"). Same doc shape as an agent skill so the model can skim the
   toolbox like a skill list. A top-level `toolbox/README.md` index maps
   scenario → tool.
2. **Thin helper scripts** where invocation is currently awkward (multi-step
   train launches, judge setup, corpus staging) — one script per rough edge,
   no framework.
3. **More extensive training tools, tinker-inspired but ours**: add the
   missing levers as plain tools with the same doc pattern — teacher→student
   distillation, rubric-reward RL (judge-backed reward for the generalized
   `pipeline/rl.py`), prompt-distillation. Wrapping axolotl/slime as today.
   Our data / eval / evolve / harness categories stay as-is (tinker has no
   equivalent — that's our edge, keep it).
4. **Endpoint LLM client** (`toolbox/llm_client.py`, ascl-style protocol):
   any OpenAI-compatible base URL. `rubric_eval` gains
   `--judge-backend endpoint`; generators likewise; the generic RL reward
   uses it. Kills the claude-CLI dependence (containers get a judge).
5. **ascl gem ports as plain tools**: finalize hygiene (cap-slam / 13-gram
   decontam / exact dedup / prefix-iid shuffle), prompts-as-data +
   fingerprints, source-support gate, two-artifact contamination firewall.
6. **Per-task setup**: a short per-task page (`toolbox/tasks/<task>.md`) —
   which tools/presets fit this task, answer format, globs, judge defaults —
   plus preset files where useful. Promote the GLM agent's invented
   statement-aware retrieval into the fav2 page/preset.

## P1-B — New task `longhealth` (closed-book knowledge injection)

Unchanged from rev 1, now framed as the flagship *big-teaches-small* task:
corpus = source-only patient notes (two-artifact firewall), dev 40 / test 80
MCQs (☐ ok?), closed-book at answer time (corpus absent at eval — retrieval
physically impossible), `unparseable_frac` secondary. Student: ☐ keep global
pin Qwen3.5-9B, or per-task override to a smaller student (ascl used
Qwen3-4B — faster laps, comparable ascl baselines)? Per-task `base_model`
override is small and needed eventually either way.

## P1-C — (Optional tonight) serve Qwen3.6-27B as a second big model

`serve/agent_llm.py`, app `lab-agent-llm`, SGLang latest (our version),
OpenAI API + tool parser; `Qwen/Qwen3.6-27B` on 2×H200 (or 35B-A3B on 1).
Gives a second learning-agent model for GLM-vs-Qwen comparisons on the same
codex harness. ☐ spend tonight, or defer and run everything on the GLM
endpoint?

## ⚠ Finding from R2 (2026-08-01, codex 6h run): shared-volume tag collisions

The codex run trained `--tag fav2_v1/v2/v3` and **silently overwrote the
07-28 OpenCode run's checkpoints at those paths** on `lab-out` —
`/out/models/<tag>/merged` is a shared flat namespace with no per-run
scoping. (The old run's scored `fav2_v11` submission is untouched; its
recorded test number stands. Its v1–v3 ablation artifacts are gone.)

Fix to make: per-run namespacing of trained weights —
`/out/models/<run_id>/<tag>/merged`, with `agents/run.sh` exporting
`LEARNING_AGENT_RUN_ID` and `pipeline/train.py` (+ `learning_agent.md`'s fixed-locations
table — pinned, needs deliberate re-freeze) prefixing the output path.
Backfill note: old artifacts keep their flat paths; only new runs namespace.

## P2 — Tonight's runs

| id | what | budget | needs |
|---|---|---|---|
| D1 | re-judge `glm52_fav2_v11_sub` with cli-claude (collapse diagnostic) | minutes | nothing |
| R1 | `codex_glm52` smoke incl. `<think>` file-write | 0.1 h | P0-A |
| R2 | GLM-5.2 **via codex harness** → 9B student, fav2 easy — harness A/B vs the finished OpenCode run (same model, same task, same budget class) | 4–6 h | P0-A |
| R3 | GLM-5.2 via codex → small student, `longhealth` closed-book, cookbook + harness-evolution recipes enabled | 4–6 h | P0-A, P1-A(1,2), P1-B |

Both light runs detached + watched; dev/test columns already on the dashboard.

## Sequencing tonight

1. D1 (runs unattended)
2. P0-A codex scaffolds + smoke → commit
3. P0-B dspy removal + RL generalization + P1-B longhealth task + **one
   combined re-freeze** → commit
4. P1-A(4) llm_client + endpoint judge → P1-A(1) skill-style READMEs for the
   tools tonight's runs touch (datagen, train, rubric_eval, evolve) → commit
5. Kick off R2, then R3 detached
6. ascl repo: commit its working tree; gem ports (P1-A(3)) as the evening
   allows — remainder tomorrow with profiles (P1-A(4))

## Decisions (☐)

1. Demo fixture keeps dspy-named sample data?
2. longhealth dev/test = 40/80?
3. longhealth student: 9B pin or per-task 4B override?
4. Serve Qwen3.6-27B tonight (2×H200) or GLM-only tonight?
5. ~~Tinker SDK backend~~ — dropped per Leon: we are not doing Tinker; it's
   style inspiration only.
