`tasks/tau2_telecom/corpus/` is a pinned snapshot of the **τ²-bench** `telecom` domain
(the dual-control domain). Primary source: the tau2-bench repository
(https://github.com/sierra-research/tau2-bench), MIT-licensed, pinned at commit
`1901a301961cbbe3fd11f3e84a2a376530c759e3` — the task's `env.pin`. (The snapshot
was originally taken at `aa74303c`; the telecom domain is byte-identical at both
commits.)

## Acquisition (hard track)

```bash
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout 1901a301961cbbe3fd11f3e84a2a376530c759e3
cp tau2-bench/data/tau2/domains/telecom/main_policy.md \
   tau2-bench/data/tau2/domains/telecom/main_policy_solo.md \
   tau2-bench/data/tau2/domains/telecom/tech_support_manual.md \
   tau2-bench/data/tau2/domains/telecom/tech_support_workflow.md \
   tau2-bench/data/tau2/domains/telecom/tech_support_workflow_solo.md \
   tau2-bench/data/tau2/domains/telecom/db.toml \
   tau2-bench/data/tau2/domains/telecom/user_db.toml \
   tasks/tau2_telecom/corpus/
```

Corpus files (byte-identical to the source; normalizer = identity):

| file | what | approx size |
|---|---|---|
| `main_policy.md` | primary agent policy (dual-control) | 5.7 KB |
| `main_policy_solo.md` | solo-mode policy variant | 5.2 KB |
| `tech_support_workflow.md` | troubleshooting workflow | 16 KB |
| `tech_support_workflow_solo.md` | solo-mode workflow variant | 15 KB |
| `tech_support_manual.md` | tech-support knowledge base | 18 KB |
| `db.toml` | agent-side accounts / lines / devices | 9.6 KB |
| `user_db.toml` | user-side state (dual-control) | 0.9 KB |

The domain also ships `tools.py` (agent + user action sets) and `tasks.json` (gold
scenarios). Tools become part of the run-time harness; scenarios are eval material
(`dev.json`/`test.json` carry the shipped split's ids; `test.json` is held out) —
neither is corpus. NOTE: DB files are **TOML**, not JSON.
