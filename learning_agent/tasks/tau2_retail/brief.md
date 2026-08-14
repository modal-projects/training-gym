`tasks/tau2_retail/corpus/` is a pinned snapshot of the **τ²-bench** `retail` domain.
Primary source: the tau2-bench repository
(https://github.com/sierra-research/tau2-bench), MIT-licensed, pinned at commit
`1901a301961cbbe3fd11f3e84a2a376530c759e3` — the task's `env.pin`. (The snapshot
was originally taken at `aa74303c`; the retail domain is byte-identical at both
commits.)

## Acquisition (hard track)

```bash
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout 1901a301961cbbe3fd11f3e84a2a376530c759e3
cp tau2-bench/data/tau2/domains/retail/policy.md \
   tau2-bench/data/tau2/domains/retail/db.json \
   tasks/tau2_retail/corpus/
```

Corpus files (byte-identical to the source; normalizer = identity):

| file | what | approx size |
|---|---|---|
| `policy.md` | retail customer-service policy | 6.7 KB |
| `db.json` | users / orders / products database | 2.7 MB |

The domain also ships `tools.py` (the agent's action set) and `tasks.json` (gold
scenarios). Tools become part of the run-time harness; scenarios are eval material
(`dev.json`/`test.json` carry the shipped split's ids; `test.json` is held out) —
neither is corpus.
