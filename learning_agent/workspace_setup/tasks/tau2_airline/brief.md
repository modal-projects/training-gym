> DRAFT — operator review pending

`tasks/tau2_airline/corpus/` is a pinned snapshot of the **τ²-bench** `airline`
domain. Primary source: the tau2-bench repository
(https://github.com/sierra-research/tau2-bench), MIT-licensed, pinned at commit
`aa74303ce5ff89a675297a3930b825bf1096f3fa` — the `release/v1.0.1` line (no `v1.0.1`
tag exists upstream; pin by SHA). NOTE: banking/other domains are still being
corrected on unmerged upstream `fix/*` branches — pin deliberately.

## Acquisition (hard track)

```bash
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout aa74303ce5ff89a675297a3930b825bf1096f3fa
cp tau2-bench/data/tau2/domains/airline/policy.md \
   tau2-bench/data/tau2/domains/airline/db.json \
   tasks/tau2_airline/corpus/
```

Corpus files (byte-identical to the source; normalizer = identity):

| file | what | approx size |
|---|---|---|
| `policy.md` | airline customer-service policy | 7.6 KB |
| `db.json` | flights / reservations / users database | 7.0 MB |

The domain also ships `tools.py` (the agent's action set) and `tasks.json` (50 gold
scenarios). The tools become part of the run-time harness; the scenarios are the eval
material (held out, operator-only) — neither is corpus.
