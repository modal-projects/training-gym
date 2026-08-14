`tasks/tau2_banking/corpus/` is a pinned snapshot of the **τ²-bench**
`banking_knowledge` domain (the τ³ knowledge/RAG domain). Primary source: the
tau2-bench repository (https://github.com/sierra-research/tau2-bench), MIT-licensed,
pinned at commit `aa74303ce5ff89a675297a3930b825bf1096f3fa` (pin by SHA).

## Acquisition (hard track)

```bash
git clone https://github.com/sierra-research/tau2-bench
git -C tau2-bench checkout aa74303ce5ff89a675297a3930b825bf1096f3fa
cp -R tau2-bench/data/tau2/domains/banking_knowledge/documents \
      tasks/tau2_banking/corpus/documents
cp    tau2-bench/data/tau2/domains/banking_knowledge/db.json \
      tasks/tau2_banking/corpus/
```

Corpus files (byte-identical to the source; normalizer = identity):

| path | what | count / size |
|---|---|---|
| `documents/doc_*.json` | product-knowledge base (checking, savings, cards, BNPL, payments, …) | 698 files, ~2.8 MB |
| `db.json` | customer / account records | 268 KB |

No `policy.md` in this domain — knowledge lives entirely in `documents/`. The domain
also ships `tools.py` (search/read over the knowledge base) and `tasks.json` (gold
scenarios). Tools/scenarios are harness + eval material, not corpus.

## Splits (Learning Agent — the domain ships no split_tasks.json)

97 scenarios in the shipped `tasks.json`. Learning Agent split, deterministic:

```python
ids = [t["id"] for t in tasks]             # shipped order
random.Random(0).shuffle(ids)
dev, test = sorted(ids[:58]), sorted(ids[58:])   # 58 dev / 39 test
```

`dev.json` / `test.json` carry these ids with each scenario's shipped
`description.purpose` verbatim; `test.json` is held out from agent workspaces.
