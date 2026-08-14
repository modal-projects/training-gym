# Task: fav2

## Corpus
`tasks/fav2/corpus/` is a pinned snapshot of **SEC EDGAR filings**: 2,415 documents
for 41 issuers, filed 2023-01-01 through 2026-07-08, as normalized plain text. Layout
is `<TICKER>/<filing_date>_<FORM>_<accession>.txt` (e.g.
`CRWD/2025-03-10_10-K_0001535527-25-000009.txt`) — the filename carries the ticker,
filing date, form type, and accession number. Forms included: 10-K, 10-Q, 20-F
(periodic reports), 8-K (material events), 425 / DEFM14A / DEFA14A (merger and
proxy-supplement transactions). The
corpus is read-only; ground every answer in the filings you find here (do not invent
figures or filings).

## Topic areas (what the questions probe)
- **general-qualitative-analysis** / **general-quantitative-analysis** — strategy,
  risk factors, segment disclosures; specific reported figures and trends.
- **comparables** — head-to-head comparisons across issuers (e.g. loyalty programs,
  margin structure), concluding which company is better positioned.
- **precedents** — M&A transactions in the record (terms, premiums, structures).
- **adjustments** — non-GAAP bridges and normalizations (e.g. Adjusted-EBITDA walks).
- **disclosure-analysis** — what a filer does and does not disclose, and changes over time.
- **financial-modeling** — building estimates strictly from disclosed inputs.
- **market-analysis** — market positioning as disclosed in filings.

Dev covers six of these one question each; adjustments and market-analysis appear
only in the hidden test, so generalize from the corpus, not just the dev topics.

## Answer format
Concise **markdown prose** (no code blocks). State the specific numbers with units
and fiscal periods; numeric claims are graded to roughly **1% relative tolerance**.
Name the filing each key fact comes from (ticker, form, period). When the question
asks for a judgment or ranking, **commit to a conclusion** — graders give no credit
for hedged non-answers. The system prompt is `sys.txt`.

## Data
- `dev.json` — 6 questions, each with `question`, `gold_answer`, `rubric`, `evidence`.
  Steer ONLY by these.
- `test.json` — the hidden 13-question test (gold + rubric). **OFF LIMITS**;
  harness-only.
