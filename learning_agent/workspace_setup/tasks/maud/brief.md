> DRAFT — operator review pending (authored from the shipped `task.md` +
> `manifest.jsonl` facts to complete the pinned task-file set; maud pack owner
> should review before any hard-track run).

`tasks/maud/corpus/` is the **Atticus MAUD v1** contract set — all 152
public-target merger agreements in the release (US public targets >$200M,
acquisitions closed 2021; CC-BY-4.0, The Atticus Project). That release zip is
the primary source for this task.

## Manifest

- Source zip: `https://github.com/TheAtticusProject/maud/raw/main/data.zip`
- Zip sha256: `75af5a33d038e9254864f043da38072490ffe11e8488d58d0a2dd39c8f554519`
- Contents used: `data/contracts/contract_<n>.txt` — one plain-text merger
  agreement per file, 152 files, 54.1 MB total (tier M).
- Normalization: **normalizer v1 = identity** — files are materialized
  byte-identical from the zip members; only the on-disk names change.
- `corpus_version: maud-M-095c61b5ecdb` — sha256 over the shipped
  `manifest.jsonl` in this directory, which lists every document with its
  `doc_id`, human-readable title, target filename (`path`), per-file `sha256`,
  byte size, and the exact zip `member` it comes from.

## Acquisition

1. Download the release zip and verify its sha256 against the value above.
2. For each row of `tasks/maud/manifest.jsonl`, extract the zip `member` named
   in `fetch.member` and write it byte-identical to
   `tasks/maud/corpus/<path>`.
3. Verify every written file's sha256 against the row's `sha256` — all 152
   must match; the corpus tree then reproduces the pinned
   `corpus_version` exactly.
