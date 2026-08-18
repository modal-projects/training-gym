#!/usr/bin/env python3
"""Build the longhealth task assets from the LongHealth benchmark.

Source: github.com/kbressem/LongHealth, data/benchmark_v5.json — 20 synthetic
patient records (clinical notes) with 400 five-option multiple-choice
questions, answer text in `correct`. This script writes the task's assets
under workspace_setup/tasks/longhealth/:

    corpus/patient_XX/info.txt        name, birthday, diagnosis
    corpus/patient_XX/<text_id>.txt   one clinical note per file
    dev.json                          25 questions (seeded sample)
    test.json                         75 questions (disjoint)

Splits follow the repo's 25/75 convention (seed 0, spread over patients).
Rows use the standard qa schema (id, topic, question, gold_answer, rubric,
evidence). Rerun after changing SEED/DEV_N/TEST_N; then re-freeze pins and
`workspace_setup/hf_tasks.py upload --task longhealth`.

    python3 dev/make_longhealth.py [--src /path/to/benchmark_v5.json]
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "workspace_setup" / "tasks" / "longhealth"
URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"
SEED, DEV_N, TEST_N = 0, 25, 75

LETTERS = ("a", "b", "c", "d", "e")


def load(src: str | None) -> dict:
    if src:
        return json.loads(Path(src).read_text())
    with urllib.request.urlopen(URL) as r:
        return json.loads(r.read())


def write_corpus(data: dict) -> int:
    n = 0
    for pid, p in sorted(data.items()):
        pdir = DEST / "corpus" / pid
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "info.txt").write_text(
            f"Patient {pid}: {p['name']}\nBorn: {p['birthday']}\n"
            f"Diagnosis: {p['diagnosis']}\nNotes: {', '.join(sorted(p['texts']))}\n")
        for tid, text in sorted(p["texts"].items()):
            (pdir / f"{tid}.txt").write_text(text)
            n += 1
    return n


def rows(data: dict) -> list[dict]:
    out = []
    for pid, p in sorted(data.items()):
        for q in p["questions"]:
            opts = "\n".join(f"{l.upper()}) {q[f'answer_{l}']}" for l in LETTERS)
            correct = q["correct"].strip()
            letter = next((l.upper() for l in LETTERS
                           if q[f"answer_{l}"].strip() == correct), "?")
            evidence = sorted((q.get("answer_location") or {}).keys())
            out.append({
                "id": f"longhealth_{pid}_{q['No']}",
                "topic": pid,
                "question": (f"Patient {pid} ({p['name']}): {q['question']}\n\n"
                             f"Options:\n{opts}\n\n"
                             "Answer with the correct option and justify it "
                             "from the patient's record."),
                "gold_answer": f"{letter}) {correct}",
                "rubric": [{"claim": f"The answer selects the option stating: {correct}",
                            "weight": 1.0}],
                "evidence": evidence,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="", help="local benchmark_v5.json (default: download)")
    args = ap.parse_args()
    data = load(args.src or None)
    n_notes = write_corpus(data)
    all_rows = rows(data)
    rng = random.Random(SEED)
    # spread the sample over patients: shuffle within an interleave by patient
    by_pid: dict[str, list] = {}
    for r in all_rows:
        by_pid.setdefault(r["topic"], []).append(r)
    for v in by_pid.values():
        rng.shuffle(v)
    interleaved = [v[i] for i in range(max(map(len, by_pid.values())))
                   for v in by_pid.values() if i < len(v)]
    picked = interleaved[:DEV_N + TEST_N]
    dev, test = picked[:DEV_N], picked[DEV_N:DEV_N + TEST_N]
    (DEST / "dev.json").write_text(json.dumps(dev, indent=1))
    (DEST / "test.json").write_text(json.dumps(test, indent=1))
    print(f"{len(data)} patients, {n_notes} notes -> corpus/")
    print(f"dev {len(dev)} / test {len(test)} of {len(all_rows)} questions "
          f"(seed {SEED}, disjoint, spread over patients)")


if __name__ == "__main__":
    main()
