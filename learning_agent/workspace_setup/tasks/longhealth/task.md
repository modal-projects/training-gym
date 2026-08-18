# longhealth — clinical-record QA

Answer five-option multiple-choice questions about synthetic patient
records (the LongHealth benchmark, 20 patients, 133 clinical notes).

## Corpus layout

    corpus/patient_XX/info.txt      name, birthday, diagnosis, note list
    corpus/patient_XX/text_N.txt    one clinical note per file

Every question names its patient id, so the record to search is always
known. The notes are long (thousands of words each); the answer is a
specific fact inside one or a few notes: a value, a date, a medication,
an event.

## Answer format

State the correct option (letter and text) and justify it from the
record. Answers are judged against a rubric keyed to the correct
option's content; naming the right option without support still scores,
support without commitment does not.

## Provenance

Questions and records come from the LongHealth benchmark
(github.com/kbressem/LongHealth, benchmark_v5). The records are
synthetic; no real patient data. The benchmark carries a canary string
in its source; the corpus files here are the note texts only.
