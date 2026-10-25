# Owo dialect benchmark

A small parallel corpus of 200 everyday sentences in English, Standard Yorùbá and Owo
(a Yorùbá dialect with no standard writing system), with every Owo word that differs from
Standard Yorùbá annotated by type, Standard word and pattern. This repository holds the
machine-learning part of the Tonative capstone: using the corpus as a **test set** to ask

1. How much worse does an existing translation model do on Owo than on Standard Yorùbá,
   and which kinds of differences hurt most? (`src/task1_dialect_gap.py`)
2. How much of Owo can simple rules predict from Standard Yorùbá? (`src/task2_baselines.py`, `src/rules.py`)

**Status:** labels come from one annotator; the second-annotator agreement check is paused.
Treat all results as preliminary.

## Data (private until the licence is settled)

| File | What it is |
|---|---|
| `data/raw/owo_annotations_2026-10-25.json` | Full Label Studio export (source of truth) |
| `data/processed/sentences.csv` | One row per sentence: English, Standard, Owo, speaker, domain, `n_diffs`, `types` |
| `data/processed/annotations.csv` | One row per annotated Owo word: position, `owo_word`, `std_word`, `type`, `pattern` |

Types: **Tonal**, **Phonological**, **Lexical**, **Loanword**. Speakers: S00 (education set), S01 to S03.
Rebuild the tables with `python src/prepare_data.py`.
Read `DATA_NOTICE.md` before sharing data or sending it to any online service.

## Quick start

```bash
git clone <repo-url> && cd owo-dialect-benchmark
conda create -n owo-ml python=3.11 -y && conda activate owo-ml
pip install -r requirements.txt
python src/prepare_data.py
python src/task2_baselines.py                       # fast, no model needed
python src/task1_dialect_gap.py --limit 20          # first run downloads the model (about 2.5 GB)
python src/task1_dialect_gap.py                     # full run
```

## Rules for working with the data

- **Never split rows at random.** 30 sentence IDs were recorded by more than one speaker. If you train
  anything, hold out a whole speaker and keep every version of the same `sentence_id` on one side.
- Report the number of sentences behind every number; with 200 sentences, use bootstrap intervals.
- Run models locally. Do not send unreleased sentences to hosted APIs (see `DATA_NOTICE.md`).

## Working together

- `main` is always runnable. Work on a branch (`git checkout -b task1-nllb`), open a pull request, and have
  one other person look at it before merging.
- Put results in `results/` as CSV and add a line to the table below.

## Results

| Date | Task | Model / method | Standard | Owo | Notes |
|---|---|---|---|---|---|
| | Task 2 | copy baseline (chrF++ vs gold Owo) | | 21.6 | all 200 sentences |
| | Task 2 | word lookup baseline | | 25.7 | leave-one-sentence-out |
| | Task 1 | NLLB-200 600M | | | to run |
