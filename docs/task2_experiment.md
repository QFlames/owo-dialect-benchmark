# Task 2 — Rule-Based Standard Yorùbá to Owo Conversion

## Research objective

This experiment investigates how effectively a small set of manually specified lexical and phonological rules can convert Standard Yorùbá text into Owo, compared with copying the source unchanged and using an annotation-derived word dictionary.

## Dataset and evaluation design

The experiment used 140 sentence IDs with 200 speaker-specific records, partitioned by sentence ID to prevent different speaker versions of the same sentence from appearing in both development and test sets.

| Partition | Sentence IDs | Speaker records |
|---|---:|---:|
| Development | 105 | 151 |
| Final test | 35 | 49 |

Four approaches were evaluated:

- **R0 — Copy:** Leave Standard Yorùbá unchanged.
- **R1 — Lookup:** Replace words using an annotation-derived dictionary.
- **R2 — Rules:** Apply seven ordered lexical and phonological transformations.
- **R3 — Hybrid:** Apply dictionary replacements to known words and rules to unseen words.

Lookup dictionaries were constructed from development annotations only, excluding the evaluated sentence ID when applicable. The rules were frozen before the final evaluation.

## Rule inventory

The seven rules targeted the following transformations: word-final `in → ẹn`, `mo → mòó`, initial `ì → ù`, `náà → nẹ́ẹ̀`, initial `ṣ → tṣ`, `ẹ → ẹ̀ghẹn`, and `fún → ghún`.

These are experimental transformations inferred from the available annotations, not claims of universal Owo linguistic rules.

## Evaluation metrics

The primary metric was the mean of sentence-level chrF++ scores against each speaker's Owo reference. Ninety-five percent confidence intervals were estimated with 1,000 bootstrap samples clustered by sentence ID.

Secondary analyses included exact-match accuracy on annotated single-word differences, estimated changes to unchanged words, and inter-speaker agreement on shared sentences.

## Results

| Approach | Development chrF++ | Final test chrF++ | Test 95% CI |
|---|---:|---:|---:|
| R0 Copy | 22.1 | 20.3 | 15.7–26.2 |
| R1 Lookup | 25.7 | 21.7 | 16.7–27.9 |
| R2 Rules | **26.4** | **23.7** | 17.9–30.9 |
| R3 Hybrid | 26.1 | 22.0 | 16.8–28.8 |

The rule-based approach achieved the highest observed sentence-level score on both partitions. On the final test set, it exceeded the copy baseline by 3.4 chrF++ points.

However, these results do not establish statistical significance for the differences between approaches.

At the word level, the hybrid correctly reproduced 32% of annotated differences, compared with 12% for the rules-only system. Conversely, the rules-only approach changed approximately 4% of inferred unchanged-word occurrences, compared with 40% for the hybrid.

This suggests a trade-off between recovering annotated substitutions and preserving words that do not require modification.

## Inter-speaker variation

For the 21 final-test records belonging to sentences shared by three speakers, mean human-to-human chrF++ was 35.3, compared with 58.0 for the corresponding development subset.

This measurement provides an inter-speaker agreement reference, not a theoretical upper bound on system performance.

## Limitations

The dataset is small, and speaker-specific variation may influence the observed scores. The rule inventory is limited and does not model sentence-level grammatical or contextual transformations.

The unchanged-word analysis relies on token overlap rather than manually verified word alignments.

Although lookup dictionaries were corrected to prevent test-annotation leakage, some full-dataset annotations had already been inspected during exploratory rule development. The final test is therefore not a completely blind evaluation.

The reported confidence intervals describe uncertainty in individual mean scores; paired significance tests for differences between approaches were not performed.

## Conclusion

A compact seven-rule converter achieved the highest observed sentence-level chrF++ among the four evaluated approaches, scoring 23.7 on the final test set compared with 20.3 for the identity baseline.

The results provide preliminary evidence that simple, annotation-informed transformations can improve Standard Yorùbá-to-Owo conversion in a low-resource setting. Larger datasets, independent validation, and contextual modeling are needed to establish broader generalizability.

## Reproducibility

Implementation: `src/task2_ladder.py`, `src/rules.py`, and `src/task2_baselines.py`.

Frozen implementation commit: `764b284`.

Development command: `python src/task2_ladder.py`

Final evaluation command: `python src/task2_ladder.py --final`
