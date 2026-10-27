"""Hand-written Standard -> Owo rules (Task 2).

Each rule is a tuple:  (name, regex, replacement)            or
                       (name, regex, replacement, words_to_skip)

Rules run IN ORDER on every Standard word (lower-cased; a leading capital is put
back afterwards). Add one rule at a time, then run

    python src/task2_ladder.py

and read results/task2_rule_report.csv: for every rule it says how many words it
fired on, how many ended CLOSER to the real Owo word (improved) and how many ended
FURTHER away (worsened). Keep a rule only if improved clearly beats worsened.

Tips
- Text is NFC: "ọ" is one character, but "ọ́" is "ọ" + a combining tone mark.
  A rule on "ọ" therefore keeps the tone mark and changes only the letter.
- r"in$" means "ends in in"; r"^i" means "starts with i"; (?<=...) / (?=...) look at
  neighbouring letters without changing them.
- Use words_to_skip for known exceptions, e.g. ("w -> gh", r"w", "gh", {"wá", "wà"}).

The two rules below are only STARTERS taken from the workbook's Patterns tab.
Test them; do not assume they are right. 'ọ -> a' is applied only to the LAST
letter on purpose (ọmọ -> ọma); check whether that is what the data really shows.
"""
RULES = [
    ("-in -> -ẹn", r"in$", "ẹn"),
    ("mo → mòó", r"^mo$", "mòó"),
    ("initial ì → ù", r"^ì", "ù"),
    ("náà → nẹ́ẹ̀", r"^náà$", "nẹ́ẹ̀"),
    ("initial ṣ → tṣ", r"^ṣ", "tṣ"),
    ("ẹ → ẹ̀ghẹn", r"^ẹ$", "ẹ̀ghẹn"),
    ("fún → ghún", r"^fún$", "ghún"),
]
