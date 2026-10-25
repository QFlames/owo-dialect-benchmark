"""Task 2 starting point: how much of Owo can be predicted from Standard Yorùbá?

Two baselines to beat with hand-written sound rules (see RULES in src/rules.py):
  A. copy      - predict that Owo = Standard
  B. lookup    - replace each Standard word by the Owo word seen for it in OTHER
                 sentences (leave-one-sentence-out, so no answer leaks)

    python src/task2_baselines.py
"""
import re, collections
import pandas as pd
from sacrebleu.metrics import CHRF

chrf = CHRF(word_order=2)
tokens = lambda s: re.findall(r"[^\s.,!?;:]+", s)
low = lambda w: w.lower()

def lookup_table(ann, exclude_id):
    t = collections.defaultdict(collections.Counter)
    for r in ann[ann.sentence_id != exclude_id].itertuples():
        if isinstance(r.std_word, str):
            t[low(r.std_word)][r.owo_word] += 1
    return {k: v.most_common(1)[0][0] for k, v in t.items()}

def predict_lookup(std, table):
    return " ".join(table.get(low(w), w) for w in tokens(std))

def main():
    sent = pd.read_csv("data/processed/sentences.csv")
    ann = pd.read_csv("data/processed/annotations.csv")
    res = []
    for r in sent.itertuples():
        copy = " ".join(tokens(r.standard))
        look = predict_lookup(r.standard, lookup_table(ann, r.sentence_id))
        gold = " ".join(tokens(r.owo))
        res.append(dict(sentence_id=r.sentence_id, speaker=r.speaker,
                        copy_chrf=chrf.sentence_score(copy, [gold]).score,
                        lookup_chrf=chrf.sentence_score(look, [gold]).score,
                        copy_exact=copy.lower() == gold.lower(), lookup_exact=look.lower() == gold.lower()))
    df = pd.DataFrame(res)
    print(df[["copy_chrf", "lookup_chrf", "copy_exact", "lookup_exact"]].mean().round(3))
    df.to_csv("results/task2_baselines.csv", index=False)

if __name__ == "__main__":
    main()
