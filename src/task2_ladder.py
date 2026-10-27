"""Task 2: how much of Owo can simple rules predict from Standard Yorùbá?

    python src/task2_ladder.py            # while writing rules: DEVELOPMENT sentences only
    python src/task2_ladder.py --final    # ONCE, when rules are frozen: the untouched TEST sentences

Every 4th sentence ID (all speakers' versions together) is locked away as the test set.
Looking at test sentences while writing rules would be cheating, because the rules would
then be tuned to the very sentences used to score them.

Four approaches:
R0: Identity baseline (copy Standard Yorùbá)
R1: Dictionary lookup
R2: Ordered phonological and lexical rules
R3: Dictionary lookup with rule-based fallback  
Also reports: a inter-speaker agreement reference (how much do the speakers agree with EACH OTHER on
the 30 shared sentences?), word-level recall by difference type, how often each
rung wrongly changes a word that Owo leaves alone, and a per-rule report.
"""
import re
import sys
import unicodedata as ud
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

from rules import RULES
from task2_baselines import lookup_table, tokens, low, predict_lookup

chrf = CHRF(word_order=2)
rng = np.random.default_rng(0)
PUNCT = ".,!?;:"
Path("results").mkdir(exist_ok=True)


# ---------- rules ----------------------------------------------------------------------
def apply_rules(word, rules=RULES):
    cap = word[:1].isupper()
    w = ud.normalize("NFC", word.lower())
    for rule in rules:
        pat, rep = rule[1], rule[2]
        if len(rule) > 3 and w in set(rule[3]):
            continue
        w = re.sub(pat, rep, w)
    return (w[:1].upper() + w[1:]) if cap else w


def lev(a, b):
    """Edit distance between two words (number of letter changes needed)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------- data -----------------------------------------------------------------------
sent = pd.read_csv("data/processed/sentences.csv")
ann = pd.read_csv("data/processed/annotations.csv")
nfc = lambda s: ud.normalize("NFC", s)
tables = {sid: lookup_table(ann, sid) for sid in sent.sentence_id.unique()}   # cache

FINAL = "--final" in sys.argv

all_ids = sorted(sent.sentence_id.unique())
test_ids = set(all_ids[::4])
dev_ids = set(all_ids) - test_ids

use_ids = test_ids if FINAL else dev_ids
TAG = "_final" if FINAL else ""

# Build dictionaries using development annotations ONLY.
dev_ann = ann[ann.sentence_id.isin(dev_ids)].copy()

# Exclude the current sentence ID to prevent same-sentence leakage.
tables = {
    sid: lookup_table(dev_ann, sid)
    for sid in use_ids
}

print(
    "=== FINAL TEST sentences ==="
    if FINAL else "=== DEVELOPMENT sentences (test set locked) ===",
    f"({len(use_ids)} sentence IDs)\n"
)

sent = sent[
    sent.sentence_id.isin(use_ids)
].reset_index(drop=True)

RUNGS = {
    "0 copy": lambda w, t: w,
    "1 lookup": lambda w, t: t.get(low(w), w),
    "2 rules": lambda w, t: apply_rules(w),
    "3 lookup+rules": lambda w, t: t[low(w)] if low(w) in t else apply_rules(w),
}

# ---------- sentence level: chrF++ against the gold Owo sentence -----------------------
rows = []
for r in sent.itertuples():
    t = tables[r.sentence_id]
    gold = " ".join(tokens(r.owo))
    row = dict(sentence_id=r.sentence_id, speaker=r.speaker, domain=r.domain,
               standard=r.standard, gold=gold)
    for name, f in RUNGS.items():
        pred = " ".join(f(w, t) for w in tokens(r.standard))
        row[name + " pred"] = pred
        row[name] = chrf.sentence_score(pred, [gold]).score
    rows.append(row)
S = pd.DataFrame(rows)

# human ceiling on the shared sentences: speaker vs speaker, same metric
core_ids = sent.groupby("sentence_id").size().loc[lambda x: x == 3].index   # shared by 3 speakers
hum = {}
for sid in core_ids:
    g = sent[sent.sentence_id == sid]
    for a in g.itertuples():
        hum[(sid, a.speaker)] = np.mean([
            chrf.sentence_score(" ".join(tokens(a.owo)), [" ".join(tokens(b.owo))]).score
            for b in g.itertuples() if b.speaker != a.speaker])
S["human vs human"] = [hum.get((r.sentence_id, r.speaker), np.nan) for r in S.itertuples()]


def boot_mean(d, col, n=1000):
    """Mean over rows; the interval resamples whole sentence IDs (all speakers together)."""
    g = d.groupby("sentence_id")[col].agg(["sum", "count"])
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    m = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return d[col].mean(), *np.percentile(m, [2.5, 97.5])


lad = []
for label, d in (("all sentences", S), ("sentences shared by 3 speakers", S[S.sentence_id.isin(core_ids)])):
    for col in list(RUNGS) + (["human vs human"] if label.startswith("sentences shared") else []):
        m, lo, hi = boot_mean(d, col)
        lad.append({"subset": label, "rung": col, "n_rows": len(d), "chrF++": round(m, 1),
                    "ci_low": round(lo, 1), "ci_high": round(hi, 1)})
lad = pd.DataFrame(lad)
lad.to_csv(f"results/task2_ladder{TAG}.csv", index=False)
print("SENTENCE LEVEL (mean chrF++ vs the speaker's own Owo; 95% CI by sentence ID)")
print(lad.to_string(index=False))

# ---------- word level -----------------------------------------------------------------
gold_pairs, unchanged = [], []
for r in sent.itertuples():
    a = ann[(ann.sentence_id == r.sentence_id) & (ann.speaker == r.speaker)]
    annotated = set()
    for x in a.itertuples():
        if isinstance(x.std_word, str):
            sw = x.std_word.strip().strip(PUNCT)
            annotated.add(low(sw))
            if sw and " " not in sw and sw != "—" and " " not in x.owo_word.strip():
                gold_pairs.append((r.sentence_id, sw, x.owo_word.strip(), x.type, x.pattern))
    owo_toks = {low(w) for w in tokens(r.owo)}
    for w in tokens(r.standard):
        if low(w) not in annotated and low(w) in owo_toks:
            unchanged.append((r.sentence_id, w))
print(f"\nWORD LEVEL: {len(gold_pairs)} annotated single-word pairs, {len(unchanged)} unchanged words")

wl = []
for name, f in RUNGS.items():
    per_type = {}
    for sid, sw, ow, typ, pat in gold_pairs:
        hit = low(nfc(f(sw, tables[sid]))) == low(nfc(ow))
        per_type.setdefault(typ, []).append(hit)
    allhits = [h for v in per_type.values() for h in v]
    changed = np.mean([low(f(w, tables[sid])) != low(w) for sid, w in unchanged])
    row = {"rung": name, "all differences got exactly right": round(np.mean(allhits), 2)}
    for typ, v in sorted(per_type.items()):
        row[f"{typ} (n={len(v)})"] = round(np.mean(v), 2)
    row["unchanged words wrongly changed"] = round(changed, 2)
    wl.append(row)
wl = pd.DataFrame(wl)
wl.to_csv(f"results/task2_word_level{TAG}.csv", index=False)
print(wl.to_string(index=False))

# ---------- per-rule report ------------------------------------------------------------
known = [(sw, ow) for _, sw, ow, _, _ in gold_pairs] + [(w, w) for _, w in unchanged]
rep = []
for rule in RULES:
    fired = better = worse = 0
    for sw, ow in known:
        new = apply_rules(sw, [rule])
        if low(new) == low(sw):
            continue
        fired += 1
        d0, d1 = lev(low(nfc(sw)), low(nfc(ow))), lev(low(nfc(new)), low(nfc(ow)))
        better += d1 < d0
        worse += d1 > d0
    rep.append({"rule": rule[0], "fired": fired, "improved": better, "worsened": worse,
                "same": fired - better - worse,
                "improved share": round(better / fired, 2) if fired else None})
rep = pd.DataFrame(rep)
rep.to_csv(f"results/task2_rule_report{TAG}.csv", index=False)
print("\nPER-RULE REPORT (applied alone; known words only)")
print(rep.to_string(index=False))

S.to_csv(f"results/task2_sentences{TAG}.csv", index=False, encoding="utf-8-sig")
print("\nWrote results/task2_*.csv  (open task2_sentences.csv to read predictions beside gold)")
