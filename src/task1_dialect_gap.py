"""Task 1: how much worse is a translation model on Owo than on Standard Yorùbá?

    python src/task1_dialect_gap.py --model facebook/nllb-200-distilled-600M --limit 20   # quick test
    python src/task1_dialect_gap.py --model facebook/nllb-200-distilled-600M              # full run

Translates the Standard and the Owo sentence to English, scores both against the
English reference with chrF++, then breaks the score drop down by the kinds of
differences annotated in each sentence. Runs locally; no data leaves the machine.
"""
import argparse
from pathlib import Path
import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate(texts, tok, model, bs=8, beams=4):
    out = []
    for i in tqdm(range(0, len(texts), bs)):
        batch = tok(texts[i:i + bs], return_tensors="pt", padding=True)
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
                                 num_beams=beams, max_new_tokens=96)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/task1")
    a = ap.parse_args()
    df = pd.read_csv("data/processed/sentences.csv")
    if a.limit: df = df.sample(a.limit, random_state=0).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.model, src_lang="yor_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).eval()
    df["hyp_standard"] = translate(df["standard"].tolist(), tok, model)
    df["hyp_owo"] = translate(df["owo"].tolist(), tok, model)
    chrf = CHRF(word_order=2)                      # word_order=2 gives chrF++
    sc = lambda h, r: chrf.sentence_score(h, [r]).score
    df["chrf_standard"] = [sc(h, r) for h, r in zip(df.hyp_standard, df.english)]
    df["chrf_owo"] = [sc(h, r) for h, r in zip(df.hyp_owo, df.english)]
    df["drop"] = df.chrf_standard - df.chrf_owo
    corpus = lambda col: chrf.corpus_score(df[col].tolist(), [df["english"].tolist()]).score
    print(f"\nCorpus chrF++  Standard: {corpus('hyp_standard'):.1f}   Owo: {corpus('hyp_owo'):.1f}")
    # which kinds of differences go with bigger drops?
    for t in ["Tonal", "Phonological", "Lexical", "Loanword"]:
        has = df["types"].fillna("").str.contains(t)
        print(f"{t:13s} sentences with it: {has.sum():3d}  mean drop {df.loc[has,'drop'].mean():6.2f}  without: {df.loc[~has,'drop'].mean():6.2f}")
    print("\nBy speaker:\n", df.groupby("speaker")[["chrf_standard", "chrf_owo", "drop"]].mean().round(1))
    print("\nBy domain:\n", df.groupby("domain")[["chrf_standard", "chrf_owo", "drop"]].mean().round(1))
    print("\nCorrelation of drop with number of annotated differences:", round(df["drop"].corr(df["n_diffs"]), 2))
    Path(a.out).parent.mkdir(exist_ok=True)
    df.to_csv(a.out + "_sentences.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
