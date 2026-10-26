"""Turn the Label Studio JSON export into two flat tables.

    python src/prepare_data.py data/raw/owo_annotations_2026-10-26.json

Writes data/processed/sentences.csv and data/processed/annotations.csv.
Highlights are cleaned: stray spaces and a trailing full stop are removed and
start/end are corrected so that owo[start:end] == owo_word always holds.
"""
import argparse, json, unicodedata
from pathlib import Path
import pandas as pd

PUNCT = ".,!?;:"
nfc = lambda s: unicodedata.normalize("NFC", s or "")

def load_corrections(path="data/corrections.csv"):
    """Fixes kept outside the raw export: rows with an empty new_value are still TODO."""
    if not Path(path).exists():
        return {}
    c = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    out = {}
    for r in c.itertuples():
        if r.new_value:
            out[(r.sentence_id, r.speaker, nfc(r.owo_word).lower(), r.field)] = (r.new_value, r.source)
    return out

def main(path, nee=None):
    corr = load_corrections()
    tasks = json.load(open(path, encoding="utf-8"))
    sent_rows, ann_rows = [], []
    for t in tasks:
        d = t["data"]
        owo = nfc(d["owo"])
        res = t["annotations"][0]["result"] if t["annotations"] else []
        regions, review = {}, None
        for r in res:
            v = r["value"]
            if r["from_name"] == "review":
                review = (v.get("choices") or [None])[0]
                continue
            e = regions.setdefault(r["id"], {})
            if r["from_name"] == "diff":
                e.update(start=v["start"], end=v["end"], type=v["labels"][0])
            elif r["from_name"] == "std_word":
                e["std_word"] = (v["text"] or [""])[0].strip() or None
            elif r["from_name"] == "pattern":
                e["pattern"] = (v.get("choices") or [None])[0]
        kept = []
        for e in regions.values():
            if "start" not in e:
                continue
            raw = owo[e["start"]:e["end"]]
            lead = len(raw) - len(raw.lstrip())
            word = raw.strip().rstrip(PUNCT).strip()
            if not word:
                continue
            s = e["start"] + lead
            kept.append(dict(sentence_id=d["id"], speaker=d["speaker"], domain=d["domain"],
                             start=s, end=s + len(word), owo_word=word,
                             std_word=nfc(e.get("std_word") or "") or None,
                             type=e["type"], pattern=e.get("pattern")))
        for k in kept:
            assert owo[k["start"]:k["end"]] == k["owo_word"], (k["sentence_id"], k["owo_word"])
            k["fix"] = ""
            for field in ("std_word", "pattern"):
                hit = corr.get((k["sentence_id"], k["speaker"], k["owo_word"].lower(), field))
                if hit and not k[field]:
                    k[field] = nfc(hit[0]); k["fix"] += f"{field}:{hit[1]};"
            if nee and nfc(k["owo_word"]).lower() == "nẹ́ẹ̀" and (k["std_word"] or "").lower() == "náà":
                k["type"], k["pattern"] = (("Lexical", "náà → nẹ́ẹ̀") if nee == "lexical" else ("Phonological", "áà → ẹ́ẹ̀"))
                k["fix"] += "nee-unified;"
        ann_rows += kept
        sent_rows.append(dict(sentence_id=d["id"], speaker=d["speaker"], domain=d["domain"],
                              english=d["english"], standard=nfc(d["standard"]), owo=owo,
                              review=review, n_diffs=len(kept),
                              types=";".join(sorted({k["type"] for k in kept}))))
    out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sent_rows).to_csv(out / "sentences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ann_rows).to_csv(out / "annotations.csv", index=False, encoding="utf-8-sig")
    ann = pd.DataFrame(ann_rows)
    print(f"{len(sent_rows)} sentences, {len(ann_rows)} annotated differences")
    print(f"still missing: {ann.std_word.isna().sum()} Standard words, {ann.pattern.isna().sum()} patterns "
          f"(fill them in data/corrections.csv)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export", nargs="?", default="data/raw/owo_annotations_2026-10-26.json")
    ap.add_argument("--unify-nee", choices=["lexical", "phonological", "none"], default="lexical",
                    help="label every náà -> nẹ́ẹ̀ the same way. Decision: Lexical, pattern 'náà → nẹ́ẹ̀'")
    a = ap.parse_args()
    main(a.export, None if a.unify_nee == "none" else a.unify_nee)
