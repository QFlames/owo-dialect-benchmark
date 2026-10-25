"""Turn the Label Studio JSON export into two flat tables.

    python src/prepare_data.py data/raw/owo_annotations_2026-10-25.json

Writes data/processed/sentences.csv and data/processed/annotations.csv.
Highlights are cleaned: stray spaces and a trailing full stop are removed and
start/end are corrected so that owo[start:end] == owo_word always holds.
"""
import json, sys, unicodedata
from pathlib import Path
import pandas as pd

PUNCT = ".,!?;:"
nfc = lambda s: unicodedata.normalize("NFC", s or "")

def main(path):
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
        ann_rows += kept
        sent_rows.append(dict(sentence_id=d["id"], speaker=d["speaker"], domain=d["domain"],
                              english=d["english"], standard=nfc(d["standard"]), owo=owo,
                              review=review, n_diffs=len(kept),
                              types=";".join(sorted({k["type"] for k in kept}))))
    out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sent_rows).to_csv(out / "sentences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ann_rows).to_csv(out / "annotations.csv", index=False, encoding="utf-8-sig")
    print(f"{len(sent_rows)} sentences, {len(ann_rows)} annotated differences")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/owo_annotations_2026-10-25.json")
