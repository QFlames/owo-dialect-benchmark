
import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

# Load saved Task 1 predictions.
df = pd.read_csv("results/task1_sentences.csv")

required = [
    "sentence_id", "english",
    "hyp_standard", "hyp_owo"
]
assert df[required].notna().all().all(), "Missing values found"

metric = CHRF(word_order=2)

def corpus_score(data, column):
    return metric.corpus_score(
        data[column].astype(str).tolist(),
        [data["english"].astype(str).tolist()]
    ).score

def gap(data):
    standard = corpus_score(data, "hyp_standard")
    owo = corpus_score(data, "hyp_owo")
    return standard - owo

# Observed result
observed = gap(df)

# Group repeated versions of each sentence together.
groups = [
    group
    for _, group in df.groupby("sentence_id", sort=False)
]

n_clusters = len(groups)
rng = np.random.default_rng(42)
n_bootstrap = 2000
bootstrap_gaps = []

for _ in range(n_bootstrap):
    sampled_indices = rng.integers(
        0, n_clusters, size=n_clusters
    )
    sample = pd.concat(
        [groups[i] for i in sampled_indices],
        ignore_index=True
    )
    bootstrap_gaps.append(gap(sample))

lower, upper = np.percentile(
    bootstrap_gaps, [2.5, 97.5]
)

print(f"Rows: {len(df)}")
print(f"Unique sentence IDs: {n_clusters}")
print(f"Standard chrF++: {corpus_score(df, 'hyp_standard'):.2f}")
print(f"Owo chrF++: {corpus_score(df, 'hyp_owo'):.2f}")
print(f"Observed gap: {observed:.2f}")
print(f"95% cluster bootstrap CI: [{lower:.2f}, {upper:.2f}]")

pd.DataFrame({
    "bootstrap_gap": bootstrap_gaps
}).to_csv(
    "results/task1_bootstrap.csv", index=False
)
