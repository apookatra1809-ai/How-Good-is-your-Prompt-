import pandas as pd
import numpy as np

MODELS = ["mistral", "phi", "qwen"]
CONDITIONS = ["original", "vague"]

DIMENSION_FILES = {
    "task_alignment": ("{model}_{cond}_task_alignment.csv", "bertscore_f1"),
    "response_richness": ("{model}_{cond}_response_richness.csv", "rouge_l_f1"),
    "readability": ("{model}_{cond}_readability.csv", "flesch_reading_ease"),
    "lexical_diversity": ("{model}_{cond}_lexical_diversity.csv", "mtld"),
    "format_compliance": ("{model}_{cond}_format_compliance.csv", "format_compliance_raw"),
    "clarity_specificity": ("{model}_{cond}_clarity_specificity.csv", "specificity_index"),
    "coherence": ("{model}_{cond}_coherence.csv", "coherence_raw"),
}

MASTER_KEYS = ["prompt_id", "primary_tag", "prompt_type", "model"]


def load_dimension(dim_name, filename_template, raw_col):
    frames = []
    for model in MODELS:
        for cond in CONDITIONS:
            path = filename_template.format(model=model, cond=cond)
            df = pd.read_csv(path)
            df["model"] = f"{model}-7b" if model != "phi" else "phi"
            df["prompt_type"] = cond
            frames.append(df[["prompt_id", "primary_tag", "prompt_type", "model", raw_col]]
                          .rename(columns={raw_col: dim_name}))
    return pd.concat(frames, ignore_index=True)


def quintile_score(series):
    pct_rank = series.rank(pct=True, method="average")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return pd.cut(pct_rank, bins=bins, labels=[1, 2, 3, 4, 5], include_lowest=True).astype(float)


master = None
for dim_name, (template, raw_col) in DIMENSION_FILES.items():
    dim_df = load_dimension(dim_name, template, raw_col)
    if master is None:
        master = dim_df
    else:
        master = master.merge(dim_df, on=MASTER_KEYS, how="outer")

for dim_name in DIMENSION_FILES.keys():
    master[f"{dim_name}_raw"] = master[dim_name]
    master[dim_name] = quintile_score(master[dim_name])

master = master.dropna(subset=list(DIMENSION_FILES.keys()), how="all")
master.to_csv("master_dataset_FINAL.csv", index=False)
print("Saved: master_dataset_FINAL.csv")
print(master.shape)
print(master[list(DIMENSION_FILES.keys())].describe())
