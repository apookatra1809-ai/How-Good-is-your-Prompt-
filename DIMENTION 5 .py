import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

tqdm.pandas()

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"
PROMPT_BANK_FILE = f"{DATA_PATH}/prompt_bank_final.csv"

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_f1(reference, candidate):
    if pd.isna(reference) or pd.isna(candidate):
        return np.nan
    reference, candidate = str(reference).strip(), str(candidate).strip()
    if not reference or not candidate:
        return np.nan
    return scorer.score(reference, candidate)["rougeL"].fmeasure


def process_file(input_file, output_file, reference_col):
    df = pd.read_csv(input_file)
    df = df.merge(prompt_bank[["prompt_id", reference_col]], on="prompt_id", how="left")

    df["rouge_l_f1"] = df.progress_apply(
        lambda r: rouge_l_f1(r[reference_col], r["response"]), axis=1
    )

    minimum = df["rouge_l_f1"].min()
    maximum = df["rouge_l_f1"].max()
    if maximum > minimum:
        df["response_richness_score"] = (df["rouge_l_f1"] - minimum) / (maximum - minimum)
    else:
        df["response_richness_score"] = 1

    df.to_csv(output_file, index=False)
    print("Saved:", output_file)
    return df


prompt_bank = pd.read_csv(PROMPT_BANK_FILE)

for model in MODELS:
    process_file(
        f"{DATA_PATH}/{model}_original_checkpoint_1024.csv",
        f"{model}_original_response_richness.csv",
        reference_col="prompt",
    )
    process_file(
        f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv",
        f"{model}_vague_response_richness.csv",
        reference_col="vague_version",
    )
