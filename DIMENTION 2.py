import pandas as pd
import numpy as np
import textstat
from tqdm.auto import tqdm

tqdm.pandas()

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"


def compute_readability(text):
    if pd.isna(text):
        return np.nan
    return textstat.flesch_reading_ease(str(text))


def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    df["flesch_reading_ease"] = df["response"].progress_apply(compute_readability)

    minimum = df["flesch_reading_ease"].min()
    maximum = df["flesch_reading_ease"].max()
    if maximum > minimum:
        df["readability_score"] = (df["flesch_reading_ease"] - minimum) / (maximum - minimum)
    else:
        df["readability_score"] = 1

    df.to_csv(output_file, index=False)
    print("Saved:", output_file)
    return df


for model in MODELS:
    process_file(
        f"{DATA_PATH}/{model}_original_checkpoint_1024.csv",
        f"{model}_original_readability.csv",
    )
    process_file(
        f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv",
        f"{model}_vague_readability.csv",
    )
