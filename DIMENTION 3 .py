import pandas as pd
import numpy as np
from lexicalrichness import LexicalRichness
from tqdm.auto import tqdm

tqdm.pandas()

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"


def compute_mtld(text):
    if pd.isna(text):
        return np.nan
    text = str(text).strip()
    if len(text.split()) < 20:
        return np.nan
    try:
        lex = LexicalRichness(text)
        return lex.mtld()
    except Exception:
        return np.nan


def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    df["mtld"] = df["response"].progress_apply(compute_mtld)
    df.to_csv(output_file, index=False)
    print("Saved:", output_file)
    return df


for model in MODELS:
    process_file(
        f"{DATA_PATH}/{model}_original_checkpoint_1024.csv",
        f"{model}_original_lexical_diversity.csv",
    )
    process_file(
        f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv",
        f"{model}_vague_lexical_diversity.csv",
    )
