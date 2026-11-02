import pandas as pd
import numpy as np
import re
import spacy
from tqdm.auto import tqdm

tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "might", "could", "may",
    "seems", "appears", "apparently", "generally",
    "approximately", "about", "around",
    "kind", "sort", "etc",
    "something", "someone", "somewhere",
    "likely", "probably",
}


def clarity_specificity(text):
    if pd.isna(text):
        return pd.Series([0, 0, 0, 0])

    text = str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)
    if total_words == 0:
        return pd.Series([0, 0, 0, 0])

    hedge_count = sum(word in HEDGE_WORDS for word in words)
    hedge_ratio = hedge_count / total_words

    doc = nlp(text)
    specific_count = sum(1 for token in doc if token.like_num)
    specific_count += len(doc.ents)
    specificity_index = specific_count / total_words

    return pd.Series([hedge_count, hedge_ratio, specific_count, specificity_index])


def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    df[
        ["hedge_word_count", "hedge_ratio", "specific_term_count", "specificity_index"]
    ] = df["response"].progress_apply(clarity_specificity)
    df.to_csv(output_file, index=False)
    print("Saved:", output_file)
    return df


for model in MODELS:
    process_file(
        f"{DATA_PATH}/{model}_original_checkpoint_1024.csv",
        f"{model}_original_clarity_specificity.csv",
    )
    process_file(
        f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv",
        f"{model}_vague_clarity_specificity.csv",
    )
