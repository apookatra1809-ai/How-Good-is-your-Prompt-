import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tqdm.pandas()

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    text = str(text).strip()
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 0]


def compute_coherence(text):
    if pd.isna(text):
        return np.nan

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return np.nan

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError:
        return np.nan

    sims = [
        cosine_similarity(tfidf[i], tfidf[i + 1])[0][0]
        for i in range(len(sentences) - 1)
    ]
    return float(np.mean(sims))


def process_file(input_file, output_file):
    df = pd.read_csv(input_file)
    df["coherence_raw"] = df["response"].progress_apply(compute_coherence)

    minimum = df["coherence_raw"].min()
    maximum = df["coherence_raw"].max()
    if maximum > minimum:
        df["coherence_score"] = (df["coherence_raw"] - minimum) / (maximum - minimum)
    else:
        df["coherence_score"] = 1

    df.to_csv(output_file, index=False)
    print("Saved:", output_file)
    return df


for model in MODELS:
    process_file(
        f"{DATA_PATH}/{model}_original_checkpoint_1024.csv",
        f"{model}_original_coherence.csv",
    )
    process_file(
        f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv",
        f"{model}_vague_coherence.csv",
    )
