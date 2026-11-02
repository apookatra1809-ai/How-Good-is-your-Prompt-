import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from bert_score import score
from transformers import AutoTokenizer

tqdm.pandas()

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"
PROMPT_BANK_FILE = f"{DATA_PATH}/prompt_bank_final.csv"

tokenizer = AutoTokenizer.from_pretrained("roberta-large")


def truncate_text(text, max_tokens=510):
    if pd.isna(text):
        return ""
    encoded = tokenizer(str(text), truncation=True, max_length=max_tokens)
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)


def compute_bertscore(df, reference_col, response_col, batch_size=8):
    df = df.copy()
    df["bertscore_precision"] = np.nan
    df["bertscore_recall"] = np.nan
    df["bertscore_f1"] = np.nan

    valid_mask = (
        df[reference_col].notna() & df[response_col].notna()
        & (df[reference_col].astype(str).str.strip() != "")
        & (df[response_col].astype(str).str.strip() != "")
    )
    valid_df = df.loc[valid_mask].copy()

    references = valid_df[reference_col].astype(str).tolist()
    candidates = valid_df[response_col].astype(str).tolist()

    P_all, R_all, F_all = [], [], []
    for i in tqdm(range(0, len(references), batch_size)):
        ref_batch = references[i:i + batch_size]
        cand_batch = candidates[i:i + batch_size]
        P, R, F = score(
            cand_batch, ref_batch, lang="en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=batch_size, verbose=False,
        )
        P_all.extend(P.cpu().numpy())
        R_all.extend(R.cpu().numpy())
        F_all.extend(F.cpu().numpy())

    valid_df["bertscore_precision"] = P_all
    valid_df["bertscore_recall"] = R_all
    valid_df["bertscore_f1"] = F_all

    df.loc[valid_df.index, ["bertscore_precision", "bertscore_recall", "bertscore_f1"]] = \
        valid_df[["bertscore_precision", "bertscore_recall", "bertscore_f1"]]
    return df


prompt_bank = pd.read_csv(PROMPT_BANK_FILE)

for model in MODELS:
    original = pd.read_csv(f"{DATA_PATH}/{model}_original_checkpoint_1024.csv")
    vague = pd.read_csv(f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv")

    original_eval = original.merge(prompt_bank[["prompt_id", "prompt"]], on="prompt_id", how="left")
    vague_eval = vague.merge(prompt_bank[["prompt_id", "vague_version"]], on="prompt_id", how="left")

    original_eval["prompt"] = original_eval["prompt"].apply(truncate_text)
    original_eval["response"] = original_eval["response"].apply(truncate_text)
    vague_eval["vague_version"] = vague_eval["vague_version"].apply(truncate_text)
    vague_eval["response"] = vague_eval["response"].apply(truncate_text)

    original_scored = compute_bertscore(original_eval, reference_col="prompt", response_col="response")
    vague_scored = compute_bertscore(vague_eval, reference_col="vague_version", response_col="response")

    original_scored.to_csv(f"{model}_original_task_alignment.csv", index=False)
    vague_scored.to_csv(f"{model}_vague_task_alignment.csv", index=False)
    print(f"Saved task alignment files for {model}")
