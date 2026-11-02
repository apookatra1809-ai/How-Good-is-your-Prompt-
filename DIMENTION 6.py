import re
import ast
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()

MODELS = ["mistral", "phi", "qwen"]
DATA_PATH = "/kaggle/input/datasets/sharathraoram/dataset-final"
PROMPT_BANK_FILE = f"{DATA_PATH}/prompt_bank_final.csv"

FORMAT_PATTERNS = {
    "numbered_list": re.compile(r"\b(numbered list|steps|step[- ]by[- ]step|step \d)\b", re.I),
    "bullet_list": re.compile(r"\b(bullet(ed)? (points|list)|list of)\b", re.I),
    "table": re.compile(r"\btable\b", re.I),
    "code_block": re.compile(
        r"\b(write|provide|give|generate|create)\b.{0,40}\b(code|function|script|snippet|program)\b"
        r"|```"
        r"|\bin (python|javascript|java|c\+\+|sql|html|css)\b",
        re.I,
    ),
    "json": re.compile(r"\bjson\b", re.I),
    "headers": re.compile(r"\b(headings?|sections?|headers?)\b", re.I),
}

RESPONSE_CHECKS = {
    "numbered_list": re.compile(r"(^|\n)\s*\d+[.)]\s+", re.M),
    "bullet_list": re.compile(r"(^|\n)\s*[-*•]\s+", re.M),
    "table": re.compile(r"\|.+\|.+\|"),
    "code_block": re.compile(r"```"),
    "json": re.compile(r"[{\[].*[}\]]", re.S),
    "headers": re.compile(r"(^|\n)\s*#{1,6}\s+\S", re.M),
}


def detect_expected_format(prompt_text):
    if pd.isna(prompt_text):
        return []
    prompt_text = str(prompt_text)
    return [fmt for fmt, pattern in FORMAT_PATTERNS.items() if pattern.search(prompt_text)]


def compliance_score(response_text, expected_format_str):
    expected = ast.literal_eval(expected_format_str) if isinstance(expected_format_str, str) else expected_format_str
    if not expected:
        return 1.0
    if pd.isna(response_text):
        return 0.0
    response_text = str(response_text)
    hits = sum(1 for fmt in expected if RESPONSE_CHECKS[fmt].search(response_text))
    return hits / len(expected)


def to_5point(x):
    if pd.isna(x):
        return np.nan
    if x <= 0.2:
        return 1
    elif x <= 0.4:
        return 2
    elif x <= 0.6:
        return 3
    elif x <= 0.8:
        return 4
    return 5


def process_file(input_file, output_file, prompt_bank_with_format):
    df = pd.read_csv(input_file)
    df = df.merge(prompt_bank_with_format[["prompt_id", "expected_format"]], on="prompt_id", how="left")

    df["format_compliance_raw"] = df.progress_apply(
        lambda r: compliance_score(r["response"], r["expected_format"]), axis=1
    )
    df["format_compliance_score"] = df["format_compliance_raw"].apply(to_5point)

    df.to_csv(output_file, index=False)
    print("Saved:", output_file)
    return df


prompt_bank = pd.read_csv(PROMPT_BANK_FILE)
prompt_bank["expected_format"] = prompt_bank["prompt"].progress_apply(detect_expected_format)

for model in MODELS:
    process_file(
        f"{DATA_PATH}/{model}_original_checkpoint_1024.csv",
        f"{model}_original_format_compliance.csv",
        prompt_bank,
    )
    process_file(
        f"{DATA_PATH}/{model}_vague_checkpoint_1024.csv",
        f"{model}_vague_format_compliance.csv",
        prompt_bank,
    )
