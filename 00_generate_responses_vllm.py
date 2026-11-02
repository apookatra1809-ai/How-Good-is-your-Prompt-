import pandas as pd
from vllm import LLM, SamplingParams

FILE_PATH = "/kaggle/input/datasets/aparnakatra/dataset/prompt_bank_final.csv"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "qwen2.5-7b"

df = pd.read_csv(FILE_PATH)

llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

results = []
for idx, row in df.iterrows():
    try:
        outputs = llm.generate([str(row["prompt"])], sampling_params)
        response = outputs[0].outputs[0].text
        results.append({
            "prompt_id": row["prompt_id"],
            "primary_tag": row["primary_tag"],
            "prompt_type": "original",
            "model": MODEL_NAME,
            "response": response,
        })
    except Exception as e:
        print(f"Error at {row['prompt_id']}: {e}")

final_df = pd.DataFrame(results)
final_df.to_csv(f"{MODEL_NAME}_original_final.csv", index=False)
print(final_df.shape)

results_vague = []
for idx, row in df.iterrows():
    try:
        outputs = llm.generate([str(row["vague_version"])], sampling_params)
        response = outputs[0].outputs[0].text
        results_vague.append({
            "prompt_id": row["prompt_id"],
            "primary_tag": row["primary_tag"],
            "prompt_type": "vague",
            "model": MODEL_NAME,
            "response": response,
        })
    except Exception as e:
        print(f"Error at {row['prompt_id']}: {e}")

vague_df = pd.DataFrame(results_vague)
vague_df.to_csv(f"{MODEL_NAME}_vague_final.csv", index=False)
print(vague_df.shape)
