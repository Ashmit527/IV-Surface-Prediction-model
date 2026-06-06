import pandas as pd
import sys

ORIGINAL_DATASET_PATH = "train.csv" 

SEPARATOR = "||"

def generate_solution(filled_path: str, output_path: str = "solution.csv"):
    original = pd.read_csv(ORIGINAL_DATASET_PATH)
    filled   = pd.read_csv(filled_path)

    feature_cols = [c for c in original.columns if c != "datetime"]

    rows = []
    for col in feature_cols:
        was_missing = original[col].isna()

        for idx in original.index[was_missing]:
            dt  = original.loc[idx, "datetime"]
            uid = f"{dt}{SEPARATOR}{col}"
            val = filled.loc[idx, col]
            rows.append({"id": uid, "value": val})

    solution = pd.DataFrame(rows, columns=["id", "value"])
    solution = solution.sort_values("id").reset_index(drop=True)
    solution.to_csv(output_path, index=False)
    print(f"✅ Solution saved → {output_path}  ({len(solution)} rows)")

generate_solution("filled_surfacebackfc65c.csv", "submissionbackfc65c.csv")
