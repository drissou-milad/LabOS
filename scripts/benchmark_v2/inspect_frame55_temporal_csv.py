from pathlib import Path
import pandas as pd

p = Path(
    "results/exp05_training_dataset/run02_20samples/"
    "run13_temporal_audit/run13_temporal_gt_context.csv"
)

df = pd.read_csv(p)

# Display the already-computed temporal context for frame 55.
row = df[
    (df["sample_id"] == "44b6_1d530831")
    & (df["candidate_frame"] == 55)
] if "candidate_frame" in df.columns else None

print("Columns:", df.columns.tolist())
print("\nFrame-55 context rows:")
print(df.to_string(index=False))
