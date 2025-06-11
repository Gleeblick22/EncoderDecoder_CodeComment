# File: code/extract_trainer_state_log.py

import os
import json
import pandas as pd

def extract_trainer_logs(trainer_state_path, output_csv_path):
    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        raise ValueError("No log history found in trainer_state.json")

    df = pd.DataFrame(log_history)
    df = df.sort_values("step").reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)

    print(f" Extracted {len(df)} log entries.")
    print(f" Saved to: {output_csv_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    trainer_state_file = os.path.abspath(os.path.join(base_dir, "../models/T5_Training_Checkpoints/trainer_state.json"))
    output_csv = os.path.join(os.path.dirname(trainer_state_file), "training_log.csv")

    extract_trainer_logs(trainer_state_file, output_csv)
