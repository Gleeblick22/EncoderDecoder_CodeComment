# File: code/extract_trainer_state_t5.py

import os
import json
import pandas as pd

def find_latest_checkpoint_trainer_state(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not subdirs:
        raise FileNotFoundError("No checkpoint folders found.")
    latest = sorted(subdirs, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(base_dir, latest, "trainer_state.json")

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
    checkpoints_dir = os.path.abspath(os.path.join(base_dir, "../models/T5_Training_Checkpoints"))
    output_csv = os.path.join(checkpoints_dir, "T5_Python+Java data_training_log.csv")

    try:
        trainer_state_file = os.path.join(checkpoints_dir, "trainer_state.json")
        if not os.path.isfile(trainer_state_file):
            print(" trainer_state.json not found at root. Searching inside checkpoints...")
            trainer_state_file = find_latest_checkpoint_trainer_state(checkpoints_dir)
        extract_trainer_logs(trainer_state_file, output_csv)
    except Exception as e:
        print(f" Error: {e}")
