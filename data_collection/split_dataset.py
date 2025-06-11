# File: data_collection/split_dataset.py

import json
import random
from pathlib import Path
from typing import List, Dict, Any

def save_split(data: List[Dict[str, Any]], name: str, out_dir: Path):
    random.shuffle(data)
    n = len(data)
    train_end = int(0.7 * n)
    valid_end = train_end + int(0.15 * n)

    splits = {
        "train_data.json": data[:train_end],
        "valid_data.json": data[train_end:valid_end],
        "test_data.json": data[valid_end:]
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_data in splits.items():
        with open(out_dir / filename, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
    print(f" {name} split saved to: {out_dir} ({n} samples)")

def split_and_save_by_language(python_data: List[Dict[str, Any]],
                                java_data: List[Dict[str, Any]],
                                full_dataset_path: Path,
                                output_root: Path):
    # Save full combined data
    with open(full_dataset_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    save_split(all_data, "Combined", output_root / "combined")
    save_split(python_data, "Python Only", output_root / "python_only")
    save_split(java_data, "Java Only", output_root / "java_only")
