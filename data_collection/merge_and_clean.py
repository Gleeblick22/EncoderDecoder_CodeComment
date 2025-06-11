# File: data_collection/merge_and_clean.py

import json
from pathlib import Path
from typing import List, Dict, Any

def merge_and_save_dataset(python_data: List[Dict[str, Any]],
                            java_data: List[Dict[str, Any]],
                            output_file: Path) -> Path:
    """Merge Python and Java datasets and save to a single JSON file."""
    all_data = python_data + java_data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f" Merged dataset saved to: {output_file} ({len(all_data)} samples)")
    return output_file
