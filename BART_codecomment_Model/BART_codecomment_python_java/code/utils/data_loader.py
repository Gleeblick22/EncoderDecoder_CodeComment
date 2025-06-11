# utils/data_loader.py

import os
import json
import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer

def load_json_dataset(file_path: str) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_hf_dataset(data: list) -> Dataset:
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)
