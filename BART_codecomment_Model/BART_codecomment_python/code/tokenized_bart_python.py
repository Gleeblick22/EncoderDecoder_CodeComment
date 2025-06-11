# File: code/tokenized_bart_python.py

import os
import json
import pandas as pd
import argparse
import logging
from datasets import Dataset
from transformers import BartTokenizer

# === [1] Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === [2] Load JSON ===
def load_json_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === [3] Tokenization ===
def tokenize_function(example, tokenizer):
    inputs = tokenizer(
        example['code'],
        max_length=512,
        padding='max_length',
        truncation=True
    )
    targets = tokenizer(
        example['comment'],
        max_length=128,
        padding='max_length',
        truncation=True
    )
    inputs['labels'] = targets['input_ids']
    return inputs

# === [4] Save Dataset ===
def save_tokenized_dataset(dataset, output_dir, split):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    dataset.save_to_disk(split_dir)

# === [5] Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize Python-only dataset for BART")
    parser.add_argument('--input_dir', type=str, default="C:/ModelTrain/Econder-Decoder/data_collection/python_only", help='Directory containing train/valid/test JSONs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save tokenized datasets')
    return parser.parse_args()

# === [6] Main ===
def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    train_path = os.path.join(input_dir, "train_data.json")
    valid_path = os.path.join(input_dir, "valid_data.json")
    test_path  = os.path.join(input_dir, "test_data.json")

    logger.info("Loading BART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    logger.info("Loading datasets...")
    train_data = load_json_dataset(train_path)
    valid_data = load_json_dataset(valid_path)
    test_data  = load_json_dataset(test_path)

    logger.info("Converting datasets to HuggingFace format...")
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data))
    test_dataset  = Dataset.from_pandas(pd.DataFrame(test_data))

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    valid_dataset = valid_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset  = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    logger.info(f"Saving tokenized datasets to: {output_dir}")
    save_tokenized_dataset(train_dataset, output_dir, "train")
    save_tokenized_dataset(valid_dataset, output_dir, "valid")
    save_tokenized_dataset(test_dataset, output_dir, "test")

    logger.info("Tokenization complete.")

if __name__ == "__main__":
    main()
