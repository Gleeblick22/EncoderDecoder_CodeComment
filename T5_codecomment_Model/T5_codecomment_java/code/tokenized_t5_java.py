# File: code/tokenized_t5_java.py

"""
Tokenize Java code-comment datasets using T5 tokenizer.
This script loads training, validation, and test JSON files,
performs tokenization using HuggingFace Transformers,
and saves the datasets for downstream training.
"""

import os
import json
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer
import argparse
import logging

# === [0] Logger setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === [1] Load JSON dataset from file ===
def load_json_dataset(file_path):
    """
    Load a JSON dataset from the given path.
    Raise an error if the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === [2] Tokenization logic ===
def tokenize_function(example, tokenizer):
    """
    Tokenize the 'code' and 'comment' fields using the T5 tokenizer.
    Returns a dictionary of input tokens and label tokens.
    """
    inputs = tokenizer(example['code'], max_length=512, padding='max_length', truncation=True)
    labels = tokenizer(example['comment'], max_length=128, padding='max_length', truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

# === [3] Save tokenized dataset to disk ===
def save_tokenized_dataset(dataset, output_dir, split):
    """
    Save the tokenized dataset for a given split (train/valid/test) to disk.
    """
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    dataset.save_to_disk(split_dir)

# === [4] Argument parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize Java dataset using T5 tokenizer")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save tokenized data')
    return parser.parse_args()

# === [5] Main logic ===
def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    train_path = os.path.join(input_dir, "train_data.json")
    valid_path = os.path.join(input_dir, "valid_data.json")
    test_path  = os.path.join(input_dir, "test_data.json")

    logger.info("Loading T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    logger.info(f"Reading datasets from: {input_dir}")
    train_data = load_json_dataset(train_path)
    valid_data = load_json_dataset(valid_path)
    test_data  = load_json_dataset(test_path)

    for sample in [train_data[0], valid_data[0], test_data[0]]:
        assert 'code' in sample and 'comment' in sample, "Missing required keys in input data."

    logger.info("Converting to HuggingFace Datasets format...")
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

    logger.info(f"Tokenization complete. Tokenized data saved in: {output_dir}")

# === [6] Entry point ===
if __name__ == "__main__":
    main()
