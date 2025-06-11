# File: code/tokenized_bart_java.py

import os
import json
import argparse
import logging
from datasets import Dataset
from transformers import BartTokenizer

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load JSON file ===
def load_json_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

# === Tokenize input/output ===
def tokenize_function(example, tokenizer):
    inputs = tokenizer(
        example['code'], max_length=512, padding='max_length', truncation=True
    )
    targets = tokenizer(
        example['comment'], max_length=128, padding='max_length', truncation=True
    )
    inputs['labels'] = targets['input_ids']
    return inputs

# === Save dataset to disk ===
def save_tokenized_dataset(dataset, output_dir, split):
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)
    dataset.save_to_disk(split_path)

# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Tokenize Java-only code-comment dataset for BART")
    parser.add_argument('--train_path', type=str, required=True, help="Path to train_data.json")
    parser.add_argument('--valid_path', type=str, required=True, help="Path to valid_data.json")
    parser.add_argument('--test_path', type=str, required=True, help="Path to test_data.json")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save tokenized splits")
    args = parser.parse_args()

    logger.info("Loading BART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    logger.info("Loading datasets...")
    train_data = load_json_dataset(args.train_path)
    valid_data = load_json_dataset(args.valid_path)
    test_data = load_json_dataset(args.test_path)

    logger.info("Tokenizing datasets...")
    tokenized_train = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_valid = valid_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    logger.info("Saving tokenized datasets...")
    save_tokenized_dataset(tokenized_train, args.output_dir, "train")
    save_tokenized_dataset(tokenized_valid, args.output_dir, "valid")
    save_tokenized_dataset(tokenized_test, args.output_dir, "test")

    logger.info(" Tokenization complete.")

if __name__ == "__main__":
    main()
