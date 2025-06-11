# File: code/tokenized_t5_dataset_python_java.py

import os
import json
import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer

# === [1] Load JSON ===
def load_json_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === [2] Tokenization Logic ===
def tokenize_function(example, tokenizer):
    inputs = tokenizer(example['code'], max_length=512, padding='max_length', truncation=True)
    labels = tokenizer(example['comment'], max_length=128, padding='max_length', truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

# === [3] Save to Disk ===
def save_tokenized_dataset(dataset, output_dir, split):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    dataset.save_to_disk(split_dir)

# === [4] Main Logic ===
def main():
    root_dir = "/mnt/c/ModelTrain/Econder-Decoder"
    input_dir = os.path.join(root_dir, "data_collection", "combined")
    output_dir = os.path.join(root_dir, "T5_codecomment_Model", "T5_codecomment_python_java", "Tokenized_data")

    train_path = os.path.join(input_dir, "train_data.json")
    valid_path = os.path.join(input_dir, "valid_data.json")
    test_path  = os.path.join(input_dir, "test_data.json")

    print("[INFO] Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    print(f"[INFO] Reading datasets from: {input_dir}")
    train_data = load_json_dataset(train_path)
    valid_data = load_json_dataset(valid_path)
    test_data  = load_json_dataset(test_path)

    for sample in [train_data[0], valid_data[0], test_data[0]]:
        assert 'code' in sample and 'comment' in sample, "Missing required keys in input data."

    print("[INFO] Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_data))
    test_dataset  = Dataset.from_pandas(pd.DataFrame(test_data))

    print("[INFO] Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    valid_dataset = valid_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    print(f"[INFO] Saving tokenized datasets to: {output_dir}")
    save_tokenized_dataset(train_dataset, output_dir, "train")
    save_tokenized_dataset(valid_dataset, output_dir, "valid")
    save_tokenized_dataset(test_dataset, output_dir, "test")

    print(f"\n Tokenization complete.\nTokenized datasets saved to:\n{output_dir}")

# === [5] Entry ===
if __name__ == "__main__":
    main()
