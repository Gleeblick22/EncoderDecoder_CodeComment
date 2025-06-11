# File: data_collection/run_all.py

import os
import json
import argparse
from pathlib import Path
from .clone_repos import clone_all_repos
from .extract_python import extract_python_from_repos
from .extract_java import extract_java_from_repos
from .merge_and_clean import merge_and_save_dataset
from .split_dataset import split_and_save_by_language

def main():
    parser = argparse.ArgumentParser(description="Run full data collection pipeline.")
    parser.add_argument("--repos_file", type=str, required=True, help="Path to repos.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output")
    args = parser.parse_args()

    base_output = Path(args.output_dir).resolve()
    repos_file = Path(args.repos_file).resolve()
    raw_repo_dir = base_output / "raw_cloned_repos"
    json_dataset_file = base_output / "json_full_dataset.json"

    print("[1/5] Cloning repositories...")
    clone_all_repos(repos_file, raw_repo_dir)

    print("[2/5] Extracting Python code-comment pairs...")
    python_data = extract_python_from_repos(raw_repo_dir)

    print("[3/5] Extracting Java code-comment pairs...")
    java_data = extract_java_from_repos(raw_repo_dir)

    print("[4/5] Merging and saving full dataset...")
    merge_and_save_dataset(python_data, java_data, json_dataset_file)

    print("[5/5] Splitting and saving language-specific datasets...")
    split_and_save_by_language(python_data, java_data, json_dataset_file, base_output)

    print(" Data collection complete.")

if __name__ == "__main__":
    main()
