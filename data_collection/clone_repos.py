# File: data_collection/clone_repos.py

import json
import subprocess
from pathlib import Path
from typing import Dict

def clone_all_repos(repos_file: Path, output_dir: Path) -> None:
    """Clone all GitHub repositories from a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(repos_file, 'r', encoding='utf-8') as f:
        repos: Dict[str, str] = json.load(f)

    for url, name in repos.items():
        repo_path = output_dir / name
        if repo_path.exists():
            print(f"[SKIP] Repo already exists: {repo_path}")
        else:
            print(f"[CLONE] Cloning {url} into {repo_path}...")
            try:
                subprocess.run(["git", "clone", url, str(repo_path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to clone {url}: {e}")
