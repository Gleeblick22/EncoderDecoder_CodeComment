# File: data_collection/extract_python.py

import ast
import os
from pathlib import Path
from typing import List, Dict, Any

def extract_python_from_repos(repo_dir: Path) -> List[Dict[str, Any]]:
    entries = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        tree = ast.parse(f.read(), filename=str(file_path))

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            docstring = ast.get_docstring(node, clean=True)
                            if not docstring or len(docstring.split()) < 10:
                                continue

                            if not any(word in docstring.lower() for word in ("return", "calculate", "check", "param", "example", "raise", "process")):
                                continue

                            try:
                                code = ast.unparse(node)
                            except Exception:
                                continue

                            if len(code.split()) < 15:
                                continue

                            entries.append({
                                "file": str(file_path),
                                "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                                "name": node.name,
                                "code": code,
                                "comment": docstring.strip()
                            })

                except SyntaxError:
                    print(f"[SKIP] Syntax error in file: {file_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to parse {file_path}: {e}")
    return entries
