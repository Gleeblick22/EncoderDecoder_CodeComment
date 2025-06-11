
#File:data_collection/extract_java.py
import os
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_java_from_repos(repo_dir: Path) -> List[Dict[str, Any]]:
    entries = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".java"):
                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    comment_buffer = []
                    in_javadoc = False
                    in_method = False
                    method_buffer = []
                    brace_level = 0
                    skip_anonymous = False
                    javadoc_only = False

                    for line in lines:
                        stripped = line.strip()

                        if 'new ' in stripped and '{' in stripped:
                            skip_anonymous = True
                        elif skip_anonymous:
                            if '}' in stripped:
                                skip_anonymous = False
                            continue

                        if stripped.startswith("/**"):
                            comment_buffer = []
                            in_javadoc = True
                            javadoc_only = True
                            continue
                        elif in_javadoc:
                            if stripped.endswith("*/"):
                                in_javadoc = False
                                line_clean = stripped.rstrip("*/ ").lstrip("* ")
                                comment_buffer.append(line_clean)
                            else:
                                line_clean = stripped.lstrip("* ")
                                comment_buffer.append(line_clean)
                            continue
                        elif stripped.startswith("//") and not javadoc_only:
                            comment_buffer.append(stripped.lstrip("/ "))
                            continue

                        if re.match(r".*\)\s*\{", stripped) and not in_method:
                            in_method = True
                            method_buffer = [line]
                            brace_level = line.count("{") - line.count("}")
                        elif in_method:
                            method_buffer.append(line)
                            brace_level += line.count("{") - line.count("}")
                            if brace_level <= 0:
                                in_method = False
                                code = "".join(method_buffer).strip()
                                comment = "\n".join(comment_buffer).strip().rstrip("/").strip()

                                name_match = re.search(r'\b(?:[\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)\s*\{', code)
                                method_name = name_match.group(1) if name_match else ""

                                if not method_name:
                                    comment_buffer = []
                                    javadoc_only = False
                                    continue

                                if not comment or all(x.strip().startswith(("@param", "@return", "@formatter", "$example")) for x in comment.splitlines()):
                                    comment_buffer = []
                                    javadoc_only = False
                                    continue

                                if not any(len(line.split()) > 3 and not line.strip().startswith("@") for line in comment.splitlines()):
                                    comment_buffer = []
                                    javadoc_only = False
                                    continue

                                keywords = ("return", "calculate", "check", "determine", "@param", "@return",
                                            "create", "generate", "fetch", "read", "write", "process",
                                            "convert", "parse", "serialize", "build")

                                if (len(code.split()) >= 12 and len(comment.split()) >= 8 and
                                    ("\n" in comment or comment.endswith(".")) and
                                    any(word in comment.lower() for word in keywords)) or \
                                   (len(code.split()) >= 8 and len(comment.split()) >= 5):

                                    entries.append({
                                        "file": str(file_path),
                                        "type": "method",
                                        "name": method_name,
                                        "code": code,
                                        "comment": comment
                                    })

                                comment_buffer = []
                                javadoc_only = False

                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")
    return entries
