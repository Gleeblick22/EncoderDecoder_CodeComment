# Data Collection Module

This module automates the collection of code-comment pairs from open-source Python and Java repositories. 
It clones repositories, extracts documented functions and classes, merges the results, and splits them into train, validation, and test sets.

---

##  Folder Structure

```
data_collection/
├── extract_python.py       # Extracts documented Python code elements
├── extract_java.py         # Extracts documented Java methods
├── repos.json              # JSON list of GitHub repositories to clone
├── split_dataset.py        # Splits data into train/valid/test
├── json_full_dataset.json  # Merged output file of all samples
├── combined/               # Combined Python + Java dataset splits
├── python_only/            # Python-only dataset splits
├── java_only/              # Java-only dataset splits
```

---

##  How to Run

From the root of the repository:

```bash
python -m data_collection --repos_file data_collection/repos.json --output_dir data_collection/
```

This will:
1. Clone repositories listed in `repos.json`
2. Extract code-comment pairs (Python via AST, Java via regex)
3. Merge into a full dataset
4. Split into language-specific train/valid/test datasets

---

##  Output Format

Each dataset file (`train_data.json`, `valid_data.json`, `test_data.json`) contains a list of entries with:

```json
{
  "file": "<source path>",
  "type": "function" or "class",
  "name": "<function/class name>",
  "code": "<full source code>",
  "comment": "<associated docstring or comment>"
}
```

---

##  Requirements

- Python 3.9+
- Standard libraries only: `ast`, `os`, `json`, `re`, `pathlib`, `typing`

---

##  Notes

- Python uses AST parsing with `ast.unparse()` to extract code
- Java uses regular expressions to match methods and associated comments
- Syntax errors in files are skipped and logged

---

##  Output Datasets

| Dataset Type   | Description                                 |
|----------------|---------------------------------------------|
| `combined/`    | All entries (Python + Java)                 |
| `python_only/` | Entries parsed only from Python files       |
| `java_only/`   | Entries parsed only from Java files         |

---

##  Example

```bash
python -m data_collection --repos_file data_collection/repos.json --output_dir data_collection/
```

Outputs:
- `data_collection/json_full_dataset.json`
- `data_collection/combined/train_data.json`
- `data_collection/python_only/test_data.json`
- `data_collection/java_only/valid_data.json`

---

##  License & Credits

- Powered by AST and Regex-based parsing
- Designed for research and ML datasets
