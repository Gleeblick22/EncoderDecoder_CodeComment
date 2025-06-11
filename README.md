<<<<<<< HEAD
# EncoderDecoder_CodeComment
"Empirical Evaluation of T5 and BART for Automated Code Comment Generation on Python and Java"
=======
# 📘 Automated Code Comment Generation Using Large Language Models  
**Empirical Evaluation of T5 and BART for Python and Java**

---

## 🧠 Abstract

Source code documentation is critical to software maintenance and comprehension. Yet, it is often under-prioritized in real-world projects. This research investigates the automatic generation of source code comments using two powerful Large Language Models (LLMs): **T5** (by Google) and **BART** (by Meta AI).

We collected code-comment datasets from open-source Python and Java repositories and fine-tuned T5 and BART for generating meaningful code summaries. The models were evaluated using standard NLP metrics: **BLEU**, **Smoothed BLEU**, **ROUGE**, **METEOR**, and **SIDE similarity**.

---

## 📁 Repository Structure

```
EncoderDecoder_CodeComment/
├── README.md
├── .gitignore
├── environment.yaml               # Conda environment file
│
├── data_collection/              # Dataset builder
│   ├── extract_python.py
│   ├── extract_java.py
│   ├── merge_and_clean.py
│   ├── split_dataset.py
│   ├── compare_test_data.py
│   ├── raw_cloned_repos/         # Ignored: cloned GitHub repos
│   ├── combined/
│   ├── python_only/
│   ├── java_only/
│   └── json_full_dataset.json
│
├── T5_codecomment_Model/
│   ├── T5_codecomment_python_java/
│   │   ├── code/
│   │   ├── models/
│   │   ├── Tokenized_data/
│   │   └── evaluation_results/
│   ├── T5_codecomment_python/
│   └── T5_codecomment_java/
│
├── BART_codecomment_Model/
│   ├── BART_codecomment_python_java/
│   │   ├── code/
│   │   ├── models/
│   │   ├── Tokenized_data/
│   │   └── evaluation_results/
│   ├── BART_codecomment_python/
│   └── BART_codecomment_java/
```

---

## 🔧 Environment Setup

```bash
conda env create -f environment.yaml
conda activate codet5_bart
```

Required: Python 3.9+, PyTorch, Transformers, Datasets, SentenceTransformers, Evaluate, Scikit-learn.

---

## 📊 Dataset Preparation

```bash
cd data_collection

# Clone repos, extract Python + Java comments, and split
python extract_python.py
python extract_java.py
python merge_and_clean.py
python split_dataset.py
```

Datasets generated in:
- `data_collection/combined/`
- `data_collection/python_only/`
- `data_collection/java_only/`

---

## 🏗️ Model Pipeline (T5 / BART)

Each model follows this 4-step pipeline:

### 1️⃣ Tokenization  
```bash
cd T5_codecomment_Model/T5_codecomment_python_java/code
python tokenize_t5_dataset_python_java.py
```

### 2️⃣ Model Training  
```bash
python modeltrain_t5_python_java.py   --train_dataset_path "../Tokenized_data/train"   --valid_dataset_path "../Tokenized_data/valid"   --output_base_dir ".."
```

### 3️⃣ Inference  
```bash
python inference_t5_python_java.py   --model_dir "../models/T5_Finetuned_model"   --test_dataset_path "../../../data_collection/combined/test_data.json"   --output_base_dir "../results"
```

### 4️⃣ Evaluation  
```bash
python evaluate_t5_python_java.py   --predictions_path "../results/t5_python_java_generated_comments.json"   --test_path "../../../data_collection/combined/test_data.json"   --output_dir "../results/evaluation"
```

---

## 📈 Evaluation Metrics

| Metric         | Description                                      |
|----------------|--------------------------------------------------|
| **BLEU**       | Measures n-gram precision                        |
| **Smoothed BLEU** | BLEU with smoothing for short sequences     |
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | Measures overlap and sequence coverage |
| **METEOR**     | Weighted harmonic mean of precision and recall  |
| **SIDE**       | Semantic textual similarity using embeddings    |

---

## 🧪 Experiments Conducted

- **Languages**: Python, Java
- **Model Types**: T5-base, BART-base
- **Comparisons**:
  - Python-only
  - Java-only
  - Combined dataset (Python + Java)

---

## 📌 Notable Findings

- BART showed stronger syntactic fluency in short sequences.
- T5 exhibited stronger contextual consistency for longer functions.
- No-repeat n-gram and beam size tuning significantly impacted BLEU and ROUGE scores.

---

## 📁 .gitignore Highlights

```bash
__pycache__/
*.py[cod]
*.log
*.ipynb_checkpoints
*.json
*.csv
.vscode/
.idea/
models/
results/
data_collection/raw_cloned_repos/
```

---

## 📜 License

This repository is for academic research and educational purposes. All datasets are collected from public open-source repositories on GitHub.

---

## 🤝 Acknowledgments

- Google AI – T5
- Meta AI – BART
- Hugging Face Transformers
- GitHub Open-Source Community

---

## 🧩 Future Work

- Fine-tune on code-comment datasets like CodeSearchNet
- Introduce multilingual code models
- Evaluate with human annotators

---

## ⭐ Contribute or Cite

If you use or build on this project, feel free to open an issue or star the repo.
>>>>>>> 5038f93 (Initial commit: Project structure, code, README, environment, and .gitignore)
