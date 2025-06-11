# File: code/evaluate_t5_java.py

import os
import json
import argparse
import evaluate
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

def load_json_comments(path, pred_key, ref_key):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, [item.get(pred_key, "").strip() for item in data], [item.get(ref_key, "").strip() for item in data]

def compute_bleu(references, predictions):
    refs = [[word_tokenize(ref)] for ref in references]
    preds = [word_tokenize(pred) for pred in predictions]
    return corpus_bleu(refs, preds)

def compute_smoothed_bleu(references, predictions):
    smoother = SmoothingFunction().method1
    scores = []
    for ref, pred in zip(references, predictions):
        try:
            scores.append(sentence_bleu([word_tokenize(ref)], word_tokenize(pred), smoothing_function=smoother))
        except:
            scores.append(0)
    return sum(scores) / len(scores)

def compute_side_score(references, predictions, model):
    scores = []
    for ref, pred in zip(references, predictions):
        ref_emb = model.encode(ref, convert_to_tensor=True)
        pred_emb = model.encode(pred, convert_to_tensor=True)
        scores.append(util.pytorch_cos_sim(ref_emb, pred_emb).item())
    return sum(scores) / len(scores)

def main(predictions_path, test_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pred_data, predictions, _ = load_json_comments(predictions_path, "model_generated_comment", "comment")
    test_data, _, references = load_json_comments(test_path, "code", "comment")

    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]
    pred_data = pred_data[:min_len]
    test_data = test_data[:min_len]

    bleu = compute_bleu(references, predictions)
    sm_bleu = compute_smoothed_bleu(references, predictions)

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    rouge1 = rouge_scores["rouge1"]
    rouge2 = rouge_scores["rouge2"]
    rougeL = rouge_scores["rougeL"]

    meteor = evaluate.load("meteor")
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]

    side_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    side_score = compute_side_score(references, predictions, side_model)

    scores = {
        "BLEU": bleu,
        "Smoothed_BLEU": sm_bleu,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "METEOR": meteor_score,
        "SIDE": side_score
    }

    with open(os.path.join(output_dir, "evaluation_t5_java_scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)

    print("\n Evaluation Scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    qual_path = os.path.join(output_dir, "t5_java_qualitative_examples.txt")
    with open(qual_path, 'w', encoding='utf-8') as f:
        f.write("Model: T5-Java Only\n")
        #f.write(f"Total Samples: {min_len}\n")
        f.write("="*70 + "\n\n")
        for i in range(min(10, min_len)):
            f.write(f"ID: {i}\n")
            f.write(f"Code Snippet: {test_data[i].get('code', '')}\n")
            f.write(f"Human Comment: {references[i]}\n")
            f.write(f"Model Generated Comment: {predictions[i]}\n")
            f.write("="*70 + "\n\n")

    print(f"\n Saved qualitative examples to: {qual_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args.predictions_path, args.test_path, args.output_dir)
