# File: code/evaluate_bart_python.py

import os
import argparse
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_bleu(references, predictions):
    refs = [[ref.split()] for ref in references]
    hyps = [pred.split() for pred in predictions]
    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothie)
    return bleu

def compute_smoothed_bleu(references, predictions):
    smoothie = SmoothingFunction().method1
    scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores)

def compute_rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
    total = len(references)
    return rouge1/total, rouge2/total, rougeL/total

def compute_meteor(references, predictions):
    score = 0
    for ref, pred in zip(references, predictions):
        ref_tokens = nltk.word_tokenize(ref)
        pred_tokens = nltk.word_tokenize(pred)
        score += meteor_score([ref_tokens], pred_tokens)
    return score / len(references)

def compute_side(references, predictions):
    vectorizer = TfidfVectorizer().fit(references + predictions)
    ref_vecs = vectorizer.transform(references)
    pred_vecs = vectorizer.transform(predictions)
    similarities = cosine_similarity(ref_vecs, pred_vecs)
    score = similarities.diagonal().mean()
    return score

def main(predictions_file, test_file, output_dir):
    full_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Evaluation'))
    os.makedirs(full_output_dir, exist_ok=True)

    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    references = [item['comment'] for item in test_data]
    predictions = [item['model_generated_comment'] for item in predictions_data]

    bleu = compute_bleu(references, predictions)
    smoothed_bleu = compute_smoothed_bleu(references, predictions)
    rouge1, rouge2, rougeL = compute_rouge(references, predictions)
    meteor = compute_meteor(references, predictions)
    side = compute_side(references, predictions)

    scores = {
        "BLEU": bleu,
        "Smoothed_BLEU": smoothed_bleu,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "METEOR": meteor,
        "SIDE": side
    }

    scores_path = os.path.join(full_output_dir, 'evaluation_bartbase_python_Scores.json')
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=4)

    qualitative_path = os.path.join(full_output_dir, 'Bartbase_python_qualitative_examples.txt')
    with open(qualitative_path, 'w', encoding='utf-8') as f:
        f.write("Model Evaluated: BART-base Python\n")
        f.write("Dataset Evaluated: test_python_data.json\n")
        #f.write(f"Total Samples: {len(references)}\n\n")
        for i in range(min(10, len(references))):
            f.write("======================================================================\n")
            f.write(f"ID: {i}\n")
            f.write(f"Code Snippet: {test_data[i]['code']}\n")
            f.write(f"Human Comment: {references[i]}\n")
            f.write(f"Model Generated Comment: {predictions[i]}\n\n")

    print(f"\nEvaluation scores:")
    for key, value in scores.items():
        print(f"{key}: {value:.4f}")
    print(f"\nScores saved to {scores_path}")
    print(f"Qualitative examples saved to {qualitative_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated comments against true references.")
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to generated comments JSON file.')
    parser.add_argument('--test_file', type=str, required=True, help='Path to original test dataset JSON file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        predictions_file=args.predictions_file,
        test_file=args.test_file,
        output_dir=args.output_dir
    )
