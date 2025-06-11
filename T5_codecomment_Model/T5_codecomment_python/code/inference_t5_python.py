# File: code/inference_t5_python.py

import os
import json
import argparse
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def load_model(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def generate_comments(model, tokenizer, device, test_json_path, output_path):
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []
    batch_size = 16

    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating comments"):
        batch = test_data[i:i+batch_size]
        code_snippets = [item['code'] for item in batch]
        inputs = tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                forced_bos_token_id=tokenizer.bos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, pred in enumerate(decoded_preds):
            results.append({
                "id": i + j,
                "code_snippet": code_snippets[j],
                "human_comment": batch[j].get("comment", ""),
                "model_generated_comment": pred
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n Generated comments saved to: {output_path}")

def save_metadata(output_dir, model_dir, test_file):
    metadata = {
        "Model Evaluated": "T5 Python Only",
        "Model Path": model_dir,
        "Test File": test_file
    }
    metadata_path = os.path.join(output_dir, "t5_python_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f" Metadata saved to: {metadata_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comments using T5 fine-tuned on Python-only data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to T5 fine-tuned model")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to tokenized test JSON file")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Directory to save generated output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tokenizer, model, device = load_model(args.model_dir)

    os.makedirs(args.output_base_dir, exist_ok=True)
    output_path = os.path.join(args.output_base_dir, "t5_python_generated_comments.json")

    generate_comments(model, tokenizer, device, args.test_dataset_path, output_path)
    save_metadata(args.output_base_dir, args.model_dir, args.test_dataset_path)
