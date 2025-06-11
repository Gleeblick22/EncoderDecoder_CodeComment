
#File: Code/modeltrain_bart_python.py

import os
import argparse
import time
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import load_from_disk
import torch
from torch.nn.utils import clip_grad_norm_
# === [1] Custom Trainer with gradient norm logging ===
class GradNormTrainer(Trainer):
    def training_step(self, model, inputs, unused=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        if self.state.global_step % self.args.logging_steps == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            self.log({"grad_norm": grad_norm})

        return loss.detach()

# === [2] Training Function ===
def train_bart_model(train_dataset_path, valid_dataset_path, output_base_dir, model_checkpoint='facebook/bart-base', seed=42):
    set_seed(seed)

    train_dataset = load_from_disk(train_dataset_path)
    valid_dataset = load_from_disk(valid_dataset_path)

    tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

    model_output_dir = os.path.join(output_base_dir, 'models', 'BART-finetune_dmodel')
    checkpoint_dir = os.path.join(output_base_dir, 'models', 'BART-training_checkpoints')
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        logging_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1.5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        num_train_epochs=6,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        logging_dir=os.path.join(checkpoint_dir, 'logs'),
        report_to="none",
        max_grad_norm=1.0 
    )

    trainer = GradNormTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer
    )

    start = time.time()
    trainer.train()
    end = time.time()

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f"\nTraining finished in {(end - start)/60:.2f} minutes.")
    print(f"Model saved to: {model_output_dir}")
    print(f"Checkpoints stored in: {checkpoint_dir}")

# === [3] Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BART model on Python-only dataset")
    parser.add_argument('--train_dataset_path', type=str, required=True, help='Path to tokenized Python training dataset')
    parser.add_argument('--valid_dataset_path', type=str, required=True, help='Path to tokenized Python validation dataset')
    parser.add_argument('--output_base_dir', type=str, required=True, help='Directory to save fine-tuned model and logs')
    return parser.parse_args()

# === [4] Entry ===
if __name__ == "__main__":
    args = parse_args()
    train_bart_model(
        train_dataset_path=args.train_dataset_path,
        valid_dataset_path=args.valid_dataset_path,
        output_base_dir=args.output_base_dir
    )
