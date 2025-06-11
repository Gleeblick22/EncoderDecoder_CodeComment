# File: code/modeltrain_t5_python_java.py

import os
import argparse
import time
from datasets import load_from_disk
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import torch
from torch.nn.utils import clip_grad_norm_ 

# === [1] Custom Trainer to log gradient norm ===
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

# === [2] Train Function ===
def train_t5_model(train_dataset_path, valid_dataset_path, output_base_dir):
    model_name = 't5-base' 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    train_dataset = load_from_disk(train_dataset_path)
    valid_dataset = load_from_disk(valid_dataset_path)

    model_output_dir = os.path.join(output_base_dir, "models", "T5_Finetuned_model")
    checkpoints_dir = os.path.join(output_base_dir, "models", "T5_Training_Checkpoints")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=500,
        save_total_limit=2,
        per_device_train_batch_size=4,       
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,       
        learning_rate=1.5e-5,                  
        num_train_epochs=6,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,           
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        logging_dir=os.path.join(checkpoints_dir, "logs"),
        report_to="none",
        max_grad_norm=1.0  
    )

    trainer = GradNormTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    trainer.save_model(model_output_dir)
    print(f"\n Model saved to: {model_output_dir}")
    print(f" Checkpoints saved under: {checkpoints_dir}")
    print(f" Training completed in {(end_time - start_time)/60:.2f} minutes.")

# === [3] CLI Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Train T5-base model on Python+Java dataset.")
    parser.add_argument('--train_dataset_path', type=str, required=True, help="Path to tokenized training dataset")
    parser.add_argument('--valid_dataset_path', type=str, required=True, help="Path to tokenized validation dataset")
    parser.add_argument('--output_base_dir', type=str, required=True, help="Base output directory (project root)")
    return parser.parse_args()

# === [4] Entry Point ===
if __name__ == "__main__":
    args = parse_args()
    train_t5_model(
        train_dataset_path=args.train_dataset_path,
        valid_dataset_path=args.valid_dataset_path,
        output_base_dir=args.output_base_dir
    )
