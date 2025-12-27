from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


@dataclass
class LoraRunResult:
    output_dir: str
    epochs: int
    steps: int


def _format_entry(entry: Dict[str, Any]) -> str:
    # Preserve CoT and JSON for trace-preserving fine-tune.
    return f"User: Optimize candidate\nAssistant: {entry['response']}\nJSON: {entry['json_payload']}"


def train_online_lora(
    entries: List[Dict[str, Any]],
    model_name: str = "Nanbeige4-3B-Thinking",
    output_dir: str = "results/llm/nanbeige-lora",
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-4,
) -> LoraRunResult:
    if not entries:
        raise ValueError("No entries provided for online LoRA.")

    texts = [_format_entry(e) for e in entries]
    ds = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="longest", max_length=512)

    tokenized = ds.map(tokenize, batched=True)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    steps = len(tokenized) * epochs // batch_size
    return LoraRunResult(output_dir=output_dir, epochs=epochs, steps=steps)


