from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.nas.utils import vector_to_text


@dataclass
class LLMRankerMetrics:
    val_pearson: float
    val_mse: float


def _prepare_dataset(csv_path: str | Path, base_image_size: int = 224) -> Dataset:
    df = pd.read_csv(csv_path)
    texts: List[str] = []
    for _, row in df.iterrows():
        text = vector_to_text(
            cand_type(row["depth_mult"], row["width_mult"], row["res_mult"]), base_image_size
        )
        texts.append(text)
    ds = Dataset.from_dict({"text": texts, "label": df["val_acc"].astype(float).tolist()})
    return ds


def cand_type(depth: float, width: float, res: float):
    from src.evaluator import Candidate

    return Candidate(depth_mult=float(depth), width_mult=float(width), res_mult=float(res))


def train_llm_ranker(
    csv_path: str | Path,
    *,
    output_dir: str | Path = "results/surrogates/llm_ranker",
    model_name: str = "distilroberta-base",
    num_train_epochs: int = 2,
    batch_size: int = 16,
    lr: float = 3e-5,
    seed: int = 42,
    base_image_size: int = 224,
) -> LLMRankerMetrics:
    ds = _prepare_dataset(csv_path, base_image_size=base_image_size)
    ds = ds.train_test_split(test_size=0.2, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    tokenized = ds.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        val_mse = float(np.mean((preds - labels) ** 2))
        val_pearson = float(np.corrcoef(preds, labels)[0, 1])
        return {"val_mse": val_mse, "val_pearson": val_pearson}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return LLMRankerMetrics(val_pearson=float(metrics.get("eval_val_pearson", 0.0)), val_mse=float(metrics.get("eval_val_mse", 0.0)))


def load_llm_ranker(path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

