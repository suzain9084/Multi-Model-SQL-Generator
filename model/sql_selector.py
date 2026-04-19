from __future__ import annotations
from typing import Iterable, List, Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

class DebertaSqlSelector:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        checkpoint_path: Optional[str] = None,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name = checkpoint_path or model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_query_text(
        self,
        question: str,
        schema: str,
        evidence: Optional[str] = None,
    ) -> str:
        parts = [
            f"Question: {question or ''}",
            f"Schema: {schema or ''}",
        ]
        if evidence:
            parts.append(f"Evidence: {evidence}")
        return "\n".join(parts)

    def _score_candidates(self, query_text, candidates):
        candidate_list = list(candidates)
        if not candidate_list:
            return []

        encoded = self.tokenizer(
            [query_text] * len(candidate_list),
            candidate_list,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probabilities = torch.softmax(logits, dim=-1)[:, 1]
        return probabilities.detach().cpu().tolist()

    def predict(
        self,
        question: str,
        schema: str,
        candidates: List[str],
        evidence: Optional[str] = None,
    ) -> str:
        if not candidates:
            return ""

        unique_candidates = list(dict.fromkeys(candidates))
        query_text = self._build_query_text(question, schema, evidence=evidence)
        scores = self._score_candidates(query_text, unique_candidates)

        if not scores:
            return unique_candidates[0]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return unique_candidates[best_idx]

    def fine_tune(
        self,
        train_dataset,
        output_dir: str = "./selector_model",
        epochs: int = 3,
        per_device_train_batch_size: int = 8,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ) -> None:
        required_columns = {"question", "schema", "candidate_sql", "label"}
        missing = required_columns.difference(set(train_dataset.column_names))
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Selector dataset is missing required columns: {missing_str}"
            )

        def preprocess(example):
            query_text = self._build_query_text(
                question=example["question"],
                schema=example["schema"],
                evidence=example.get("evidence"),
            )
            tokenized = self.tokenizer(
                query_text,
                example["candidate_sql"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            tokenized["labels"] = int(example["label"])
            return tokenized

        tokenized_dataset = train_dataset.map(preprocess)

        keep_columns = {"input_ids", "attention_mask", "labels"}
        remove_columns = [
            c for c in tokenized_dataset.column_names if c not in keep_columns
        ]
        if remove_columns:
            tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)

        if not hasattr(self.model, "peft_config"):
            target_modules = lora_target_modules or [
                "query_proj", "key_proj", "value_proj",
                "pos_key_proj", "pos_query_proj"
            ]
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.train()

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=epochs,
            learning_rate=2e-5,
            logging_steps=50,
            save_steps=200,
            fp16=torch.cuda.is_available(),
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()
        self.model.eval()

    def save(self, path: str = "./selector_model") -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
