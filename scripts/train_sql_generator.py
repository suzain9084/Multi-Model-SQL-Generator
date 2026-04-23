from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from functools import lru_cache
from dataset.dataset import FilteredSchemaDataset
from model.sql_generator import CausalSQLModel
from model.sql_selector import DebertaSqlSelector
from sqlglot import parse
from service.multiple_sql_generator_pipeline.multiple_sql_generator import (
    MultipleSqlGenerator,
)
from service.sql_execution.sql_execution import SqlExecutionService
from service.sql_selector.sql_selector import SqlSelector

try:
    from datasets import Dataset as HFDataset
except Exception:
    HFDataset = None

def main():
    config_path = Path("config/train_sql_generator.json")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    json_path = config["json_path"]
    parse_dialect = config.get("parse_dialect", "sqlite")
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 4))
    save_root = Path(config.get("save_root", "checkpoints"))
    selector_epochs = int(config.get("selector_epochs", epochs))
    selector_batch_size = int(config.get("selector_batch_size", 8))
    selector_output_dir = Path(config.get("selector_output_dir", save_root / "sql_selector"))
    max_examples = config.get("max_examples")
    max_schemas_per_example = config.get("max_schemas_per_example")
    run_execution = bool(config.get("run_execution", True))
    train_selector = bool(config.get("train_selector", True))
    use_gold_targets = bool(config.get("use_gold_targets", False))
    enable_multitask_training_records = bool(config.get("enable_multitask_training_records", False))
    evidence_negative_candidates = int(config.get("evidence_negative_candidates", 3))
    task_weights = config.get(
        "task_weights",
        {
            "text_to_sql": 1.0,
            "question_inference": 1.0,
            "evidence_inference": 1.0,
            "self_refine": 1.0,
        },
    )

    models: List[CausalSQLModel] = []
    output_dirs: List[Path] = []
    for model_cfg in config.get("models", []):
        model_name = model_cfg["model_name"]
        finetune_type = model_cfg.get("finetune_type", "lora")
        output_dir = Path(model_cfg.get("output_dir", save_root / model_name.split("/")[-1]))
        output_dirs.append(output_dir)
        models.append(CausalSQLModel(model_name=model_name, finetune_type=finetune_type))

    if not models:
        raise ValueError("No models configured in config/train_sql_generator.json")

    multi_sql_generator = MultipleSqlGenerator(models=models, parse_dialect=parse_dialect)
    dataset = FilteredSchemaDataset(json_path=json_path)
    sql_selector = None
    if not use_gold_targets:
        selector_model = DebertaSqlSelector(
            model_name=config.get("selector_model_name", "microsoft/deberta-v3-small"),
            checkpoint_path=config.get("selector_checkpoint_path"),
            max_length=int(config.get("selector_max_length", 512)),
        )
        sql_selector = SqlSelector(selection_model=selector_model, parse_dialect=parse_dialect)
    training_records: List[Dict[str, Any]] = []
    selector_training_records: List[Dict[str, Any]] = []
    rng = random.Random(int(config.get("seed", 42)))

    @lru_cache(maxsize=20000)
    def _canonical_sql_cached(text: str) -> str:
        if not text:
            return ""
        try:
            exprs = parse(text, dialect=parse_dialect)
            parts = [
                expr.sql(dialect=parse_dialect)
                for expr in exprs
                if expr is not None
            ]
            return " | ".join(parts) if parts else text
        except Exception:
            return text

    def _canonical_sql(sql: str) -> str:
        return _canonical_sql_cached((sql or "").strip())

    total = len(dataset)
    evidence_pool: List[str] = []
    if enable_multitask_training_records:
        seen_evidence = set()
        for idx in range(total):
            ex, _ = dataset[idx]
            evidence_text = (getattr(ex, "evidence", None) or "").strip()
            if evidence_text and evidence_text not in seen_evidence:
                seen_evidence.add(evidence_text)
                evidence_pool.append(evidence_text)

    for start_idx in range(0, total, batch_size):
        batch = [dataset[idx] for idx in range(start_idx, min(start_idx + batch_size, total))]

        for example, gold_sql in batch:
            schema_texts = [str(schema) for schema in example.schemas]
            if max_schemas_per_example is not None:
                schema_texts = schema_texts[: int(max_schemas_per_example)]
            schema_joined = "\n".join(schema_texts)
            evidence_text = (example.evidence or "").strip() if getattr(example, "evidence", None) else None

            if use_gold_targets:
                if enable_multitask_training_records:
                    text_to_sql_record = {
                        "task_type": "text_to_sql",
                        "question": example.question,
                        "schema": schema_joined,
                        "sql": gold_sql,
                    }
                    if evidence_text:
                        text_to_sql_record["evidence"] = evidence_text
                    training_records.append(text_to_sql_record)

                    question_inference_record = {
                        "task_type": "question_inference",
                        "sql": gold_sql,
                        "schema": schema_joined,
                        "question": example.question,
                    }
                    if evidence_text:
                        question_inference_record["evidence"] = evidence_text
                    training_records.append(question_inference_record)

                    if evidence_text:
                        negative_pool = [ev for ev in evidence_pool if ev != evidence_text]
                        negative_count = min(max(0, evidence_negative_candidates), len(negative_pool))
                        negatives = rng.sample(negative_pool, negative_count) if negative_count > 0 else []
                        evidence_candidates = [evidence_text] + negatives
                        rng.shuffle(evidence_candidates)
                        training_records.append(
                            {
                                "task_type": "evidence_inference",
                                "question": example.question,
                                "schema": schema_joined,
                                "sql": gold_sql,
                                "evidence_candidates": evidence_candidates,
                                "correct_evidence": evidence_text,
                            }
                        )
                else:
                    training_records.append(
                        {
                            "question": example.question,
                            "schema": schema_joined,
                            "sql": gold_sql,
                        }
                    )
                continue

            sql_results = multi_sql_generator.generateMultipleSql(
                schemas=schema_texts,
                question=example.question,
                evidence=example.evidence,
                sqlfile_path=example.sqlfile_path if run_execution else None,
            )

            if train_selector:
                selected_sql = sql_selector.select(
                    sql_results=sql_results,
                    question=example.question,
                    schemas=schema_texts,
                    evidence=example.evidence,
                    isExecuted=run_execution,
                )
            else:
                selected_sql = (sql_results[0][0] if sql_results else "").strip() or gold_sql

            if train_selector:
                canonical_gold = _canonical_sql(gold_sql)
                labeled_candidates: List[Dict[str, Any]] = []

                for candidate_sql, _ in sql_results:
                    candidate_text = (candidate_sql or "").strip()
                    if not candidate_text:
                        continue
                    label = int(_canonical_sql(candidate_text) == canonical_gold)
                    labeled_candidates.append(
                        {
                            "question": example.question,
                            "schema": schema_joined,
                            "evidence": example.evidence,
                            "candidate_sql": candidate_text,
                            "label": label,
                        }
                    )

                if labeled_candidates and not any(item["label"] == 1 for item in labeled_candidates):
                    selected_canonical = _canonical_sql(selected_sql)
                    for item in labeled_candidates:
                        if _canonical_sql(item["candidate_sql"]) == selected_canonical:
                            item["label"] = 1
                            break

                selector_training_records.extend(labeled_candidates)

            target_sql = (selected_sql or "").strip() or gold_sql
            canonical_gold = _canonical_sql(gold_sql)

            if enable_multitask_training_records:
                text_to_sql_record = {
                    "task_type": "text_to_sql",
                    "question": example.question,
                    "schema": schema_joined,
                    "sql": target_sql,
                }
                if evidence_text:
                    text_to_sql_record["evidence"] = evidence_text
                training_records.append(text_to_sql_record)

                question_inference_record = {
                    "task_type": "question_inference",
                    "sql": gold_sql,
                    "schema": schema_joined,
                    "question": example.question,
                }
                if evidence_text:
                    question_inference_record["evidence"] = evidence_text
                training_records.append(question_inference_record)

                if evidence_text:
                    negative_pool = [ev for ev in evidence_pool if ev != evidence_text]
                    negative_count = min(max(0, evidence_negative_candidates), len(negative_pool))
                    negatives = rng.sample(negative_pool, negative_count) if negative_count > 0 else []
                    evidence_candidates = [evidence_text] + negatives
                    rng.shuffle(evidence_candidates)
                    training_records.append(
                        {
                            "task_type": "evidence_inference",
                            "question": example.question,
                            "schema": schema_joined,
                            "sql": gold_sql,
                            "evidence_candidates": evidence_candidates,
                            "correct_evidence": evidence_text,
                        }
                    )

                for candidate_sql, result in sql_results:
                    candidate_text = (candidate_sql or "").strip()
                    if not candidate_text:
                        continue
                    if _canonical_sql(candidate_text) == canonical_gold:
                        continue
                    self_refine_record = {
                        "task_type": "self_refine",
                        "question": example.question,
                        "schema": schema_joined,
                        "previous_sql": candidate_text,
                        "execution_result": str(result),
                        "sql": gold_sql,
                    }
                    if evidence_text:
                        self_refine_record["evidence"] = evidence_text
                    training_records.append(self_refine_record)
            else:
                training_records.append(
                    {
                        "question": example.question,
                        "schema": schema_joined,
                        "sql": target_sql,
                    }
                )
        print(f"\nProgress: {min(start_idx + batch_size, total)}/{total}")

    if not training_records:
        raise ValueError("No valid training records were built from dataset.")
    if train_selector and not selector_training_records:
        raise ValueError("No valid selector training records were built from dataset.")

    if HFDataset is None:
        raise ImportError(
            "Package 'datasets' is required for fine-tuning. "
            "Install with: pip install datasets"
        )

    hf_train_dataset = HFDataset.from_list(training_records)
    hf_selector_dataset = HFDataset.from_list(selector_training_records) if train_selector else None

    if train_selector and hf_selector_dataset is not None:
        selector_output_dir.mkdir(parents=True, exist_ok=True)
        selector_model.fine_tune(
            hf_selector_dataset,
            output_dir=str(selector_output_dir),
            epochs=selector_epochs,
            per_device_train_batch_size=selector_batch_size,
        )
        selector_model.save(str(selector_output_dir))
        print(f"Saved selector model to: {selector_output_dir}")

    for model, out_dir in zip(models, output_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        model.fine_tune(
            hf_train_dataset,
            output_dir=str(out_dir),
            epochs=epochs,
            task_weights=task_weights,
        )
        model.save(str(out_dir))
        print(f"Saved model to: {out_dir}")


if __name__ == "__main__":
    print("START MODEL TRAINING")
    main()
