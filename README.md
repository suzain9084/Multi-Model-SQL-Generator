# SQL Query Generator — Project Report

This repository implements a **text-to-SQL research pipeline** inspired by the **XiYan-SQL** multi-generator framework. It combines **schema filtering** (to shrink and diversify schema context), **sequence-to-sequence SQL generation** with **LoRA fine-tuning**, **syntax-aware refinement** via **sqlglot**, and a **clustering-based SQL selection** stage. Training and evaluation-style data are aligned with the **BIRD** benchmark-style format.

This document functions as both **repository documentation** and a **concise technical report**, including a **literature review** that situates the design choices in published work.

---

## Project snapshot (for external reviewers / LLM report generation)

- **Current core model code:** `model/sql_generator.py` (`CausalSQLModel`, LoRA/QLoRA on causal backbones such as Qwen/DeepSeek).
- **Schema filtering pipeline:** `service/schema_filter/` (keyword extraction, column selection, multi-path retrieval).
- **Candidate generation & selection:** `service/multiple_sql_generator_pipeline/multiple_sql_generator.py` and `service/sql_selector/sql_selector.py`.
- **Execution-based validation:** `service/sql_execution/sql_execution.py` (SQLite execution + result signatures).
- **Primary scripts:** `scripts/multiple_schema_pipeline_run.py`, `scripts/train_sql_generator.py`, `scripts/ablation_evaluation.py`.
- **Evaluation outputs:** `data/ablation/` (CSV/MD metrics), `plot/ablation/` (figures), `plot/schema_filter/` (schema-filter analysis plots).

### Scope and status (important)

- This is a **research/engineering prototype**, not a production-ready text-to-SQL platform.
- The repo includes both **real evaluation mode** and **synthetic simulation mode** in `scripts/ablation_evaluation.py`.
- BIRD data files are expected to be present locally (not necessarily committed to this repository).
- Report readers should treat results as **reproducible locally with correct data paths/config**, not as official leaderboard submissions.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Literature review](#2-literature-review)
3. [Problem statement and contributions](#3-problem-statement-and-contributions)
4. [System architecture](#4-system-architecture)
5. [Module reference](#5-module-reference)
6. [Data artifacts and training](#6-data-artifacts-and-training)
7. [Algorithms and design rationale](#7-algorithms-and-design-rationale)
8. [Installation and usage](#8-installation-and-usage)
9. [Limitations and future work](#9-limitations-and-future-work)
10. [References](#10-references)

---

## 1. Executive summary

**Task.** *Text-to-SQL* maps a natural-language question (and optional evidence) plus a database schema to an executable SQL query. Industrial and research systems must cope with **large schemas**, **ambiguous wording**, and **domain-specific values** (especially in realistic benchmarks).

**Approach in this repo.**

1. **Schema filter** (`service/schema_filter/`): Parse and normalize the schema; extract keywords; score and select columns iteratively; run **multi-path retrieval** to emit **several compact schema views** (paths) per question instead of one monolithic schema string.
2. **SQL generation** (`model/sql_generator.py`, `service/multiple_sql_generator_pipeline/`): One or more **causal decoder backbones** (for example Qwen/DeepSeek) generate SQL for each **(question, schema path)** pair. Optional **sqlglot** parsing checks catch syntax errors; a **refine** prompt asks the model to fix invalid SQL.
3. **SQL selection** (`service/sql_selector/`): Candidates are **clustered** by execution/syntax outcome (when available) or error type; a **majority / shortest-SQL heuristic** reduces the list; a **selection model** chooses the final query from reorganized candidates.
4. **Training** (`scripts/train_sql_generator.py`): Gold SQL from filtered JSON is paired with **every** schema path string to form supervised **input_text → target_text** examples; **LoRA** adapts each backbone efficiently.

**Primary influences.** The overall story follows **XiYan-SQL** (multi-generator + schema filtering). The dataset framing follows **BIRD** (evidence, realistic SQL). Parameter-efficient training follows **LoRA** for large pretrained seq2seq models.

---

## 2. Literature review

### 2.1 Text-to-SQL: task and modeling trends

Text-to-SQL is a structured semantic parsing problem: the model must produce a query that correctly reflects the intent over a **relational schema** without executing against hidden test data at training time (in benchmark settings). Early neural systems used sequence-to-sequence or sequence-to-tree decoders over linearized schemas; later work incorporated **graph encoders**, **schema linking**, and **pretrained language models** (BERT, T5, etc.) to improve alignment between question tokens and schema elements.

**Implication for this project:** A **seq2seq T5-style** backbone is a well-established baseline: the schema and question are concatenated (or templated) into a single input, and the SQL string is the target. This matches common “text-to-text” formulations used in public checkpoints such as T5-small variants fine-tuned for SQL.

### 2.2 Benchmarks: Spider and BIRD

- **Spider** (Yu et al.) popularized cross-domain text-to-SQL with a focus on compositional SQL and many databases. It remains a reference for **execution accuracy** and **exact match** metrics.
- **BIRD** (Li et al.) stresses **real-world** noise: **evidence** text, **dirty values**, and harder questions requiring reasoning over data values, not only schema structure.

**Implication for this project:** The batch pipeline loads **`xu3kev/BIRD-SQL-data-train`** via Hugging Face `datasets`, using fields such as `question`, `db_schema`, `evidence`, and gold `SQL`. The schema filter therefore must treat **evidence** as first-class context alongside the question—consistent with BIRD’s design.

### 2.3 Schema encoding, linking, and filtering

A central difficulty in text-to-SQL is **schema linking**: associating phrases in the question with the correct tables and columns (and sometimes values). Feeding the **entire** schema into the model can **dilute attention** and increase hallucinated columns. A complementary strategy is to **retrieve or select** a subgraph or subset of columns before generation.

**XiYan-SQL** (see §2.5) explicitly introduces a **Schema Filter** module to produce **multiple filtered schema candidates** for downstream **multi-generator** decoding. This repository’s `service/schema_filter` implements that **conceptual role**: keyword extraction, column scoring, iterative selection, and **multi-path** schema sets.

### 2.4 Multi-generator and reranking / selection ensembles

Beyond a single decode, systems often:

- Sample or beam-search multiple SQL hypotheses.
- Use **execution feedback** (success/failure, result consistency) to **rerank** or **vote**.
- Train a **selector** or **reranker** conditioned on the question and candidates.

XiYan-SQL frames **multiple generators** plus selection as a first-class architecture. Here, **multiple generators** are realized as **multiple backbone checkpoints** (configurable in training) and **multiple schema paths** (from the filter). **Selection** is realized in `SqlSelector` via **clustering by outcome**, **size-based ordering**, and a **selection_model.predict(...)** API over candidate strings.

### 2.5 XiYan-SQL: multi-generator framework and schema filter

**XiYan-SQL: A Novel Multi-Generator Framework for Text-to-SQL** (arXiv:2507.04701) proposes a pipeline in which **schema filtering** reduces irrelevant schema content and **multiple generators** explore complementary hypotheses before a selection stage. **Section III-B** discusses the schema filter: keywords from question and evidence, retrieval over schema elements, and **multiple structured views** of the schema for downstream modules.

**Relation to this codebase:** The implementation is **research-style** and **not** a full reproduction of every XiYan-SQL component (e.g., industrial-scale LLM calls, full execution-guided training, or all generator types may differ). The **design debt** is intentional: proxies (heuristics, embedding-based retrieval) stand in where the paper uses general LLM subroutines. The **architectural alignment** is: **filter → generate many → select/refine**.

### 2.6 Parameter-efficient fine-tuning (LoRA)

**LoRA** (Low-Rank Adaptation) injects trainable low-rank matrices into attention projections (and optionally other layers) while freezing most of the pretrained weights. For seq2seq text-to-SQL, this enables **domain adaptation** on BIRD-style pairs without full-model fine-tuning cost.

**In this repo:** `CausalSQLModel.fine_tune` uses Hugging Face `peft` with `TaskType.CAUSAL_LM` and LoRA targets **`q_proj`, `k_proj`, `v_proj`, `o_proj`**.

### 2.7 SQL validation and canonicalization

**sqlglot** provides a **parser** and **SQL transpilation** surface. Parsing does not guarantee semantic correctness against a live database, but it **filters malformed strings** and supports **canonical forms** for grouping similar queries.

**In this repo:** `MultipleSqlGenerator.check_sql_syntax` uses `sqlglot.parse` with a configurable **dialect** (default aligned with `sqlite` in config). `SqlSelector._canonical_sql` uses parsing to cluster **syntactically equivalent** formulations when possible.

---

## 3. Problem statement and contributions

### 3.1 Problem

Given **(question, evidence, full schema)**, produce SQL that answers the question. Challenges:

- **Scale:** Large schemas exceed practical context limits or overwhelm small models.
- **Noise:** Irrelevant tables/columns steer generation wrong.
- **Ambiguity:** Multiple valid SQL strings; benchmarks expect a specific gold or equivalent execution.
- **Reliability:** Single-pass decoding often yields syntax errors.

### 3.2 What this repository contributes (engineering / research artifacts)

- A **working schema-filter pipeline** with **multi-path** outputs persisted to JSON.
- A **training script** that expands **(question, multiple schema paths) → gold SQL** into a Hugging Face `Dataset` and runs **LoRA/QLoRA** for one or more causal backbones.
- A **multi-SQL generator** with **parse-check** and **refinement loop** scaffolding.
- A **selector** that combines **outcome clustering**, **majority heuristics**, and a **pluggable selection model**.

---

## 4. System architecture

### 4.1 Repository layout (actual)

```text
.
├── config/
│   └── train_sql_generator.json
├── dataset/
│   ├── bird_dataset.py                 # BIRD train/dev loader + schema index helpers
│   └── dataset.py                      # FilteredSchemaDataset for schema-filter outputs
├── model/
│   ├── sql_generator.py                # CausalSQLModel (LoRA / QLoRA)
│   └── sql_selector.py                 # DebertaSqlSelector
├── scripts/
│   ├── multiple_schema_pipeline_run.py # run schema filtering over BIRD-style JSON
│   ├── train_sql_generator.py          # generator + optional selector training
│   ├── ablation_evaluation.py          # real BIRD dev ablation + synthetic mode
│   └── ablation_evaluation_runs.py
├── service/
│   ├── schema_filter/
│   ├── multiple_sql_generator_pipeline/
│   │   └── multiple_sql_generator.py
│   ├── sql_selector/
│   │   └── sql_selector.py
│   └── sql_execution/
│       └── sql_execution.py            # execution service on SQLite DB files
├── plot/
│   ├── ablation/
│   └── schema_filter/
├── data/
│   └── ablation/
└── requirements.txt
```

### 4.2 End-to-end data flow

```text
BIRD-style JSON + schemas + sqlite DBs
    → SchemaFilterPipeline.process_sample(...)
        → keywords, selected columns, values map, multiple schema path structures
    → save_results → schema_filter_results.json
        → FilteredSchemaDataset
        → HF Dataset records for CausalSQLModel fine-tuning
        → LoRA/QLoRA fine-tune each configured backbone
```

At inference time (conceptually): **filter → for each model × each schema path → SQL → parse/refine → cluster → select**.

---

## 5. Module reference

### 5.1 `service/schema_filter/pipeline.py` — `SchemaFilterPipeline`

**Responsibilities:**

- **Keyword extraction** from question + evidence (`KeywordExtractor`).
- **Schema parsing** to an internal dict (`SchemaParser`).
- **Iterative column selection** (`ColumnSelector`) producing scored column candidates.
- **Value retrieval** hints keyed by `table.column` (`ValueRetriever`).
- **Multi-path retrieval** (`MultiPathSchemaRetriever`) producing `ps` schema sets (default `num_schemas`).
- **Persistence** via `save_results`: normalizes schema sets into JSON-friendly `{ "tables": [ { "name", "columns" } ] }` lists and attaches `actual_result` when present for supervised training.

**Constructor parameters (typical):** `top_k_retrieval`, `num_schemas`.

### 5.2 `service/schema_filter/multi_path_retriever.py` — `MultiPathSchemaRetriever`

Implements iterative rounds: score and take top-`k` column tuples, expand with **primary/foreign keys**, accumulate growing schema sets, and **remove** consumed columns from the candidate pool (except key-driven expansion). This realizes a **discrete** analogue of “multiple schema paths” for downstream generators.

### 5.3 `model/sql_generator.py` — `CausalSQLModel`

- **Prompt format:** `question: {question} table: {schema_string}`.
- **Inference:** Hugging Face `text-generation` pipeline, sampling with temperature.
- **Training:** Prompt+target causal training with optional multi-task records; wraps base model with **LoRA/QLoRA** and trains with `Trainer` + `TrainingArguments`.

### 5.4 `service/multiple_sql_generator_pipeline/multiple_sql_generator.py` — `MultipleSqlGenerator`

- **`generateMultipleSql`:** Nested loops over **models** and **schema paths**; optional **sqlglot** syntax check; **refine** on parse failure with error text in the prompt.
- **`fine_tune_all_backbones`:** Trains each model with its own `output_dir`.

### 5.5 `service/sql_selector/sql_selector.py` — `SqlSelector`

- **Clustering key:** Success vs failure vs syntax error string vs unchecked, with **canonical SQL** when parse succeeds.
- **Majority logic:** If the largest cluster has ≥ ⌈N/2⌉ candidates, flatten clusters in sorted order; else take **shortest SQL per cluster**.
- **Final choice:** `selection_model.predict(question, schema_union, evidence, candidates=...)`.

### 5.6 `dataset/dataset.py` — `FilteredSchemaDataset`

Reads the JSON list produced by the filter; each record yields `SchemaFilterExample` + gold string from `actual_result` / `SQL` / `gold_sql`. Skips items with empty `schemas`.

---

## 6. Data artifacts and training

### 6.1 `schema_filter_results.json`

Written by `SchemaFilterPipeline.save_results`. Each element includes at least: `question_id`, `question`, `evidence`, `keywords`, `values`, `schemas` (list of table/column structures), `statistics`, and **`actual_result`** when the batch script supplied gold SQL.

### 6.2 Training expansion

For each filtered example, **every** schema path string is paired with the **same** gold SQL (`scripts/train_sql_generator.py`, `_build_hf_dataset`). This teaches the model that **multiple schema views** can map to one correct query—aligning with multi-path supervision ideas.

### 6.3 `config/train_sql_generator.json`

Example fields:

- `json_path`: source filtered JSON (default `schema_filter_results.json`).
- `epochs`: training epochs per backbone.
- `parse_dialect`: e.g. `sqlite` (should match BIRD SQL dialect and evaluator settings).
- `models`: list of `{ "model_name", "finetune_type", "output_dir" }` entries (for example Qwen/DeepSeek backbones).

---

## 7. Algorithms and design rationale

### 7.1 Why multi-path schemas?

A single retrieved subgraph can **omit** a necessary join or column. Multiple paths explore **alternative column subsets** under the same scoring machinery, improving recall at the cost of more downstream generations—trading compute for robustness, consistent with multi-candidate decoding literature.

### 7.2 Why LoRA?

Full fine-tuning of T5-small is feasible, but LoRA **reduces memory and storage** (per-backbone adapters in separate `output_dir`s) and isolates experiments across multiple checkpoints.

### 7.3 Why sqlglot in the loop?

Syntax errors are common in small seq2seq models. Parsing provides a **cheap filter** before execution; the **refine** prompt supplies actionable feedback. This echoes **analyze-then-repair** patterns in program synthesis.

### 7.4 Why cluster before selection?

Execution or syntax outcomes induce **equivalence classes** over candidates. Sorting clusters by frequency approximates **majority consensus** when an external executor is available; otherwise, clusters still separate **hard failures** from **plausible** strings.

---

## 8. Installation and usage

### 8.1 Dependencies

```bash
pip install -r requirements.txt
```

Optional (recommended for keyword extraction):

```bash
python -m spacy download en_core_web_sm
```

Key libraries: `torch`, `transformers`, `peft`, `datasets`, `sentence-transformers`, `sqlglot`, `spacy`, `numpy`, `scikit-learn`, `huggingface-hub`.

### 8.2 Download / prepare BIRD data (required)

Place BIRD data locally so the project can read JSON and SQLite files. One workable layout is:

```text
dataset/
├── train/
│   ├── train.json
│   ├── train_tables.json
│   └── train_databases/
│       └── train_databases/
│           └── <db_id>/<db_id>.sqlite
└── dev/
    ├── dev.json
    ├── dev_tables.json
    └── dev_databases/
        └── dev_databases/
            └── <db_id>/<db_id>.sqlite
```

Download from the official [BIRD benchmark site](https://bird-bench.github.io/) and keep filenames consistent with script arguments.

### 8.2.1 Quick reproducibility checklist

Before running scripts, confirm:

- Python environment has `requirements.txt` installed.
- BIRD train/dev JSON + SQLite files exist at the paths passed to scripts.
- `config/train_sql_generator.json` has valid model names and output directories.
- `parse_dialect` in config and evaluation command are aligned (typically `sqlite`).
- GPU availability is sufficient for selected backbones (or use smaller checkpoints).

### 8.3 Run schema filtering on BIRD train split

From the repository root:

```bash
python scripts/multiple_schema_pipeline_run.py \
  --data-root dataset \
  --output-file schema_filter_results_2.json
```

This runs `SchemaFilterPipeline` and writes filtered training records.

> **Note:** Full-dataset runs are long and produce a **large** JSON file. For development, consider temporarily limiting samples in the script.

### 8.4 Train SQL generators (LoRA / QLoRA)

After filtered JSON exists:

```bash
python scripts/train_sql_generator.py
```

Edit `config/train_sql_generator.json` for JSON path, dialect, selector options, and output checkpoints.

### 8.5 End-to-end inference/evaluation on BIRD dev

Run real ablation/evaluation metrics (EX, valid SQL rate, schema-filter recall):

```bash
python scripts/ablation_evaluation.py \
  --mode real \
  --config config/train_sql_generator.json \
  --dev-json dataset/dev/dev.json \
  --dev-tables dataset/dev/dev_tables.json \
  --sqlite-root dataset/dev/dev_databases/dev_databases \
  --data-dir data/ablation
```

This produces:
- `data/ablation/pipeline_ablation_real_metrics.csv`
- `data/ablation/pipeline_ablation_real_metrics.md`

You can still run synthetic plots (kept for simulation/debug baselines):

```bash
python scripts/ablation_evaluation.py --mode synthetic
```

### 8.6 Programmatic schema filtering (minimal)

```python
from service.schema_filter.pipeline import SchemaFilterPipeline

pipeline = SchemaFilterPipeline(top_k_retrieval=20, num_schemas=2)
result = pipeline.process_sample(
    question="...",
    schema_data=bird_schema_dict,
    actual_result="SELECT ...",
    evidence="...",
    question_id="...",
)
```

`actual_result` is required for gold SQL in the saved JSON used by training.

---

## 9. Limitations and future work

1. **Not a full XiYan-SQL reproduction:** LLM-heavy subroutines from the paper are approximated with **heuristics**, **embeddings**, and **keyword models** where noted in code.
2. **Value retrieval:** Value-level grounding may be partial or dataset-dependent; full BIRD-style value grounding often needs **database access** or cached value statistics.
3. **Selection model:** The interface is present; **training** the selector with execution feedback is left as future work.
4. **Evaluation scope:** Scripted evaluation is available for local dev data; official leaderboard reporting still depends on benchmark protocol and submission tooling.
5. **Dialect mismatch:** Gold SQL dialect vs `sqlglot` dialect must be kept consistent to avoid false parse failures.

---

## 10. References

1. XiYan-SQL — *A Novel Multi-Generator Framework for Text-to-SQL*, arXiv:2507.04701. [https://arxiv.org/abs/2507.04701](https://arxiv.org/abs/2507.04701)
2. BIRD benchmark — Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation. [https://bird-bench.github.io/](https://bird-bench.github.io/)
3. T. Yu et al. — Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. (Spider dataset / baseline reference.)
4. J. Li et al. — BIRD: Benchmarking Instruction-following Reasoning for Text-to-SQL with Grounded Execution. (BIRD paper; venue/version per citation style you use.)
5. E. J. Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
6. Hugging Face `datasets` — `xu3kev/BIRD-SQL-data-train` card (mirror / preprocessing of BIRD-style training split).
7. sqlglot — SQL parser and transpiler. [https://github.com/tobymao/sqlglot](https://github.com/tobymao/sqlglot)

---

## License

This implementation is intended for **research and educational** use. Verify dataset and model licenses separately (BIRD, Hugging Face model cards, and your fine-tuned checkpoints).
