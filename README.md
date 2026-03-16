# XiYan-SQL Schema Filter Implementation

This repository implements the **Schema Filter Module** from the XiYan-SQL paper (arXiv:2507.04701), specifically Section III-B.

## Scope

This implementation includes **ONLY** the Schema Filter pipeline:
- ✅ Schema parsing and normalization
- ✅ Keyword extraction (LLM proxy)
- ✅ Multi-path retrieval (table, column, value-level)
- ✅ Iterative column selection (Algorithm 1)
- ✅ Multiple filtered schema generation

**NOT included** (as per scope constraint):
- ❌ SQL generation
- ❌ Multi-generator ensemble
- ❌ SQL selection model
- ❌ Execution or evaluation

## Paper Reference

**XiYan-SQL: A Novel Multi-Generator Framework for Text-to-SQL**  
arXiv:2507.04701v1 [cs.CL] 7 Jul 2025

**Section Implemented:** III-B (Schema Filter)

## Architecture

The implementation is modular with the following folder structure:

```
SQL-Query-Generator/
├── src/
│   └── schema_filter/          # Main schema filter module
│       ├── __init__.py
│       ├── schema_parser.py     # Parses and normalizes BIRD schema format
│       ├── keyword_extractor.py # Extracts keywords from questions/evidence (LLM proxy)
│       ├── retriever.py         # Multi-path retrieval (Equation 1 from paper) - Uses PyTorch
│       ├── column_selector.py    # Iterative column selection (Algorithm 1)
│       └── pipeline.py           # Main orchestrator
├── scripts/
│   └── demo.py                  # Demo script
├── dataset/                      # Dataset utilities
├── script/                       # Training scripts
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy English model (optional, for better keyword extraction):
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

Run the demo script:
```bash
python scripts/demo.py
```

This will:
1. Load the BIRD training dataset
2. Process 1-2 samples
3. Display results
4. Save results to `schema_filter_results.json`

### Programmatic Usage

```python
from src.schema_filter import SchemaFilterPipeline

# Initialize pipeline
pipeline = SchemaFilterPipeline(
    use_embeddings=True,
    use_spacy=True,
    top_k_retrieval=20,
    num_schemas=2
)

# Process a single sample
result = pipeline.process_sample(
    question="What is the total revenue?",
    schema_data=bird_schema_dict,
    evidence="Additional context if available"
)

# Access results
schemas = result['schemas']  # List of filtered schemas
statistics = result['statistics']
```

## Algorithm Details

### Multi-path Retrieval (Equation 1)

The retrieval score is computed as:
```
Score(ki, cj) = VQ||E · VT ab(cj) · Vki · Vcj
```

Where:
- `VQ||E`: Embedding of concatenated question and evidence
- `VT ab(cj)`: Embedding of table metadata
- `Vki`: Embedding of keyword
- `Vcj`: Embedding of column metadata

### Iterative Column Selection (Algorithm 1)

The algorithm generates `ps` schema variants through iterative selection:
1. Extract keywords from question and evidence
2. Retrieve top-k columns using multi-path retrieval
3. For each iteration `i`:
   - Select columns from remaining set
   - Identify primary/foreign keys
   - Generate schema Si by unifying previous schemas
   - Remove selected columns (except PKs/FKs)
4. Return schema set S = {S1, S2, ..., Sps}

## Output Format

Each processed sample produces:
```json
{
  "question_id": "...",
  "question": "...",
  "evidence": "...",
  "keywords": ["keyword1", "keyword2", ...],
  "num_retrieved_columns": 15,
  "schemas": [
    {
      "table_name": {
        "columns": [
          {
            "name": "column_name",
            "type": "text",
            "is_primary_key": false,
            "is_foreign_key": true,
            "references": "other_table"
          }
        ]
      }
    }
  ],
  "statistics": {
    "num_retrieved_columns": 15,
    "num_schemas": 2,
    "schema_stats": [
      {
        "num_tables": 3,
        "num_columns": 8,
        "num_primary_keys": 2,
        "num_foreign_keys": 1
      }
    ]
  }
}
```

## Implementation Notes

### LLM Proxy Functions

The following functions are LLM proxies (using deterministic baselines):

1. **Keyword Extraction** (`src/schema_filter/keyword_extractor.py`):
   - Paper: `K = f_Ms(Q, E)` where Ms is an LLM
   - Implementation: Uses spaCy (if available) or regex-based extraction

2. **Column Selection** (`src/schema_filter/column_selector.py`):
   - Paper: `Sslct_i ← f_Ms(Srtrv, Q, E)`
   - Implementation: Uses heuristic scoring based on keyword matching

### PyTorch Implementation

- **Retriever** (`src/schema_filter/retriever.py`):
  - All tensor operations use PyTorch
  - TF-IDF implemented using PyTorch tensors (no scikit-learn)
  - Cosine similarity computed using PyTorch's `F.normalize` and tensor operations
  - Supports GPU acceleration via `device` parameter

### Embeddings

- **Default**: Uses `sentence-transformers` with model `all-MiniLM-L6-v2`
- **Fallback**: Uses PyTorch-based TF-IDF implementation if sentence transformers unavailable
- **Framework**: All tensor operations use PyTorch (no scikit-learn dependency)

## Dataset

The implementation uses the BIRD dataset:
```python
from datasets import load_dataset
ds = load_dataset("xu3kev/BIRD-SQL-data-train")
```

Each sample should contain:
- `question`: Natural language question
- `schema` or `db_schema`: Database schema
- `evidence` or `context`: Optional evidence/context
- `question_id` or `id`: Question identifier

## Limitations

1. **Value-level retrieval**: Currently a mock implementation (no actual database value scanning)
2. **LLM proxies**: Uses deterministic baselines instead of actual LLMs
3. **Schema format**: Supports multiple BIRD schema formats but may need adaptation for custom formats

## References

- Paper: [XiYan-SQL: A Novel Multi-Generator Framework for Text-to-SQL](https://arxiv.org/abs/2507.04701)
- BIRD Dataset: [BIRD Benchmark](https://bird-bench.github.io/)

## License

This implementation is for research purposes, following the scope constraints specified in the task description.
