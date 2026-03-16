import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from service.schema_filter.pipeline import SchemaFilterPipeline
import json
import sys
import os

# Add service to path if needed
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 80)
    print("Schema Filter Pipeline Demo")
    print("=" * 80)
    print()
    
    # Load BIRD dataset
    print("Loading BIRD training dataset...")
    try:
        ds = load_dataset("xu3kev/BIRD-SQL-data-train", split="train")
        print(f"✓ Loaded dataset with {len(ds)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and datasets library installed.")
        return
    
    print()
    
    # Initialize pipeline
    print("Initializing Schema Filter Pipeline...")
    print("Using all-MiniLM-L6-v2 for embeddings")
    pipeline = SchemaFilterPipeline(
        top_k_retrieval=20,
        num_schemas=2,  # Generate 2 schema variants (S1, S2)
    )
    print("✓ Pipeline initialized")
    print()

    # Process the whole dataset
    num_samples = len(ds)
    print(f"Processing full training split: {num_samples} samples")
    print()

    results = []
    for i, sample in enumerate(ds):
        question = sample.get("question", sample.get("nl", ""))
        schema_sql = sample.get("db_schema", sample.get("schema", ""))
        evidence = sample.get("evidence", sample.get("context", None))
        question_id = sample.get("question_id", sample.get("id", f"sample_{i}"))

        if num_samples < i:
            break;

        if not question or not schema_sql:
            # Skip malformed entries
            continue

        try:
            result = pipeline.process_sample(
                question=question,
                schema_data=schema_sql,
                evidence=evidence,
                question_id=question_id,
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing sample {question_id}: {e}")
            continue

        # Simple progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

    print()

    print("=" * 80)
    print("RESULTS (first few samples)")
    print("=" * 80)
    print()

    for i, result in enumerate(results[:5], 1):
        print(f"Sample {i}:")
        print(f"  Question ID: {result['question_id']}")
        print(f"  Question: {result['question']}")
        if result.get("evidence"):
            ev = result["evidence"]
            print(f"  Evidence: {ev[:100]}..." if len(ev) > 100 else f"  Evidence: {ev}")

        stats = result.get("statistics", {})
        print(f"  Keywords extracted: {len(result.get('keywords', []))} keywords")
        print(f"    Keywords: {result.get('keywords', [])}")
        print(f"  Selected columns: {stats.get('num_selected_col')}")
        print(f"  Retrieved values: {stats.get('num_retrieved_val')}")
        print(f"  Generated schemas: {stats.get('num_schemas')}")
        print()

        # Display schema details from sets of (table, column)
        print("  Schema Details:")
        for j, schema in enumerate(result.get("schemas", []), 1):
            print(f"    Schema S{j}:")
            table_map = {}
            for table, column in schema:
                table_map.setdefault(table, set()).add(column)

            for table, cols in sorted(table_map.items()):
                print(f"      Table: {table}")
                print(f"        Columns: {', '.join(sorted(cols))}")
            print()

        print("-" * 80)
        print()

    # Save results to file
    output_file = "schema_filter_results.json"
    print(f"Saving {len(results)} results to {output_file}...")
    pipeline.save_results(results, output_file)
    print("✓ Results saved")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
