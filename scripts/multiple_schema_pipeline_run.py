import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.bird_dataset import BirdDataset
from service.schema_filter.pipeline import SchemaFilterPipeline

# Add service to path if needed
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_batch(ds, batch_size):
    batch = []

    for i in range(len(ds)):
        sample = ds[i]
        question = sample.get("question", sample.get("nl", ""))
        schema_sql = sample.get("db_schema", sample.get("schema", ""))
        evidence = sample.get("evidence", sample.get("context", None))
        question_id = sample.get("question_id", sample.get("id", f"sample_{i}"))
        actual_result = sample.get("SQL", None)
        sqlfile_path = sample.get('sqlfile_path', None)

        if question and actual_result:
            batch.append((question, schema_sql, evidence, question_id, actual_result, sqlfile_path))

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

def parse_args():
    parser = argparse.ArgumentParser(description="Run schema filter pipeline on BIRD dataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset root that contains train/train.json and train/train_tables.json.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="schema_filter_results_2.json",
        help="Path to write JSON pipeline results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for iteration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("Schema Filter Pipeline")
    print("=" * 80)
    print()
    
    # Load BIRD dataset
    print("Loading BIRD training dataset...")
    try:
        ds = BirdDataset(data_root=args.data_root) if args.data_root else BirdDataset()
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
        num_schemas=3,
    )
    print("✓ Pipeline initialized")
    print()

    # Process the whole dataset
    num_samples = len(ds)
    batch_size = args.batch_size
    print(f"Processing full training split: {num_samples} samples")
    print()

    results = []
    with tqdm(total=num_samples, desc="Processing", dynamic_ncols=True) as pbar:
        for batch in create_batch(ds, batch_size):
            for question, schema_sql, evidence, question_id, actual_result, sqlfile_path in batch:
                try:
                    with torch.no_grad():
                        result = pipeline.process_sample(
                            question=question,
                            schema_data=schema_sql,
                            evidence=evidence,
                            question_id=question_id,
                            actual_result=actual_result,
                            sqlfile_path=sqlfile_path
                        )
                    
                    results.append(result)

                except Exception as e:
                    print(f"Error processing sample {question_id}: {e}")

                pbar.update(1)

    print()

    # Save results to file
    output_file = args.output_file
    print(f"Saving {len(results)} results to {output_file}...")
    pipeline.save_results(results, output_file)
    print("✓ Results saved")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
