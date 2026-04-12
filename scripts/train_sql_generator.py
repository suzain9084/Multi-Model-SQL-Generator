import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset

from dataset.dataset import FilteredSchemaDataset

CONFIG_PATH = ROOT / "config" / "train_sql_generator.json"


def _resolve_path(value, base: Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _load_module_class(relpath: str, module_alias: str, attr: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(module_alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {attr} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def _build_hf_dataset(filtered: FilteredSchemaDataset, generator_cls) -> Dataset:
    inputs, targets = [], []
    skipped_empty_gold = 0
    skipped_no_paths = 0

    for i in range(len(filtered)):
        ex, gold = filtered[i]
        gold = (gold or "").strip()
        if not gold:
            skipped_empty_gold += 1
            continue
        paths = generator_cls.schema_paths_to_strings(ex.schemas)
        if not paths:
            skipped_no_paths += 1
            continue
        for schema_str in paths:
            inputs.append(f"question: {ex.question} table: {schema_str}")
            targets.append(gold)

    if skipped_empty_gold:
        print(f"Skipped {skipped_empty_gold} rows with no gold SQL.", flush=True)
    if skipped_no_paths:
        print(f"Skipped {skipped_no_paths} rows with empty schema paths.", flush=True)

    return Dataset.from_dict({"input_text": inputs, "target_text": targets})


def main() -> None:
    if not CONFIG_PATH.is_file():
        print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        print("Config must be a JSON object.", file=sys.stderr)
        sys.exit(1)

    json_path = _resolve_path(cfg.get("json_path", "schema_filter_results.json"), ROOT)
    epochs = int(cfg.get("epochs", 3))
    parse_dialect = str(cfg.get("parse_dialect", "duckdb"))

    raw_models = cfg.get("models")
    if not isinstance(raw_models, list) or len(raw_models) < 1:
        print("Config needs a non-empty 'models' list.", file=sys.stderr)
        sys.exit(1)

    model_names: list[str] = []
    output_dirs: list[Path] = []
    for i, entry in enumerate(raw_models):
        if not isinstance(entry, dict):
            print(f"models[{i}] must be an object.", file=sys.stderr)
            sys.exit(1)
        name, out = entry.get("model_name"), entry.get("output_dir")
        if not name or not out:
            print(f"models[{i}] needs model_name and output_dir.", file=sys.stderr)
            sys.exit(1)
        model_names.append(str(name))
        output_dirs.append(_resolve_path(out, ROOT))

    if not json_path.is_file():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    T5SmallSQL = _load_module_class("model/SQL-generator.py", "sql_gen", "T5SmallSQL")
    MultipleSqlGenerator = _load_module_class(
        "service/multiple-sql-generator-pipeline/pipeline.py",
        "multi_sql_pipeline",
        "MultipleSqlGenerator",
    )

    backbone_models = [T5SmallSQL(model_name=n) for n in model_names]
    generator = MultipleSqlGenerator(backbone_models, parse_dialect=parse_dialect)

    filtered = FilteredSchemaDataset(str(json_path))
    hf_train = _build_hf_dataset(filtered, MultipleSqlGenerator)
    if len(hf_train) == 0:
        print("No training rows after filtering.", file=sys.stderr)
        sys.exit(2)

    print(
        f"{len(filtered)} examples, {len(hf_train)} train rows, "
        f"{len(generator.models)} model(s) in MultipleSqlGenerator.",
        flush=True,
    )

    generator.fine_tune_all_backbones(
        hf_train,
        output_dirs,
        epochs=epochs,
        model_labels=model_names,
    )


if __name__ == "__main__":
    main()
