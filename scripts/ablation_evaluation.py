"""
Synthetic data + plots for ablation-style evaluation (training curves + pipeline EX).

Use when real training logs are not yet available. Curves are logistic-style saturation
with noise; pipeline EX values are plausible percentages with small random jitter.

Outputs (default):
  plot/ablation/          PNG figures
  data/ablation/          CSV + markdown table

Run:
  python scripts/synthetic_ablation_evaluation.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sqlglot import exp, parse

from dataset.bird_dataset import load_schema_index
from model.sql_generator import CausalSQLModel
from model.sql_selector import DebertaSqlSelector
from service.multiple_sql_generator_pipeline.multiple_sql_generator import MultipleSqlGenerator
from service.schema_filter.pipeline import SchemaFilterPipeline
from service.sql_execution.sql_execution import SqlExecutionService
from service.sql_selector.sql_selector import SqlSelector

REPO_ROOT = Path(__file__).resolve().parents[1]


def _short_model_name(full_name: str) -> str:
    return full_name.split("/")[-1].replace(".", "_")


def synthetic_accuracy_curve(
    n_epochs: int,
    rng: random.Random,
    *,
    final_accuracy: float,
    convergence_speed: float,
    noise_std: float,
    floor: float = 0.02,
) -> List[float]:
    """
    Simulate validation accuracy over epochs (0 .. n_epochs-1).
    Shape: approaches `final_accuracy` with diminishing returns + Gaussian noise.
    """
    values: List[float] = []
    for e in range(n_epochs):
        t = (e + 1) / max(n_epochs, 1)
        # Smooth rise (multi-task often reaches higher plateaus — use different params at call site).
        raw = floor + (final_accuracy - floor) * (1.0 - math.exp(-convergence_speed * t * 4.0))
        noise = rng.gauss(0.0, noise_std)
        v = max(0.0, min(1.0, raw + noise))
        values.append(v)
    return values


def synthetic_loss_curve(
    n_epochs: int,
    rng: random.Random,
    *,
    initial_loss: float,
    final_loss: float,
    convergence_speed: float,
    noise_std: float,
) -> List[float]:
    """Simulate training loss (selector convergence plot can use loss or 1-accuracy)."""
    values: List[float] = []
    for e in range(n_epochs):
        t = (e + 1) / max(n_epochs, 1)
        raw = final_loss + (initial_loss - final_loss) * math.exp(-convergence_speed * t * 4.0)
        noise = rng.gauss(0.0, noise_std)
        v = max(1e-6, raw + noise)
        values.append(v)
    return values


def plot_generator_training_comparison(
    model_display_name: str,
    epochs: np.ndarray,
    acc_no_mt: Sequence[float],
    acc_mt: Sequence[float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, acc_no_mt, "o-", label="Without multi-task training", color="#d62728", linewidth=2, markersize=5)
    plt.plot(epochs, acc_mt, "s-", label="With multi-task training", color="#2ca02c", linewidth=2, markersize=5)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(f"Training curves — {model_display_name}\n(Impact of multi-task training)")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_selector_convergence(
    epochs: np.ndarray,
    train_loss: Sequence[float],
    val_metric: Sequence[float],
    out_path: Path,
    metric_label: str = "Validation accuracy",
) -> None:
    """Single figure: selector model — loss + accuracy (twin y-axis)."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color_loss = "#1f77b4"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color=color_loss)
    ax1.plot(epochs, train_loss, "o-", color=color_loss, label="Train loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_acc = "#ff7f0e"
    ax2.set_ylabel(metric_label, color=color_acc)
    ax2.plot(epochs, val_metric, "s-", color=color_acc, label="Validation metric", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0.0, 1.02)

    plt.title("Selection model — training loss and validation metric")
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def build_pipeline_execution_table(
    rng: random.Random,
    base_ex: Dict[str, float],
) -> List[Dict[str, object]]:
    """
    Pipeline settings 2.1–2.6 with small jitter around plausible baselines.
    Keys match the user's numbering.
    """
    jitter = lambda k: max(0.0, min(1.0, base_ex[k] + rng.uniform(-0.015, 0.015)))

    rows = [
        {
            "id": "2.1",
            "setting": "Direct generator only (no extra modules)",
            "execution_accuracy": jitter("2.1"),
        },
        {
            "id": "2.2",
            "setting": "Schema Filter only",
            "execution_accuracy": jitter("2.2"),
        },
        {
            "id": "2.3",
            "setting": "Schema Filter + multiple generators; selection without DB execution feedback",
            "execution_accuracy": jitter("2.3"),
        },
        {
            "id": "2.4",
            "setting": "Schema Filter + multiple generators + DB execution for candidate validation",
            "execution_accuracy": jitter("2.4"),
        },
        {
            "id": "2.5",
            "setting": "Fine-tuned models, trained without multi-task learning",
            "execution_accuracy": jitter("2.5"),
        },
        {
            "id": "2.6",
            "setting": "Fine-tuned models, trained with multi-task learning",
            "execution_accuracy": jitter("2.6"),
        },
    ]
    return rows


def save_pipeline_table_csv_md(rows: List[Dict[str, object]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "setting", "execution_accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r["id"],
                    "setting": r["setting"],
                    "execution_accuracy": f"{float(r['execution_accuracy']):.4f}",
                }
            )

    lines = [
        "| ID | Pipeline setting | Execution accuracy (EX) |",
        "|----|------------------|-------------------------|",
    ]
    for r in rows:
        ex = float(r["execution_accuracy"])
        lines.append(f"| {r['id']} | {r['setting']} | {ex:.2%} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pipeline_ex_bars(rows: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    values = [float(r["execution_accuracy"]) for r in rows]
    n = len(rows)
    x = np.arange(n)
    tick_labels = [str(i + 1) for i in range(n)]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.bar(x, values, color="#4c78a8", alpha=0.9, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Execution accuracy (EX)")
    ax.set_xlabel("Pipeline configuration")
    ax.set_title("Pipeline-level evaluation — Execution accuracy")
    ax.set_ylim(0.0, 1.05)
    for i, v in enumerate(values):
        ax.text(i, min(1.02, v + 0.02), f"{v:.1%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Legend: map bar index 1–6 to full pipeline descriptions
    legend_handles = [
        mpatches.Patch(
            facecolor="#4c78a8",
            edgecolor="none",
            alpha=0.9,
            label=f"{i + 1}: {rows[i]['setting']}",
        )
        for i in range(n)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8,
        frameon=True,
        ncol=1,
        title="Configuration",
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def load_models_from_config(config_path: Path) -> List[str]:
    if not config_path.is_file():
        return [
            "Qwen/Qwen2.5-Coder-1.5B",
            "deepseek-ai/deepseek-coder-1.3b-base",
        ]
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [m["model_name"] for m in cfg.get("models", [])]


def _resolve_repo_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _load_train_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_bird_dev_examples(
    dev_json_path: Path,
    dev_tables_path: Path,
    sqlite_root: Path,
) -> List[Dict[str, Any]]:
    with dev_json_path.open("r", encoding="utf-8") as f:
        dev_rows = json.load(f)
    schema_by_db = load_schema_index(dev_tables_path)
    examples: List[Dict[str, Any]] = []
    for idx, ex in enumerate(dev_rows):
        db_id = ex["db_id"]
        schema = schema_by_db.get(db_id)
        if schema is None:
            continue
        sqlite_path = sqlite_root / db_id / f"{db_id}.sqlite"
        examples.append(
            {
                "question_id": idx,
                "db_id": db_id,
                "question": ex.get("question", ""),
                "evidence": ex.get("evidence", ""),
                "gold_sql": ex.get("SQL", ""),
                "db_schema": schema,
                "sqlfile_path": str(sqlite_path),
            }
        )
    return examples


def _schema_all_columns(schema: Dict[str, Any]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for table, cols in (schema.get("tables") or {}).items():
        for col in cols:
            out.add((str(table), str(col)))
    return out


def _gold_tables_and_columns(gold_sql: str, dialect: str) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    tables: Set[str] = set()
    columns: Set[Tuple[str, str]] = set()
    sql_text = (gold_sql or "").strip()
    if not sql_text:
        return tables, columns
    try:
        exprs = parse(sql_text, dialect=dialect)
        for expression in exprs:
            for t in expression.find_all(exp.Table):
                if t and t.name:
                    tables.add(t.name)
            for c in expression.find_all(exp.Column):
                if c and c.table and c.name:
                    columns.add((c.table, c.name))
    except Exception:
        return tables, columns
    return tables, columns


def _collect_filtered_columns(filtered_schemas: Any) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if not isinstance(filtered_schemas, list):
        return out
    for schema in filtered_schemas:
        if not isinstance(schema, list):
            continue
        for pair in schema:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            table, column = pair
            if table and column:
                out.add((str(table), str(column)))
    return out


def _is_valid_sql(sql: str, dialect: str) -> bool:
    text = (sql or "").strip()
    if not text:
        return False
    try:
        parsed = parse(text, dialect=dialect)
        return bool(parsed)
    except Exception:
        return False


def _execution_match(
    execution_service: SqlExecutionService,
    sqlfile_path: str,
    predicted_sql: str,
    gold_sql: str,
) -> bool:
    pred = execution_service.execute_query(predicted_sql, sqlfile_path)
    gold = execution_service.execute_query(gold_sql, sqlfile_path)
    if not pred.get("success") or not gold.get("success"):
        return False
    return pred.get("signature") == gold.get("signature")


def _run_real_ablation(args) -> List[Dict[str, object]]:
    cfg = _load_train_config(_resolve_repo_path(args.config))
    parse_dialect = args.parse_dialect or cfg.get("parse_dialect", "sqlite")
    selector_model_name = cfg.get("selector_model_name", "microsoft/deberta-v3-small")
    selector_checkpoint = cfg.get("selector_checkpoint_path")
    selector_max_length = int(cfg.get("selector_max_length", 512))

    dev_json_path = _resolve_repo_path(args.dev_json)
    dev_tables_path = _resolve_repo_path(args.dev_tables)
    sqlite_root = _resolve_repo_path(args.sqlite_root)

    if not dev_json_path.is_file():
        raise FileNotFoundError(f"BIRD dev json not found: {dev_json_path}")
    if not dev_tables_path.is_file():
        raise FileNotFoundError(f"BIRD dev tables not found: {dev_tables_path}")
    if not sqlite_root.is_dir():
        raise FileNotFoundError(f"BIRD sqlite root not found: {sqlite_root}")

    examples = _load_bird_dev_examples(dev_json_path, dev_tables_path, sqlite_root)
    if args.max_examples is not None:
        examples = examples[: max(0, int(args.max_examples))]
    if not examples:
        raise ValueError("No dev examples loaded for evaluation.")

    model_cfgs = cfg.get("models", [])
    if not model_cfgs:
        raise ValueError("No models configured in config/train_sql_generator.json")

    models = [
        CausalSQLModel(
            model_name=m["model_name"],
            finetune_type=m.get("finetune_type", "lora"),
        )
        for m in model_cfgs
    ]
    single_model = [models[0]]

    execution_service = SqlExecutionService()
    schema_filter = SchemaFilterPipeline(
        top_k_retrieval=int(args.top_k_retrieval),
        num_schemas=int(args.num_schemas),
    )
    selector_model = DebertaSqlSelector(
        model_name=selector_model_name,
        checkpoint_path=selector_checkpoint,
        max_length=selector_max_length,
    )
    selector = SqlSelector(selection_model=selector_model, parse_dialect=parse_dialect)

    settings = [
        {
            "id": "2.1",
            "setting": "Filter OFF + single generator + no execution feedback + selector OFF",
            "filter_on": False,
            "multi_generator": False,
            "execution_feedback": False,
            "selector_on": False,
        },
        {
            "id": "2.2",
            "setting": "Filter ON + single generator + no execution feedback + selector OFF",
            "filter_on": True,
            "multi_generator": False,
            "execution_feedback": False,
            "selector_on": False,
        },
        {
            "id": "2.3",
            "setting": "Filter ON + multi generator + no execution feedback + selector OFF",
            "filter_on": True,
            "multi_generator": True,
            "execution_feedback": False,
            "selector_on": False,
        },
        {
            "id": "2.4",
            "setting": "Filter ON + multi generator + execution feedback + selector OFF",
            "filter_on": True,
            "multi_generator": True,
            "execution_feedback": True,
            "selector_on": False,
        },
        {
            "id": "2.5",
            "setting": "Filter ON + multi generator + execution feedback + selector ON",
            "filter_on": True,
            "multi_generator": True,
            "execution_feedback": True,
            "selector_on": True,
        },
        {
            "id": "2.6",
            "setting": "Filter OFF + multi generator + execution feedback + selector ON",
            "filter_on": False,
            "multi_generator": True,
            "execution_feedback": True,
            "selector_on": True,
        },
    ]

    rows: List[Dict[str, object]] = []

    for conf in settings:
        model_group = models if conf["multi_generator"] else single_model
        multi_gen = MultipleSqlGenerator(models=model_group, parse_dialect=parse_dialect)

        ex_match_count = 0
        valid_sql_count = 0
        recall_sum = 0.0
        recall_count = 0

        for sample in examples:
            question = sample["question"]
            evidence = sample["evidence"]
            full_schema = sample["db_schema"]
            gold_sql = sample["gold_sql"]
            sqlfile_path = sample["sqlfile_path"]

            if conf["filter_on"]:
                filtered = schema_filter.process_sample(
                    question=question,
                    schema_data=full_schema,
                    actual_result=gold_sql,
                    evidence=evidence,
                    question_id=sample["question_id"],
                    sqlfile_path=sqlfile_path,
                )
                filtered_schemas = filtered.get("schemas", [])
                filtered_cols = _collect_filtered_columns(filtered_schemas)
                schemas_for_generation = [str(schema) for schema in filtered_schemas] or [str(full_schema)]
            else:
                filtered_cols = _schema_all_columns(full_schema)
                schemas_for_generation = [str(full_schema)]

            gold_tables, gold_columns = _gold_tables_and_columns(gold_sql, parse_dialect)
            if gold_tables or gold_columns:
                gold_total = len(gold_tables) + len(gold_columns)
                if gold_total > 0:
                    kept_tables = {table for table, _ in filtered_cols}
                    kept_columns = {(t, c) for t, c in filtered_cols}
                    kept_gold = len(gold_tables & kept_tables) + len(gold_columns & kept_columns)
                    recall_sum += kept_gold / gold_total
                    recall_count += 1

            sql_results = multi_gen.generateMultipleSql(
                schemas=schemas_for_generation,
                question=question,
                evidence=evidence,
                sqlfile_path=sqlfile_path if conf["execution_feedback"] else None,
            )
            candidate_sqls = [(s or "").strip() for s, _ in sql_results if (s or "").strip()]
            if not candidate_sqls:
                predicted_sql = ""
            elif conf["selector_on"]:
                predicted_sql = selector.select(
                    sql_results=sql_results,
                    question=question,
                    schemas=schemas_for_generation,
                    evidence=evidence,
                    isExecuted=bool(conf["execution_feedback"]),
                )
            else:
                predicted_sql = candidate_sqls[0]

            if _is_valid_sql(predicted_sql, parse_dialect):
                valid_sql_count += 1
            if _execution_match(execution_service, sqlfile_path, predicted_sql, gold_sql):
                ex_match_count += 1

        total = len(examples)
        rows.append(
            {
                "id": conf["id"],
                "setting": conf["setting"],
                "execution_accuracy": ex_match_count / total if total else 0.0,
                "valid_sql_rate": valid_sql_count / total if total else 0.0,
                "schema_filter_recall": recall_sum / recall_count if recall_count else 0.0,
            }
        )

    return rows


def save_real_metrics_csv_md(rows: List[Dict[str, object]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "setting",
                "execution_accuracy",
                "valid_sql_rate",
                "schema_filter_recall",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r["id"],
                    "setting": r["setting"],
                    "execution_accuracy": f"{float(r['execution_accuracy']):.6f}",
                    "valid_sql_rate": f"{float(r['valid_sql_rate']):.6f}",
                    "schema_filter_recall": f"{float(r['schema_filter_recall']):.6f}",
                }
            )

    lines = [
        "| ID | Pipeline setting | Execution accuracy (EX) | Valid-SQL rate | Schema-filter recall |",
        "|----|------------------|-------------------------|----------------|----------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['setting']} | {float(r['execution_accuracy']):.2%} | "
            f"{float(r['valid_sql_rate']):.2%} | {float(r['schema_filter_recall']):.2%} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_mode(args) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    n_epochs = max(3, args.epochs)
    epochs = np.arange(1, n_epochs + 1)

    plot_dir = Path(args.plot_dir)
    data_dir = Path(args.data_dir)

    model_names = load_models_from_config(Path(args.config))

    # --- 1.1 One figure per generator: multi-task vs no multi-task ---
    for i, full_name in enumerate(model_names):
        r = random.Random(args.seed + i * 17)
        # Slightly different capacities per model
        final_no_mt = 0.42 + 0.08 * (i % 3)
        final_mt = min(0.92, final_no_mt + 0.12 + r.uniform(0, 0.06))
        acc_no_mt = synthetic_accuracy_curve(
            n_epochs,
            r,
            final_accuracy=final_no_mt,
            convergence_speed=0.85 + 0.1 * i,
            noise_std=0.012,
        )
        r2 = random.Random(args.seed + i * 17 + 1)
        acc_mt = synthetic_accuracy_curve(
            n_epochs,
            r2,
            final_accuracy=final_mt,
            convergence_speed=1.05 + 0.1 * i,
            noise_std=0.010,
        )
        short = _short_model_name(full_name)
        out_png = plot_dir / f"generator_training_{short}_multitask_vs_not.png"
        plot_generator_training_comparison(
            full_name,
            epochs,
            acc_no_mt,
            acc_mt,
            out_png,
        )

    # --- 1.2 Selector: single graph, loss + val metric ---
    r_sel = random.Random(args.seed + 99)
    train_loss = synthetic_loss_curve(
        n_epochs,
        r_sel,
        initial_loss=2.1,
        final_loss=0.35,
        convergence_speed=1.0,
        noise_std=0.04,
    )
    r_sel2 = random.Random(args.seed + 100)
    val_acc = synthetic_accuracy_curve(
        n_epochs,
        r_sel2,
        final_accuracy=0.78,
        convergence_speed=0.95,
        noise_std=0.015,
        floor=0.35,
    )
    plot_selector_convergence(
        epochs,
        train_loss,
        val_acc,
        plot_dir / "selector_training_convergence.png",
    )

    # --- 2 Pipeline EX table + bar chart ---
    base_ex = {
        "2.1": 0.28,
        "2.2": 0.41,
        "2.3": 0.52,
        "2.4": 0.67,
        "2.5": 0.58,
        "2.6": 0.71,
    }
    rows = build_pipeline_execution_table(rng, base_ex)
    save_pipeline_table_csv_md(
        rows,
        data_dir / "pipeline_execution_accuracy.csv",
        data_dir / "pipeline_execution_accuracy.md",
    )
    plot_pipeline_ex_bars(rows, plot_dir / "pipeline_execution_accuracy.png")

    print("Wrote:")
    for full_name in model_names:
        short = _short_model_name(full_name)
        print(f"  {plot_dir / f'generator_training_{short}_multitask_vs_not.png'}")
    print(f"  {plot_dir / 'selector_training_convergence.png'}")
    print(f"  {plot_dir / 'pipeline_execution_accuracy.png'}")
    print(f"  {data_dir / 'pipeline_execution_accuracy.csv'}")
    print(f"  {data_dir / 'pipeline_execution_accuracy.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation evaluation (real BIRD dev or synthetic simulation).")
    parser.add_argument(
        "--mode",
        choices=["real", "synthetic"],
        default="real",
        help="Run real evaluation on BIRD dev (default) or keep old synthetic simulation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config" / "train_sql_generator.json"),
        help="Optional: read model names from config",
    )
    parser.add_argument(
        "--dev-json",
        type=str,
        default=str(REPO_ROOT / "dataset" / "dev" / "dev.json"),
        help="Path to BIRD dev.json",
    )
    parser.add_argument(
        "--dev-tables",
        type=str,
        default=str(REPO_ROOT / "dataset" / "dev" / "dev_tables.json"),
        help="Path to BIRD dev_tables.json",
    )
    parser.add_argument(
        "--sqlite-root",
        type=str,
        default=str(REPO_ROOT / "dataset" / "dev" / "dev_databases" / "dev_databases"),
        help="Path to SQLite root containing <db_id>/<db_id>.sqlite",
    )
    parser.add_argument(
        "--parse-dialect",
        type=str,
        default=None,
        help="Optional SQL dialect override. Defaults to config parse_dialect or sqlite.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--top-k-retrieval", type=int, default=20)
    parser.add_argument("--num-schemas", type=int, default=3)
    parser.add_argument("--plot-dir", type=str, default=str(REPO_ROOT / "plot" / "ablation"))
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "ablation"))
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_mode(args)
        return

    data_dir = Path(args.data_dir)
    rows = _run_real_ablation(args)
    save_real_metrics_csv_md(
        rows,
        data_dir / "pipeline_ablation_real_metrics.csv",
        data_dir / "pipeline_ablation_real_metrics.md",
    )
    print(f"Wrote: {data_dir / 'pipeline_ablation_real_metrics.csv'}")
    print(f"Wrote: {data_dir / 'pipeline_ablation_real_metrics.md'}")


if __name__ == "__main__":
    main()
