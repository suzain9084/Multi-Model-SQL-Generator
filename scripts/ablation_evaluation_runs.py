"""
Build ablation plots from REAL evaluation outputs (not simulated).

This script mirrors the figures created by `scripts/synthetic_ablation_evaluation.py`,
but reads actual metrics collected from testing predictions.

Expected inputs
---------------
1) Generator epoch prediction files (JSON), one file per:
   (model_name, multitask flag, epoch)

   Example file content:
   {
     "model_name": "Qwen/Qwen2.5-Coder-1.5B",
     "multitask": true,
     "epoch": 3,
     "predictions": [
       {
         "predicted_sql": "SELECT ...",
         "actual_result": "SELECT ...",      // gold SQL
         "sqlfile_path": "path/to/db.sqlite"
       }
     ]
   }

2) Selector curve CSV (optional, but needed for selector convergence plot):
   Required columns: epoch, train_loss, val_accuracy

3) Pipeline EX CSV (optional, but needed for pipeline bar chart):
   Required columns: id, setting, execution_accuracy
   - `id` can be 2.1..2.6 or 1..6
   - `execution_accuracy` must be in [0, 1]

Outputs
-------
- plot/ablation/generator_training_<model>_multitask_vs_not.png
- plot/ablation/selector_training_convergence.png
- plot/ablation/pipeline_execution_accuracy.png
- data/ablation/generator_epoch_execution_accuracy.csv (computed)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from service.sql_execution.sql_execution import SqlExecutionService


def _short_model_name(full_name: str) -> str:
    return full_name.split("/")[-1].replace(".", "_")


def _execution_match(exec_service: SqlExecutionService, predicted_sql: str, gold_sql: str, sqlfile_path: str) -> bool:
    pred = exec_service.execute_query(predicted_sql, sqlfile_path)
    gold = exec_service.execute_query(gold_sql, sqlfile_path)
    if not pred.get("success") or not gold.get("success"):
        return False
    return pred.get("signature") == gold.get("signature")


def compute_generator_epoch_metrics(
    prediction_files: List[Path],
    timeout_seconds: float,
    max_rows: int,
) -> List[Dict[str, object]]:
    """
    Compute execution accuracy for each (model_name, multitask, epoch) from raw predictions.
    """
    exec_service = SqlExecutionService(timeout_seconds=timeout_seconds, max_rows=max_rows)
    rows: List[Dict[str, object]] = []

    for fpath in prediction_files:
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        model_name = str(payload["model_name"])
        multitask = bool(payload["multitask"])
        epoch = int(payload["epoch"])
        preds = payload.get("predictions", [])

        total = 0
        correct = 0
        for item in preds:
            predicted_sql = (item.get("predicted_sql") or "").strip()
            gold_sql = (item.get("actual_result") or "").strip()
            db_path = (item.get("sqlfile_path") or "").strip()
            if not predicted_sql or not gold_sql or not db_path:
                continue
            total += 1
            if _execution_match(exec_service, predicted_sql, gold_sql, db_path):
                correct += 1

        ex = (correct / total) if total > 0 else 0.0
        rows.append(
            {
                "model_name": model_name,
                "multitask": multitask,
                "epoch": epoch,
                "execution_accuracy": ex,
                "num_examples": total,
                "source_file": str(fpath),
            }
        )

    return rows


def write_generator_metrics_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (str(r["model_name"]), bool(r["multitask"]), int(r["epoch"])))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "multitask", "epoch", "execution_accuracy", "num_examples", "source_file"],
        )
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(
                {
                    "model_name": r["model_name"],
                    "multitask": str(r["multitask"]).lower(),
                    "epoch": int(r["epoch"]),
                    "execution_accuracy": f"{float(r['execution_accuracy']):.6f}",
                    "num_examples": int(r["num_examples"]),
                    "source_file": r["source_file"],
                }
            )


def load_generator_metrics_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "model_name": r["model_name"],
                    "multitask": str(r["multitask"]).lower() in {"1", "true", "yes"},
                    "epoch": int(r["epoch"]),
                    "execution_accuracy": float(r["execution_accuracy"]),
                }
            )
    return rows


def plot_generator_curves_from_metrics(rows: List[Dict[str, object]], out_dir: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    out_paths: List[Path] = []
    grouped: Dict[str, Dict[bool, List[Tuple[int, float]]]] = {}
    for r in rows:
        model_name = str(r["model_name"])
        mt = bool(r["multitask"])
        grouped.setdefault(model_name, {}).setdefault(mt, []).append((int(r["epoch"]), float(r["execution_accuracy"])))

    for model_name, by_mt in grouped.items():
        if True not in by_mt or False not in by_mt:
            # Need both lines to mirror the synthetic plot structure.
            continue

        no_mt = sorted(by_mt[False], key=lambda x: x[0])
        yes_mt = sorted(by_mt[True], key=lambda x: x[0])

        epochs_no = np.array([x[0] for x in no_mt], dtype=int)
        vals_no = np.array([x[1] for x in no_mt], dtype=float)
        epochs_yes = np.array([x[0] for x in yes_mt], dtype=int)
        vals_yes = np.array([x[1] for x in yes_mt], dtype=float)

        plt.figure(figsize=(9, 5.5))
        plt.plot(epochs_no, vals_no, "o-", label="Without multi-task training", color="#d62728", linewidth=2, markersize=5)
        plt.plot(epochs_yes, vals_yes, "s-", label="With multi-task training", color="#2ca02c", linewidth=2, markersize=5)
        plt.xlabel("Epoch")
        plt.ylabel("Execution accuracy (EX)")
        plt.title(f"Training curves — {model_name}\n(Impact of multi-task training)")
        plt.ylim(0.0, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()

        out_path = out_dir / f"generator_training_{_short_model_name(model_name)}_multitask_vs_not.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
        out_paths.append(out_path)

    return out_paths


def plot_selector_convergence_from_csv(selector_csv: Path, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    epochs: List[int] = []
    train_loss: List[float] = []
    val_acc: List[float] = []
    with selector_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs.append(int(r["epoch"]))
            train_loss.append(float(r["train_loss"]))
            val_acc.append(float(r["val_accuracy"]))

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#1f77b4")
    ax1.plot(epochs, train_loss, "o-", color="#1f77b4", label="Train loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation accuracy", color="#ff7f0e")
    ax2.plot(epochs, val_acc, "s-", color="#ff7f0e", label="Validation metric", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim(0.0, 1.02)

    plt.title("Selection model — training loss and validation metric")
    fig.tight_layout()
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def plot_pipeline_ex_from_csv(pipeline_csv: Path, out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows: List[Dict[str, object]] = []
    with pipeline_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "id": r["id"],
                    "setting": r["setting"],
                    "execution_accuracy": float(r["execution_accuracy"]),
                }
            )

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

    legend_handles = [
        mpatches.Patch(facecolor="#4c78a8", edgecolor="none", alpha=0.9, label=f"{i + 1}: {rows[i]['setting']}")
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
    return out_path


def _glob_prediction_files(pattern: str) -> List[Path]:
    # Use Path.glob with a repo-relative pattern by default.
    p = Path(pattern)
    if p.is_absolute():
        base = p.parent
        return sorted(base.glob(p.name))
    return sorted(REPO_ROOT.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation figures from REAL evaluation outputs")
    parser.add_argument(
        "--generator-prediction-glob",
        type=str,
        default="data/ablation/generator_eval/*.json",
        help="Glob for generator epoch prediction JSON files",
    )
    parser.add_argument(
        "--generator-metrics-csv",
        type=str,
        default=str(REPO_ROOT / "data" / "ablation" / "generator_epoch_execution_accuracy.csv"),
        help="Path to save/load generator epoch metrics CSV",
    )
    parser.add_argument(
        "--selector-curve-csv",
        type=str,
        default=str(REPO_ROOT / "data" / "ablation" / "selector_curve.csv"),
        help="CSV with columns: epoch,train_loss,val_accuracy",
    )
    parser.add_argument(
        "--pipeline-csv",
        type=str,
        default=str(REPO_ROOT / "data" / "ablation" / "pipeline_execution_accuracy.csv"),
        help="CSV with columns: id,setting,execution_accuracy",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(REPO_ROOT / "plot" / "ablation"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument(
        "--skip-recompute-generator",
        action="store_true",
        help="Load generator metrics from --generator-metrics-csv instead of recomputing from raw predictions",
    )
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    generator_metrics_csv = Path(args.generator_metrics_csv)

    # 1) Generator training curves from real epoch predictions/metrics.
    if args.skip_recompute_generator:
        gen_rows = load_generator_metrics_csv(generator_metrics_csv)
    else:
        prediction_files = _glob_prediction_files(args.generator_prediction_glob)
        if not prediction_files:
            raise FileNotFoundError(
                f"No generator prediction files found by pattern: {args.generator_prediction_glob}"
            )
        gen_rows = compute_generator_epoch_metrics(
            prediction_files=prediction_files,
            timeout_seconds=args.timeout_seconds,
            max_rows=args.max_rows,
        )
        write_generator_metrics_csv(gen_rows, generator_metrics_csv)

    generator_plots = plot_generator_curves_from_metrics(gen_rows, plot_dir)

    # 2) Selector convergence plot from real CSV.
    selector_plot = None
    selector_csv = Path(args.selector_curve_csv)
    if selector_csv.is_file():
        selector_plot = plot_selector_convergence_from_csv(
            selector_csv=selector_csv,
            out_path=plot_dir / "selector_training_convergence.png",
        )

    # 3) Pipeline EX bar from real CSV.
    pipeline_plot = None
    pipeline_csv = Path(args.pipeline_csv)
    if pipeline_csv.is_file():
        pipeline_plot = plot_pipeline_ex_from_csv(
            pipeline_csv=pipeline_csv,
            out_path=plot_dir / "pipeline_execution_accuracy.png",
        )

    print("Generated:")
    print(f"  {generator_metrics_csv}")
    for p in generator_plots:
        print(f"  {p}")
    if selector_plot:
        print(f"  {selector_plot}")
    else:
        print(f"  (skipped selector plot; missing file: {selector_csv})")
    if pipeline_plot:
        print(f"  {pipeline_plot}")
    else:
        print(f"  (skipped pipeline plot; missing file: {pipeline_csv})")


if __name__ == "__main__":
    main()

