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
from typing import Dict, List, Sequence

import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic ablation plots + pipeline EX table")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config" / "train_sql_generator.json"),
        help="Optional: read model names from config",
    )
    parser.add_argument("--plot-dir", type=str, default=str(REPO_ROOT / "plot" / "ablation"))
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data" / "ablation"))
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
