import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

REPO_ROOT = Path(__file__).resolve().parents[1]

def normalize_identifier(s: str) -> str:
    """
    Normalize schema column identifiers and SQL identifiers so they can be matched.
    Keeps punctuation like '?' because your schema uses it (e.g., "Timely response?").
    """
    if s is None:
        return ""
    s = str(s).strip()
    # Remove common quoting wrappers.
    s = s.strip("`\"'[]")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def extract_selected_columns(question_obj: Dict) -> Tuple[Set[str], Dict[str, str]]:
    """
    Returns:
      - normalized selected columns (lowercased, whitespace-normalized)
      - map normalized->display text (for labels/embeddings)
    """
    selected_norm: Set[str] = set()
    selected_display: Dict[str, str] = {}

    for schema in question_obj.get("schemas", []) or []:
        tables = schema.get("tables", []) or []
        if isinstance(tables, dict):
            # Be defensive if the JSON format changes.
            tables = [{"name": k, "columns": v} for k, v in tables.items()]

        for table in tables:
            cols = table.get("columns", []) or []
            for col in cols:
                norm = normalize_identifier(col)
                if not norm:
                    continue
                selected_norm.add(norm)
                selected_display.setdefault(norm, str(col).strip())

    return selected_norm, selected_display


def extract_columns_from_schema(schema_obj: Dict) -> Set[str]:
    """
    Extract normalized column set from one schema candidate object:
      {"tables": [{"name": "...", "columns": [...]}, ...]}
    """
    cols: Set[str] = set()
    tables = schema_obj.get("tables", []) or []
    if isinstance(tables, dict):
        tables = [{"name": k, "columns": v} for k, v in tables.items()]
    for table in tables:
        for col in table.get("columns", []) or []:
            norm = normalize_identifier(col)
            if norm:
                cols.add(norm)
    return cols


def extract_gold_columns(actual_result_sql: str) -> Tuple[Set[str], Dict[str, str]]:
    """
    Extract column identifiers referenced in the SQL via sqlglot.
    Returns normalized column names and a normalized->display map.

    Notes:
    - We match columns by name only (ignoring table aliases) to keep TP/FP/FN usable.
    """
    gold_norm: Set[str] = set()
    gold_display: Dict[str, str] = {}

    if not actual_result_sql:
        return gold_norm, gold_display

    try:
        expr = parse_one(actual_result_sql, read="sqlite")
    except (SqlglotError, Exception):
        # Fallback: very rough extraction; primarily helps avoid hard crashes.
        # The sqlglot path above should work for most of these datasets.
        for m in re.finditer(r"(?:`([^`]+)`)|\b([A-Za-z_][A-Za-z0-9_]*\b))", actual_result_sql):
            candidate = m.group(1) or m.group(2)
            if not candidate:
                continue
            norm = normalize_identifier(candidate)
            gold_norm.add(norm)
            gold_display.setdefault(norm, candidate.strip())
        return gold_norm, gold_display

    for col in expr.find_all(exp.Column):
        # sqlglot identifiers can be quoted; col.name gives the bare identifier text.
        name = col.name
        if not name or name.strip() == "*":
            continue
        norm = normalize_identifier(name)
        gold_norm.add(norm)
        gold_display.setdefault(norm, str(name).strip())

    return gold_norm, gold_display


def safe_question_id(q: Dict) -> str:
    # Deprecated: kept only to avoid breaking older local experiments.
    qid = q.get("question_id", "unknown")
    return str(qid) if isinstance(qid, str) else "unknown"


def choose_umap_question(
    results: List[Dict],
    question_id: Optional[str],
    question_index: int,
) -> Dict:
    if question_id is not None:
        for r in results:
            if str(r.get("question_id")) == str(question_id):
                return r
        raise ValueError(f"question_id={question_id!r} not found in input JSON")
    # Default: first sample in file.
    question_index = max(0, min(question_index, len(results) - 1))
    return results[question_index]


def plot_tsne_tp_fp_fn(
    question_obj: Dict,
    method: str,
    perplexity: int,
    random_state: int,
    out_path: Path,
    label_top_k: int = 8,
    max_points: int = 200,
) -> None:
    """
    Create one scatter plot for one question:
      - Points = selected/gold column identifiers
      - TP/FP/FN coloring
      - Question embedding as a star
    """
    question = question_obj.get("question", "") or ""
    evidence = question_obj.get("evidence", "") or ""
    question_text = f"{question} {evidence}".strip()

    selected_norm, selected_display = extract_selected_columns(question_obj)
    gold_norm, gold_display = extract_gold_columns(question_obj.get("actual_result", ""))

    tp = selected_norm & gold_norm
    fp = selected_norm - gold_norm
    fn = gold_norm - selected_norm
    union = sorted(tp | fp | fn)
    tp_total, fp_total, fn_total = len(tp), len(fp), len(fn)

    if not union:
        raise ValueError("No columns found to plot for the selected question.")

    # Display text for embedding/labeling.
    display_text: Dict[str, str] = {}
    for c in union:
        display_text[c] = selected_display.get(c) or gold_display.get(c) or c

    col_texts = [display_text[c] for c in union]

    # Convert question/columns to vectors.
    # Using TF-IDF so this script runs even when sentence-transformers isn't installed.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w?]+\b",
        max_features=4096,
        min_df=1,
    )

    vectorizer.fit([question_text] + col_texts)
    col_emb_sparse = vectorizer.transform(col_texts)
    q_emb_sparse = vectorizer.transform([question_text])

    # If this question has lots of columns, t-SNE will be slow.
    # We cap points by similarity to the question (using sparse dot-product).
    if col_emb_sparse.shape[0] > max_points:
        sims = (col_emb_sparse @ q_emb_sparse.T).toarray().reshape(-1)
        keep_idx = np.argsort(sims)[::-1][:max_points]
        union_shown = [union[i] for i in keep_idx]
        col_texts = [col_texts[i] for i in keep_idx]
        col_emb = col_emb_sparse[keep_idx].toarray().astype(np.float32)
        q_emb = q_emb_sparse.toarray().astype(np.float32).reshape(1, -1)
        tp = {c for c in union_shown if c in tp}
        fp = {c for c in union_shown if c in fp}
        fn = {c for c in union_shown if c in fn}
    else:
        union_shown = union
        col_emb = col_emb_sparse.toarray().astype(np.float32)
        q_emb = q_emb_sparse.toarray().astype(np.float32).reshape(1, -1)

    all_emb = np.vstack([col_emb, q_emb])

    if method.lower() == "tsne":
        from sklearn.manifold import TSNE

        n_samples = all_emb.shape[0]
        # TSNE perplexity must be < n_samples.
        perp = int(min(perplexity, max(2, (n_samples - 1) // 3)))
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        xy = tsne.fit_transform(all_emb)
    else:
        # UMAP is optional; your environment may not have umap-learn installed.
        try:
            import umap  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "UMAP requested but umap-learn isn't installed. Install `umap-learn` "
                "or run with --method tsne."
            ) from e

        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=min(15, max(2, all_emb.shape[0] // 5)),
        )
        xy = reducer.fit_transform(all_emb)

    col_xy = xy[:-1]
    q_xy = xy[-1]

    # Distinguish points by set membership (shown/capped view).
    colors = []
    for c in union_shown:
        if c in tp:
            colors.append("#2ca02c")  # green
        elif c in fp:
            colors.append("#d62728")  # red
        else:
            colors.append("#1f77b4")  # blue

    plt.figure(figsize=(11, 8))
    plt.scatter(col_xy[:, 0], col_xy[:, 1], c=colors, alpha=0.85, s=35, edgecolors="none")
    plt.scatter([q_xy[0]], [q_xy[1]], marker="*", s=250, c="black", label="Question")

    # Optional: label closest columns to the question in 2D.
    if label_top_k > 0:
        d2 = np.linalg.norm(col_xy - q_xy.reshape(1, -1), axis=1)
        top_idx = np.argsort(d2)[: min(label_top_k, len(union_shown))]
        for idx in top_idx:
            plt.text(
                col_xy[idx, 0],
                col_xy[idx, 1],
                col_texts[idx],
                fontsize=8,
                alpha=0.9,
            )

    plt.title(
        f"Schema Filter Columns (capped)\n"
        f"Total TP={tp_total} FP={fp_total} FN={fn_total} | Shown={len(union_shown)}\n"
        f"{method.upper()} space colored by correctness"
    )
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.2)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="TP", markerfacecolor="#2ca02c", markersize=10),
            plt.Line2D([0], [0], marker="o", color="w", label="FP", markerfacecolor="#d62728", markersize=10),
            plt.Line2D([0], [0], marker="o", color="w", label="FN", markerfacecolor="#1f77b4", markersize=10),
        ],
        loc="best",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_precision_recall(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      precision_per_question, recall_per_question, summary_stats
    """
    precisions: List[float] = []
    recalls: List[float] = []

    tp_total = fp_total = fn_total = 0

    for r in results:
        selected_norm, _ = extract_selected_columns(r)
        gold_norm, _ = extract_gold_columns(r.get("actual_result", ""))

        tp = selected_norm & gold_norm
        fp = selected_norm - gold_norm
        fn = gold_norm - selected_norm

        tp_total += len(tp)
        fp_total += len(fp)
        fn_total += len(fn)

        pred = len(tp) + len(fp)
        gold = len(tp) + len(fn)

        precision = (len(tp) / pred) if pred > 0 else 0.0
        recall = (len(tp) / gold) if gold > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    precisions_arr = np.asarray(precisions, dtype=np.float32)
    recalls_arr = np.asarray(recalls, dtype=np.float32)

    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    summary_stats = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_precision_mean": float(precisions_arr.mean()) if len(precisions_arr) else 0.0,
        "macro_recall_mean": float(recalls_arr.mean()) if len(recalls_arr) else 0.0,
    }
    return precisions_arr, recalls_arr, summary_stats


def plot_precision_recall(
    precisions: np.ndarray,
    recalls: np.ndarray,
    stats: Dict[str, float],
    out_path: Path,
    bins: int = 20,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(recalls, bins=bins, color="#1f77b4", alpha=0.9)
    axes[0].set_title(f"Recall distribution (n={len(recalls)})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(precisions, recalls, alpha=0.6, s=25, c="#7f7f7f")
    axes[1].set_title("Precision vs Recall (per question)")
    axes[1].set_xlabel("Precision")
    axes[1].set_ylabel("Recall")
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(
        "Schema Filter Performance\n"
        f"macro_prec_mean={stats['macro_precision_mean']:.3f}, macro_recall_mean={stats['macro_recall_mean']:.3f}, "
        f"micro_prec={stats['micro_precision']:.3f}, micro_recall={stats['micro_recall']:.3f}",
        y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_average_recall_by_schema_index(results: List[Dict]) -> Tuple[List[float], List[int]]:
    """
    For each schema index i, compute average recall across questions that have:
      - at least one gold column
      - schema candidate at index i

    recall_i(q) = |schema_i_columns ∩ gold_columns| / |gold_columns|
    """
    recall_sums: Dict[int, float] = {}
    recall_counts: Dict[int, int] = {}

    for r in results:
        gold_norm, _ = extract_gold_columns(r.get("actual_result", ""))
        if not gold_norm:
            continue

        schemas = r.get("schemas", []) or []
        for idx, schema_obj in enumerate(schemas):
            schema_cols = extract_columns_from_schema(schema_obj)
            tp = len(schema_cols & gold_norm)
            recall = tp / len(gold_norm)
            recall_sums[idx] = recall_sums.get(idx, 0.0) + recall
            recall_counts[idx] = recall_counts.get(idx, 0) + 1

    if not recall_sums:
        return [], []

    max_idx = max(recall_sums.keys())
    avg_recalls: List[float] = []
    counts: List[int] = []
    for idx in range(max_idx + 1):
        count = recall_counts.get(idx, 0)
        avg = (recall_sums[idx] / count) if count > 0 and idx in recall_sums else 0.0
        avg_recalls.append(avg)
        counts.append(count)
    return avg_recalls, counts


def plot_schema_index_average_recall(
    avg_recalls: List[float],
    counts: List[int],
    out_path: Path,
) -> None:
    if not avg_recalls:
        raise ValueError("No schema-index recall values to plot (possibly no gold columns were parsed).")

    x = np.arange(len(avg_recalls))
    labels = [f"Schema {i+1}" for i in x]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, avg_recalls, color="#4c78a8", alpha=0.9)
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Average recall")
    plt.xlabel("Schema index")
    plt.title("Average Recall by Schema Candidate Index")
    plt.grid(axis="y", alpha=0.2)

    # Annotate bars with value and sample count.
    for i, b in enumerate(bars):
        y = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            min(1.03, y + 0.03),
            f"{y:.3f}\n(n={counts[i]})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "schema_filter_results_2.json"),
        help="Path to schema_filter_results_2.json",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "plot"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        default=None,
        help="If provided, generate TP/FP/FN scatter plot for this question_id",
    )
    parser.add_argument(
        "--question_index",
        type=int,
        default=0,
        help="Used when --question_id is not provided (0-based index into JSON list)",
    )
    parser.add_argument(
        "--question_index_start",
        type=int,
        default=0,
        help="Start index for multiple-question plotting (used when --question_id is not provided)",
    )
    parser.add_argument(
        "--question_count",
        type=int,
        default=4,
        help="How many consecutive questions to plot (>=4). Used when --question_id is not provided",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200,
        help="Cap number of columns plotted per question for speed",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="Dimensionality reduction method for scatter plot",
    )
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (auto-adjusted if needed)")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used for embeddings",
    )
    parser.add_argument(
        "--label_top_k",
        type=int,
        default=6,
        help="Label top-k closest columns to the question in 2D (scatter plot only)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    with input_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if not isinstance(results, list) or not results:
        raise ValueError("Input JSON must be a non-empty list of question result objects.")

    plot_paths: List[Path] = []

    if args.question_id is not None:
        chosen = choose_umap_question(results, args.question_id, args.question_index)
        chosen_id = str(chosen.get("question_id", "unknown"))
        chosen_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", chosen_id)

        plot1 = outdir / f"schema_filter_tp_fp_fn_{args.method}_question_{chosen_safe}.png"
        plot_tsne_tp_fp_fn(
            question_obj=chosen,
            method=args.method,
            perplexity=args.perplexity,
            random_state=args.random_state,
            out_path=plot1,
            label_top_k=args.label_top_k,
            max_points=args.max_points,
        )
        plot_paths.append(plot1)
    else:
        q_count = max(4, int(args.question_count))
        start = max(0, int(args.question_index_start))
        end = min(len(results), start + q_count)
        for idx in range(start, end):
            q = results[idx]
            qid = str(q.get("question_id", "unknown"))
            qid_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", qid)
            plot_path = outdir / f"schema_filter_tp_fp_fn_{args.method}_question_{qid_safe}_i{idx}.png"
            plot_tsne_tp_fp_fn(
                question_obj=q,
                method=args.method,
                perplexity=args.perplexity,
                random_state=args.random_state + idx,  # slight jitter across questions
                out_path=plot_path,
                label_top_k=args.label_top_k,
                max_points=args.max_points,
            )
            plot_paths.append(plot_path)

    # Plot 2: Precision/Recall distributions across all questions.
    precisions, recalls, stats = compute_precision_recall(results)
    plot2 = outdir / "schema_filter_precision_recall.png"
    plot_precision_recall(precisions, recalls, stats, plot2, bins=20)
    plot_paths.append(plot2)

    # Plot 3: Average recall by schema index.
    avg_recalls, counts = compute_average_recall_by_schema_index(results)
    plot3 = outdir / "schema_filter_schema_index_avg_recall.png"
    plot_schema_index_average_recall(avg_recalls, counts, plot3)
    plot_paths.append(plot3)

    print(f"Saved plots to: {', '.join(str(p) for p in plot_paths)}")


if __name__ == "__main__":
    # Ensure repo root is on sys.path when running as a script.
    sys.path.insert(0, str(REPO_ROOT))
    main()

