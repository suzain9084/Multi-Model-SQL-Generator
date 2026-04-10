from typing import Dict, List, Optional, Tuple
import torch

from service.schema_filter.model_resources import get_shared_resources, SharedNLPResources


class ColumnSelector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 5,
        top_per_keyword: int = 8,
        top_per_table: int = 20,
        min_score: float = 0.05,
        resources: Optional[SharedNLPResources] = None,
    ):
        self.k = k
        self.top_per_keyword = top_per_keyword
        self.top_per_table = top_per_table
        self.min_score = min_score
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resources = resources or get_shared_resources()
        self.encoder = self.resources.get_encoder(model_name=model_name, device=self.device)

    def embed(self, text: str) -> torch.Tensor:
        return self.encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )

    def embed_many(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0)
        return self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )

    def select_columns_iterative(
        self,
        schema: Dict,
        keywords: List[str],
        clean_question: str
    ):
        if not schema.get("tables"):
            return []

        unique_keywords = list(dict.fromkeys(keywords or []))
        if not unique_keywords:
            unique_keywords = [clean_question]

        table_names = list(schema["tables"].keys())
        column_pairs: List[Tuple[str, str]] = []
        for table, cols in schema["tables"].items():
            column_pairs.extend((table, col) for col in cols)

        if not column_pairs:
            return []

        v_qe = self.embed(clean_question)
        keyword_embeddings = self.embed_many(unique_keywords)
        table_embeddings = self.embed_many(table_names)
        column_embeddings = self.embed_many([col for _, col in column_pairs])

        table_sim = torch.matmul(table_embeddings, v_qe)
        table_idx = {table: idx for idx, table in enumerate(table_names)}
        col_table_sim = torch.stack([table_sim[table_idx[table]] for table, _ in column_pairs])

        # Embeddings are normalized; matrix multiply is cosine similarity.
        kw_col_sim = torch.matmul(keyword_embeddings, column_embeddings.T)
        combined_scores = kw_col_sim * col_table_sim.unsqueeze(0)

        scored_results = []
        for kw_idx, keyword in enumerate(unique_keywords):
            kw_scores = []
            for col_idx, (table, col) in enumerate(column_pairs):
                score = float(combined_scores[kw_idx, col_idx].item())
                if score < self.min_score:
                    continue
                kw_scores.append(
                    {
                        "keyword": keyword,
                        "table": table,
                        "column": col,
                        "score": score,
                    }
                )

            kw_scores.sort(key=lambda x: x["score"], reverse=True)
            scored_results.extend(kw_scores[: self.top_per_keyword])

        if not scored_results:
            return []

        by_table: Dict[str, List[Dict]] = {}
        for item in scored_results:
            by_table.setdefault(item["table"], []).append(item)

        pruned = []
        for table_items in by_table.values():
            table_items.sort(key=lambda x: x["score"], reverse=True)
            pruned.extend(table_items[: self.top_per_table])

        dedup = {}
        for item in pruned:
            key = (item["table"], item["column"], item["keyword"])
            prev = dedup.get(key)
            if prev is None or item["score"] > prev["score"]:
                dedup[key] = item

        return list(dedup.values())
