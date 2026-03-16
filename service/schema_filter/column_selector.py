from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class ColumnSelector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", k: int = 5):
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(model_name).to(self.device)

    def embed(self, text: str) -> torch.Tensor:
        return self.encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )

    def compute_score(
        self,
        v_key: torch.Tensor,
        v_col: torch.Tensor,
        v_tab: torch.Tensor,
        v_qe: torch.Tensor,
    ) -> float:
        return (
            F.cosine_similarity(v_key, v_col, dim=0)
            * F.cosine_similarity(v_qe, v_tab, dim=0)
        ).item()

    def select_columns_iterative(
        self,
        schema: Dict,
        keywords: List[str],
        clean_question: str
    ):

        v_qe = self.embed(clean_question)

        keyword_embeddings = {
            kw: self.embed(kw) for kw in keywords
        }

        table_embeddings = {}
        column_embeddings = {}

        for table, cols in schema["tables"].items():
            table_embeddings[table] = self.embed(table)
            for col in cols:
                column_embeddings[(table, col)] = self.embed(col)

        scored_results = []

        for kw, v_key in keyword_embeddings.items():
            for (table, col), v_col in column_embeddings.items():
                v_tab = table_embeddings[table]

                score = self.compute_score(v_key, v_col, v_tab, v_qe)

                scored_results.append({
                    "keyword": kw,
                    "table": table,
                    "column": col,
                    "score": score
                })

        return scored_results
