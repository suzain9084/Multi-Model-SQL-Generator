from typing import Dict, List, Optional
import torch

from service.schema_filter.model_resources import get_shared_resources, SharedNLPResources

class ValueRetriever:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 3,
        resources: Optional[SharedNLPResources] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.k = k
        self.resources = resources or get_shared_resources()

        try:
            self.encoder = self.resources.get_encoder(model_name=model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model '{model_name}': {e}")

        try:
            self.nlp = self.resources.get_spacy("en_core_web_sm")
        except Exception as e:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.type_hints = {
            "NUMBER": ["id", "count", "num", "amount", "price", "year", "score", "rate", "age"],
            "DATE": ["date", "time", "year", "month", "day"],
            "TIME": ["time", "hour", "minute", "second"],
            "PERSON": ["name", "person", "author", "director", "inspector", "owner", "customer"],
            "ORG": ["company", "org", "organization", "publisher", "school", "university"],
            "GPE": ["country", "city", "state", "region", "location", "address"],
            "PROPN": ["name", "title", "type", "category"],
        }

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0)
        return self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )
    
    def extract_values_from_question(self, cleanQuestion: str) -> List[Dict]:
        values = []

        if not self.nlp:
            return []

        doc = self.nlp(cleanQuestion)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY"]:
                values.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "token": ent.root
                })

        for token in doc:
            if token.pos_ == "NUM":
                values.append({
                    "text": token.text,
                    "type": "NUMBER",
                    "token": token
                })

        for token in doc:
            if token.pos_ == "PROPN":
                values.append({
                    "text": token.text,
                    "type": "PROPN",
                    "token": token
                })

        return values


    def retrieve_values(self, cleanQuestion: str, selected_col: List[Dict]):
        values = self.extract_values_from_question(cleanQuestion)
        results = {}
        if not values or not selected_col:
            return results

        column_metas = []
        for sample in selected_col:
            col = sample["column"]
            tab = sample["table"]
            column_metas.append(f"{tab}.{col}")

        value_payloads = [(f"{value['type']}-{value['text']}", value["text"]) for value in values]

        column_embeddings = self.get_embeddings(column_metas)
        value_embeddings = self.get_embeddings([item[0] for item in value_payloads])
        similarity = torch.matmul(value_embeddings, column_embeddings.T)

        topk_sims, topk_idx = torch.topk(similarity, k=min(2, similarity.shape[1]), dim=1)
        for i, (_, raw_value_text) in enumerate(value_payloads):
            best_sim = float(topk_sims[i, 0].item())
            best_col = column_metas[int(topk_idx[i, 0].item())]
            second_best = float(topk_sims[i, 1].item()) if topk_sims.shape[1] > 1 else -1.0
            margin = best_sim - second_best

            value_type = values[i]["type"]
            adaptive_threshold = 0.30 if value_type in {"NUMBER", "DATE", "TIME"} else 0.35
            is_type_compatible = self._is_type_compatible(best_col, value_type)

            if best_col and best_sim >= adaptive_threshold and margin >= 0.03 and is_type_compatible:
                results[best_col] = raw_value_text

        return results

    def _is_type_compatible(self, column_name: str, value_type: str) -> bool:
        table_col = column_name.lower()
        hints = self.type_hints.get(value_type, [])
        if not hints:
            return True
        return any(h in table_col for h in hints)