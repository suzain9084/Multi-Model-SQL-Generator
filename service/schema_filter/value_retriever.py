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

        best_sims, best_idx = torch.max(similarity, dim=1)
        for i, (_, raw_value_text) in enumerate(value_payloads):
            best_sim = float(best_sims[i].item())
            best_col = column_metas[int(best_idx[i].item())]
            if best_col and best_sim >= 0.2:
                results[best_col] = raw_value_text

        return results