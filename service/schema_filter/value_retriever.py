from typing import List, Tuple, Dict, Set, Optional
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer
import re
import spacy

class ValueRetriever:    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', k: int = 3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        try:
            self.encoder = SentenceTransformer(model_name)
            self.encoder = self.encoder.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model '{model_name}': {e}")
        
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        except Exception as e:
            print(f"Warning: Could not load RoBERTa tokenizer: {e}")
            self.tokenizer = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    
    def get_embedding(self, text: str ) -> torch.Tensor:        
        embedding = self.encoder.encode([text], convert_to_tensor=True, device=self.device)[0]        
        return embedding
    
    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        if vec1.dim() > 1:
            vec1 = vec1.squeeze()
        if vec2.dim() > 1:
            vec2 = vec2.squeeze()
        
        vec1_norm = F.normalize(vec1, p=2, dim=0)
        vec2_norm = F.normalize(vec2, p=2, dim=0)
        
        similarity = torch.dot(vec1_norm, vec2_norm).item()
        return similarity
    
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


    def retrieve_values(self, cleanQuestion: str, selected_col: Dict):
        values = self.extract_values_from_question(cleanQuestion)
        results = {}

        sample_embeddings = []
        for sample in selected_col:
            col = sample["column"]
            tab = sample["table"]
            column_meta = f"{tab}.{col}"
            emb = self.get_embedding(column_meta)
            sample_embeddings.append((column_meta, emb))

        for value in values:
            value_text = f"{value['type']}-{value['text']}"
            value_embedding = self.get_embedding(value_text)

            best_col = None
            best_sim = 0.0

            for column_meta, column_embedding in sample_embeddings:
                sim = self.cosine_similarity(
                    value_embedding,
                    column_embedding
                )

                if sim > best_sim:
                    best_sim = sim
                    best_col = column_meta

            if best_col and best_sim >= 0.2:
                results[best_col] = value_text.split("-")[1]

        return results