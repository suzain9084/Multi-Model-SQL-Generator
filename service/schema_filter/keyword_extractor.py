from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import spacy

class KeywordExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, percent: int = 0.8):
        self.percent = percent
        self.nlp = spacy.load("en_core_web_sm")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def cleanQuestion(self, question: str, evidence: Optional[str] = None) -> str:
        text = question
        if evidence:
            text = f"{question} {evidence}"

        doc = self.nlp(text.lower())

        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and not token.like_num
            and token.pos_ != "SYM"
        ]
        return " ".join(tokens)

    def generate_candidates(self, text: str) -> List[str]:
        tokens = text.split()
        return list(set(tokens))

    def embed(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeddings

    def extract_keywords(self, question: str, evidence: Optional[str] = None) -> List[Tuple[str, float]]:
        clean_text = self.cleanQuestion(question, evidence)
        candidates = self.generate_candidates(clean_text)

        if not candidates:
            return []

        question_embedding = self.embed([clean_text])
        candidate_embeddings = self.embed(candidates)

        scores = F.cosine_similarity(
            question_embedding,
            candidate_embeddings,
        )

        scored_keywords = list(zip(candidates, scores.tolist()))
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        scored_keywords = [keywords for keywords, _ in scored_keywords]
        return scored_keywords[: int(len(scored_keywords)*self.percent)]

