from typing import List, Optional
import torch
import torch.nn.functional as F

from service.schema_filter.model_resources import get_shared_resources, SharedNLPResources

class KeywordExtractor:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        percent: int = 0.8,
        resources: Optional[SharedNLPResources] = None,
    ):
        self.percent = percent
        self.resources = resources or get_shared_resources()
        self.nlp = self.resources.get_spacy("en_core_web_sm")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.resources.get_encoder(model_name=model_name, device=self.device)
        self.noise_terms = {"refer", "list", "show", "give", "get", "number"}

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
        doc = self.nlp(text)
        candidates = []

        for token in doc:
            lemma = token.lemma_.strip()
            if (
                lemma
                and not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(lemma) > 1
                and lemma not in self.noise_terms
            ):
                candidates.append(lemma)

        # Preserve informative phrases (e.g. "out of business", "timely response").
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if phrase and len(phrase.split()) <= 4:
                candidates.append(phrase)

        # Include named entities as atomic candidates.
        for ent in doc.ents:
            phrase = ent.text.strip().lower()
            if phrase:
                candidates.append(phrase)

        # De-duplicate while preserving order.
        return list(dict.fromkeys(candidates))

    def embed(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeddings

    def extract_keywords(self, question: str, evidence: Optional[str] = None) -> List[str]:
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
        keep_n = max(1, int(len(scored_keywords) * self.percent))
        return scored_keywords[:keep_n]

