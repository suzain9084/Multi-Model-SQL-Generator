from typing import Dict, Optional, Tuple
import threading

import torch
import spacy
from sentence_transformers import SentenceTransformer


class SharedNLPResources:
    def __init__(self):
        self._lock = threading.Lock()
        self._encoders: Dict[Tuple[str, str], SentenceTransformer] = {}
        self._spacy_models: Dict[str, object] = {}

    def get_encoder(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> SentenceTransformer:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        key = (model_name, resolved_device)

        with self._lock:
            if key not in self._encoders:
                self._encoders[key] = SentenceTransformer(model_name, device=resolved_device)
            return self._encoders[key]

    def get_spacy(self, model_name: str = "en_core_web_sm"):
        with self._lock:
            if model_name not in self._spacy_models:
                self._spacy_models[model_name] = spacy.load(model_name)
            return self._spacy_models[model_name]


_shared_resources = SharedNLPResources()


def get_shared_resources() -> SharedNLPResources:
    return _shared_resources
