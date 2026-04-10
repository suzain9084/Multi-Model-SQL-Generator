from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import json
from torch.utils.data import Dataset

@dataclass
class SchemaFilterExample:
    question: str
    evidence: Optional[str]
    schemas: List[Dict[str, Any]]

class FilteredSchemaDataset(Dataset):
    def __init__(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, Sequence):
            raise ValueError("Expected JSON file to contain a list of examples.")

        self._examples: List[SchemaFilterExample] = []
        self._actual_result: List[str] = []
        for item in data:
            schemas = item.get("schemas", [])
            if (len(schemas) != 0):
                self._actual_result.append(item.get("actual_result", ""))
                self._examples.append(
                    SchemaFilterExample(
                        question=item.get("question", ""),
                        evidence=item.get("evidence"),
                        schemas=schemas,
                    )
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> (SchemaFilterExample, str):
        return self._examples[idx], self._actual_result[idx]

