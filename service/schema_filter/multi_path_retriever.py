from typing import List, Dict, Tuple, Set
from collections import defaultdict

class MultiPathSchemaRetriever:
    def __init__(self,  k: int = 20, coverage_bonus: float = 0.1, min_score: float = 0.05):
        self.k = k
        self.coverage_bonus = coverage_bonus
        self.min_score = min_score

    def identify_keys(
        self,
        selected: Set[Tuple[str, str]],
        schema: Dict,
    ) -> Set[Tuple[str, str]]:

        expanded = set()

        for table, _ in selected:
            pks = schema.get("primary_keys", {}).get(table, [])
            if isinstance(pks, str):
                pks = [pks]
            for pk in pks:
                expanded.add((table, pk))

        for fk, ref in schema.get("foreign_keys", []):
            t1, c1 = fk.split(".")
            t2, c2 = ref.split(".")

            if (t1, c1) in selected:
                expanded.add((t2, c2))
            if (t2, c2) in selected:
                expanded.add((t1, c1))

        return expanded

    def fMs(self, candidates):
        column_scores = defaultdict(float)
        keyword_hits = defaultdict(int)

        for item in candidates:
            key = (item["table"], item["column"])

            if item["score"] >= self.min_score:
                column_scores[key] += item["score"]
                keyword_hits[key] += 1

        ranked = []

        for key in column_scores:
            score = column_scores[key]
            score += self.coverage_bonus * keyword_hits[key]
            ranked.append((key, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        return set([x[0] for x in ranked[: self.k]])

    def retrieve(
        self,
        schema: Dict,
        scored_candidates: List[Dict],
        ps: int = 3,
    ) -> List[Set[Tuple[str, str]]]:

        Srtrv = list(scored_candidates)
        S = []
        accumulated_schema = set()

        for _ in range(ps):
            if not Srtrv:
                break

            Sslct_i = self.fMs(Srtrv)

            if not Sslct_i:
                break

            Pi = self.identify_keys(Sslct_i, schema)

            before_size = len(accumulated_schema)
            accumulated_schema |= Sslct_i | Pi
            S.append(accumulated_schema.copy())
            Srtrv = [
                x for x in Srtrv
                if (x["table"], x["column"]) not in (Sslct_i | Pi)
            ]

            if len(accumulated_schema) == before_size:
                break

        return S