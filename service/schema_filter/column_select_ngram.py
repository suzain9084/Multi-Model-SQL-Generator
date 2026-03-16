from __future__ import annotations

from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
import math
import re


class ColumnSelectNGram:
    _CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
    _NON_ALNUM = re.compile(r"[^a-z0-9]+")

    def __init__(
        self,
        schema: Dict[str, List[str]],
        training_data: List[Dict],
        n: int = 2,
        k: int = 5,
        alpha: float = 0.5,
    ):
        if n < 2:
            raise ValueError("n must be >= 2 for an N-gram model.")
        if k < 1:
            raise ValueError("k must be >= 1.")

        self.schema: Dict[str, List[str]] = schema
        self.training_data: List[Dict] = training_data
        self.n: int = n
        self.k: int = k
        self.alpha: float = alpha

        self._context_counts: Counter[Tuple[str, ...]] = Counter()
        self._ngram_counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self._next_vocab: set[str] = set()
        self._vocab_size: int = 1

        self._table_counts: Counter[str] = Counter()
        self._column_counts: Counter[Tuple[str, str]] = Counter()
        self._total_column_events: int = 0

        self._trained: bool = False

    @classmethod
    def normalize_and_tokenize(cls, text: str) -> List[str]:
        if not text:
            return []

        text = cls._CAMEL_BOUNDARY.sub(" ", text)
        text = text.replace("_", " ")
        text = text.lower()
        text = cls._NON_ALNUM.sub(" ", text)
        return [t for t in text.split() if t]

    def _contexts_from_tokens(self, tokens: Sequence[str]) -> List[Tuple[str, ...]]:
        ctx_len = self.n - 1
        if ctx_len <= 0:
            return [tuple()]

        if not tokens:
            return [tuple(["<s>"] * ctx_len)]

        padded = ["<s>"] * ctx_len + list(tokens)
        contexts: List[Tuple[str, ...]] = []
        for i in range(ctx_len, len(padded)):
            ctx = tuple(padded[i - ctx_len : i])
            contexts.append(ctx)
        return contexts

    @staticmethod
    def _split_table_column(qualified: str) -> Tuple[str, str]:
        if not qualified:
            return "", ""
        if "." not in qualified:
            return "", qualified
        table, col = qualified.split(".", 1)
        return table, col

    def _column_context(self, table: str, column: str) -> Tuple[str, ...]:
        tokens = self.normalize_and_tokenize(table) + self.normalize_and_tokenize(column)
        ctxs = self._contexts_from_tokens(tokens)
        return ctxs[-1] if ctxs else tuple(["<s>"] * (self.n - 1))

    def train(self) -> None:
        self._context_counts.clear()
        self._ngram_counts.clear()
        self._next_vocab.clear()
        self._table_counts.clear()
        self._column_counts.clear()
        self._total_column_events = 0

        for sample in self.training_data:
            question = str(sample.get("question", "") or "")
            selected_cols = sample.get("columns", []) or []

            q_tokens = self.normalize_and_tokenize(question)
            if not q_tokens:
                continue

            for w in q_tokens:
                self._next_vocab.add(w)

            for qualified in selected_cols:
                table, col = self._split_table_column(str(qualified))
                if not col:
                    continue

                self._table_counts[table] += 1
                self._column_counts[(table, col)] += 1
                self._total_column_events += 1

                ctx = self._column_context(table, col)
                for w in q_tokens:
                    self._ngram_counts[ctx][w] += 1
                    self._context_counts[ctx] += 1

        self._vocab_size = max(1, len(self._next_vocab) or 1)
        self._trained = True

    def _prob(self, next_token: str, context: Tuple[str, ...]) -> float:
        if not self._trained:
            self.train()

        count_ctx = self._context_counts.get(context, 0)
        count_pair = self._ngram_counts.get(context, Counter()).get(next_token, 0)
        return (count_pair + 1.0) / (count_ctx + float(self._vocab_size))

    def _table_prior(self, table: str) -> float:
        num_tables = max(1, len(self.schema))
        return (self._table_counts.get(table, 0) + 1.0) / (
            self._total_column_events + float(num_tables)
        )

    def _column_given_table(self, table: str, column: str) -> float:
        table_total = self._table_counts.get(table, 0)
        num_cols_in_table = max(1, len(self.schema.get(table, [])))
        return (self._column_counts.get((table, column), 0) + 1.0) / (
            table_total + float(num_cols_in_table)
        )

    def score_column(self, query_tokens: Sequence[str], table: str, column: str) -> float:
        if not self._trained:
            self.train()

        ctx = self._column_context(table, column)

        log_likelihood = 0.0
        for w in query_tokens:
            if w not in self._next_vocab:
                self._next_vocab.add(w)
                self._vocab_size = max(1, len(self._next_vocab))
            log_likelihood += math.log(self._prob(w, ctx))

        p_t = self._table_prior(table)
        p_c_given_t = self._column_given_table(table, column)

        return math.log(p_t) + math.log(p_c_given_t) + log_likelihood

    def select_columns(self, question: str, keywords: List[str]) -> List[Tuple[str, float]]:
        if not self._trained:
            self.train()

        query_tokens: List[str] = []
        for kw in keywords or []:
            query_tokens.extend(self.normalize_and_tokenize(kw))

        if not query_tokens:
            query_tokens = self.normalize_and_tokenize(question)
        if not query_tokens:
            query_tokens = ["<s>"]

        scored: List[Tuple[str, float]] = []

        for table, cols in self.schema.items():
            for col in cols:
                score = self.score_column(query_tokens, table, col)
                scored.append((f"{table}.{col}", score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.k]


if __name__ == "__main__":
    schema = {
        "publishers": ["publisher_id", "publisher_name", "country"],
        "games": ["game_id", "publisher_id", "global_sales", "num_sales", "release_year"],
    }

    training_data = [
        {
            "question": "List publishers with sales less than 10000",
            "columns": ["publishers.publisher_name", "games.global_sales", "games.num_sales"],
        },
        {
            "question": "Show publisher names and their total sales",
            "columns": ["publishers.publisher_name", "games.global_sales", "games.num_sales"],
        },
        {
            "question": "Find games released in 2020 with high sales",
            "columns": ["games.release_year", "games.global_sales"],
        },
        {
            "question": "Which publishers have the highest global sales?",
            "columns": ["publishers.publisher_name", "games.global_sales"],
        },
        {
            "question": "Count sales per publisher",
            "columns": ["publishers.publisher_name", "games.num_sales"],
        },
    ]

    selector = ColumnSelectNGram(schema=schema, training_data=training_data, n=2, k=5, alpha=0.5)
    selector.train()

    question = "List publishers with sales less than 10000"
    keywords = ["publishers", "sales"]

    top = selector.select_columns(question=question, keywords=keywords)
    for qualified, score in top:
        print(f"{qualified}\t{score:.4f}")


