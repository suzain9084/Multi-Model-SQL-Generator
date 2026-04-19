from collections import defaultdict
import math
from sqlglot import parse


class SqlSelector:
    def __init__(self, selection_model, generator_order=None, parse_dialect: str = "duckdb"):
        self.selection_model = selection_model
        self.generator_order = generator_order or {}
        self.parse_dialect = parse_dialect

    def _canonical_sql(self, sql: str) -> str:
        text = (sql or "").strip()
        if not text:
            return ""
        try:
            exprs = parse(text, dialect=self.parse_dialect)
            parts = [
                e.sql(dialect=self.parse_dialect)
                for e in exprs
                if e is not None
            ]
            return " | ".join(parts) if parts else text
        except Exception:
            return text

    def _cluster_key(self, sql: str, result, is_executed: bool = False):
        if isinstance(result, dict):
            if is_executed:
                if result.get("success") is True:
                    signature = result.get("signature")
                    if signature:
                        return ("exec_signature", str(signature))
                    return ("exec_signature", self._canonical_sql(sql))
                err = result.get("error") or "execution_error"
                return ("failed", str(err))
            status = result.get("status")
            if status == "success":
                return ("ok", self._canonical_sql(sql))
            err = result.get("error") or status or "error"
            return ("failed", str(err))
        if isinstance(result, str):
            return ("syntax_error", result)
        if result is None:
            return ("unchecked", self._canonical_sql(sql))
        return ("ok", self._canonical_sql(sql))

    def cluster_by_result(self, sql_results, is_executed: bool = False):
        clusters = defaultdict(list)

        for idx, (sql, result) in enumerate(sql_results):
            key = self._cluster_key(sql, result, is_executed=is_executed)
            clusters[key].append((idx, sql, result))

        return list(clusters.values())

    def sort_clusters(self, clusters):
        return sorted(clusters, key=lambda c: len(c), reverse=True)

    def sort_within_cluster(self, cluster):
        return sorted(
            cluster,
            key=lambda x: self.generator_order.get(x[0], 0),
            reverse=True,
        )

    def shortest_sql(self, cluster):
        return min(cluster, key=lambda x: len(x[1]))

    def select(self, sql_results, question, schemas, evidence=None, isExecuted: bool = False):
        clusters = self.cluster_by_result(sql_results, is_executed=isExecuted)

        schema_union = " ".join(map(str, schemas))

        if not isExecuted:
            non_syntax_error_candidates = [
                sql
                for sql, result in sql_results
                if self._cluster_key(sql, result, is_executed=False)[0] != "syntax_error"
            ]
            if non_syntax_error_candidates:
                return self.selection_model.predict(
                    question=question,
                    schema=schema_union,
                    evidence=evidence,
                    candidates=non_syntax_error_candidates,
                )

        if len(clusters) == 1:
            best = self.shortest_sql(clusters[0])
            return best[1]

        clusters = self.sort_clusters(clusters)

        sorted_clusters = []
        for cluster in clusters:
            sorted_clusters.append(self.sort_within_cluster(cluster))

        total_sqls = len(sql_results)
        majority_threshold = math.ceil(total_sqls / 2)

        reorganized_sqls = []

        if len(sorted_clusters[0]) >= majority_threshold:
            for cluster in sorted_clusters:
                for item in cluster:
                    reorganized_sqls.append(item[1])

        else:
            for cluster in sorted_clusters:
                best = self.shortest_sql(cluster)
                reorganized_sqls.append(best[1])

        final_sql = self.selection_model.predict(
            question=question,
            schema=schema_union,
            evidence=evidence,
            candidates=reorganized_sqls,
        )

        return final_sql