from collections import defaultdict
import math

class SqlSelector:
    def __init__(self, selection_model, generator_order=None):
        self.selection_model = selection_model
        self.generator_order = generator_order or {}

    def cluster_by_result(self, sql_results):
        clusters = defaultdict(list)

        for idx, (sql, result) in enumerate(sql_results):
            key = str(result)
            clusters[key].append((idx, sql, result))

        return list(clusters.values())

    def sort_clusters(self, clusters):
        return sorted(clusters, key=lambda c: len(c), reverse=True)

    def sort_within_cluster(self, cluster):
        return sorted(
            cluster,
            key=lambda x: self.generator_order.get(x[0], 0),
            reverse=True
        )

    def shortest_sql(self, cluster):
        return min(cluster, key=lambda x: len(x[1]))

    def select(self, sql_results, question, schemas, evidence=None):
        clusters = self.cluster_by_result(sql_results)

        schema_union = " ".join(map(str, schemas))

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
            candidates=reorganized_sqls
        )

        return final_sql