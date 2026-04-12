from typing import Dict, List, Optional
import json

from service.schema_filter.parse_schema import SchemaParser
from service.schema_filter.keyword_extractor import KeywordExtractor
from service.schema_filter.column_selector import ColumnSelector
from service.schema_filter.value_retriever import ValueRetriever
from service.schema_filter.multi_path_retriever import MultiPathSchemaRetriever
from service.schema_filter.model_resources import get_shared_resources

class SchemaFilterPipeline:
    def __init__(
        self,
        top_k_retrieval: int = 20,
        num_schemas: int = 2
    ):
        shared_resources = get_shared_resources()
        self.schema_parser = SchemaParser()
        self.keyword_extractor = KeywordExtractor(resources=shared_resources)
        self.column_selector = ColumnSelector(resources=shared_resources)
        self.value_retriever = ValueRetriever(resources=shared_resources)
        self.multi_path_retriever = MultiPathSchemaRetriever(k=top_k_retrieval)
        self.top_k_retrieval = top_k_retrieval
        self.num_schemas = num_schemas
    
    def process_sample(
        self,
        question: str,
        schema_data: Dict,
        actual_result: str,
        evidence: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> Dict:
        keywords = self.keyword_extractor.extract_keywords(question, evidence)
        parsed_schema = self.schema_parser.parse(schema_data)
        
        selected_col = self.column_selector.select_columns_iterative(
            parsed_schema, 
            keywords,
            f"{question} {evidence}"
        )
        
        retrieve_value = self.value_retriever.retrieve_values(
            cleanQuestion=f"{question} {evidence}",
            selected_col=selected_col
        )

        filtered_schemas = self.multi_path_retriever.retrieve(
            schema=parsed_schema,
            scored_candidates=selected_col,
            ps=self.num_schemas,
        )

        # Keep valid table.column shape only.
        retrieve_value = {
            key: value
            for key, value in retrieve_value.items()
            if "." in key and not key.startswith(".")
        }

        result = {
            "question_id": question_id or "unknown",
            "question": question,
            "evidence": evidence,
            "keywords": keywords,
            "schemas": filtered_schemas,
            "actual_result": actual_result,
            "values": retrieve_value,
            "statistics": {
                "num_retrieved_val": len(retrieve_value),
                "num_selected_col": len(selected_col),
                "num_schemas": len(filtered_schemas)
            }
        }

        return result
    
    def save_results(self, results: List[Dict], output_path: str):
        formatted_results = []

        for result in results:
            formatted_schemas = []
            for schema in result.get("schemas", []):
                table_map = {}
                for table, column in schema:
                    if not table:
                        continue
                    table_map.setdefault(table, set()).add(column)

                if table_map:
                    formatted_schemas.append({
                        "tables": [
                            {
                                "name": table,
                                "columns": sorted(list(columns)),
                            }
                            for table, columns in sorted(table_map.items())
                        ]
                    })
            
            formatted_result = {
                "question_id": result["question_id"],
                "question": result["question"],
                "evidence": result.get("evidence"),
                "keywords": result.get("keywords", []),
                "values": result["values"],
                "num_selected_columns": result["statistics"].get("num_selected_col"),
                "schemas": formatted_schemas,
                "statistics": result["statistics"],
            }
            if result.get("actual_result") is not None:
                formatted_result["actual_result"] = result["actual_result"]
            formatted_results.append(formatted_result)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(formatted_results)} results to {output_path}")
