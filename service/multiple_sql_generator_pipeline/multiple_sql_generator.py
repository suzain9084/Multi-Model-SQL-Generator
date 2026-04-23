from pathlib import Path
from typing import Optional, Sequence, Union
from sqlglot import parse
from sqlglot.errors import ParseError
from service.sql_execution.sql_execution import SqlExecutionService

class MultipleSqlGenerator:
    def __init__(self, models, parse_dialect: str = "sqlite"):
        self.models = models
        self.parse_dialect = parse_dialect
        self.sql_execution_service = SqlExecutionService()

    def is_valid(self, result):
        return not isinstance(result, str)

    def check_sql_syntax(self, sql: str):
        text = (sql or "").strip()
        if not text:
            return "empty query"
        try:
            parsed = parse(text, dialect=self.parse_dialect)
            if not parsed:
                return "no SQL statements parsed"
            return []
        except ParseError as e:
            return str(e)
        except Exception as e:
            return str(e)

    def refine(self, model, question, schema, evidence, sql, error):
        try:
            prompt = f"""
            Question: {question}
            Evidence: {evidence}
            Schema: {schema}
            Previous SQL: {sql}
            Error: {error}
            Fix the SQL query.
            """
            return model.predict(prompt, schema)
        except:
            return sql

    def generateMultipleSql(
        self,
        schemas,
        question,
        evidence=None,
        db=None,
        check_syntax=None,
        sqlfile_path=None
    ):
        if check_syntax is None:
            check_syntax = bool(db)

        sqls = []

        for model in self.models:
            for schema in schemas:
                sql = model.predict(question, schema)
                result = None

                if sqlfile_path is None:
                    result = self.check_sql_syntax(sql)
                    if not self.is_valid(result):
                        sql = self.refine(
                            model, question, schema, evidence, sql, result
                        )
                    result = self.check_sql_syntax(sql)
                elif sqlfile_path is not None:
                    result = self.sql_execution_service.execute_query(
                        sql=sql, sqlfile_path=sqlfile_path
                    )
                    
                sqls.append((sql, result))
        return sqls

    def fine_tune_all_backbones(
        self,
        train_dataset,
        output_dirs: Sequence[Union[str, Path]],
        epochs: int = 3,
        *,
        model_labels: Optional[Sequence[str]] = None,
    ) -> None:
        dirs = list(output_dirs)
        if len(dirs) != len(self.models):
            raise ValueError(
                f"Need one output_dir per model (got {len(dirs)} dirs, {len(self.models)} models)."
            )
        labels = list(model_labels) if model_labels is not None else []
        for i, model in enumerate(self.models):
            out = Path(dirs[i])
            label = labels[i] if i < len(labels) else f"model_{i + 1}"
            print(
                f"=== backbone {i + 1}/{len(self.models)} ({label}) -> {out} ===",
                flush=True,
            )
            out.mkdir(parents=True, exist_ok=True)
            model.fine_tune(train_dataset, output_dir=str(out), epochs=epochs)
            model.save(str(out))
            print(f"Saved {out}", flush=True)
