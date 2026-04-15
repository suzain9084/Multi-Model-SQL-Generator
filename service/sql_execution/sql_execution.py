from __future__ import annotations
import sqlite3
import hashlib
import json
from typing import Any, Dict, List


class SqlExecutionService:
    def __init__(self, timeout_seconds: float = 30.0, max_rows: int = 100) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def execute_query(self, sql: str, sqlfile_path: str) -> Dict[str, Any]:
        query = (sql or "").strip()

        if not query:
            return {"success": False, "error": "empty query", "rows": []}
        if not sqlfile_path:
            return {"success": False, "error": "missing sqlfile_path", "rows": []}

        connection = None
        try:
            connection = sqlite3.connect(sqlfile_path, timeout=self.timeout_seconds)
            cursor = connection.cursor()

            cursor.execute(query)

            rows = cursor.fetchall()
            columns = [item[0] for item in (cursor.description or [])]

            rows = self._normalize_rows(rows)
            rows = self._sort_rows(rows)
            rows = self._truncate_rows(rows)

            signature = self._result_signature(columns, rows)

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "signature": signature, 
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "rows": [],
            }

        finally:
            if connection is not None:
                connection.close()

    def _normalize_rows(self, rows: List[tuple]) -> List[List[Any]]:
        def normalize_value(v):
            if isinstance(v, float):
                return round(v, 6)
            if v is None:
                return None
            return v

        return [[normalize_value(v) for v in row] for row in rows]

    def _sort_rows(self, rows: List[List[Any]]) -> List[List[Any]]:
        try:
            return sorted(rows)
        except Exception:
            return rows

    def _truncate_rows(self, rows: List[List[Any]]) -> List[List[Any]]:
        return rows[: self.max_rows]

    def _result_signature(self, columns: List[str], rows: List[List[Any]]) -> str:
        try:
            payload = {
                "columns": columns,
                "rows": rows,
            }
            result_str = json.dumps(payload, sort_keys=True, default=str)
            return hashlib.md5(result_str.encode()).hexdigest()
        except Exception:
            return "hash_error"