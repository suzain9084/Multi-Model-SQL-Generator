import duckdb
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class SQLValidator:
    def __init__(self, schema_sql: str, data_sql: str = None, timeout: float = 0.5, max_workers: int = 4):
        self.schema_sql = schema_sql
        self.data_sql = data_sql
        self.timeout = timeout
        self.max_workers = max_workers
        self.cache = {}
        self.local = threading.local()

    def get_connection(self):
        if not hasattr(self.local, "conn"):
            conn = duckdb.connect(database=':memory:')
            conn.execute(self.schema_sql)
            if self.data_sql:
                conn.execute(self.data_sql)
            self.local.conn = conn
        return self.local.conn

    def reset_connection(self):
        conn = self._get_connection()
        try:
            conn.execute("ROLLBACK;")
        except:
            pass

    def validate_syntax(self, query: str):
        conn = self._get_connection()
        try:
            conn.execute(f"EXPLAIN {query}")
            return True, None
        except Exception as e:
            return False, str(e)

    def execute_query(self, query: str):
        conn = self._get_connection()
        result_container = {"result": None, "error": None}

        def target():
            try:
                conn.execute("BEGIN;")
                res = conn.execute(query).fetchall()
                result_container["result"] = res
                conn.execute("ROLLBACK;")
            except Exception as e:
                result_container["error"] = str(e)
                try:
                    conn.execute("ROLLBACK;")
                except:
                    pass

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(self.timeout)

        if thread.is_alive():
            return False, "Execution timeout"

        if result_container["error"]:
            return False, result_container["error"]

        return True, result_container["result"]

    def validate_and_execute(self, query: str):
        if query in self.cache:
            return self.cache[query]

        is_valid, error = self.validate_syntax(query)
        if not is_valid:
            result = {
                "status": "syntax_error",
                "error": error
            }
            self.cache[query] = result
            return result

        success, output = self.execute_query(query)
        if not success:
            result = {
                "status": "runtime_error",
                "error": output
            }
        else:
            result = {
                "status": "success",
                "result": output
            }

        self.cache[query] = result
        return result

    def process_batch(self, queries):
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.validate_and_execute, q): q for q in queries}

            for future in as_completed(futures):
                query = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "internal_error", "error": str(e)}
                results.append((query, result))
        return results