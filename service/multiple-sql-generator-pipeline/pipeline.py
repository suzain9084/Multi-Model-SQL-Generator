class MultipleSqlGenerator:
    def __init__(self, models):
        self.models = models

    def is_valid(self, result):
        return not isinstance(result, str)

    def execute_sql(self, sql, db):
        try:
            result = db.execute(sql)
            return result.fetchall() if result else []
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

    def generateMultipleSql(self, schemas, question, evidence=None, db=None):
        sqls = []

        for model in self.models:
            for schema in schemas:
                sql = model.predict(question, schema)
                if db:
                    result = self.execute_sql(sql, db)
                    if not self.is_valid(result):
                        sql = self.refine(
                            model, question, schema, evidence, sql, result
                        )
                sqls.append((sql, result))
        return sqls