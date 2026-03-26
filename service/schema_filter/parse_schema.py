from typing import Dict, List, Tuple
import sqlglot
from sqlglot import exp


class SchemaParser:
    def __init__(self):
        pass

    def parse_tables(self, schema_sql: str) -> Dict[str, List[str]]:
        tables = {}
        statements = sqlglot.parse(schema_sql)

        for stmt in statements:
            if isinstance(stmt, exp.Create) and stmt.this:
                table_name = stmt.this.name.lower()

                columns = []
                for col in stmt.find_all(exp.ColumnDef):
                    columns.append(col.name.lower())

                tables[table_name] = columns
        return tables

    def extract_primary_keys(self, schema_sql: str) -> Dict[str, str]:
        pk_map = {}
        statements = sqlglot.parse(schema_sql)

        for stmt in statements:
            if isinstance(stmt, exp.Create) and stmt.this:
                table_name = stmt.this.name.lower()

                for col in stmt.find_all(exp.ColumnDef):
                    constraints = col.args.get("constraints") or []
                    for c in constraints:
                        if isinstance(c, exp.PrimaryKeyColumnConstraint):
                            pk_map[table_name] = col.name.lower()

                for constraint in stmt.find_all(exp.PrimaryKey):
                    cols = constraint.expressions
                    if cols:
                        pk_map[table_name] = cols[0].name.lower()
        return pk_map

    def extract_foreign_keys(self, schema_sql: str) -> List[Tuple[str, str]]:
        foreign_keys: List[Tuple[str, str]] = []
        statements = sqlglot.parse(schema_sql)

        for stmt in statements:
            if isinstance(stmt, exp.Create) and stmt.this:
                table_name = stmt.this.name.lower()

                for fk in stmt.find_all(exp.ForeignKey):
                    local_cols = fk.expressions
                    ref = fk.args.get("reference")

                    if ref:
                        ref_table = ref.this.name.lower()
                        ref_cols = ref.expressions

                        for lc, rc in zip(local_cols, ref_cols):
                            local = f"{table_name}.{lc.name.lower()}"
                            remote = f"{ref_table}.{rc.name.lower()}"
                            foreign_keys.append((local, remote))

                for col in stmt.find_all(exp.ColumnDef):
                    constraints = col.args.get("constraints") or []

                    for c in constraints:
                        if isinstance(c, exp.ForeignKey):
                            ref = c.args.get("reference")

                            if ref:
                                ref_table = ref.this.name.lower()
                                ref_col = ref.expressions[0].name.lower()

                                local = f"{table_name}.{col.name.lower()}"
                                remote = f"{ref_table}.{ref_col}"

                                foreign_keys.append((local, remote))

        return foreign_keys

    def parse(self, schema_sql: str) -> Dict:
        return {
            "tables": self.parse_tables(schema_sql),
            "primary_keys": self.extract_primary_keys(schema_sql),
            "foreign_keys": self.extract_foreign_keys(schema_sql),
        }