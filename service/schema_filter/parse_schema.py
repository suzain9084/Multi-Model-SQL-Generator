from typing import Dict, List, Optional, Tuple
import sqlglot
from sqlglot import exp


class SchemaParser:
    def __init__(self):
        pass

    def _parse_statements(self, schema_sql: str):
        # Try multiple dialects because BIRD schemas mix quoting styles.
        best_statements = []
        best_create_count = -1
        for dialect in (None, "sqlite", "mysql"):
            try:
                if dialect is None:
                    statements = sqlglot.parse(schema_sql)
                else:
                    statements = sqlglot.parse(schema_sql, read=dialect)
            except Exception:
                continue

            create_count = sum(1 for stmt in statements if isinstance(stmt, exp.Create))
            if create_count > best_create_count:
                best_create_count = create_count
                best_statements = statements

        return best_statements

    def _extract_table_name(self, create_stmt: exp.Create) -> Optional[str]:
        target = create_stmt.this
        if not target:
            return None

        table_expr = target.this if hasattr(target, "this") and isinstance(target.this, exp.Table) else target
        if isinstance(table_expr, exp.Table):
            table_name = table_expr.name
            if table_name:
                return table_name.lower()
        return None

    def parse_tables(self, schema_sql: str) -> Dict[str, List[str]]:
        tables = {}
        statements = self._parse_statements(schema_sql)

        for stmt in statements:
            if isinstance(stmt, exp.Create) and stmt.this:
                table_name = self._extract_table_name(stmt)
                if not table_name:
                    continue

                columns = []
                for col in stmt.find_all(exp.ColumnDef):
                    columns.append(col.name.lower())

                tables[table_name] = columns
        return tables

    def extract_primary_keys(self, schema_sql: str) -> Dict[str, List[str]]:
        pk_map: Dict[str, List[str]] = {}
        statements = self._parse_statements(schema_sql)

        for stmt in statements:
            if isinstance(stmt, exp.Create) and stmt.this:
                table_name = self._extract_table_name(stmt)
                if not table_name:
                    continue

                for col in stmt.find_all(exp.ColumnDef):
                    constraints = col.args.get("constraints") or []
                    for c in constraints:
                        if isinstance(c, exp.PrimaryKeyColumnConstraint):
                            pk_map.setdefault(table_name, [])
                            col_name = col.name.lower()
                            if col_name not in pk_map[table_name]:
                                pk_map[table_name].append(col_name)

                for constraint in stmt.find_all(exp.PrimaryKey):
                    cols = constraint.expressions
                    if cols:
                        pk_map.setdefault(table_name, [])
                        for c in cols:
                            col_name = c.name.lower()
                            if col_name not in pk_map[table_name]:
                                pk_map[table_name].append(col_name)
        return pk_map

    def extract_foreign_keys(self, schema_sql: str) -> List[Tuple[str, str]]:
        foreign_keys: List[Tuple[str, str]] = []
        statements = self._parse_statements(schema_sql)

        for stmt in statements:
            if isinstance(stmt, exp.Create) and stmt.this:
                table_name = self._extract_table_name(stmt)
                if not table_name:
                    continue

                for fk in stmt.find_all(exp.ForeignKey):
                    local_cols = fk.expressions
                    ref = fk.args.get("reference")

                    if ref:
                        ref_table = ref.this.name.lower() if ref.this and ref.this.name else None
                        ref_cols = ref.expressions

                        if not ref_table:
                            continue
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
                                ref_table = ref.this.name.lower() if ref.this and ref.this.name else None
                                if not ref_table or not ref.expressions:
                                    continue
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