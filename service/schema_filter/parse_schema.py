import re
from typing import Dict, List, Tuple

class SchemaParser:
    def __init__(self):
        pass

    def parse_tables(self, schema_sql: str) -> Dict[str, List[str]]:
        tables = {}
        create_table_blocks = re.findall(
            r"CREATE TABLE\s+(\w+)\s*\((.*?)\);",
            schema_sql,
            re.S | re.I,
        )

        for table_name, body in create_table_blocks:
            columns = []
            lines = body.split("\n")

            for line in lines:
                line = line.strip().lower()

                if (
                    line.startswith("primary key")
                    or line.startswith("foreign key")
                    or line.startswith("unique")
                    or line.startswith("constraint")
                    or line == ""
                ):
                    continue

                col_match = re.match(r"(\w+)\s+", line)
                if col_match:
                    columns.append(col_match.group(1))

            tables[table_name.lower()] = columns

        return tables

    def extract_primary_keys(self, schema_sql: str) -> Dict[str, str]:
        pk_map = {}

        table_blocks = re.findall(
            r"CREATE TABLE\s+(\w+)\s*\((.*?)\);",
            schema_sql,
            re.S | re.I,
        )

        for table, body in table_blocks:
            body = body.lower()

            pk_col = None

            lines = [l.strip() for l in body.split("\n") if l.strip()]

            for i in range(len(lines) - 1):
                if "primary key" in lines[i + 1]:
                    col_match = re.match(r"(\w+)\s+", lines[i])
                    if col_match:
                        pk_col = col_match.group(1)
                        break

            if pk_col is None:
                table_pk = re.search(
                    r"primary key\s*\((\w+)\)",
                    body,
                    re.I,
                )
                if table_pk:
                    pk_col = table_pk.group(1)

            if pk_col:
                pk_map[table.lower()] = pk_col.lower()

        return pk_map

    def extract_foreign_keys(self, schema_sql: str) -> List[Tuple[str, str]]:
        foreign_keys: List[Tuple[str, str]] = []

        table_blocks = re.findall(
            r"CREATE TABLE\s+(\w+)\s*\((.*?)\);",
            schema_sql,
            re.S | re.I,
        )

        for table, body in table_blocks:
            matches = re.findall(
                r"foreign key\s*\((\w+)\)\s*references\s*(\w+)\s*\((\w+)\)",
                body,
                re.I,
            )

            for local_col, ref_table, ref_col in matches:
                fk = f"{table.lower()}.{local_col.lower()}"
                ref = f"{ref_table.lower()}.{ref_col.lower()}"
                foreign_keys.append((fk, ref))

        return foreign_keys

    def parse(self, schema_sql: str) -> Dict:
        return {
            "tables": self.parse_tables(schema_sql),
            "primary_keys": self.extract_primary_keys(schema_sql),
            "foreign_keys": self.extract_foreign_keys(schema_sql),
        }
