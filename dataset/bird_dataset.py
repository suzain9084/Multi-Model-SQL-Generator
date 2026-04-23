from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from torch.utils.data import Dataset

SchemaDict = Dict[str, Any]
RowDict = Dict[str, Any]


def _build_tables(meta: Dict[str, Any]) -> Dict[str, List[str]]:
    table_names = meta["table_names_original"]
    col_names = meta["column_names_original"]
    tables: Dict[str, List[str]] = {t: [] for t in table_names}
    for tid, cname in col_names:
        if tid < 0 or cname == "*":
            continue
        tables[table_names[tid]].append(cname)
    return tables


def _build_primary_keys(meta: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
    pk_raw = meta["primary_keys"]
    cols = meta["column_names_original"]
    table_names = meta["table_names_original"]
    out: Dict[str, Union[str, List[str]]] = {}
    for p in pk_raw:
        if isinstance(p, int):
            tid, cname = cols[p]
            if tid < 0:
                continue
            out[table_names[tid]] = cname
        elif isinstance(p, list):
            tname = None
            names: List[str] = []
            for idx in p:
                tid, cname = cols[idx]
                if tid < 0:
                    continue
                if tname is None:
                    tname = table_names[tid]
                names.append(cname)
            if tname is not None:
                out[tname] = names if len(names) > 1 else names[0]
    return out


def _build_foreign_keys(meta: Dict[str, Any]) -> List[Tuple[str, str]]:
    fks = meta["foreign_keys"]
    cols = meta["column_names_original"]
    table_names = meta["table_names_original"]
    out: List[Tuple[str, str]] = []
    for child, parent in fks:
        ctid, cc = cols[child]
        ptid, pc = cols[parent]
        out.append((f"{table_names[ctid]}.{cc}", f"{table_names[ptid]}.{pc}"))
    return out


def schema_from_table_entry(meta: Dict[str, Any]) -> SchemaDict:
    return {
        "tables": _build_tables(meta),
        "primary_keys": _build_primary_keys(meta),
        "foreign_keys": _build_foreign_keys(meta),
    }


def load_schema_index(train_tables_path: Path) -> Dict[str, SchemaDict]:
    raw = json.loads(train_tables_path.read_text(encoding="utf-8"))
    return {entry["db_id"]: schema_from_table_entry(entry) for entry in raw}


def _resolve_under_data_root(data_root: Path, path: Path | str | None, default: Path) -> Path:
    if path is None:
        return default
    p = Path(path)
    if p.is_absolute():
        return p
    return (data_root / p).resolve()


class BirdDataset(Dataset):
    def __init__(
        self,
        train_json_path: Path | str | None = None,
        train_tables_path: Path | str | None = None,
        schema_by_db: Dict[str, SchemaDict] | None = None,
        data_root: Path | str | None = None,
        sqlite_root: Path | str | None = None,
    ) -> None:
        # Directory that contains `train/` (defaults to this file's parent, i.e. `dataset/`)
        self._data_root = Path(data_root).resolve() if data_root is not None else Path(__file__).resolve().parent
        self._train_json_path = _resolve_under_data_root(
            self._data_root, train_json_path, self._data_root / "train" / "train.json"
        )
        self._train_tables_path = _resolve_under_data_root(
            self._data_root, train_tables_path, self._data_root / "train" / "train_tables.json"
        )
        default_sqlite = self._data_root / "train" / "train_databases" / "train_databases"
        self._sqlite_root = Path(sqlite_root).resolve() if sqlite_root is not None else default_sqlite

        with self._train_json_path.open(encoding="utf-8") as f:
            self._examples: List[Dict[str, Any]] = json.load(f)

        self._schema_by_db = schema_by_db or load_schema_index(self._train_tables_path)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> RowDict:
        if idx < 0 or idx >= len(self._examples):
            raise IndexError(idx)
        ex = self._examples[idx]
        db_id = ex["db_id"]
        schema = self._schema_by_db.get(db_id)
        if schema is None:
            raise KeyError(f"No schema for db_id={db_id!r} in train_tables.json")
        sqlite_path = self._sqlite_root / db_id / f"{db_id}.sqlite"
        return {
            "question_id": idx,
            "db_id": db_id,
            "question": ex["question"],
            "evidence": ex.get("evidence", ""),
            "SQL": ex["SQL"],
            "db_schema": schema,
            "sqlfile_path": str(sqlite_path),
        }

if __name__ == "__main__":
    ds = BirdDataset()
    print(ds[0]["db_schema"]["primary_keys"])