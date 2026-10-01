import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.expander.config import load_settings
from pool.pg_connect import connect_postgresql


def read_sample_values(
    root_dir: Path,
    database_name: str,
    table_name: str,
    column_name: str,
    limit: int,
) -> list[str]:

    sample_path = root_dir / database_name / table_name / f"{column_name}.txt"
    if not sample_path.exists():
        return []

    values: list[str] = []
    with sample_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            value = line.strip()
            if value == "":
                continue
            values.append(value)
            if len(values) >= limit:
                break
    return values


def fetch_table_names(cursor) -> list[str]:

    cursor.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    return [row[0] for row in cursor.fetchall()]


def fetch_foreign_keys(cursor) -> dict[str, list[dict]]:

    cursor.execute(
        """
        SELECT
            kcu.table_name,
            kcu.column_name,
            ccu.table_name AS referenced_table_name,
            ccu.column_name AS referenced_column_name
        FROM information_schema.referential_constraints AS rc
        JOIN information_schema.key_column_usage AS kcu
          ON rc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON rc.unique_constraint_name = ccu.constraint_name
        WHERE kcu.table_schema = 'public'
        ORDER BY kcu.table_name, kcu.column_name
        """
    )
    foreign_keys_by_table: dict[str, list[dict]] = {}
    for table_name, column_name, referenced_table_name, referenced_column_name in cursor.fetchall():
        foreign_keys_by_table.setdefault(table_name, []).append(
            {
                "column": column_name,
                "referenced_table": referenced_table_name,
                "referenced_column": referenced_column_name,
            }
        )
    return foreign_keys_by_table


def fetch_columns(cursor, table_name: str) -> list[tuple[str, str]]:

    cursor.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def fetch_schema_columns(cursor, table_names: list[str]) -> dict[str, dict[str, list[str]]]:

    schema: dict[str, dict[str, list[str]]] = {}
    for table_name in table_names:
        schema[table_name] = {}
        for column_name, data_type in fetch_columns(cursor, table_name):
            schema[table_name][column_name] = [data_type]
    return schema


def fetch_row_estimates(cursor) -> dict[str, float]:

    cursor.execute(
        """
        SELECT c.relname, c.reltuples
        FROM pg_class AS c
        JOIN pg_namespace AS n
          ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        ORDER BY c.relname
        """
    )
    return {row[0]: float(row[1]) for row in cursor.fetchall()}


def build_database_summary(summary_path: Path, sample_value_limit: int) -> dict:

    connection = connect_postgresql()
    database_name = connection.info.dbname
    root_dir = summary_path.parents[1]
    try:
        with connection.cursor() as cursor:
            table_names = fetch_table_names(cursor)
            foreign_keys_by_table = fetch_foreign_keys(cursor)
            row_estimates = fetch_row_estimates(cursor)

        tables: dict[str, dict] = {}
        with connection.cursor() as cursor:
            for table_name in table_names:
                columns = []
                for column_name, data_type in fetch_columns(cursor, table_name):
                    columns.append(
                        {
                            "name": column_name,
                            "data_type": data_type,
                            "samples": read_sample_values(
                                root_dir=root_dir,
                                database_name=database_name,
                                table_name=table_name,
                                column_name=column_name,
                                limit=sample_value_limit,
                            ),
                        }
                    )
                tables[table_name] = {
                    "columns": columns,
                    "foreign_keys": foreign_keys_by_table.get(table_name, []),
                    "row_estimate": row_estimates.get(table_name, 0.0),
                }
    finally:
        connection.close()

    summary = {
        "database": database_name,
        "tables": tables,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_or_build_summary(summary_path: Path, sample_value_limit: int) -> dict:

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if all("row_estimate" in table_info for table_info in summary.get("tables", {}).values()):
            return summary
    return build_database_summary(summary_path=summary_path, sample_value_limit=sample_value_limit)


def extract_seed_tables(seed_sql: str, summary: dict) -> list[str]:

    seed_lower = seed_sql.lower()
    tables = []
    for table_name in summary["tables"].keys():
        if f" {table_name.lower()}" in seed_lower or f"{table_name.lower()}." in seed_lower:
            tables.append(table_name)
    return tables


def build_related_summary(seed_sql: str, summary: dict) -> dict:

    seed_tables = extract_seed_tables(seed_sql, summary)
    if not seed_tables:
        return summary

    related_tables = set(seed_tables)
    for table_name in seed_tables:
        for foreign_key in summary["tables"][table_name]["foreign_keys"]:
            related_tables.add(foreign_key["referenced_table"])
        for other_table_name, other_table in summary["tables"].items():
            for foreign_key in other_table["foreign_keys"]:
                if foreign_key["referenced_table"] == table_name:
                    related_tables.add(other_table_name)

    compact_tables = {table_name: summary["tables"][table_name] for table_name in sorted(related_tables)}
    return {
        "database": summary["database"],
        "seed_tables": seed_tables,
        "tables": compact_tables,
    }


def main() -> None:

    settings = load_settings()
    summary = build_database_summary(
        summary_path=settings.summary_file,
        sample_value_limit=settings.sample_value_limit,
    )
    print(f"database={summary['database']}")
    print(f"table_count={len(summary['tables'])}")
    print(f"output={settings.summary_file}")


if __name__ == "__main__":
    main()
