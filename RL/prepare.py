from collections import defaultdict
from pathlib import Path
import sys

from psycopg2 import sql

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.expander.config import load_settings
from pool import connect_postgresql

MAX_SAMPLE_ROWS_PER_TABLE = 80


def get_table_structure(cursor):

    cursor.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name != 'pg_stat_statements'
        ORDER BY table_name;
        """
    )
    tables = [row[0] for row in cursor.fetchall()]

    schema = defaultdict(list)
    for table_name in tables:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (table_name,),
        )
        schema[table_name].extend(row[0] for row in cursor.fetchall())
    return schema


def fetch_table_rows(cursor, table_name, columns, sample_rows):

    select_columns = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    query = sql.SQL(
        """
        SELECT {columns}
        FROM {table}
        ORDER BY RANDOM()
        LIMIT %s
        """
    ).format(
        columns=select_columns,
        table=sql.Identifier(table_name),
    )
    cursor.execute(query, (sample_rows,))
    return cursor.fetchall()


def split_rows_by_column(columns, rows):

    column_values = defaultdict(list)
    for row in rows:
        for column, value in zip(columns, row):
            if value is None:
                continue
            if value not in column_values[column]:
                column_values[column].append(value)
    return column_values


def write_into_txt(db_name, table_name, column_name, data):

    path = Path(db_name) / table_name / f"{column_name.lower()}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in data:
            file.write(f"{item}\n")


def main():

    db_name = load_settings().seed_generator.dbname
    connection = connect_postgresql()
    try:
        cursor = connection.cursor()
        schema = get_table_structure(cursor)

        for table_name, columns in schema.items():
            rows = fetch_table_rows(cursor, table_name, columns, MAX_SAMPLE_ROWS_PER_TABLE)
            column_data = split_rows_by_column(columns, rows)
            for column_name in columns:
                write_into_txt(db_name, table_name, column_name, column_data.get(column_name, []))

        print(schema)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
