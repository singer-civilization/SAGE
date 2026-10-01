import psycopg2
from psycopg2.pool import ThreadedConnectionPool

from agent.expander.config import load_settings


_CONNECTION_POOL = None


def get_connection_config() -> dict[str, str]:

    database = load_settings().database
    return {
        "host": database.host,
        "port": database.port,
        "user": database.user,
        "password": database.password,
        "database": database.database,
    }


def get_pool_bounds() -> tuple[int, int]:

    database = load_settings().database
    return database.min_conn, database.max_conn


def connect_postgresql():

    return psycopg2.connect(**get_connection_config())


def get_connection_pool() -> ThreadedConnectionPool:

    global _CONNECTION_POOL
    if _CONNECTION_POOL is None:
        min_conn, max_conn = get_pool_bounds()
        _CONNECTION_POOL = ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            **get_connection_config(),
        )
    return _CONNECTION_POOL


def return_connection(connection) -> None:

    pool = get_connection_pool()
    pool.putconn(connection)


def close_connection_pool() -> None:

    global _CONNECTION_POOL
    if _CONNECTION_POOL is not None:
        _CONNECTION_POOL.closeall()
        _CONNECTION_POOL = None


def test_connection() -> dict[str, str]:

    connection = connect_postgresql()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user;")
            database_name, user_name = cursor.fetchone()
        return {
            "status": "ok",
            "database": str(database_name),
            "user": str(user_name),
        }
    finally:
        connection.close()


if __name__ == "__main__":
    min_conn, max_conn = get_pool_bounds()
    result = test_connection()
    print(
        f"status={result['status']} "
        f"database={result['database']} "
        f"user={result['user']} "
        f"pool_min={min_conn} "
        f"pool_max={max_conn}"
    )
