"""Read-only access to the local SADINhub PostgreSQL warehouse."""
import getpass
import os

import psycopg

from . import config


def connect():
    """Open a connection that cannot write, and prove it before returning."""
    password = os.environ.get("SADINHUB_PASSWORD") or os.environ.get("PGPASSWORD")
    if not password:
        password = getpass.getpass("SADINhub database password: ")
    conn = psycopg.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=password,
        options="-c default_transaction_read_only=on -c statement_timeout=900000",
    )
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        if cur.fetchone()[0] != "on":
            conn.close()
            raise RuntimeError("Database session is not read-only; refusing to continue.")
    return conn
