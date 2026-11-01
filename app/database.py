import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


def get_connection() -> psycopg.Connection:
    """Connect to NexaGraph's dedicated PostgreSQL database."""
    required_settings = (
        "NEXAGRAPH_DB_HOST",
        "NEXAGRAPH_DB_PORT",
        "NEXAGRAPH_DB_NAME",
        "NEXAGRAPH_DB_USER",
        "NEXAGRAPH_DB_PASSWORD",
    )

    missing_settings = [
        name for name in required_settings
        if not os.getenv(name)
    ]

    if missing_settings:
        raise RuntimeError(
            "Missing required database configuration: "
            + ", ".join(missing_settings)
        )

    try:
        port = int(os.environ["NEXAGRAPH_DB_PORT"])
    except ValueError as exc:
        raise RuntimeError(
            "NEXAGRAPH_DB_PORT must be an integer"
        ) from exc

    return psycopg.connect(
        host=os.environ["NEXAGRAPH_DB_HOST"],
        port=port,
        dbname=os.environ["NEXAGRAPH_DB_NAME"],
        user=os.environ["NEXAGRAPH_DB_USER"],
        password=os.environ["NEXAGRAPH_DB_PASSWORD"],
        connect_timeout=5,
        application_name="nexagraph-next",
    )


def check_database_connection() -> dict:
    """Verify connectivity without modifying database contents."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), current_user"
            )
            row = cursor.fetchone()

    return {
        "connected": True,
        "database": row[0],
        "user": row[1],
    }
