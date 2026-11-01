from app.database import get_connection


CREATE_EMPLOYEE_ACCOUNTS = """
CREATE TABLE IF NOT EXISTS employee_accounts (
    user_id TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'employee'
        CHECK (role IN ('employee', 'manager')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ
);
"""


def main():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_EMPLOYEE_ACCOUNTS)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_employee_accounts_active
                ON employee_accounts (is_active)
            """)

            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'employee_accounts'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()

    print("EMPLOYEE ACCOUNT SCHEMA CREATED")
    for name, data_type, nullable in columns:
        print(f"{name}: {data_type}, nullable={nullable}")


if __name__ == "__main__":
    main()
