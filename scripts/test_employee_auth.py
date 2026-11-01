import os
import uuid

from app.auth.service import (
    AccountProvisioningError,
    AuthenticationError,
    authenticate_employee,
    provision_employee,
)
from app.database import get_connection


def main():
    user_id = f"auth_test_{uuid.uuid4().hex[:12]}"
    password = "NexaGraph-Test-Password-2026!"
    created = False

    try:
        account = provision_employee(
            user_id=user_id,
            password=password,
            role="employee",
        )
        created = True
        print("Account created:", account["user_id"] == user_id)
        print("Role assigned:", account["role"] == "employee")
        print("Account active:", account["is_active"] is True)

        login = authenticate_employee(user_id=user_id, password=password)
        print("Valid login accepted:", bool(login["access_token"]))
        print("Token type:", login["token_type"])
        print("Identity matches:", login["user"]["user_id"] == user_id)

        try:
            authenticate_employee(
                user_id=user_id,
                password="Incorrect-Test-Password-2026!",
            )
            print("Incorrect password rejected: False")
        except AuthenticationError:
            print("Incorrect password rejected: True")

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE employee_accounts
                    SET is_active = FALSE
                    WHERE user_id = %s
                    """,
                    (user_id,),
                )

        try:
            authenticate_employee(user_id=user_id, password=password)
            print("Inactive account rejected: False")
        except AuthenticationError:
            print("Inactive account rejected: True")

    finally:
        if created:
            with get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM employee_accounts WHERE user_id = %s",
                        (user_id,),
                    )
            print("Temporary test account cleaned up: True")


if __name__ == "__main__":
    main()
