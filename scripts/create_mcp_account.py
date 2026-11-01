import sys
from pathlib import Path
from getpass import getpass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.auth.service import provision_employee

user_id = input("Enter a new MCP employee ID: ").strip()
password = getpass("Enter a new password (minimum 12 characters): ")
confirmation = getpass("Confirm password: ")

if password != confirmation:
    raise SystemExit("Passwords do not match.")

account = provision_employee(
    user_id=user_id,
    password=password,
    role="employee",
)

print("MCP account created successfully.")
print("User ID:", account["user_id"])
print("Role:", account["role"])
print("Active:", account["is_active"])
