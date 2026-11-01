
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class User:
    user_id: str
    role: str


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str


@dataclass(frozen=True)
class Document:
    document_id: str
    project_id: str
    source_file: str
    name: str


# Development and production identities.
USERS = {
    "manager": User(user_id="manager", role="manager"),
    **{
        f"emp{i}": User(user_id=f"emp{i}", role="employee")
        for i in range(1, 10)
    },
    "nexagraph_mcp": User(user_id="nexagraph_mcp", role="employee"),
    "n8n_service": User(user_id="n8n_service", role="employee"),
}

PROJECTS = {
    "corporate": Project(
        project_id="corporate",
        name="Corporate Knowledge",
    ),
    "sustainability": Project(
        project_id="sustainability",
        name="Sustainability",
    ),
    "industrial_automation": Project(
        project_id="industrial_automation",
        name="Industrial Automation",
    ),
}

# Explicit project memberships.
PROJECT_MEMBERSHIPS = {
    "emp1": {"corporate"},
    "emp2": {"corporate"},
    "emp3": {"sustainability"},
    "emp4": {"sustainability"},
    "emp5": {"industrial_automation"},
    "emp6": {"industrial_automation"},
    "nexagraph_mcp": {"corporate"},
    "n8n_service": {"corporate"},
}

# Explicit document registry. Unknown IDs fail closed.
DOCUMENTS: Mapping[str, Document] = {
    "siemens-annual-report-2025": Document(
        document_id="siemens-annual-report-2025",
        project_id="corporate",
        source_file="siemens_annual_report_2025.pdf",
        name="Siemens Annual Report 2025",
    ),
    "siemens-sustainability-statement-2025": Document(
        document_id="siemens-sustainability-statement-2025",
        project_id="sustainability",
        source_file="sustainability-statement.pdf",
        name="Siemens Sustainability Statement 2025",
    ),
    "siemens-simatic-cybersecurity-2026": Document(
        document_id="siemens-simatic-cybersecurity-2026",
        project_id="industrial_automation",
        source_file="siemens_industrial_automation_manual.pdf",
        name="Siemens Industrial Automation Manual 2026",
    ),
}


def can_access_project(user_id: str, project_id: str) -> bool:
    """Check whether a known user can access a known project."""
    user = USERS.get(user_id)
    if user is None or project_id not in PROJECTS:
        return False
    if user.role == "manager":
        return True
    return project_id in PROJECT_MEMBERSHIPS.get(user_id, set())


def can_access_document(
    user_id: str,
    document_id: str,
    project_id: str | None = None,
) -> bool:
    """Check access to a registered document, failing closed for unknown IDs."""
    document = DOCUMENTS.get(document_id)
    if document is None:
        return False
    if project_id is not None and document.project_id != project_id:
        return False
    return can_access_project(user_id, document.project_id)


def documents_for_project(project_id: str) -> list[Document]:
    """Return registered documents assigned to a known project."""
    if project_id not in PROJECTS:
        return []
    return [doc for doc in DOCUMENTS.values() if doc.project_id == project_id]


def can_manage_documents(user_id: str) -> bool:
    """Only the manager may manage company documents."""
    user = USERS.get(user_id)
    return user is not None and user.role == "manager"
