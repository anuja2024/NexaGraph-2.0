import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tool_response = await session.list_tools()
            tools = [tool.name for tool in tool_response.tools]
            expected = {
                "list_projects",
                "list_project_documents",
                "check_document_access",
                "search_project_documents",
                "ask_project_question",
            }

            print("REGISTERED TOOLS:", ", ".join(tools))
            missing = expected - set(tools)
            if missing:
                raise RuntimeError(f"Missing MCP tools: {sorted(missing)}")

            async def call_tool(name, arguments=None):
                response = await session.call_tool(name, arguments or {})
                texts = [
                    item.text
                    for item in response.content
                    if getattr(item, "text", None) is not None
                ]
                combined = "\n".join(texts)
                try:
                    parsed = json.loads(combined)
                except json.JSONDecodeError:
                    parsed = {"raw_response": combined}
                if response.is_error:
                    raise RuntimeError(f"{name} failed: {parsed}")
                if isinstance(parsed, dict) and "error" in parsed:
                    raise RuntimeError(f"{name} returned error: {parsed['error']}")
                return parsed

            projects = await call_tool("list_projects")
            print("\n1. list_projects: PASS")
            print(json.dumps(projects, ensure_ascii=False)[:1500])

            # The configured test identity is emp1; corporate should be accessible.
            project_id = "corporate"

            documents = await call_tool(
                "list_project_documents", {"project_id": project_id}
            )
            print("\n2. list_project_documents: PASS")
            print(json.dumps(documents, ensure_ascii=False)[:1500])

            document_id = "siemens-annual-report-2025"
            access = await call_tool(
                "check_document_access",
                {"project_id": project_id, "document_id": document_id},
            )
            print("\n3. check_document_access (corporate): PASS")
            print(json.dumps(access, ensure_ascii=False)[:1000])

            search = await call_tool(
                "search_project_documents",
                {
                    "project_id": project_id,
                    "query": "Siemens revenue fiscal 2025",
                    "limit": 3,
                },
            )
            print("\n4. search_project_documents: PASS")
            print(json.dumps(search, ensure_ascii=False)[:2000])

            answer = await call_tool(
                "ask_project_question",
                {
                    "project_id": project_id,
                    "query": (
                        "For the consolidated Siemens Group in fiscal 2025, "
                        "what were reported revenue and net income? Cite the source."
                    ),
                },
            )
            print("\n5. ask_project_question: PASS")
            print(json.dumps(answer, ensure_ascii=False)[:2500])

            # emp1 must not access the sustainability document.
            denied = await call_tool(
                "check_document_access",
                {
                    "project_id": "sustainability",
                    "document_id": "siemens-sustainability-statement-2025",
                },
            )
            print("\n6. Unauthorized document access tool response:")
            print(json.dumps(denied, ensure_ascii=False)[:1000])

            print("\nMCP TEST RUN COMPLETED.")

asyncio.run(main())
