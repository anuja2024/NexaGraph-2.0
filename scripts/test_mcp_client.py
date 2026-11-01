import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    required = [
        "NEXAGRAPH_USER_ID",
        "NEXAGRAPH_PASSWORD",
        "NEXAGRAPH_API_BASE_URL",
    ]

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        print("Missing environment variables:", ", ".join(missing))
        return

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("\n=== MCP TOOLS ===")
            for tool in tools.tools:
                print("-", tool.name)

            print("\n=== PROJECTS ===")
            projects = await session.call_tool("list_projects", {})
            for item in projects.content:
                if hasattr(item, "text"):
                    print(item.text)

            if projects.is_error:
                return

            print("\n=== PENSION QUESTION ===")
            result = await session.call_tool(
                "ask_project_question",
                {
                    "project_id": "corporate",
                    "query": "What pension provisions did Siemens report for fiscal 2025?"
                },
            )

            for item in result.content:
                if hasattr(item, "text"):
                    print(item.text)

            if result.is_error:
                print("Pension question returned an MCP error.")


if __name__ == "__main__":
    asyncio.run(main())
