import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    server_params = StdioServerParameters(
        command="python3",
        args=["medical_mcp_server.py"],
    )

    print("--- Connecting to MCP Server ---")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. List tools
            print("\n1. Listing Tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}")

            # 2. Call a tool
            print("\n2. Calling 'get_Patient' for Peter Stafford:")
            result = await session.call_tool(
                "get_Patient",
                arguments={
                    "given": "Peter",
                    "family": "Stafford",
                    "birthdate": "1932-12-29"
                }
            )
            print(f"Result: {result.content[0].text[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
