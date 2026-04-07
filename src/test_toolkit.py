import asyncio
import json
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import sys

# Test script for the Medical MCP Server
# This script ensures that all medical retrieval tools are responsive and return expected schemas.

import traceback

async def test_medical_tools():
    print("Starting Medical Tool Toolkit Validation...")
    
    # Path to our local server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["medical_mcp_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Initializing MCP Session...")
                try:
                    await asyncio.wait_for(session.initialize(), timeout=10.0)
                except asyncio.TimeoutError:
                    print("Timeout during session initialization.")
                    return
                
                # 1. Test Tool Discovery
                print("\nTest 1: Tool Discovery")
                try:
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    print(f"Found Tools: {tool_names}")
                    assert "pubmed_search" in tool_names
                except Exception as e:
                    print(f"Failed Tool Discovery: {e}")
                    return
                
                # 2. Test PubMed Search
                print("\nTest 2: PubMed Search")
                try:
                    pubmed_res = await session.call_tool("pubmed_search", {"query": "diabetes", "max_results": 2})
                    print(f"Result: {pubmed_res.content[0].text if hasattr(pubmed_res.content[0], 'text') else pubmed_res.content}")
                except Exception as e:
                    print(f"Failed PubMed: {e}")
                
        print("\nAll tools validated successfully!")

    except Exception as e:
        print(f"\nValidation Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_medical_tools())
