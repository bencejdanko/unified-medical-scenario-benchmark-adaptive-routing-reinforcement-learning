import json
import os
import requests
import asyncio
from mcp.server import Server
import mcp.types as types
from typing import Dict, Any, List

app = Server("medical-mcp-server")

FHIR_API_BASE = os.getenv("FHIR_API_BASE", "http://localhost:8080/api/")

def load_schema():
    schema_path = "medagentbench_cube/data/medagentbench/funcs_v1.json"
    if not os.path.exists(schema_path):
        # Fallback for different working directory
        schema_path = "/home/bence/medqa-cube-adaptive-curriculum-tool-calling/medagentbench_cube/data/medagentbench/funcs_v1.json"
    with open(schema_path, "r") as f:
        return json.load(f)

def derive_tool_name(tool_def: Dict[str, Any]) -> str:
    path = tool_def["name"].replace("GET ", "get_").replace("POST ", "post_").replace("{api_base}/", "").replace("/", "_")
    desc = tool_def["description"].lower()
    
    if "labs" in desc:
        suffix = "_labs"
    elif "vitals" in desc:
        suffix = "_vitals"
    elif "problems" in desc:
        suffix = "_problems"
    elif "orders" in desc:
        suffix = "_orders"
    elif "patient" in desc:
        suffix = ""
    else:
        suffix = ""
        
    if suffix and not path.endswith(suffix):
        return f"{path}{suffix}"
    return path

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    schema = load_schema()
    tools = []
    for tool_def in schema:
        tools.append(types.Tool(
            name=derive_tool_name(tool_def),
            description=tool_def["description"],
            inputSchema=tool_def["parameters"]
        ))
    return tools

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    schema = load_schema()
    target_tool = None
    for tool_def in schema:
        if derive_tool_name(tool_def) == name:
            target_tool = tool_def
            break
    
    if not target_tool:
        return [types.TextContent(type="text", text=f"Tool {name} not found")]

    method = "GET" if "GET" in target_tool["name"] else "POST"
    path = target_tool["name"].replace("GET ", "").replace("POST ", "").replace("{api_base}", "").strip()
    url = FHIR_API_BASE.rstrip("/") + path
    
    try:
        if method == "GET":
            res = requests.get(url, params=arguments, timeout=10)
        else:
            res = requests.post(url, json=arguments, timeout=10)
        
        if res.status_code == 200:
            return [types.TextContent(type="text", text=json.dumps(res.json(), indent=2))]
        else:
            return [types.TextContent(type="text", text=f"HTTP {res.status_code}: {res.text}")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Connection error: {e}")]

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
