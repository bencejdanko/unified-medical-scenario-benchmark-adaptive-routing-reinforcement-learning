import json
import asyncio
from medical_mcp_server import load_schema, derive_tool_name

async def verify_port():
    print("Verifying tool port from funcs_v1.json to MCP...")
    original_schema = load_schema()
    
    ported_tools = []
    for tool_def in original_schema:
        ported_name = derive_tool_name(tool_def)
        ported_tools.append({
            "original_name": tool_def["name"],
            "ported_name": ported_name,
            "description_snippet": tool_def["description"][:50],
            "params_count": len(tool_def["parameters"].get("properties", {}))
        })
    
    print(f"\nTotal original tools: {len(original_schema)}")
    print(f"Total ported tools: {len(ported_tools)}")
    
    # Check for name collisions
    names = [t["ported_name"] for t in ported_tools]
    if len(names) != len(set(names)):
        print("ALERT: Name collisions detected!")
        from collections import Counter
        counts = Counter(names)
        for name, count in counts.items():
            if count > 1:
                print(f"  - {name}: {count} occurrences")
    else:
        print("No name collisions detected.")

    print("\nMapping Details:")
    print(f"{'Original Name':<40} | {'Ported Name':<30} | Params")
    print("-" * 85)
    for t in ported_tools:
        print(f"{t['original_name']:<40} | {t['ported_name']:<30} | {t['params_count']}")

if __name__ == "__main__":
    asyncio.run(verify_port())
