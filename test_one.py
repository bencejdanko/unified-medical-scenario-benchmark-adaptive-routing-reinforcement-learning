import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Import the benchmarks
from healthbench_cube.cube_healthbench import HealthBenchBenchmark
from medagentbench_cube.cube_medagentbench import MedAgentBenchBenchmark
from cube.core import Action

load_dotenv()

TARGET_MODEL = "google/gemini-3-flash-preview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

# Clean URL for OpenAI client
clean_url = OPENAI_API_URL
if clean_url and clean_url.endswith("/chat/completions"):
    clean_url = clean_url[:-len("/chat/completions")]
if clean_url and clean_url.endswith("/"):
    clean_url = clean_url[:-1]

client = OpenAI(base_url=clean_url, api_key=OPENAI_API_KEY)

def get_model_response(messages, tools=None):
    print(f"Calling {TARGET_MODEL}...")
    try:
        kwargs = {
            "model": TARGET_MODEL,
            "messages": messages,
            "temperature": 0.0
        }
        if tools:
            # Convert tools to OpenAI format
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"]
                    }
                })
            kwargs["tools"] = openai_tools
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message
    except Exception as e:
        print(f"Error calling model: {e}")
        return None

def test_healthbench():
    print("\n--- Testing HealthBench ---")
    benchmark = HealthBenchBenchmark(num_examples=1)
    task = benchmark.get_task(0)
    obs, info = task.reset()
    
    # Extract prompt from observation
    prompt = None
    for content in obs.contents:
        if content.name == "prompt":
            prompt = content.data
            break
            
    if not prompt:
        print("Error: Prompt not found in observation")
        return
        
    print(f"Prompt: {prompt[0]['content'][:100]}...")
    
    message = get_model_response(prompt)
    if not message: return
    answer = message.content
    print(f"Answer: {answer[:100]}...")
    
    # Action creation
    action = Action(name="answer", arguments={"content": answer})
    res = task.step(action)
    print(f"Reward: {res.reward}")
    print(f"Done: {res.done}")

def test_medagentbench():
    print("\n--- Testing MedAgentBench ---")
    benchmark = MedAgentBenchBenchmark(
        data_path="medagentbench_cube/data/medagentbench/test_data_v2.json",
        func_path="medagentbench_cube/data/medagentbench/funcs_v1.json"
    )
    task = benchmark.get_task(0)
    obs, info = task.reset()
    
    instruction = None
    tools = []
    for content in obs.contents:
        if content.name == "instruction":
            instruction = content.data
        elif content.name == "tools":
            tools = content.data
            
    print(f"Instruction: {instruction}")
    print(f"Available Ported Tools: {[t['name'] for t in tools]}")
    
    messages = [{"role": "user", "content": instruction}]
    message = get_model_response(messages, tools=tools)
    if not message: return

    if message.tool_calls:
        for tool_call in message.tool_calls:
            print(f"Model called tool: {tool_call.function.name} with {tool_call.function.arguments}")
            action = Action(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            )
            res = task.step(action)
            print(f"Observation: {res.obs.contents[0].data if res.obs.contents else 'No observation'}")
    else:
        print(f"Answer: {message.content}")

if __name__ == "__main__":
    test_healthbench()
    test_medagentbench()
