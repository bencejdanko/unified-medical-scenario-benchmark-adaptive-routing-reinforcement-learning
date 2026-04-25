import streamlit as st
import json
import os
import pandas as pd
from healthbench_cube.cube_healthbench import HealthBenchBenchmark
from medagentbench_cube.cube_medagentbench import MedAgentBenchBenchmark
from cube.core import Observation, StructuredContent, Action

# Page configuration
st.set_page_config(
    page_title="CUBE Benchmark Explorer",
    page_icon="🏥",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stHeader {
        color: #00d4ff;
    }
    .task-card {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #303742;
        margin-bottom: 20px;
    }
    .observation-result {
        background-color: #161b22;
        padding: 15px;
        border-left: 5px solid #00d4ff;
        border-radius: 0 5px 5px 0;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 CUBE Benchmark Explorer")
st.markdown("Explore, confirm, and **test tools** for ported CUBE benchmarks.")

# --- Sidebar Configuration ---
st.sidebar.header("Benchmark Selection")
benchmark_type = st.sidebar.selectbox(
    "Select Benchmark",
    ["HealthBench", "MedAgentBench"]
)

@st.cache_resource
def load_benchmark(b_type):
    if b_type == "HealthBench":
        return HealthBenchBenchmark()
    else:
        return MedAgentBenchBenchmark(
            data_path="medagentbench_cube/data/medagentbench/test_data_v2.json",
            func_path="medagentbench_cube/data/medagentbench/funcs_v1.json"
        )

benchmark = load_benchmark(benchmark_type)

# Initialize Session State for interactive testing
if "history" not in st.session_state:
    st.session_state.history = []
if "current_task_idx" not in st.session_state:
    st.session_state.current_task_idx = 0
if "current_benchmark" not in st.session_state:
    st.session_state.current_benchmark = benchmark_type

# Reset state if benchmark changes
if benchmark_type != st.session_state.current_benchmark:
    st.session_state.current_benchmark = benchmark_type
    st.session_state.current_task_idx = 0
    st.session_state.history = []

def reset_history():
    st.session_state.history = []

# --- Main Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📋 Benchmark Info")
    meta = benchmark.benchmark_metadata
    meta_dict = meta.model_dump()
    
    st.write(f"**Name:** {meta_dict.get('name', 'N/A')}")
    if 'extra_info' in meta_dict and 'id' in meta_dict['extra_info']:
        st.write(f"**ID:** {meta_dict['extra_info']['id']}")
        
    st.write(f"**Version:** {meta_dict.get('version', 'N/A')}")
    st.write(f"**Description:** {meta_dict.get('description', 'N/A')}")
    
    st.divider()
    
    st.header("Search Tasks")
    total_tasks = len(benchmark)
    
    # Ensure current_task_idx is within bounds for the current benchmark
    st.session_state.current_task_idx = max(0, min(st.session_state.current_task_idx, total_tasks - 1))
    
    new_task_idx = st.number_input(
        "Task Index", 
        min_value=0, 
        max_value=total_tasks-1, 
        value=st.session_state.current_task_idx
    )
    
    if new_task_idx != st.session_state.current_task_idx:
        st.session_state.current_task_idx = new_task_idx
        reset_history()
        st.rerun()
    
    st.info(f"Showing task {st.session_state.current_task_idx + 1} of {total_tasks}")

with col2:
    st.header("🔍 Task Details")
    task = benchmark.get_task(st.session_state.current_task_idx)
    
    # Task Metadata
    with st.expander("Task Metadata", expanded=False):
        st.json(task.metadata.model_dump())
    
    # Task Observation (What the agent sees at start)
    obs, info = task.reset()
    
    def find_content(observation, name):
        if not hasattr(observation, "contents"):
            st.error(f"Inconsistent State Detected: Expected CUBE Observation but got {type(observation).__name__}. This usually means an old benchmark class was loaded from cache. Please click 'Clear History & Refresh' in the sidebar.")
            return None
        for c in observation.contents:
            if c.name == name:
                return c.data
        return None

    if benchmark_type == "HealthBench":
        prompt = find_content(obs, "prompt")
        rubrics = find_content(obs, "rubrics")
        
        st.subheader("Prompt")
        if prompt:
            for msg in prompt:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        if rubrics:
            with st.expander("Show Rubrics"):
                st.table(pd.DataFrame(rubrics))
        
    else: # MedAgentBench
        instruction = find_content(obs, "instruction")
        tools = find_content(obs, "tools")
        
        st.subheader("Instruction")
        st.info(instruction or "No instruction found")
        
        # --- TOOL SANDBOX ---
        st.divider()
        st.subheader("🛠️ Tool Sandbox")
        st.markdown("Test the ported FHIR tools against the local MedAgentBench server.")
        
        if tools:
            tool_names = [t["name"] for t in tools]
            selected_tool_name = st.selectbox("Select Tool to Test", tool_names)
            selected_tool = next(t for t in tools if t["name"] == selected_tool_name)
            
            st.caption(selected_tool["description"])
            
            # Form for tool parameters
            with st.form(key="tool_form"):
                params_str = st.text_area("Parameters (JSON)", value=json.dumps(selected_tool["parameters"].get("properties", {}), indent=2))
                submit_button = st.form_submit_button(label="🚀 Execute Tool Call")
                
                if submit_button:
                    try:
                        params = json.loads(params_str)
                        # We use the actual task instance to call step
                        action = Action(name=selected_tool_name, arguments=params)
                        result = task.step(action)
                        
                        # Add to history
                        st.session_state.history.append({
                            "action": f"{selected_tool_name}({params})",
                            "result": result.obs.contents[0].data if result.obs.contents else "No data"
                        })
                    except Exception as e:
                        st.error(f"Execution Error: {e}")

            # Display History
            if st.session_state.history:
                st.write("### 📜 Execution History")
                for i, item in enumerate(reversed(st.session_state.history)):
                    with st.expander(f"Step {len(st.session_state.history) - i}: {item['action']}", expanded=(i==0)):
                        st.json(item["result"])

        with st.expander("View All Tool Definitions"):
            for tool in tools:
                st.write(f"**{tool['name']}**")
                st.write(tool["description"])
                st.json(tool["parameters"])
                st.markdown("---")

# --- Footer / Debug ---
st.sidebar.divider()
if st.sidebar.button("Clear History & Refresh"):
    reset_history()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity CUBE Explorer v0.3")
