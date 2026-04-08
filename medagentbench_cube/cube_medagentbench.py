import json
import os
import requests
from typing import Dict, Any, List, Optional, Tuple, ClassVar, Type
from pydantic import PrivateAttr, Field
from cube.task import Task, TaskMetadata, TaskConfig
from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.tool import ToolboxConfig
from cube.core import Observation, StructuredContent, EnvironmentOutput, Action, Content
from dotenv import load_dotenv

load_dotenv()

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
   else:
       suffix = ""
       
   if suffix and not path.endswith(suffix):
       return f"{path}{suffix}"
   return path

class MedAgentBenchTask(Task):
   _data: Dict[str, Any] = PrivateAttr()
   _tools_def: List[Dict[str, Any]] = PrivateAttr()
   _done: bool = PrivateAttr(default=False)
   _history: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
   _fhir_api_base: str = PrivateAttr(default="http://localhost:8080/fhir/")

   def __init__(self, data: Dict[str, Any], tools_def: List[Dict[str, Any]], **kwargs):
       super().__init__(**kwargs)
       self._data = data
       self._tools_def = tools_def
       self._done = False
       self._history = []

   def reset(self) -> Tuple[Observation, Dict[str, Any]]:
       self._done = False
       self._history = []
       
       # Format tools for the agent (MCP format)
       available_tools = []
       for t in self._tools_def:
           available_tools.append({
               "name": derive_tool_name(t),
               "description": t["description"],
               "parameters": t["parameters"]
           })
           
       obs = Observation(contents=[
           Content.from_data(self._data["instruction"], name="instruction"),
           Content.from_data(self._data.get("context", ""), name="context"),
           Content.from_data(available_tools, name="tools")
       ])
       return obs, {}

   def filter_actions(self, actions: Any) -> Any:
       # For now, we return the tools in the observation during reset.
       # In a full CUBE tool implementation, this would return ActionSchema objects.
       return actions

   def step(self, action: Action | List[Action]) -> EnvironmentOutput:
       if self._done:
           raise RuntimeError("Task finished")
       
       actions = action if isinstance(action, list) else [action]
       last_obs = Observation()
       total_reward = 0.0
       finished = False
       
       for act in actions:
           self._history.append(act.model_dump())
           
           if act.name == "answer":
               prediction = act.arguments.get("content", "")
               sol = self._data.get("sol", [])
               is_correct = False
               if isinstance(sol, list):
                   is_correct = any(str(prediction).strip().lower() == str(s).strip().lower() for s in sol)
               else:
                   is_correct = str(prediction).strip().lower() == str(sol).strip().lower()
               
               total_reward = 1.0 if is_correct else 0.0
               self._done = True
               finished = True
               last_obs += Observation.from_text(f"Task completed. Correct: {is_correct}")
               break
               
           # Handle tool calls
           tool_name = act.name
           parameters = act.arguments
           
           target_tool = None
           for t in self._tools_def:
               if derive_tool_name(t) == tool_name or t["name"] == tool_name:
                   target_tool = t
                   break
           
           if not target_tool:
               last_obs += Observation.from_text(f"Error: Tool {tool_name} not found")
               continue

           try:
               original_name = target_tool["name"]
               path = original_name.replace("GET ", "").replace("POST ", "").replace("{api_base}", "").strip()
               method = "GET" if "GET" in original_name else "POST"
               url = self._fhir_api_base.rstrip("/") + path
               
               if method == "GET":
                   res = requests.get(url, params=parameters, timeout=10)
               else:
                   res = requests.post(url, json=parameters, timeout=10)
               
               if res.status_code == 200:
                   last_obs += Observation(contents=[StructuredContent(name=f"observation_{tool_name}", data=res.json())])
               else:
                   last_obs += Observation.from_text(f"Error: HTTP {res.status_code}: {res.text}")
           except Exception as e:
               last_obs += Observation.from_text(f"Error: Failed to connect to FHIR server: {e}")
       
       return EnvironmentOutput(
           obs=last_obs,
           reward=total_reward,
           done=finished,
           info={"history": self._history}
       )

   def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
       return 0.0, {"task_id": self.metadata.id, "done": self._done}

class MedAgentBenchTaskConfig(TaskConfig):
    data: Dict[str, Any] = Field(..., description="Task data record")
    tools_def: List[Dict[str, Any]] = Field(..., description="Available tools definition")

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> MedAgentBenchTask:
        from cube.task import TaskMetadata
        metadata = TaskMetadata(id=self.task_id, abstract_description="MedAgentBench EHR task")
        
        return MedAgentBenchTask(
            data=self.data,
            tools_def=self.tools_def,
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context
        )

class MedAgentBenchBenchmark(Benchmark):
   benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
       name="MedAgentBench-Benchmark",
       description="Benchmark for medical agents (EHR tasks).",
       version="1.0.0",
       extra_info={"id": "medagentbench-benchmark-v1"}
   )
   
   task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
   task_config_class: ClassVar[Type[TaskConfig]] = MedAgentBenchTaskConfig

   _data_path: str = PrivateAttr()
   _func_path: str = PrivateAttr()
   _task_data: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
   _tools_def: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

   def __init__(self, data_path: str, func_path: str, **kwargs):
       super().__init__(**kwargs)
       self._data_path = data_path
       self._func_path = func_path
       self._load_tasks()

   def _setup(self):
       pass

   def _load_tasks(self):
       with open(self._func_path, "r") as f:
           self._tools_def = json.load(f)
           
       new_metadata = {}
       with open(self._data_path, "r") as f:
           data_list = json.load(f)
           for i, data in enumerate(data_list):
               task_id = f"mb_{i}"
               self._task_data[task_id] = data
               new_metadata[task_id] = TaskMetadata(
                   id=task_id,
                   abstract_description=f"MedAgentBench task {i}",
               )
               
       object.__setattr__(self, "task_metadata", new_metadata)

   def get_task_configs(self) -> List[MedAgentBenchTaskConfig]:
       configs = []
       for task_id in self.task_metadata:
           configs.append(MedAgentBenchTaskConfig(
               task_id=task_id,
               data=self._task_data[task_id],
               tools_def=self._tools_def,
               tool_config=ToolboxConfig()
           ))
       return configs

   def get_task(self, index: int) -> MedAgentBenchTask:
       task_id = list(self.task_metadata.keys())[index]
       config = MedAgentBenchTaskConfig(
           task_id=task_id,
           data=self._task_data[task_id],
           tools_def=self._tools_def,
           tool_config=ToolboxConfig()
       )
       return config.make()

   def __len__(self):
       return len(self.task_metadata)

   def close(self):
       pass

MedAgentBenchTask.model_rebuild()
MedAgentBenchBenchmark.model_rebuild()
MedAgentBenchTaskConfig.model_rebuild()
