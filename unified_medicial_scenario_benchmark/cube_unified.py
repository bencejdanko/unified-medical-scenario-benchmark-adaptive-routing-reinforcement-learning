"""CUBE wrapper for bdanko/unified_medical_scenario_benchmark.

This wrapper keeps the benchmark loading and task surfaces unified. The full
reward implementation lives in `run_unified_eval.py` because HealthBench and
MedAgentBench require external judges/servers.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from datasets import load_dataset
from pydantic import Field, PrivateAttr

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.core import Action, Content, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import ToolboxConfig


DEFAULT_DATASET_ID = "bdanko/unified_medical_scenario_benchmark"


class UnifiedMedicalScenarioTask(Task):
    _row: Dict[str, Any] = PrivateAttr()
    _done: bool = PrivateAttr(default=False)

    def __init__(self, row: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._row = row

    @property
    def row(self) -> Dict[str, Any]:
        return self._row

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._done = False
        benchmark = self._row["benchmark"]
        contents = [
            Content.from_data(benchmark, name="benchmark"),
            Content.from_data(self._row["source_id"], name="source_id"),
            Content.from_data(json.loads(self._row["reward_spec_json"]), name="reward_spec"),
            Content.from_data(json.loads(self._row["eval_spec_json"]), name="eval_spec"),
        ]
        if benchmark == "mmlu_medical":
            contents.extend(
                [
                    Content.from_data(self._row["question"], name="question"),
                    Content.from_data(json.loads(self._row["choices_json"]), name="choices"),
                ]
            )
        elif benchmark == "healthbench":
            contents.extend(
                [
                    Content.from_data(json.loads(self._row["messages_json"]), name="messages"),
                    Content.from_data(json.loads(self._row["rubrics_json"]), name="rubrics"),
                ]
            )
        elif benchmark == "medagentbench":
            contents.extend(
                [
                    Content.from_data(self._row["instruction"], name="instruction"),
                    Content.from_data(self._row["context"], name="context"),
                    Content.from_data(json.loads(self._row["tools_json"]), name="tools"),
                ]
            )
        return Observation(contents=contents), {"row": self._row}

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        self._done = True
        actions = action if isinstance(action, list) else [action]
        return EnvironmentOutput(
            obs=Observation.from_text("Task output recorded. Use run_unified_eval.py for external scoring."),
            reward=0.0,
            done=True,
            info={"actions": [a.model_dump() for a in actions], "row": self._row},
        )

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {"note": "External scoring required for HealthBench and MedAgentBench."}


class UnifiedMedicalScenarioTaskConfig(TaskConfig):
    row: Dict[str, Any] = Field(..., description="Unified medical scenario row")

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: Any = None,
    ) -> UnifiedMedicalScenarioTask:
        return UnifiedMedicalScenarioTask(
            row=self.row,
            metadata=TaskMetadata(
                id=self.task_id,
                abstract_description=f"{self.row['benchmark']} unified medical scenario",
            ),
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context,
        )


class UnifiedMedicalScenarioBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="Unified-Medical-Scenario-Benchmark",
        description="Unified MMLU-Medical, HealthBench, and MedAgentBench CUBE benchmark.",
        version="1.0.0",
        extra_info={"id": "unified-medical-scenario-benchmark-v1", "source": DEFAULT_DATASET_ID},
    )
    task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[Type[TaskConfig]] = UnifiedMedicalScenarioTaskConfig

    _rows: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        split: str = "test",
        dataset_id: str = DEFAULT_DATASET_ID,
        benchmarks: Optional[List[str]] = None,
        num_examples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ds = load_dataset(dataset_id, split=split)
        rows = [dict(row) for row in ds]
        if benchmarks:
            allowed = set(benchmarks)
            rows = [row for row in rows if row["benchmark"] in allowed]
        if num_examples is not None:
            rows = rows[:num_examples]
        self._rows = rows
        object.__setattr__(
            self,
            "task_metadata",
            {
                row["id"]: TaskMetadata(
                    id=row["id"],
                    abstract_description=f"{row['benchmark']} unified medical scenario",
                )
                for row in rows
            },
        )

    def _setup(self):
        pass

    def close(self):
        pass

    def get_task_configs(self) -> List[UnifiedMedicalScenarioTaskConfig]:
        return [
            UnifiedMedicalScenarioTaskConfig(task_id=row["id"], row=row, tool_config=ToolboxConfig())
            for row in self._rows
        ]

    def get_task(self, index: int) -> UnifiedMedicalScenarioTask:
        row = self._rows[index]
        return UnifiedMedicalScenarioTaskConfig(
            task_id=row["id"], row=row, tool_config=ToolboxConfig()
        ).make()

    def __len__(self):
        return len(self._rows)


UnifiedMedicalScenarioTask.model_rebuild()
UnifiedMedicalScenarioTaskConfig.model_rebuild()
UnifiedMedicalScenarioBenchmark.model_rebuild()

