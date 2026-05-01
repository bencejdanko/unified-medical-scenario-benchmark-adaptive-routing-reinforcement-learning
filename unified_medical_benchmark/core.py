from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from pydantic import PrivateAttr

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.core import Action, Content, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import ToolboxConfig


@dataclass(frozen=True)
class ScenarioMetadata:
    benchmark: str
    source_task_id: str
    unified_task_id: str
    scenario_type: str
    clinical_domain: str
    complexity: str
    answer_format: str
    requires_tools: bool = False
    requires_rubric_grading: bool = False
    category: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    action_name: str
    action_arguments: Dict[str, Any]
    reward: float
    done: bool
    elapsed_s: float
    info: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    benchmark_factory: Callable[..., Benchmark]
    scenario_type: str
    clinical_domain: str
    complexity: str
    answer_format: str
    requires_tools: bool = False
    requires_rubric_grading: bool = False
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    category_extractor: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None

    def build(self, **overrides: Any) -> Benchmark:
        kwargs = dict(self.default_kwargs)
        kwargs.update(overrides)
        return self.benchmark_factory(**kwargs)

    def metadata_for(self, source_task_id: str, unified_task_id: str, task_data: Dict[str, Any]) -> ScenarioMetadata:
        category = self.category_extractor(task_data) if self.category_extractor else None
        return ScenarioMetadata(
            benchmark=self.name,
            source_task_id=source_task_id,
            unified_task_id=unified_task_id,
            scenario_type=self.scenario_type,
            clinical_domain=self.clinical_domain,
            complexity=self.complexity,
            answer_format=self.answer_format,
            requires_tools=self.requires_tools,
            requires_rubric_grading=self.requires_rubric_grading,
            category=category,
            source=self.source,
            tags=list(self.tags),
        )


class UnifiedMedicalTask(Task):
    _base_task: Task = PrivateAttr()
    _scenario_metadata: ScenarioMetadata = PrivateAttr()
    _step_records: List[StepRecord] = PrivateAttr(default_factory=list)

    def __init__(self, base_task: Task, scenario_metadata: ScenarioMetadata, **kwargs: Any):
        super().__init__(**kwargs)
        self._base_task = base_task
        self._scenario_metadata = scenario_metadata
        self._step_records = []

    @property
    def scenario_metadata(self) -> ScenarioMetadata:
        return self._scenario_metadata

    @property
    def step_records(self) -> List[StepRecord]:
        return list(self._step_records)

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._step_records = []
        obs, info = self._base_task.reset()
        obs.contents.append(Content.from_data(asdict(self._scenario_metadata), name="scenario_metadata"))
        info = dict(info or {})
        info["scenario_metadata"] = asdict(self._scenario_metadata)
        return obs, info

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        started = time.perf_counter()
        output = self._base_task.step(action)
        elapsed = time.perf_counter() - started
        actions = action if isinstance(action, list) else [action]
        action_name = "+".join(act.name for act in actions)
        action_arguments = {str(i): act.arguments for i, act in enumerate(actions)}
        info = dict(output.info or {})
        info["scenario_metadata"] = asdict(self._scenario_metadata)
        record = StepRecord(
            step_index=len(self._step_records),
            action_name=action_name,
            action_arguments=action_arguments,
            reward=float(output.reward),
            done=bool(output.done),
            elapsed_s=elapsed,
            info=info,
        )
        self._step_records.append(record)
        info["step_record"] = asdict(record)
        info["step_records"] = [asdict(r) for r in self._step_records]
        return EnvironmentOutput(obs=output.obs, reward=output.reward, done=output.done, info=info)

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        score, info = self._base_task.evaluate(obs)
        info = dict(info or {})
        info["scenario_metadata"] = asdict(self._scenario_metadata)
        info["step_records"] = [asdict(r) for r in self._step_records]
        return score, info


class UnifiedMedicalTaskConfig(TaskConfig):
    base_config: TaskConfig
    scenario_metadata: Dict[str, Any]

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> UnifiedMedicalTask:
        metadata = TaskMetadata(id=self.task_id, abstract_description="Unified medical scenario task")
        base_task = self.base_config.make(runtime_context=runtime_context, container_backend=container_backend)
        return UnifiedMedicalTask(
            base_task=base_task,
            scenario_metadata=ScenarioMetadata(**self.scenario_metadata),
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context,
        )


class UnifiedMedicalBenchmark(Benchmark):
    benchmark_metadata = BenchmarkMetadata(
        name="Unified-Medical-Scenario-Benchmark",
        description="Unified CUBE benchmark spanning medical QA, rubric safety, and EHR tool-use scenarios.",
        version="1.0.0",
        extra_info={
            "id": "unified-medical-scenario-benchmark-v1",
            "purpose": "benchmark substrate for adaptive medical routing systems",
        },
    )
    task_metadata: Dict[str, TaskMetadata] = {}
    task_config_class = UnifiedMedicalTaskConfig

    _adapters: Dict[str, BenchmarkAdapter] = PrivateAttr(default_factory=dict)
    _benchmarks: Dict[str, Benchmark] = PrivateAttr(default_factory=dict)
    _entries: List[Tuple[str, str, TaskConfig, ScenarioMetadata]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        adapters: Iterable[BenchmarkAdapter],
        benchmark_names: Optional[Iterable[str]] = None,
        per_benchmark_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        num_examples_per_benchmark: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._adapters = {adapter.name: adapter for adapter in adapters}
        self._benchmarks = {}
        self._entries = []
        self._load_tasks(benchmark_names, per_benchmark_kwargs or {}, num_examples_per_benchmark)

    def _setup(self):
        pass

    def close(self):
        for benchmark in self._benchmarks.values():
            close = getattr(benchmark, "close", None)
            if close:
                close()

    def _load_tasks(
        self,
        benchmark_names: Optional[Iterable[str]],
        per_benchmark_kwargs: Dict[str, Dict[str, Any]],
        num_examples_per_benchmark: Optional[int],
    ) -> None:
        selected = list(benchmark_names) if benchmark_names else list(self._adapters.keys())
        new_metadata: Dict[str, TaskMetadata] = {}
        for benchmark_name in selected:
            adapter = self._adapters[benchmark_name]
            overrides = dict(per_benchmark_kwargs.get(benchmark_name, {}))
            if num_examples_per_benchmark is not None and "num_examples" not in overrides:
                overrides["num_examples"] = num_examples_per_benchmark
            benchmark = adapter.build(**overrides)
            self._benchmarks[benchmark_name] = benchmark
            configs = benchmark.get_task_configs()
            if num_examples_per_benchmark is not None:
                configs = configs[:num_examples_per_benchmark]
            task_data = getattr(benchmark, "_task_data", {})
            for config in configs:
                unified_task_id = f"{benchmark_name}::{config.task_id}"
                source_data = task_data.get(config.task_id, {}) if isinstance(task_data, dict) else {}
                scenario_metadata = adapter.metadata_for(config.task_id, unified_task_id, source_data)
                self._entries.append((unified_task_id, benchmark_name, config, scenario_metadata))
                new_metadata[unified_task_id] = TaskMetadata(
                    id=unified_task_id,
                    abstract_description=f"{adapter.scenario_type} scenario from {benchmark_name}",
                )
        object.__setattr__(self, "task_metadata", new_metadata)

    def get_task_configs(self) -> List[UnifiedMedicalTaskConfig]:
        return [
            UnifiedMedicalTaskConfig(
                task_id=unified_task_id,
                base_config=config,
                scenario_metadata=asdict(scenario_metadata),
                tool_config=ToolboxConfig(),
            )
            for unified_task_id, _benchmark_name, config, scenario_metadata in self._entries
        ]

    def get_task(self, index: int) -> UnifiedMedicalTask:
        return self.get_task_configs()[index].make()

    def get_scenario_metadata(self) -> List[ScenarioMetadata]:
        return [entry[3] for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)


UnifiedMedicalTask.model_rebuild()
UnifiedMedicalTaskConfig.model_rebuild()
UnifiedMedicalBenchmark.model_rebuild()
