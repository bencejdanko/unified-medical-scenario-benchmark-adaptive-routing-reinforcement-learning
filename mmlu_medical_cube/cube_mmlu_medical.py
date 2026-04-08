"""
MMLU Medical Subset CUBE Environment
=====================================
Wraps the medical-related subjects from MMLU as a CUBE benchmark.
Subjects: anatomy, clinical_knowledge, college_biology, college_medicine,
          medical_genetics, professional_medicine, nutrition, virology,
          high_school_biology.

Dataset: cais/mmlu on HuggingFace (filtered to medical subjects)
"""

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from pydantic import Field, PrivateAttr

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.core import Action, Content, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import ToolboxConfig

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATA_PATH = DATA_DIR / "mmlu_medical_test.jsonl"

IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

MEDICAL_SUBJECTS = [
    "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
    "medical_genetics", "professional_medicine", "nutrition", "virology",
    "high_school_biology",
]


def _normalize_option(text: str) -> str:
    return text.strip().upper()[:1]


class MMLUMedicalTask(Task):
    """A single MMLU medical question as a CUBE Task."""

    _data: Dict[str, Any] = PrivateAttr()
    _done: bool = PrivateAttr(default=False)

    def __init__(self, data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._data = data
        self._done = False

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._done = False
        d = self._data
        question = d["question"]
        choices = d["choices"]

        formatted = f"{question}\n\n"
        for idx, choice in enumerate(choices):
            letter = IDX_TO_LETTER[idx]
            formatted += f"{letter}. {choice}\n"
        formatted += "\nAnswer with just the letter (A, B, C, or D)."

        obs = Observation(contents=[
            Content.from_data(formatted, name="question"),
            Content.from_data({"subject": d.get("subject", "")}, name="metadata"),
        ])
        return obs, {}

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        if self._done:
            raise RuntimeError("Task already completed.")

        actions = action if isinstance(action, list) else [action]

        for act in actions:
            if act.name == "answer":
                answer = _normalize_option(act.arguments.get("content", ""))
                correct_idx = self._data["answer"]
                correct = IDX_TO_LETTER.get(correct_idx, "?")
                score = 1.0 if answer == correct else 0.0
                self._done = True

                correct_text = self._data["choices"][correct_idx] if correct_idx < len(self._data["choices"]) else ""

                return EnvironmentOutput(
                    obs=Observation.from_text(
                        f"{'Correct' if score == 1.0 else 'Incorrect'}. "
                        f"The answer was {correct}: {correct_text}"
                    ),
                    reward=score,
                    done=True,
                    info={
                        "score": score,
                        "predicted": answer,
                        "correct": correct,
                        "subject": self._data.get("subject", ""),
                    },
                )

        return EnvironmentOutput(
            obs=Observation.from_text("Please provide your answer with action name 'answer'."),
            reward=0.0,
            done=False,
        )

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}


class MMLUMedicalTaskConfig(TaskConfig):
    data: Dict[str, Any] = Field(..., description="MMLU medical question record")

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> MMLUMedicalTask:
        metadata = TaskMetadata(id=self.task_id, abstract_description="MMLU medical question")
        return MMLUMedicalTask(
            data=self.data,
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context,
        )


class MMLUMedicalBenchmark(Benchmark):
    """MMLU Medical Subset benchmark (1871 questions across 9 medical subjects)."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="MMLU-Medical-Benchmark",
        description="Medical knowledge subset of MMLU across 9 subjects.",
        version="1.0.0",
        extra_info={
            "id": "mmlu-medical-benchmark-v1",
            "source": "cais/mmlu",
            "subjects": MEDICAL_SUBJECTS,
        },
    )

    task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[Type[TaskConfig]] = MMLUMedicalTaskConfig

    _task_data: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        num_examples: Optional[int] = None,
        data_path: Optional[str] = None,
        subjects: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._load_tasks(num_examples, data_path, subjects)

    def _setup(self):
        pass

    def close(self):
        pass

    def _load_tasks(
        self,
        num_examples: Optional[int],
        data_path: Optional[str] = None,
        subjects: Optional[List[str]] = None,
    ):
        path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self._task_data = {}
        new_metadata = {}

        with open(path) as f:
            count = 0
            for i, line in enumerate(f):
                if num_examples is not None and count >= num_examples:
                    break
                data = json.loads(line)
                # Filter by subject if specified
                if subjects and data.get("subject", "") not in subjects:
                    continue
                task_id = f"mmlu_med_{i}"
                self._task_data[task_id] = data
                new_metadata[task_id] = TaskMetadata(
                    id=task_id,
                    abstract_description=f"MMLU {data.get('subject', '')} question {i}",
                )
                count += 1

        object.__setattr__(self, "task_metadata", new_metadata)

    def get_task_configs(self) -> List[MMLUMedicalTaskConfig]:
        return [
            MMLUMedicalTaskConfig(task_id=tid, data=self._task_data[tid], tool_config=ToolboxConfig())
            for tid in self.task_metadata
        ]

    def get_task(self, index: int) -> MMLUMedicalTask:
        task_id = list(self.task_metadata.keys())[index]
        return MMLUMedicalTaskConfig(
            task_id=task_id, data=self._task_data[task_id], tool_config=ToolboxConfig()
        ).make()

    def __len__(self):
        return len(self.task_metadata)


MMLUMedicalTask.model_rebuild()
MMLUMedicalBenchmark.model_rebuild()
MMLUMedicalTaskConfig.model_rebuild()
