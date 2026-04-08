"""
MedQA CUBE Environment
======================
Wraps the MedQA (USMLE-style) dataset as a CUBE benchmark.
Each task presents a clinical vignette with 4 options (A-D).
Scoring is exact-match on the correct option letter.

Dataset: openlifescienceai/medqa on HuggingFace
"""

import json
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from pydantic import Field, PrivateAttr

from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.core import Action, Content, EnvironmentOutput, Observation
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import ToolboxConfig

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DATA_PATH = DATA_DIR / "medqa_test.jsonl"


def _normalize_option(text: str) -> str:
    """Normalize an answer option letter."""
    return text.strip().upper()[:1]


class MedQATask(Task):
    """A single MedQA multiple-choice question as a CUBE Task."""

    _data: Dict[str, Any] = PrivateAttr()
    _done: bool = PrivateAttr(default=False)

    def __init__(self, data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._data = data
        self._done = False

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._done = False
        q = self._data["data"]
        question_text = q["Question"]
        options = q["Options"]

        # Format the prompt as a clean multiple-choice question
        formatted = f"{question_text}\n\n"
        for letter in sorted(options.keys()):
            formatted += f"{letter}. {options[letter]}\n"
        formatted += "\nAnswer with just the letter (A, B, C, or D)."

        obs = Observation(contents=[
            Content.from_data(formatted, name="question"),
            Content.from_data(
                {"options": options, "subject": self._data.get("subject_name", "")},
                name="metadata",
            ),
        ])
        return obs, {}

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        if self._done:
            raise RuntimeError("Task already completed.")

        actions = action if isinstance(action, list) else [action]

        for act in actions:
            if act.name == "answer":
                answer = _normalize_option(act.arguments.get("content", ""))
                correct = _normalize_option(self._data["data"]["Correct Option"])
                score = 1.0 if answer == correct else 0.0
                self._done = True
                return EnvironmentOutput(
                    obs=Observation.from_text(
                        f"{'Correct' if score == 1.0 else 'Incorrect'}. "
                        f"The answer was {correct}: {self._data['data']['Correct Answer']}"
                    ),
                    reward=score,
                    done=True,
                    info={
                        "score": score,
                        "predicted": answer,
                        "correct": correct,
                        "correct_answer": self._data["data"]["Correct Answer"],
                    },
                )

        return EnvironmentOutput(
            obs=Observation.from_text("Please provide your answer with action name 'answer'."),
            reward=0.0,
            done=False,
        )

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}


class MedQATaskConfig(TaskConfig):
    data: Dict[str, Any] = Field(..., description="MedQA question record")

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> MedQATask:
        metadata = TaskMetadata(id=self.task_id, abstract_description="MedQA USMLE question")
        return MedQATask(
            data=self.data,
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context,
        )


class MedQABenchmark(Benchmark):
    """MedQA USMLE-style medical QA benchmark (1273 test questions)."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="MedQA-Benchmark",
        description="USMLE-style medical multiple-choice QA (MedQA).",
        version="1.0.0",
        extra_info={"id": "medqa-benchmark-v1", "source": "openlifescienceai/medqa"},
    )

    task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[Type[TaskConfig]] = MedQATaskConfig

    _task_data: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, num_examples: Optional[int] = None, data_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._load_tasks(num_examples, data_path)

    def _setup(self):
        pass

    def close(self):
        pass

    def _load_tasks(self, num_examples: Optional[int], data_path: Optional[str] = None):
        path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self._task_data = {}
        new_metadata = {}

        with open(path) as f:
            for i, line in enumerate(f):
                if num_examples is not None and i >= num_examples:
                    break
                data = json.loads(line)
                task_id = f"medqa_{i}"
                self._task_data[task_id] = data
                new_metadata[task_id] = TaskMetadata(
                    id=task_id,
                    abstract_description=f"MedQA question {i}",
                )

        object.__setattr__(self, "task_metadata", new_metadata)

    def get_task_configs(self) -> List[MedQATaskConfig]:
        return [
            MedQATaskConfig(task_id=tid, data=self._task_data[tid], tool_config=ToolboxConfig())
            for tid in self.task_metadata
        ]

    def get_task(self, index: int) -> MedQATask:
        task_id = list(self.task_metadata.keys())[index]
        return MedQATaskConfig(task_id=task_id, data=self._task_data[task_id], tool_config=ToolboxConfig()).make()

    def __len__(self):
        return len(self.task_metadata)


MedQATask.model_rebuild()
MedQABenchmark.model_rebuild()
MedQATaskConfig.model_rebuild()
