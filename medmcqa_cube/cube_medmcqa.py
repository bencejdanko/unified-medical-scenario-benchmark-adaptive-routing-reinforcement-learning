"""
MedMCQA CUBE Environment
========================
Wraps the MedMCQA dataset as a CUBE benchmark.
Each task presents a medical question with 4 options (A-D).
Scoring is exact-match on the correct option letter.

Dataset: openlifescienceai/medmcqa on HuggingFace
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
DEFAULT_DATA_PATH = DATA_DIR / "medmcqa_validation.jsonl"

# MedMCQA uses integer cop (0-3) for correct option
COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def _normalize_option(text: str) -> str:
    return text.strip().upper()[:1]


class MedMCQATask(Task):
    """A single MedMCQA question as a CUBE Task."""

    _data: Dict[str, Any] = PrivateAttr()
    _done: bool = PrivateAttr(default=False)

    def __init__(self, data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._data = data
        self._done = False

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._done = False
        d = self._data
        options = {"A": d["opa"], "B": d["opb"], "C": d["opc"], "D": d["opd"]}

        formatted = f"{d['question']}\n\n"
        for letter in ["A", "B", "C", "D"]:
            formatted += f"{letter}. {options[letter]}\n"
        formatted += "\nAnswer with just the letter (A, B, C, or D)."

        obs = Observation(contents=[
            Content.from_data(formatted, name="question"),
            Content.from_data(
                {
                    "options": options,
                    "subject": d.get("subject_name", ""),
                    "topic": d.get("topic_name", ""),
                    "explanation": d.get("exp", ""),
                },
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
                cop = self._data.get("cop", -1)
                correct = COP_TO_LETTER.get(cop, "?")
                score = 1.0 if answer == correct else 0.0
                self._done = True

                correct_text = {"A": self._data["opa"], "B": self._data["opb"],
                                "C": self._data["opc"], "D": self._data["opd"]}.get(correct, "")

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
                        "explanation": self._data.get("exp", ""),
                    },
                )

        return EnvironmentOutput(
            obs=Observation.from_text("Please provide your answer with action name 'answer'."),
            reward=0.0,
            done=False,
        )

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}


class MedMCQATaskConfig(TaskConfig):
    data: Dict[str, Any] = Field(..., description="MedMCQA question record")

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> MedMCQATask:
        metadata = TaskMetadata(id=self.task_id, abstract_description="MedMCQA medical question")
        return MedMCQATask(
            data=self.data,
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context,
        )


class MedMCQABenchmark(Benchmark):
    """MedMCQA medical entrance exam QA benchmark (4183 validation questions)."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="MedMCQA-Benchmark",
        description="Indian medical entrance exam multiple-choice QA (MedMCQA).",
        version="1.0.0",
        extra_info={"id": "medmcqa-benchmark-v1", "source": "openlifescienceai/medmcqa"},
    )

    task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[Type[TaskConfig]] = MedMCQATaskConfig

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
                # Skip items with cop=-1 (no correct answer)
                if data.get("cop", -1) == -1:
                    continue
                task_id = f"medmcqa_{i}"
                self._task_data[task_id] = data
                new_metadata[task_id] = TaskMetadata(
                    id=task_id,
                    abstract_description=f"MedMCQA question {i}",
                )

        object.__setattr__(self, "task_metadata", new_metadata)

    def get_task_configs(self) -> List[MedMCQATaskConfig]:
        return [
            MedMCQATaskConfig(task_id=tid, data=self._task_data[tid], tool_config=ToolboxConfig())
            for tid in self.task_metadata
        ]

    def get_task(self, index: int) -> MedMCQATask:
        task_id = list(self.task_metadata.keys())[index]
        return MedMCQATaskConfig(task_id=task_id, data=self._task_data[task_id], tool_config=ToolboxConfig()).make()

    def __len__(self):
        return len(self.task_metadata)


MedMCQATask.model_rebuild()
MedMCQABenchmark.model_rebuild()
MedMCQATaskConfig.model_rebuild()
