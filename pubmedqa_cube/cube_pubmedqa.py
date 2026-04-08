"""
PubMedQA CUBE Environment
=========================
Wraps PubMedQA as a CUBE benchmark.
Each task presents a biomedical research question with context (abstract excerpts)
and 3 options: yes / no / maybe.

Dataset: openlifescienceai/pubmedqa on HuggingFace
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
DEFAULT_DATA_PATH = DATA_DIR / "pubmedqa_test.jsonl"

ANSWER_MAP = {"A": "yes", "B": "no", "C": "maybe"}
REVERSE_MAP = {"yes": "A", "no": "B", "maybe": "C"}


def _normalize_answer(text: str) -> str:
    """Accept either letter (A/B/C) or word (yes/no/maybe)."""
    t = text.strip().lower()
    if t in REVERSE_MAP:
        return REVERSE_MAP[t]
    return text.strip().upper()[:1]


class PubMedQATask(Task):
    """A single PubMedQA question as a CUBE Task."""

    _data: Dict[str, Any] = PrivateAttr()
    _done: bool = PrivateAttr(default=False)

    def __init__(self, data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._data = data
        self._done = False

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._done = False
        d = self._data["data"]
        question = d["Question"]
        context = d.get("Context", [])
        options = d.get("Options", {"A": "yes", "B": "no", "C": "maybe"})

        context_str = "\n\n".join(context) if isinstance(context, list) else str(context)

        formatted = f"Context:\n{context_str}\n\nQuestion: {question}\n\n"
        for letter in sorted(options.keys()):
            formatted += f"{letter}. {options[letter]}\n"
        formatted += "\nAnswer with the letter (A, B, or C) or the word (yes, no, maybe)."

        obs = Observation(contents=[
            Content.from_data(formatted, name="question"),
            Content.from_data(
                {
                    "context": context,
                    "long_answer": d.get("Long Answer", ""),
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
                answer = _normalize_answer(act.arguments.get("content", ""))
                correct = self._data["data"]["Correct Option"]
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
                        "long_answer": self._data["data"].get("Long Answer", ""),
                    },
                )

        return EnvironmentOutput(
            obs=Observation.from_text("Please provide your answer with action name 'answer'."),
            reward=0.0,
            done=False,
        )

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}


class PubMedQATaskConfig(TaskConfig):
    data: Dict[str, Any] = Field(..., description="PubMedQA question record")

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> PubMedQATask:
        metadata = TaskMetadata(id=self.task_id, abstract_description="PubMedQA biomedical question")
        return PubMedQATask(
            data=self.data,
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context,
        )


class PubMedQABenchmark(Benchmark):
    """PubMedQA biomedical research QA benchmark (500 test questions)."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="PubMedQA-Benchmark",
        description="Biomedical research yes/no/maybe QA (PubMedQA).",
        version="1.0.0",
        extra_info={"id": "pubmedqa-benchmark-v1", "source": "openlifescienceai/pubmedqa"},
    )

    task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[Type[TaskConfig]] = PubMedQATaskConfig

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
                task_id = f"pubmedqa_{i}"
                self._task_data[task_id] = data
                new_metadata[task_id] = TaskMetadata(
                    id=task_id,
                    abstract_description=f"PubMedQA question {i}",
                )

        object.__setattr__(self, "task_metadata", new_metadata)

    def get_task_configs(self) -> List[PubMedQATaskConfig]:
        return [
            PubMedQATaskConfig(task_id=tid, data=self._task_data[tid], tool_config=ToolboxConfig())
            for tid in self.task_metadata
        ]

    def get_task(self, index: int) -> PubMedQATask:
        task_id = list(self.task_metadata.keys())[index]
        return PubMedQATaskConfig(task_id=task_id, data=self._task_data[task_id], tool_config=ToolboxConfig()).make()

    def __len__(self):
        return len(self.task_metadata)


PubMedQATask.model_rebuild()
PubMedQABenchmark.model_rebuild()
PubMedQATaskConfig.model_rebuild()
