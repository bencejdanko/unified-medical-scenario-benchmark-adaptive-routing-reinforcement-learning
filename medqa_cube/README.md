# MedQA CUBE Environment

CUBE (Common Unified Benchmark Environments) wrapper for the [MedQA](https://github.com/jind11/MedQA) dataset — USMLE-style medical multiple-choice questions.

## Dataset

- **Source**: `openlifescienceai/medqa` on HuggingFace
- **Test split**: 1,273 questions
- **Format**: Clinical vignette → 4 options (A–D) → single correct answer

## Usage

```python
from medqa_cube import MedQABenchmark
from cube.core import Action

bench = MedQABenchmark(num_examples=10)
task = bench.get_task(0)
obs, info = task.reset()

# The observation contains the question text and options
print(obs.contents[0].to_markdown())

# Submit an answer
result = task.step(Action(name="answer", arguments={"content": "B"}))
print(f"Score: {result.reward}, Done: {result.done}")
```

## Task Interface

- **reset()** → Observation with question text + metadata
- **step(Action(name="answer", arguments={"content": "<letter>"}))** → EnvironmentOutput with exact-match score (0.0 or 1.0)

## Citation

```bibtex
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}
```
