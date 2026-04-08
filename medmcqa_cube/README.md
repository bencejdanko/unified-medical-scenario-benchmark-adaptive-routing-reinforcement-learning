# MedMCQA CUBE Environment

CUBE wrapper for the [MedMCQA](https://medmcqa.github.io/) dataset — multiple-choice questions from Indian medical entrance exams (AIIMS/NEET).

## Dataset

- **Source**: `openlifescienceai/medmcqa` on HuggingFace
- **Validation split**: 4,183 questions (used as test — official test has no labels)
- **Format**: Medical question → 4 options (A–D) → single correct answer
- **Subjects**: Anatomy, Biochemistry, Pathology, Pharmacology, Microbiology, etc.

## Usage

```python
from medmcqa_cube import MedMCQABenchmark
from cube.core import Action

bench = MedMCQABenchmark(num_examples=10)
task = bench.get_task(0)
obs, info = task.reset()
print(obs.contents[0].to_markdown())

result = task.step(Action(name="answer", arguments={"content": "C"}))
print(f"Score: {result.reward}")
```

## Citation

```bibtex
@inproceedings{pmlr-v174-pal22a,
  title={MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle={Proceedings of the Conference on Health, Inference, and Learning},
  year={2022}
}
```
