# PubMedQA CUBE Environment

CUBE wrapper for [PubMedQA](https://pubmedqa.github.io/) — biomedical research question answering with yes/no/maybe answers.

## Dataset

- **Source**: `openlifescienceai/pubmedqa` on HuggingFace
- **Test split**: 500 expert-annotated questions
- **Format**: PubMed abstract context → research question → yes / no / maybe

## Usage

```python
from pubmedqa_cube import PubMedQABenchmark
from cube.core import Action

bench = PubMedQABenchmark(num_examples=10)
task = bench.get_task(0)
obs, info = task.reset()
print(obs.contents[0].to_markdown())

result = task.step(Action(name="answer", arguments={"content": "yes"}))
print(f"Score: {result.reward}")
```

## Citation

```bibtex
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={EMNLP},
  year={2019}
}
```
