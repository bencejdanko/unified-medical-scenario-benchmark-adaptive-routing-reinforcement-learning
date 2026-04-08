# MMLU Medical Subset CUBE Environment

CUBE wrapper for the medical-related subjects from [MMLU](https://github.com/hendrycks/test) (Massive Multitask Language Understanding).

## Dataset

- **Source**: `cais/mmlu` on HuggingFace
- **Test split**: 1,871 questions across 9 medical subjects
- **Subjects**: anatomy, clinical_knowledge, college_biology, college_medicine, medical_genetics, professional_medicine, nutrition, virology, high_school_biology
- **Format**: Knowledge question → 4 choices (A–D) → single correct answer

## Usage

```python
from mmlu_medical_cube import MMLUMedicalBenchmark
from cube.core import Action

# Load all subjects
bench = MMLUMedicalBenchmark(num_examples=10)

# Or filter to specific subjects
bench = MMLUMedicalBenchmark(subjects=["anatomy", "clinical_knowledge"])

task = bench.get_task(0)
obs, info = task.reset()
print(obs.contents[0].to_markdown())

result = task.step(Action(name="answer", arguments={"content": "A"}))
print(f"Score: {result.reward}")
```

## Citation

```bibtex
@article{hendrycks2021measuring,
  title={Measuring Massive Multitask Language Understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={ICLR},
  year={2021}
}
```
