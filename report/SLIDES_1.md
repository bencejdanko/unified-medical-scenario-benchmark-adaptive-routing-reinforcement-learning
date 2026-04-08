---
marp: true
theme: uncover
---

# Adaptive Learning Framework for Medical Question Answering with Tool Calling

---

## Problem Definition

- Integrating medical domain knowledge to LLMs
- Adding agentic ability and tool use to medical LLMs
- Models that know the medical facts cannot navigate the complex real medical records or follow multi-step safety protocols
<!--- Notes

--->

---

### QA Datasets

| Evaluation | Size | 
| :-- | ---: |
| MedMCQA | 4,183 |
| MedQA | 1,273 |
| PubMedQA (expert-annotated test split) | 500 |
| MMLU-Medical (9 MMLU subjects) | 1,871 |

accuracy, F1, ECE metrics

---

### HealthBench

- Released by OpenAI May 2025
- Alternative to saturated QA benchmarks
- 5,000 multi-turn clinical conversations (open ended)
- synthetic generation and human adversarial testing
- Each sample has it's own doctor-annotated rubric

---

### MedAgentBench

- Stanford Machine Learning Group early 2025
- 100 real world patients and real clinical data
- 785,000 unique data points (labs, medications, vitals, diagnoses)
- 300 tasks written by human physicians across 10 medical categories

---

## Methodology

- Zero-shot baseline
- RAG

---

![alt text](image-1.png)

---

## Next Steps and Expected Contributions

---

**Originality:** A meaningful new idea or improvement
**Rigorous Evaluation:** Proper baselines and metrics
**Robustness:** Stability across settings
**Reproducibility:** Clear, repeatable results