# Unified Medical Scenario Benchmark: Training an Adaptive Routing LLM Framework with Reinforcement Learning

In recent years, several major NLP and LLM-focused benchmarks have emerged to test language models on medical domain knowledge and proficiency. Early medical benchmarks and datasets were based on zero-shot, one turn question-answering (QA) [^1]. However, for practical medical enviroment use, new evaluation frameworks have emerged that test not just factuality, but strict human-aligned guardrailing [^2], and real-world medical systems interaction (automated tool use) [^3].

Despite the growing breadth of evaluation frameworks, the community has not yet adopted one single, unified technical approach to conducting reproducible evaluations. CUBE [^4] is one such proposed library standard, in which incompatible benchmarks are meant to be wrapped uniformly used anywhere.

There is a practical research need for such a unified, multi-scenario benchmark. In practical applied LLM systems, for economic needs, engineers are creating router-based LLM orchestrations that route requests based on request needs. Simple queries routing to less powerful but capable models, and more complex requests need to be allocated to more powerful agentic harnesses.

This implementation attempts not only to unify these benchmarks for the purposes of demonstrating the utility of CUBE wrapper, but also it's practical and industry-relevant purpose in training economic LLM systems, not just powerful model benchmarking.

[^1]: MedMCQA, MedQA, PubMedQA, MMLU (Medical sets)
[^2]: Healthbench (OpenAI)
[^3]: MedAgentBench (Stanford)

Mid Progress Slide Deck:

https://docs.google.com/presentation/d/1MtnFUPUmQn5fVKny17UsET8ZU8APDlzg6u4PLNv_DAE/edit?usp=sharing

## QA

Zero shot, {"answer": [`A,B,C,D`], "confidence"": `x.xx`} format

| Evaluation | Size | 
| --- | --- |
| MedMCQA | 4,183 |
| MedQA | 1,273 |
| PubMedQA (expert-annotated) | 500 |
| MMLU-Medical (9 MMLU subjects) | 1,871 |

| Evaluation | Model | Accuracy | F1-Macro | ECE |
| --- | --- | --- | --- | --- |
| MedQA | gemini-flash-preview | 0.9442 | 0.9438 | 0.0332 |
| MedQA | gemma-4-31b-it | 0.8409 | 0.8394 | 0.1344 |
| MedQA | gpt-oss-120b | 0.9018 | 0.906 | 0.0632 |
| MedMCQA | gemini-flash-preview | 0.8505 | 0.8499 | 0.1047 |
| MedMCQA | gemma-4-31b-it | 0.7496 | 0.7475 | 0.2104 |
| MedMCQA | gpt-oss-120b | 0.7352 | 0.7377 | 0.1923 |
| PubMedQA | gemini-flash-preview | 0.8048 | 0.6238 | 0.1443 |
| PubMedQA | gemma-4-31b-it | 0.7940 | 0.5901 | 0.1709 |
| PubMedQA | gpt-oss-120b | 0.6680 | 0.5800 | 0.2241 |
| MMLU-Medical | gemini-flash-preview | 0.9221 | 0.9198 | 0.0483 |
| MMLU-Medical | gemma-4-31b-it | 0.8938 | 0.8914 | 0.0868 |
| MMLU-Medical | gpt-oss-120b | 0.8859 | 0.8866 | 0.0819 |

## RAG

| Evaluation | Model | Accuracy | F1-Macro | ECE |
| --- | --- | --- | --- | --- |
| MedQA | gpt-oss-120b | 0.9018 | 0.906 | 0.0632 |
| MedQA RAG | gpt-oss-120b | 0.8995 | 0.9019 | 0.0612 |
| MedMCQA | gpt-oss-120b | 0.7352 | 0.7377 | 0.1923 |
| MedMCQA RAG | gpt-oss-120b | 0.7322 | 0.7333 | 0.1925 |
| PubMedQA | gpt-oss-120b | 0.6680 | 0.5800 | 0.2241 |
| PubMedQA RAG | gpt-oss-120b | 0.6440 | 0.5544 | 0.2413 |
| MMLU-Medical | gpt-oss-120b | 0.8859 | 0.8866 | 0.0819 |
| MMLU-Medical RAG | gpt-oss-120b | 0.8862  | 0.8868 | 0.0773 |

## HealthBench (50 hard examples)

| Model | Score |
| --- | --- |
| gemini-3-flash-preview | 0.3939 |
| gemma-4-31b-it | 0.3611 |
| gpt-oss-120b | 0.4129 | 

## Unified Benchmark Framework

This repository now includes a unified CUBE-compatible benchmark layer in `unified_medical_benchmark/`.

The unified benchmark is designed as a benchmark product, not a routing policy. It preserves the existing CUBE wrappers for MedQA, MedMCQA, PubMedQA, MMLU-Medical, HealthBench, and MedAgentBench, then composes them behind one shared interface suitable for future adaptive routing experiments.

The framework adds:

- **Scenario metadata**: every task receives a normalized `scenario_metadata` payload with benchmark name, source task ID, scenario type, clinical domain, complexity, answer format, category, and operational requirements.
- **Complexity/category awareness**: QA, literature-grounded QA, rubric-scored conversation, and EHR tool-use tasks can be stratified before any router is implemented.
- **Reward preservation**: each source CUBE wrapper remains responsible for native scoring, so exact-match QA, rubric grading, and tool-use task scoring are not reimplemented in the unified layer.
- **Step tracking**: every call to `step()` records action name, action arguments, reward, done state, elapsed time, source info, and cumulative trace history in `EnvironmentOutput.info`.
- **Stable task identity**: unified task IDs use `<benchmark>::<source_task_id>`, allowing benchmark-level aggregation and source-level debugging.

Example:

```python
from cube.core import Action
from unified_medical_benchmark import build_benchmark

bench = build_benchmark(
    benchmark_names=["medqa", "medmcqa", "pubmedqa", "mmlu_medical"],
    num_examples_per_benchmark=10,
)

task = bench.get_task(0)
obs, info = task.reset()
print(info["scenario_metadata"])

result = task.step(Action(name="answer", arguments={"content": "A"}))
print(result.reward)
print(result.info["step_records"])
```

See `unified_medical_benchmark/README.md` for the architecture, taxonomy, runtime requirements, and CUBE integration details.


## Requirements

### CUBE-Standard

Benchmarks were conducted with this fork of CUBE-Standard:

```bash
git clone https://github.com/bencejdanko/cube-standard

# then install it directly
pip install -e cube-standard
```

### MedAgentBench

MedAgentBench [@jiang2025medagentbenchrealisticvirtualehr] has over 300 tasks across the 10 categories. It consists of a docker server containing data for the realistic profiles of 100 patients, and 700,000 data elements relevant to the tasks.

Ensure you pull the patient data needed first:

```bash
# pull the image
docker pull jyxsu6/medagentbench:latest

# ensure it's running on port 8080
docker run -d --name medagentbench_server -p 8080:8080 medagentbench
```

### HealthBench

HealthBench [] (TODO)

5000 samples of questions, with corresponding rubrics.

### Additional QA Benchmarks to test

from `openlifescienceai` on huggingface:

- MedQA (must convert to CUBE)
- MedMCQA (must convert to CUBE)
- PubMedQA (must convert to CUBE)
- MMLU (Medical Subset) (must convert to CUBE)

Corpus: 

- PubMed
- UMLS

Goal:

Compare:

- For QA: Baseline prompting (zero shot)
- For QA: RAG without adaption (naive)
- For QA: Compare RAG against existing tool calling frameworks (pure MCP). Also compare against Bash (letting agent loose on just raw files).
- For Agent: Compare with randomized noisy tools (MCP, original 9, -> 20 -> 50), seeing if adding more noisy tools degrades performance on these models.

Target models:

`gemini/gemini-3-flash-preview` (OpenRouter)
`google/gemma-4-31b-it` (OpenRouter)
`gpt-oss-120b` (Cerebras)

For noisy tools, let's download the `Nanbeige/ToolMind` repository. inside are random synthetic - sometimes relevant - tools we can use as noise. 

Use LangFuse for full traceability. Secrets set in `.env`. The idea is we eventually use the Langfuse "Datasets" feature to convert annotated traces into your training data for fine tuned model iterations.

My HF token is available locally at HF_TOKEN - use this to synchronize datasets. 

Use the modal CLI to run heavy preprocessing jobs if necessary. Ensure synchronization with HuggingFace. HuggingFace key available at secrets=[modal.Secret.from_name("my-huggingface-secret")], for example.

---

```
 modal run modal_rag_eval.py --model gpt-oss-120b --top-k 5

modal run modal_rag_eval.py --model gpt-oss-120b --benchmark "medqa,medmcqa,pubmedqa,mmlu_medical" --top-k 5

modal run modal_rag_eval.py --model gpt-oss-120b --benchmark "pubmedqa,mmlu_medical" --top-k 5
 ```

```
python run_healthbench_eval.py --model gemini-3-flash-preview --subset hard --num-examples 50

```
