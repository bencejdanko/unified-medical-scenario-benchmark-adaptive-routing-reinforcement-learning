# Unified Medical Scenario Benchmark: Training an Adaptive Routing LLM Framework with Reinforcement Learning

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

Possible corpus: 

- PubMed
- UMLS
- SOTA needed

Unsure if maybe useful (needs manual labor/effort/scrape):
- Clinical Guidelines (https://www.guidelinecentral.com/guidelines/)
- MedlinePlus
- ClinicalTrials.gov

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
