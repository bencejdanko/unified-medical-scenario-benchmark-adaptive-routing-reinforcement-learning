# medqa-cube-adaptive-curriculum-tool-calling

##

| Evaluation | Model | Score |
| --- | --- | --- |
| [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) | |
| [HealthBench](https://github.com/openai/simple-evals) | | |

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