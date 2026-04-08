# Adaptive Learning Framework for Medical Scenarios with Tool Calling

## Abstract

In our research, we apply aim to test and evaluate on multiple medical QA benchmarks

### Methodology

#### Data Sources

- MedQA (must convert to CUBE)
- MedMCQA (must convert to CUBE)
- PubMedQA (must convert to CUBE)
- MMLU (Medical Subset) (must convert to CUBE)

- HealthBench (must convert to CUBE - already done locally?)
- MedAgentBench (must convert to CUBE - already done locally?)

#### MedAgentBench

MedAgentBench [@jiang2025medagentbenchrealisticvirtualehr] has over 300 tasks across the 10 categories. It consists of a docker server containing data for the realistic profiles of 100 patients, and 700,000 data elements relevant to the tasks.

Ensure you pull the patient data needed first:

```bash
# pull the image
docker pull jyxsu6/medagentbench:latest

# ensure it's running on port 8080
docker run -d --name medagentbench_server -p 8080:8080 medagentbench
```

#### CUBE (Common Unified Benchmark Environments)

AI research community is suffering from severe fragmentation when it comes to evaluating AI agents. Every time a new benchmark is created, it requires a lot of custom integration work

CUBE (Common Unified Benchmark Environments), which serves as a universal protocol standard built on top of MCP (Model Context Protocol) and OpenAI's Gym. The goal is to allow developers to wrap benchmarks just once so they can be accessed universally for evaluation, reinforcement learning training, or data generation across any compliant platform.

#### Observability and Data Collection Stack

#### Contributions 

- Using industry tools to standardize Process Reward Modeling
  - Fully reproducible infrastructure (github, docker-compose)
  - Process Reward Model (PRM) as formally-verifable computer code (Lean-Lang)
- Mathematical formalization of long form tool-use and curriculum reward training
- Robust evaluation on a unified evaluation protocol (CUBE)

Compare:

- RAG without adaption
- Compare existing tool calling frameworks (pure MCP)

Metrics:
- Accuracy
- F1
- Calibration (ECE)
- Faithfulness
- Semantic Similarity (SBERT suggested)