# Adaptive Learning Framework for Medical Scenarios with Tool Calling

## Abstract

## Background

In our research, we apply curriculum learning, given unbalanced question domains and scaling irrelevant tools.

### Staged Training

To fine tune a model to this dataset, we'll train first on standard QA benchmarks.

### Adaptive Curriculum

- Start by rewarding the agent for simple, single-step exploits.
- Slowly increase the difficulty to multi-step, multi-tool exploit chains.

An ongoing issue with adaptive curriculum is the concept of catastrophic forgetting. This is especially an issue when your problem has a wide variety of domains it is meant to adapt to.

### Process Reward Modeling

At it's most basic form, QA can take the form of zero-shot reasoning. The limitation of such tuning is that this may not generalize well - the model simply memorizes the QA pairs it's trained on. It's also significantly more difficult to assess how the model arrives at it's conclusions.

For a baseline, we perform a pure fine tune on MedQA, no special weighing.
For adaptive curriculum, we change and schedule what the model trains on epochs based on information gain, weighing and exploration.

Citations

https://arxiv.org/abs/2501.07301

### Formalism of Unix-Style Tool Use

LLMs are fundamentally auto-regressive text-stream engines. Forcing them to generate deeply nested JSON objects with strict brackets and commas frequently breaks their token-prediction flow, leading to syntax hallucinations.

Unix shell operators (|, &&, >) align perfectly with how LLMs "think"—linearly and compositionally. By mapping tools to CSP, you are treating the LLM not as a master orchestrator of disparate APIs, but as a pipeline constructor. The "Rule of Composition" means the LLM only has to predict the next logical transformation of the data stream, drastically reducing the context load.

Tool-use among agents has become industry standard, but long form use still lacks the academic formal rigor. 

If we can combine intuitive data processing with explainable, structured logic to create AI that can both "think" and "reason," aiming for more trustworthy, efficient, and human-like AI systems.

### Data Contracts and Unix Pipes

> ...a contract, at its irreducible core, is a promise with consequences. Nothing more. It says: I will give you this, in this form, by this time. And if I don’t, something happens.

#### Unix Syntax

- `|` (Pipe - Data Flow): Conceptually sequential for the LLM generating the text, but it represents state transformation. cmd_A | cmd_B means the output of A becomes the input of B.
- `&&` (Logical AND - Control Flow): Strictly sequential. cmd_A && cmd_B means "Wait for A to finish completely. If its exit status is 0 (success), then run B." This is vital for state-based reasoning where step B relies on a database mutation that happens in step A.
- `>` or `>>` (Redirection - State Mutation): Writes the stream to a persistent state (a file or memory buffer) rather than passing it to the next tool.
- `<<` Here-Document flexibility - two way flexibility for agents to persist state after causal generation has already begun.
- `||` (OR): `cmd_A || cmd_B` means "if cmd_A fails, try cmd_B". An agent can have multiple tries if given feedback.
- `$()` (Command Substitution): Allows for nested heirarchical reasoning. A perfect interface for agent delegation. 
- `tee`: could be used for verifiable auditing
- `&`: background execution

Once an error occurs, they are routed to `stderr` - the pipe is broken (EOF or SIGPIPE is recieved), the pipeline ends.

The way we structure this is - the agent starts generating. We have a single Python interpreter reading the stream asynchronously, checking for pipes for execution - when it does, it kills the stream, then prompts for the LLM to continue after processing as it wishes.

- `trap`: an LLM can reason about it's own error handling ahead of time as well. In fact, this could be used to schedule it's own Ralph loops - also can be used for an LLM to define safety fallbacks ahead of time. Overall, these are rich commands for areas such as governance and data safety
- `coproc` for asynchronously starting processes

https://lean-lang.org/

### Methodology

#### Data Sources

We collect from a wide availability of benchmarks and available corpus collections. MedQA [@jin2020diseasedoespatienthave] contains 12,723 English samples [^1].

[^1]: The authors acknowledge 

#### CUBE (Common Unified Benchmark Environments)

AI research community is suffering from severe fragmentation when it comes to evaluating AI agents. Every time a new benchmark is created, it requires a lot of custom integration work

CUBE (Common Unified Benchmark Environments), which serves as a universal protocol standard built on top of MCP (Model Context Protocol) and OpenAI's Gym. The goal is to allow developers to wrap benchmarks just once so they can be accessed universally for evaluation, reinforcement learning training, or data generation across any compliant platform.

#### Observability and Data Collection Stack


#### Enterprise Grade Fine tuning with Databricks

- Multi-Objective RL (MORL).

https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-gpt-oss-120b-ddp-fsdp

#### Contributions 

- Using industry tools to standardize Process Reward Modeling
  - Fully reproducible infrastructure (github, docker-compose)
  - Process Reward Model (PRM) as formally-verifable computer code (Lean-Lang)
- Mathematical formalization of long form tool-use and curriculum reward training
- Robust evaluation on a unified evaluation protocol (CUBE)

#### Data sources

- MedQA
- MedMCQA
- PubMedQA
- MMLU (Medical Subset)
- PubMed
- Clinical Guidelines (https://www.guidelinecentral.com/guidelines/)
- MedlinePlus
- UMLS
- ClinicalTrials.gov

Compare:

- Standard fine tuning (LORA, SFT)
- RAG without adapttion
- Compare existing tool calling frameworks (pure MCP)

Metrics:
- Accuracy
- F1
- Calibration (ECE)
- Faithfulness
- Semantic Similarity with SBERT

---


### 3. Data Persistence
Results are automatically saved to your Modal Volume (`medqa-data-volume`) in the `/results` directory as JSON reports, including accuracy, macro-F1, and Expected Calibration Error (ECE).

---

## 📊 Project Structure
- `modal_app.py`: Main entrypoint for remote execution on Modal.
- `medical_mcp_server.py`: MCP server for medical tool integration (PubMed, etc.).
- `cube_medqa.py`: CUBE-compliant task and benchmark wrappers.

## Data Availability Statement