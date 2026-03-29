# MedQA Research Suite

This repository contains an extensible evaluation framework for benchmarking Large Language Models on medical question-answering datasets (MedQA, MedMCQA, PubMedQA, MMLU).

---

To run the original research baseline using the **Qwen 3.5 9B** model on OpenRouter, use the following `modal run` command. 

> [!IMPORTANT]
> You must have an OpenRouter API key configured in your Modal secrets as `openrouter-secret`.

### Basic Zero-Shot Run
```bash
modal run modal_app.py --model qwen/qwen3.5-9b --strategy zero_shot --samples 100
```

### Run All Prompting Strategies
To evaluate the model across Zero-Shot, Few-Shot, CoT, and Tree-of-Thoughts:
```bash
modal run modal_app.py --model qwen/qwen3.5-9b --strategy all --samples 100
```

---

## 🛠️ Advanced Usage

### 1. Multiple Models & Strategies
You can evaluate multiple models and multiple strategies in a single command by passing comma-separated lists:
```bash
modal run modal_app.py \
  --model "qwen/qwen3.5-9b,google/gemini-flash-1.5" \
  --strategy "zero_shot,cot" \
  --samples 20
```

### 2. Supported Prompting Strategies
| Strategy | Description |
| :--- | :--- |
| `zero_shot` | Direct question answering without examples. |
| `few_shot` | Includes 5 clinical examples from the training set. |
| `cot` | Chain-of-Thought (step-by-step reasoning). |
| `cot_few_shot` | Combines Few-Shot examples with CoT reasoning. |
| `tree_of_thoughts` | Multi-path differential diagnosis approach. |

### 3. Data Persistence
Results are automatically saved to your Modal Volume (`medqa-data-volume`) in the `/results` directory as JSON reports, including accuracy, macro-F1, and Expected Calibration Error (ECE).

---

## 📊 Project Structure
- `modal_app.py`: Main entrypoint for remote execution on Modal.
- `medical_mcp_server.py`: MCP server for medical tool integration (PubMed, etc.).
- `cube_medqa.py`: CUBE-compliant task and benchmark wrappers.
