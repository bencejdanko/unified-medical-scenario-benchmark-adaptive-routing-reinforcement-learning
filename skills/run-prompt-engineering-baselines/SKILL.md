To run the original research baseline using the **Qwen 3.5 9B** model on OpenRouter, use the following `modal run` command. 

### Basic Zero-Shot Run
```bash
modal run src/modal_app.py --model qwen/qwen3.5-9b --strategy zero_shot --samples 100
```

### Run All Prompting Strategies
To evaluate the model across Zero-Shot, Few-Shot, CoT, and Tree-of-Thoughts:
```bash
modal run src/modal_app.py --model qwen/qwen3.5-9b --strategy all --samples 100
```

### Supported Prompting Strategies
| Strategy | Description |
| :--- | :--- |
| `zero_shot` | Direct question answering without examples. |
| `few_shot` | Includes 5 clinical examples from the training set. |
| `cot` | Chain-of-Thought (step-by-step reasoning). |
| `cot_few_shot` | Combines Few-Shot examples with CoT reasoning. |
| `tree_of_thoughts` | Multi-path differential diagnosis approach. |

### Data Persistence
Results are automatically saved to your Modal Volume (`medqa-data-volume`) in the `/results` directory as JSON reports, including:

- Accuracy
- Macro-F1
- Expected Calibration Error (ECE).
