# medqa-cube-adaptive-curriculum-tool-calling

##

| Evaluation | Model | Score |
| --- | --- | --- |
| [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) | |
| [HealthBench](https://github.com/openai/simple-evals) | | |

## Requirements

### MedAgentBench

Ensure you pull the patient data needed first:

```bash
# pull the image
docker pull jyxsu6/medagentbench:latest

# ensure it's running on port 8080
docker run -d --name medagentbench_server -p 8080:8080 medagentbench
```
