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

# install streamlit for exploring the 
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

HealthBench []