# Medical QA Data & Tool Manifest

This manifest standardizes the sourcing and integration of medical datasets and retrieval tools for the curriculum learning and tool-calling framework.

## 1. Dataset Loaders (Hugging Face)

All datasets should be ingested into a standardized schema for training and evaluation.

| Dataset | Hugging Face Path | Subsets / Categories | Task Type |
| :--- | :--- | :--- | :--- |
| **MedQA (USMLE)** | `openlifescienceai/medqa` | N/A | Multiple Choice |
| **MedMCQA** | `openlifescienceai/medmcqa` | N/A | Multiple Choice |
| **PubMedQA** | `qiaojin/pubmed_qa` | `pqa_labeled` | Yes/No/Maybe + Reasoning |
| **MMLU (Medical)**| `cais/mmlu` | `anatomy`, `clinical_knowledge`, `college_medicine`, `college_biology`, `medical_genetics`, `professional_medicine`, `nutrition` | Multiple Choice |

### Standardized Ingestion Schema
```json
{
  "id": "string",
  "question": "string",
  "options": ["string"], 
  "answer_idx": "integer",
  "answer_label": "string",
  "context": "string (optional)",
  "source": "string (e.g., 'medmcqa')",
  "metadata": {
    "subject": "string",
    "difficulty": "float (optional)",
    "task_type": "string"
  }
}
```

### Local Data Storage
All datasets are stored in `medqa/data/` as JSON files:
- `openlifescienceai_medqa_*.json` (USMLE)
- `openlifescienceai_medmcqa_*.json`
- `qiaojin_pubmed_qa_pqa_labeled_*.json`
- `cais_mmlu_*.json` (Specific medical subsets)

---

## 2. Tool Source Indexing (API & Vector Schemas)

These tools are designed for the "agentic" component to augment reasoning with external evidence.

### A. PubMed (NCBI E-utilities)
*   **Purpose:** Retrieve latest biomedical research abstracts.
*   **API Base:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
*   **Tool Schema:**
```json
{
  "name": "pubmed_search",
  "description": "Searches PubMed for relevant research papers and returns abstracts.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "Medical search query" },
      "max_results": { "type": "integer", "default": 5 }
    },
    "required": ["query"]
  }
}
```

### B. MedlinePlus
*   **Purpose:** Patient-friendly consolidated health topics.
*   **API Base:** `https://wsearch.nlm.nih.gov/ws/query`
*   **Tool Schema:**
```json
{
  "name": "medlineplus_lookup",
  "description": "Retrieves consumer health information for a given medical topic or symptom.",
  "parameters": {
    "type": "object",
    "properties": {
      "term": { "type": "string", "description": "Common name for condition or drug" }
    },
    "required": ["term"]
  }
}
```

### C. UMLS (Unified Medical Language System)
*   **Purpose:** Entity normalization and concept mapping (CUIs).
*   **API Base:** `https://uts-ws.nlm.nih.gov/rest/`
*   **Tool Schema:**
```json
{
  "name": "umls_entity_linker",
  "description": "Maps clinical terms to Concept Unique Identifiers (CUIs) and retrieves definitions.",
  "parameters": {
    "type": "object",
    "properties": {
      "string": { "type": "string", "description": "Medical term to link" }
    },
    "required": ["string"]
  }
}
```

### D. ClinicalTrials.gov (v2.0)
*   **Purpose:** Search for active clinical trials and recruitment status.
*   **API Base:** `https://clinicaltrials.gov/api/v2/`
*   **Tool Schema:**
```json
{
  "name": "clinical_trials_search",
  "description": "Searches for clinical trials based on condition, location, or status.",
  "parameters": {
    "type": "object",
    "properties": {
      "condition": { "type": "string", "description": "Medical condition (e.g., 'Diabetes')" },
      "status": { "type": "string", "enum": ["RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING"], "default": "RECRUITING" },
      "max_results": { "type": "integer", "default": 5 }
    },
    "required": ["condition"]
  }
}
```

---

## 3. Vector Store Schema (Proposed)

For RAG components using local indexes of clinical guidelines.

*   **Collection Name:** `clinical_guidelines`
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (or medical specific like `SapBERT`)
*   **Metadata Fields:**
    - `source_url`: Link to guideline
    - `specialty`: e.g., Cardiology, Oncology
    - `last_updated`: Date
    - `evidence_level`: (A, B, C, etc.)

---

## 4. Implementation Notes

1.  **Authentication:** UMLS requires an API Key via UTS. PubMed/NCBI suggests API keys for higher rate limits (10 req/sec).
2.  **Tool Pollution Strategy:** The agent will be tested with "distractor" tools from common domains (e.g., weather, news) to evaluate its ability to select the correct medical source.
3.  **Curriculum Link:** Uncertainty scores from the LLM will trigger tool calls. If internal confidence is low, the agent initiates `pubmed_search` or `umls_entity_linker`.
