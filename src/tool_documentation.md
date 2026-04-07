# Medical API Documentation for Agentic Tool-Augmented LLMs

This document summarizes the technical requirements and documentation for integrating the proposed medical retrieval tools.

## 1. PubMed (NCBI E-utilities)
*   **Documentation:** [NCBI E-utilities Overview](https://www.ncbi.nlm.nih.gov/books/NBK25500/)
*   **Key Operations:**
    *   `ESearch`: Map queries to PMIDs.
    *   `EFetch`: Retrieve abstracts/metadata for PMIDs.
*   **Best Practices:** Use API keys for increased rate limits (10 req/s).

## 2. MedlinePlus
*   **Web Service:** [MedlinePlus Web Service](https://medlineplus.gov/web_services.html)
*   **Schema:** Returns results in XML (requires parsing for LLM input).
*   **Usage:** Best for grounding answers in consumer-friendly language.

## 3. UMLS (UTS REST API)
*   **Documentation:** [UMLS REST API](https://documentation.uts.nlm.nih.gov/rest/home.html)
*   **Key Endpoints:**
    *   `/search`: Map strings to CUIs.
    *   `/content`: CUI to atoms/definitions.
*   **Auth:** Requires UTS API Key (Header: `X-API-KEY`).

## 4. ClinicalTrials.gov (v2.0)
*   **Documentation:** [ClinicalTrials.gov Data API](https://clinicaltrials.gov/data-api/about-api)
*   **Format:** RESTful JSON.
*   **Usage:** Identifying active trials or recruitment status.

## 5. MMLU (Medical Subsets)
*   **Source:** [Hugging Face `cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)
*   **Subsets:** `anatomy`, `clinical_knowledge`, `college_medicine`, `college_biology`, `medical_genetics`, `professional_medicine`, `nutrition`.
*   **Format:** Standardized MCQ.
