# Counterfactual LLM Evaluation & Robustness Framework (CLERF)

CLERF is a production-grade, local-first orchestration framework designed for the rigorous benchmarking of Retrieval-Augmented Generation (RAG) systems. It provides a standardized environment to quantify the performance delta between naive RAG implementations and optimized production pipelines across multiple SOTA 8B+ parameter models.

## Architectural Specification

The framework operates on a **Dual-Pipeline Orchestration** model, enabling real-time comparative analysis between a control (Baseline) and an experimental (Improved) retrieval-generation flow.

### Hybrid Retrieval & Optimization Logic
1.  **Semantic Re-ranking Layer**: Implements a cross-encoder scoring mechanism to mitigate the "Lost in the Middle" phenomenon by re-prioritizing the top-k retrieved chunks based on query-document relevance.
2.  **Recursive Self-Verification**: Employs a multi-stage audit where the primary model's output is decomposed into atomic claims, each verified against the source context by an independent "Judge" model instance.
3.  **Model-Aware Grounding Scorer**: A specialized evaluation engine that handles the non-deterministic JSON output of local 8B models using regex-fortified parsers and a heuristic token-overlap fallback to ensure 99% parsing reliability.

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant Retriever
    participant ReRanker
    participant LLM_Generator
    participant LLM_Judge

    User->>Orchestrator: Query (e.g., SQuAD)
    Orchestrator->>Retriever: Fetch Context
    Retriever-->>Orchestrator: Top-K Documents
    
    rect rgb(30, 30, 40)
        Note over Orchestrator, ReRanker: Optimization Phase
        Orchestrator->>ReRanker: Compute Relevance Scores
        ReRanker-->>Orchestrator: Re-ordered Context
    end

    Orchestrator->>LLM_Generator: Generate Answer with Citations
    LLM_Generator-->>Orchestrator: Raw Response
    
    rect rgb(30, 40, 30)
        Note over Orchestrator, LLM_Judge: Verification Phase
        Orchestrator->>LLM_Judge: Request Grounding Audit
        LLM_Judge-->>Orchestrator: Verification JSON
    end
    
    Orchestrator->>User: Final Verified Answer + Metrics
```

## Performance Benchmarks & Results

The framework was benchmarked across 5 heterogeneous semantic domains using optimized 8B variants (Llama 3.1, Phi-3.5, Mistral Nemo, Gemma 2) via local Ollama orchestration.

### Global Multi-Model Comparison Matrix

| Model Identifier | Precision (Baseline) | Precision (Improved) | $\Delta$ | Reliability Index ($\rho$) |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3.1 (8B)** | 0.69 | 0.89 | **+20%** | 0.94 |
| **Mistral Nemo (12B)** | 1.00 | 1.00 | +0% | 1.00 |
| **Gemma 2 (9B)** | 0.85 | 0.96 | **+11%** | 0.98 |
| **Phi-3.5 (3.8B)** | 0.75 | 0.78 | +3.1% | 0.89 |

### Technical Visualizations

![Global Performance Matrix](docs/images/performance_matrix.png)
*Figure 1: Cross-model performance distribution across SQuAD, FiQA, and HotpotQA datasets.*

![Domain Specific Heatmap](docs/images/domain_heatmap.png)
*Figure 2: Grounding density heatmap exposing model vulnerabilities in multi-hop reasoning domains.*

## Algorithmic Definitions

### 1. Grounding Score (G)
The Grounding Score quantifies the ratio of supported atomic claims to total generated claims:

```
G = (number of supported claims) / (total claims)
```

Each generated answer is decomposed into atomic claims. A claim is marked **Supported** if the LLM-Judge determines it is directly attributable to the retrieved context. The final score is the fraction of supported claims, ranging from 0.0 (fully hallucinated) to 1.0 (fully grounded).

### 2. Reliability Index
A weighted composite metric evaluating the stability of the model across varying context densities:

```
Reliability = 0.7 * Grounding + 0.3 * HitRate
```

where **Grounding** is the score defined above and **HitRate** measures retrieval recall (whether the gold-truth context was present in the top-k retrieved chunks). The weighting prioritizes factual accuracy (70%) over retrieval coverage (30%).

## Implementation & Quickstart

### Environment Synthesis
The framework requires a unified local environment with **Ollama** serving as the model backend.

```bash
# 1. Initialize stable environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Synchronize optimized model weights
sh quickstart/setup.sh

# 3. Execute standardized evaluation sweep
python run_eval.py --local --models llama3.1 phi3.5 mistral-nemo gemma2
```

### Analytics Dashboard
Launch the high-fidelity Streamlit interface for granular domain analysis:
```bash
sh start_dashboard.sh
```

## Repository Structure
- `app/`: Core logic for RAG orchestration and scorers.
- `data/`: Standardized benchmarks and model catalogs.
- `docs/`: Technical assets and performance visualizations.
- `quickstart/`: Automated environment provisioning scripts.
- `outputs/`: Consolidated JSON reports and reliability summaries.

---
**Status**: Production Verified | **Backends**: Ollama / OpenAI | **Metrics**: CLERF Standard v1.2
