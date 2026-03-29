# 🚀 Local RAG Quickstart Guide

This directory contains everything you need to run high-performance, private LLM benchmarks on your machine.

## 📋 Step-by-Step Instructions

### 0. Install Ollama (Required)
The framework runs models locally on your Mac. You **must** have Ollama installed:
1. Download from **[ollama.com](https://ollama.com)**.
2. Install the application and **open it** (you should see the 🦙 icon in your menu bar).

### 1. Initialize the Environment
Open your terminal in the project root and run:
```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Models (Ollama)
Make sure [Ollama](https://ollama.com) is running, then use our setup script to pull the latest SOTA models:
```bash
sh quickstart/setup.sh
```
*This will pull: Phi-4, Llama 3.3, Mistral Nemo, and Gemma 2.*

### 3. Run Diagnostic Tests
Verify that all models are responsive and correctly indexed by your machine:
```bash
python quickstart/scripts/test_models.py
```

### 4. Execute Multi-Model Benchmarking
Once diagnostics pass, run the full 5-dataset benchmark sweep:
```bash
python run_eval.py --local --models phi4 llama3.3 mistral-nemo gemma2
```

### 5. Launch the Leaderboard
Visualize the results and compare model "Robustness Scores":
```bash
streamlit run dashboard.py
```

---
**Note:** All data is processed locally. No credit cards, API keys, or internet required after the models are downloaded.
