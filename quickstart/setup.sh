#!/bin/bash
# --- LLM Evaluation Framework: Local Setup Guide ---
# This script prepares your environment for local benchmarking with Ollama.

echo "🚀 Starting Local RAG Setup..."

# 1. Environment Activation
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python3.14 -m venv .venv' first."
    exit 1
fi

source .venv/bin/activate
echo "✅ Virtual environment activated."

# 3. Model Management
# These are the 4 optimized models we use for benchmarking
MODELS=("phi3.5" "llama3.1" "mistral-nemo" "gemma2")

echo "🤖 Syncing Ollama models..."
for model in "${MODELS[@]}"; do
    echo "  - Ensuring $model is present..."
    ollama pull "$model"
done

# 4. Optional Cleanup (Removes heavy 70B models if they were partially downloaded)
echo "🧹 Cleaning up unnecessary heavy models (optional)..."
ollama rm llama3.3 phi4 2>/dev/null || true
echo "✅ Cleanup done."

echo "✨ Setup complete! You are ready to run comparisons."
echo "▶️ Run diagnostics: python quickstart/scripts/test_models.py"
