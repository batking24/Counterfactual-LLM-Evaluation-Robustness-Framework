#!/bin/bash
# One-click script to run the stable dashboard
# This script handles port conflicts and Python 3.14 compatibility

PORT=8504
echo "Preparing RAG Multi-Model Evaluation on port $PORT..."

# 1. Kill any existing process on the port to prevent "Port not available"
PID=$(lsof -t -i:$PORT)
if [ ! -z "$PID" ]; then
    echo "Cleaning up existing dashboard session (PID: $PID)..."
    kill -9 $PID
fi

# 2. Force pure-python protobuf to avoid Python 3.14 C-API crashes
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 3. Launch from the stable Python 3.13 environment
echo "Launching dashboard..."
.venv_stable/bin/streamlit run dashboard.py --server.port $PORT
