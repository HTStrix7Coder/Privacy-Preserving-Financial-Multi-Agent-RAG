#!/bin/bash
set -e

echo "========================================================"
echo " Starting GDPR-Sovereign Financial Compliance Services  "
echo "========================================================"

# 1. Start Ollama service in the background
echo "[1/4] Starting Ollama daemon..."
ollama serve &

# 2. Wait for Ollama service to become responsive
echo "[2/4] Waiting for Ollama to be ready on localhost:11434..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done
echo "✅ Ollama daemon is active and responding."

# 3. Register fine-tuned GGUF model in Ollama
echo "[3/4] Registering fine-tuned model (gdpr-slm-qwen3.5)..."
if [ -f "/app/gguf_q8_0_gguf_qwen3.5/Modelfile" ]; then
    ollama create gdpr-slm-qwen3.5 -f /app/gguf_q8_0_gguf_qwen3.5/Modelfile
    echo "✅ gdpr-slm-qwen3.5 registered successfully."
else
    echo "⚠️ Modelfile not found in /app/gguf_q8_0_gguf_qwen3.5/Modelfile"
fi

# 4. Pull compliance reasoning model
echo "[4/4] Checking compliance reasoning model (qwen2.5:7b)..."
ollama pull qwen2.5:7b || echo "⚠️ Warning: Failed to pull qwen2.5:7b, check internet connectivity."

# 5. Launch Streamlit on Hugging Face Spaces port 7860
echo "========================================================"
echo " Launching Streamlit Dashboard on port 7860            "
echo "========================================================"
exec streamlit run app.py --server.port 7860 --server.address 0.0.0.0
