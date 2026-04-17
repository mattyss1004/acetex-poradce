#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Acetex Chatbot — startup script
# Usage: ./start.sh
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ☀️  Acetex Poradce — spouštím..."
echo ""

# 1. Start Ollama if not already running
if ! curl -s http://localhost:11434/ > /dev/null 2>&1; then
    echo "  → Spouštím Ollama server..."
    ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    # Wait up to 10 seconds for it to be ready
    for i in $(seq 1 10); do
        sleep 1
        if curl -s http://localhost:11434/ > /dev/null 2>&1; then
            echo "  ✓ Ollama server běží (PID $OLLAMA_PID)"
            break
        fi
    done
else
    echo "  ✓ Ollama server již běží"
fi

# 2. Verify the model is available
if ! ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
    echo "  → Stahuji model llama3.2:3b (cca 2 GB, chvíli to potrvá)..."
    ollama pull llama3.2:3b
fi
echo "  ✓ Model llama3.2:3b připraven"

# 3. Launch Streamlit
echo ""
echo "  → Spouštím chatbot UI..."
echo "  ✓ Otevřete prohlížeč na: http://localhost:8501"
echo ""
echo "  Pro ukončení stiskněte Ctrl+C"
echo ""

streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#0066cc"
