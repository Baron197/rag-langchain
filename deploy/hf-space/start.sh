#!/usr/bin/env sh
# Boot both processes in one container for a Hugging Face Space:
#   1) run the FastAPI service on 127.0.0.1:8000 (internal only)
#   2) run the Streamlit UI on 0.0.0.0:7860 (the Space's public port)
# The in-memory index is built by the pipeline on first use, so the /health probe
# below doubles as a warm-up (no separate ingest step needed for the memory backend).
set -e

echo ">> Starting API on 127.0.0.1:8000..."
uvicorn src.rag_lc.api:app --host 127.0.0.1 --port 8000 &

echo ">> Waiting for the API to become healthy (this also builds the in-memory index)..."
i=0
while [ "$i" -lt 90 ]; do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" >/dev/null 2>&1; then
    echo ">> API is up."
    break
  fi
  i=$((i + 1))
  sleep 1
done

echo ">> Starting Streamlit UI on 0.0.0.0:7860..."
exec streamlit run ui/streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
