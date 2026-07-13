# Deploy the live demo to Hugging Face Spaces (Docker)

> ⚠️ **As of 2026, Hugging Face Docker (and Gradio) Spaces require a paid PRO plan**
> — only *Static* Spaces are free. For a **free, no-credit-card** demo, use
> [`../streamlit-cloud/DEPLOY_GUIDE.md`](../streamlit-cloud/DEPLOY_GUIDE.md) instead.
> The files in this folder still work if you have HF PRO.

Runs the whole LangChain app — FastAPI service **and** Streamlit UI — in one
Hugging Face Docker Space. Defaults to the keyless `fake` providers + in-memory
vector store, so it needs no API key. ~10 minutes, mostly the build.

The Space needs only **two files** (`Dockerfile` + `README.md`); the app code is
cloned from GitHub at build time.

---

## 1. Create a Hugging Face account
Sign up at <https://huggingface.co/join> — free, **no credit card**.

## 2. Create a new Space
Go to <https://huggingface.co/new-space> and set:
- **Owner:** your username
- **Space name:** `rag-langchain`
- **License:** MIT
- **SDK:** **Docker** → **Blank** template
- **Hardware:** *CPU basic* (Docker Spaces now require a PRO plan — no longer free)
- **Visibility:** Public

Click **Create Space** (this creates an empty git repo for the Space).

## 3. Add the two files
```bash
git clone https://huggingface.co/spaces/<your-username>/rag-langchain
cd rag-langchain

# from your local clone of the GitHub repo:
cp "D:/Portfolios/rag-langchain/deploy/hf-space/Dockerfile"       ./Dockerfile
cp "D:/Portfolios/rag-langchain/deploy/hf-space/space_README.md"  ./README.md
```

> The `Dockerfile` clones the app from GitHub (`Baron197/rag-langchain`) at build
> time, so you do **not** copy the source into the Space. `README.md` carries the
> Space's config (the YAML frontmatter at the top).

## 4. Push (authenticate with an access token)
Create a **write** token at <https://huggingface.co/settings/tokens>, then:

```bash
git add Dockerfile README.md
git commit -m "Deploy RAG Knowledge Assistant — LangChain (keyless demo)"
git push
# Username: <your-username>   Password: <paste the write token>
```

## 5. Watch it build
Open your Space page; the **Building** logs stream live (first build ~3–5 min).
When the badge turns **Running**, the app is live at:

```
https://<your-username>-rag-langchain.hf.space
```

## 6. Link it from your GitHub README
Add a demo line near the top of the main README (send me the URL and I'll add it):

```markdown
**▶ [Try the live demo](https://huggingface.co/spaces/<your-username>/rag-langchain)** — keyless mode, runs in your browser.
```

---

## Optional: switch this Space to real answers (`hf` mode)
Free, still no API key, slower on the CPU tier (~5–15 s per answer, 0.5B model).

1. In the Space's `Dockerfile`, **uncomment** the `hf mode` `RUN` line (it installs
   `torch` + `langchain-huggingface` + `sentence-transformers` + `transformers`).
2. In **Settings → Variables and secrets**, add `LLM_PROVIDER = hf` and
   `EMBEDDING_PROVIDER = hf`.
3. **Restart** the Space (Settings → Factory reboot). The first answer downloads
   the model (~1–2 min); subsequent answers are cached-fast.

## Notes
- **Backend:** the demo uses `VECTOR_BACKEND=memory` (LangChain's
  `InMemoryVectorStore`) — zero external services; the index is built in-process on
  first request. The pgvector backend is for the Docker/VM deploy, not this Space.
- **Sleeping:** a free Space sleeps after ~48 h idle and wakes in a few seconds.
- **No password:** the keyless demo runs open (nothing to abuse — no key, no cost).
  Set `APP_PASSWORD` as a Variable if you enable `hf` mode and want to limit load.
