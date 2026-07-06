"""Load the document corpus and split it into chunks with LangChain.

Uses `RecursiveCharacterTextSplitter` (character-based) to turn each source file
into overlapping `Document` chunks carrying `source` + `chunk_index` metadata.
Supported inputs: `.md` / `.txt` (read directly), `.html` (tags stripped) and
`.pdf` (text extracted per page).
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings, get_settings

SUFFIXES = {".md", ".txt", ".html", ".htm", ".pdf"}


def _read_file(path: Path) -> str:
    """Extract plain text from one file based on its extension (heavy deps lazy)."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def load_and_split(settings: Settings | None = None) -> list[Document]:
    """Read every supported file under `docs_dir` and split into chunk Documents."""
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    out: list[Document] = []
    for path in sorted(Path(settings.docs_dir).rglob("*")):
        if not (path.is_file() and path.suffix.lower() in SUFFIXES):
            continue
        text = _read_file(path).strip()
        if not text:
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            out.append(
                Document(page_content=chunk, metadata={"source": path.name, "chunk_index": i})
            )
    return out


def ingest(settings: Settings | None = None, *, reset: bool = True) -> int:
    """Load + split the corpus and, on the pgvector backend, embed and persist it.

    The in-memory backend has no persistence (the pipeline builds it in-process), so
    this just reports the chunk count there. On pgvector it (re)writes the collection.
    Returns the number of chunks. Mirrors the from-scratch `ingest(reset=...)`.
    """
    settings = settings or get_settings()
    docs = load_and_split(settings)
    if settings.vector_backend == "pgvector":
        from .components import build_vectorstore, get_embeddings

        build_vectorstore(settings, get_embeddings(settings), docs, reset=reset)
    return len(docs)


def main() -> None:
    """CLI: `python -m src.rag_lc.ingest --reset` (persists to pgvector when configured)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest the corpus (persists embeddings to pgvector when VECTOR_BACKEND=pgvector)."
    )
    parser.add_argument("--reset", action="store_true", help="Rewrite the vector store from scratch.")
    parser.parse_args()  # accepted for parity; ingest always rewrites the store
    settings = get_settings()
    n = ingest(settings, reset=True)
    print(f"Ingested {n} chunks (vector_backend={settings.vector_backend}).")


if __name__ == "__main__":
    main()
