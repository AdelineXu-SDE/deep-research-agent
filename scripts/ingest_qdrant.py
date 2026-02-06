#!/usr/bin/env python
# coding: utf-8

"""
Ingest extracted RAG data into local Qdrant with HYBRID retrieval (dense + sparse).

Expected data layout (from the course notebooks):
  Multi-Agent-Deep-RAG/data/rag-data/
    ├── markdown/<company>/*.md          (page-level chunks separated by '<!-- page break -->')
    ├── tables/<company>/<doc>/*.md      (tables as separate docs)
    └── images_desc/<company>/<doc>/*.md (image descriptions as separate docs)

This script:
- Creates (or reuses) Qdrant collection: financial_docs
- Adds docs with nested metadata payload: {"metadata": {...}}
  so that your rag_tools filter keys like "metadata.company_name" will work.
- Deduplicates by file_hash.
"""

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm
import os
import re
import hashlib
from pathlib import Path
from typing import Optional, Dict, Set, Tuple, List

from dotenv import load_dotenv
load_dotenv()


# -------------------------
# Configuration
# -------------------------

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# If set to "true", recreate collection (WARNING: deletes existing vectors!)
FORCE_RECREATE = os.getenv("FORCE_RECREATE", "false").lower() == "true"

# Resolve the rag-data directory based on your current folder structure:
# AI playground/
#   ├── deep-research-agent/
#   └── Multi-Agent-Deep-RAG/
# .../deep-research-agent/scripts
SCRIPT_DIR = Path(__file__).resolve().parent
# .../deep-research-agent
DEEP_AGENT_DIR = SCRIPT_DIR.parent
AI_PLAYGROUND_DIR = DEEP_AGENT_DIR.parent                   # .../AI playground
DEFAULT_RAG_DATA_DIR = AI_PLAYGROUND_DIR / \
    "Multi-Agent-Deep-RAG" / "data" / "rag-data"
RAG_DATA_DIR = Path(
    os.getenv("RAG_DATA_DIR", str(DEFAULT_RAG_DATA_DIR))).resolve()

MARKDOWN_DIR = RAG_DATA_DIR / "markdown"
TABLES_DIR = RAG_DATA_DIR / "tables"
IMAGES_DESC_DIR = RAG_DATA_DIR / "images_desc"


# -------------------------
# Helpers
# -------------------------

def compute_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()


def extract_page_number(file_path: Path) -> Optional[int]:
    # e.g. page_28.md -> 28
    m = re.search(r"page_(\d+)", file_path.stem.lower())
    return int(m.group(1)) if m else None


def normalize_doc_type(raw: str) -> str:
    s = raw.strip().lower()
    s = s.replace("_", "-")
    if s in {"10k", "10-k"}:
        return "10-k"
    if s in {"10q", "10-q"}:
        return "10-q"
    if s in {"8k", "8-k"}:
        return "8-k"
    return s


def normalize_quarter(raw: str) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip().lower()
    # allow "Q1" / "q1" / "quarter1" (we only normalize q1-q4)
    m = re.search(r"q([1-4])", s)
    return f"q{m.group(1)}" if m else None


def extract_metadata_from_docname(doc_name: str) -> Dict:
    """
    Extract metadata from doc name like:
      - 'amazon 10-k 2023'
      - 'amazon 10-q q3 2024'
    Returns normalized:
      company_name: lowercase
      doc_type: 10-k/10-q/8-k
      fiscal_quarter: q1-q4 or None
      fiscal_year: int or None
    """
    name = doc_name.lower().replace(".pdf", "").replace(".md", "").strip()
    parts = [p for p in name.split() if p]

    meta: Dict = {
        "company_name": None,
        "doc_type": None,
        "fiscal_quarter": None,
        "fiscal_year": None,
    }

    if len(parts) >= 2:
        meta["company_name"] = parts[0].strip().lower()
        meta["doc_type"] = normalize_doc_type(parts[1])

    # year: last 4-digit year in string
    m_year = re.search(r"(20\d{2})", name)
    if m_year:
        meta["fiscal_year"] = int(m_year.group(1))

    # quarter: any q[1-4] token
    m_q = re.search(r"\bq([1-4])\b", name)
    if m_q:
        meta["fiscal_quarter"] = f"q{m_q.group(1)}"

    return meta


def infer_content_type(file_path: Path) -> Tuple[str, str]:
    """
    Returns (content_type, doc_name):
      content_type in {"text","tables","image","unknown"}
      doc_name is used to extract base metadata
    """
    path_str = str(file_path).lower()
    if "markdown" in path_str:
        return "text", file_path.name
    if "tables" in path_str:
        # tables/<company>/<doc>/xxx.md -> doc folder name holds the doc name
        return "tables", file_path.parent.name
    if "images_desc" in path_str:
        return "image", file_path.parent.name
    return "unknown", file_path.name


# -------------------------
# Qdrant interaction
# -------------------------

def init_vector_store() -> QdrantVectorStore:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # create or connect to collection
    vs = QdrantVectorStore.from_documents(
        documents=[],
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=FORCE_RECREATE,
    )
    return vs


def get_processed_hashes(vs: QdrantVectorStore) -> Set[str]:
    """
    Collect already ingested file hashes from Qdrant.
    We stored them in payload['metadata']['file_hash'].
    """
    processed: Set[str] = set()
    offset = None

    while True:
        points, offset = vs.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10_000,
            with_payload=True,
            offset=offset,
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            md = payload.get("metadata") if isinstance(
                payload.get("metadata"), dict) else None
            if md and "file_hash" in md:
                processed.add(md["file_hash"])

        if offset is None:
            break

    return processed


def ingest_one_file(vs: QdrantVectorStore, file_path: Path, processed_hashes: Set[str]) -> None:
    file_hash = compute_file_hash(file_path)
    if file_hash in processed_hashes:
        # already ingested
        return

    content_type, doc_name = infer_content_type(file_path)
    content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return

    base_meta = extract_metadata_from_docname(doc_name)
    # attach common fields
    base_meta.update({
        "content_type": content_type,
        "file_hash": file_hash,
        "source_file": doc_name,
    })

    documents: List[Document] = []

    if content_type == "text":
        # markdown files contain multiple pages separated by '<!-- page break -->'
        pages = content.split("<!-- page break -->")
        for idx, page in enumerate(pages, start=1):
            page_txt = page.strip()
            if not page_txt:
                continue
            md = base_meta.copy()
            md.update({"page": idx})
            # IMPORTANT: nest under "metadata" to match rag_tools.py filter keys
            documents.append(Document(page_content=page_txt,
                             metadata=md))
    else:
        page_num = extract_page_number(file_path)
        md = base_meta.copy()
        md.update({"page": page_num})
        documents.append(Document(page_content=content,
                         metadata=md))

    if documents:
        vs.add_documents(documents)
        processed_hashes.add(file_hash)


def main():
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME} (force_recreate={FORCE_RECREATE})")
    print(f"RAG_DATA_DIR: {RAG_DATA_DIR}")

    if not RAG_DATA_DIR.exists():
        raise RuntimeError(f"RAG_DATA_DIR not found: {RAG_DATA_DIR}")

    vs = init_vector_store()

    processed_hashes = get_processed_hashes(vs)
    print(f"Already ingested file hashes: {len(processed_hashes)}")

    all_md_files = []
    if RAG_DATA_DIR.exists():
        all_md_files = list(RAG_DATA_DIR.rglob("*.md"))

    if not all_md_files:
        raise RuntimeError(f"No .md files found under: {RAG_DATA_DIR}")

    print(f"Found .md files: {len(all_md_files)}")
    for fp in tqdm(all_md_files, desc="Ingesting"):
        ingest_one_file(vs, fp, processed_hashes)

    # quick verify
    info = vs.client.get_collection(COLLECTION_NAME)
    print("Ingestion complete. Collection info:")
    print(info)


if __name__ == "__main__":
    main()
