"""
RAG Corpus Ingestion Script
Chunks, embeds, and stores German financial law documents into a ChromaDB vector store.

Usage:
    python GDPR_SLM/rag/ingest_corpus.py              # Ingest Law/ folder (default)
    python GDPR_SLM/rag/ingest_corpus.py --all        # Ingest Law/ + Annual_reports/ + Base_prospectuses/
    python GDPR_SLM/rag/ingest_corpus.py --reset      # Delete existing DB and re-ingest

Collections created:
    - law_corpus          (KWG, MiFID II, WpHG, etc.)
    - annual_reports      (issuer financial reports)
    - base_prospectuses   (base prospectus documents)
"""

import os
import sys
import argparse
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CORPUS_DIR = PROJECT_DIR / "dataset_main" / "safe_corpus" / "txt"
CHROMA_DB_DIR = str(SCRIPT_DIR / "chroma_db")

# Embedding model — multilingual, optimized for German
# Downloads automatically on first run (~500MB, cached locally)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Chunking config (in characters)
CHUNK_SIZE = 1500         # ~250-300 tokens per chunk
CHUNK_OVERLAP = 200       # overlap to preserve context across chunks

# ============================================================================
# HELPERS
# ============================================================================

def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("(Downloads ~500MB on first run, cached after that)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Embedding model loaded\n")
    return model


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_txt_files(folder: Path) -> list[Path]:
    """Recursively collect all .txt files in a folder."""
    return sorted(folder.rglob("*.txt"))


def ingest_folder(
    folder: Path,
    collection_name: str,
    client: chromadb.ClientAPI,
    model: SentenceTransformer,
    reset: bool = False,
):
    """Ingest all .txt files from a folder into a ChromaDB collection."""

    txt_files = get_txt_files(folder)
    if not txt_files:
        print(f"⚠️  No .txt files found in {folder}")
        return

    # Create or get collection
    if reset and collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
        print(f"   Deleted existing collection: {collection_name}")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    print(f"\n📚 Ingesting '{collection_name}': {len(txt_files)} files | {existing_count} existing chunks")

    doc_ids = []
    doc_texts = []
    doc_metas = []
    batch_size = 50   # embed in batches to manage memory

    for filepath in tqdm(txt_files, desc=f"  Loading {collection_name}"):
        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception as e:
            print(f"\n   ⚠️  Could not read {filepath.name}: {e}")
            continue

        if len(text) < 100:
            continue  # skip near-empty files

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{filepath.stem}__chunk_{i}"

            # Skip if already in DB
            if existing_count > 0:
                existing = collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue

            doc_ids.append(doc_id)
            doc_texts.append(chunk)
            doc_metas.append({
                "source": filepath.name,
                "chunk": i,
                "total_chunks": len(chunks),
            })

            # Flush batch
            if len(doc_ids) >= batch_size:
                embeddings = model.encode(doc_texts, show_progress_bar=False, normalize_embeddings=True).tolist()
                collection.add(ids=doc_ids, embeddings=embeddings, documents=doc_texts, metadatas=doc_metas)
                doc_ids, doc_texts, doc_metas = [], [], []

    # Final batch
    if doc_ids:
        embeddings = model.encode(doc_texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        collection.add(ids=doc_ids, embeddings=embeddings, documents=doc_texts, metadatas=doc_metas)

    final_count = collection.count()
    print(f"   ✅ '{collection_name}': {final_count} total chunks in DB\n")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest German financial corpus into ChromaDB")
    parser.add_argument("--all", action="store_true", help="Ingest all corpus folders (not just Law)")
    parser.add_argument("--reset", action="store_true", help="Delete and re-ingest existing collections")
    args = parser.parse_args()

    # Init ChromaDB persistent client
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    print(f"ChromaDB storage: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Load embedding model
    model = load_embedding_model()

    # Always ingest Law/
    ingest_folder(
        folder=CORPUS_DIR / "Law",
        collection_name="law_corpus",
        client=client,
        model=model,
        reset=args.reset,
    )

    if args.all:
        ingest_folder(
            folder=CORPUS_DIR / "Annual_reports",
            collection_name="annual_reports",
            client=client,
            model=model,
            reset=args.reset,
        )
        ingest_folder(
            folder=CORPUS_DIR / "Base_prospectuses",
            collection_name="base_prospectuses",
            client=client,
            model=model,
            reset=args.reset,
        )

    # Summary
    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    for col in client.list_collections():
        print(f"   {col.name}: {client.get_collection(col.name).count()} chunks")
    print(f"\nDB path: {CHROMA_DB_DIR}")
    print("Ready for RAG queries!")
