"""
Quick RAG retrieval test — queries the ChromaDB law_corpus.
Run: python GDPR_SLM/rag/test_retrieval.py
"""
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_DB_DIR = str(Path(__file__).parent / "chroma_db")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Test queries — things your compliance agent will ask
TEST_QUERIES = [
    "Eigenkapitalanforderungen für Schuldverschreibungen KWG",   # Capital requirements for bonds
    "Prospektpflicht Emittent Wertpapierprospektgesetz",         # Prospectus requirements
    "ISIN Zulassung Börsenhandel Finanzinstrumente",             # ISIN exchange listing rules
    "Nullkupon Schuldverschreibungen Zinsen Steuer",             # Zero coupon tax treatment
    "Nachrangige Verbindlichkeiten Insolvenz Haftung",           # Subordinated bonds insolvency
]

N_RESULTS = 3  # top-k results per query

def main():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection("law_corpus")
    print(f"✅ Connected — {collection.count()} chunks in law_corpus\n")
    print("=" * 70)

    for query in TEST_QUERIES:
        print(f"\n🔍 Query: {query}")
        print("-" * 70)
        
        # Embed query with the "query:" prefix (required for E5 models)
        query_embedding = model.encode(f"query: {query}", normalize_embeddings=True).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=N_RESULTS,
            include=["documents", "metadatas", "distances"],
        )

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            similarity = 1 - dist  # cosine similarity (0-1)
            source = meta.get("source", "unknown")
            print(f"\n  [{i+1}] Source: {source}  |  Similarity: {similarity:.3f}")
            print(f"  Text: {doc[:300]}...")

    print("\n" + "=" * 70)
    print("✅ RAG retrieval test complete!")

if __name__ == "__main__":
    main()
