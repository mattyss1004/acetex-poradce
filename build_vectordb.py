"""
build_vectordb.py
-----------------
Embeds all chunks from data/chunks.jsonl and stores them in a
local ChromaDB vector database at data/chroma_db/.

Uses sentence-transformers model:
  paraphrase-multilingual-MiniLM-L12-v2
  - 50M parameters, runs fast on CPU
  - Trained on 50+ languages including Czech
  - 384-dimensional embeddings

After building, runs 3 test queries to verify retrieval quality.

Run with:
    python3.11 build_vectordb.py
"""

import json
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "data" / "chunks.jsonl"
CHROMA_DIR  = BASE_DIR / "data" / "chroma_db"

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION    = "acetex_knowledge"
BATCH_SIZE    = 64   # chunks per embedding batch

# ── Test queries (Czech) ──────────────────────────────────────────────────────
TEST_QUERIES = [
    "Jak spustit systém Wattsonic krok za krokem?",
    "Jaká je výše dotace na fotovoltaiku v programu NZÚ Light?",
    "Jaká je záruka na baterie GoodWe Lynx Home?",
]


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_database(chunks: list[dict], model: SentenceTransformer) -> chromadb.Collection:
    print(f"\nInitialising ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Drop and recreate collection for a clean build
    try:
        client.delete_collection(COLLECTION)
        print("  Dropped existing collection for fresh build.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts     = [c["text"]   for c in chunks]
    ids       = [c["id"]     for c in chunks]
    metadatas = [
        {
            "source":    c["source"],
            "title":     c["title"][:200],   # ChromaDB metadata values must be strings
            "type":      c["type"],
            "chunk_idx": str(c["chunk_idx"]),
        }
        for c in chunks
    ]

    total   = len(chunks)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Embedding {total} chunks in {batches} batches of {BATCH_SIZE}...")
    t0 = time.time()

    for i in range(batches):
        start = i * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)

        batch_texts = texts[start:end]
        batch_ids   = ids[start:end]
        batch_meta  = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

        elapsed = time.time() - t0
        rate    = (end) / elapsed
        eta     = (total - end) / rate if rate > 0 else 0
        print(f"  Batch {i+1:>3}/{batches}  [{end:>4}/{total}]  "
              f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    total_time = time.time() - t0
    print(f"\n  ✓ Done in {total_time:.1f}s  ({total/total_time:.1f} chunks/sec)")
    return collection


def test_retrieval(collection: chromadb.Collection, model: SentenceTransformer):
    print(f"\n{'='*60}")
    print("RETRIEVAL TEST — 3 Czech queries")
    print(f"{'='*60}")

    for query in TEST_QUERIES:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)

        q_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        results = collection.query(
            query_embeddings=q_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), 1):
            score = 1 - dist   # cosine similarity (higher = better)
            print(f"\n  #{rank}  score={score:.3f}  type={meta['type']}")
            print(f"       source: {meta['source'][:70]}")
            print(f"       text:   {doc[:200].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    print(f"{'='*60}")
    print("VECTOR DATABASE BUILDER")
    print(f"{'='*60}")

    # 1. Load chunks
    print(f"\nLoading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    print(f"  Loaded {len(chunks)} chunks.")

    # 2. Load embedding model (downloads ~120MB on first run, cached after)
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  (First run downloads ~120 MB — cached for all future runs)")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  ✓ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # 3. Build the database
    collection = build_database(chunks, model)

    # 4. Verify with test queries
    test_retrieval(collection, model)

    # 5. Final stats
    print(f"\n{'='*60}")
    print("DATABASE READY")
    print(f"  Collection : {COLLECTION}")
    print(f"  Documents  : {collection.count()}")
    print(f"  Location   : {CHROMA_DIR}")
    print(f"{'='*60}\n")
