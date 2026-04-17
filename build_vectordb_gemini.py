"""
Rebuild ChromaDB vector database using Gemini text-embedding-001.
Zero RAM usage at runtime — embeddings are pre-computed and stored.
"""
import json
import os
import time
from pathlib import Path
from google import genai

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

CHUNKS_FILE = Path("/home/ubuntu/acetex_bot/data/chunks.jsonl")
CHROMA_DIR = Path("/home/ubuntu/acetex_deploy/data/chroma_db_gemini")
EMBED_MODEL = "gemini-embedding-001"
BATCH_SIZE = 20  # conservative to avoid rate limits

import chromadb

# Load all chunks
print("Loading chunks...")
chunks = []
with open(CHUNKS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))
print(f"Loaded {len(chunks)} chunks")

# Set up ChromaDB
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    chroma_client.delete_collection("acetex_gemini")
    print("Deleted existing collection")
except:
    pass

collection = chroma_client.create_collection(
    name="acetex_gemini",
    metadata={"hnsw:space": "cosine"}
)

# Embed and insert in batches
print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
total_inserted = 0

for batch_idx in range(total_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(chunks))
    batch = chunks[start:end]

    texts = [c["text"] for c in batch]
    ids = [c["id"] for c in batch]
    metadatas = [
        {
            "source": c.get("source", ""),
            "title": c.get("title", ""),
            "type": c.get("type", ""),
            "chunk_idx": str(c.get("chunk_idx", 0))
        }
        for c in batch
    ]

    # Get embeddings from Gemini (one at a time to avoid rate limits)
    embeddings = []
    for text in texts:
        try:
            resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
            embeddings.append(resp.embeddings[0].values)
            time.sleep(0.05)
        except Exception as e:
            print(f"  Embedding error: {e}, retrying in 5s...")
            time.sleep(5)
            resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
            embeddings.append(resp.embeddings[0].values)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    total_inserted += len(batch)
    print(f"  Batch {batch_idx + 1}/{total_batches} done ({total_inserted}/{len(chunks)} chunks)")

print(f"\nDone! Collection has {collection.count()} vectors.")
print(f"Saved to: {CHROMA_DIR}")

# Quick test query
print("\nRunning test query...")
test_emb = client.models.embed_content(model=EMBED_MODEL, contents="Jaká je dotace na fotovoltaiku v NZÚ Light?")
results = collection.query(
    query_embeddings=[test_emb.embeddings[0].values],
    n_results=3
)
print("Top 3 results:")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"  [{i+1}] [{meta.get('type','?')}] {meta.get('title','?')[:60]}")
    print(f"       {doc[:100]}...")
