"""
Rebuild ChromaDB vector database using OpenAI text-embedding-3-small.
This eliminates the 470 MB sentence-transformers model from RAM.
Cost: ~$0.00002 per 990 chunks = essentially free.
"""
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

CHUNKS_FILE = Path("/home/ubuntu/acetex_bot/data/chunks.jsonl")
CHROMA_DIR = Path("/home/ubuntu/acetex_deploy/data/chroma_db_openai")
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI allows up to 2048 per request

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load all chunks
print("Loading chunks...")
chunks = []
with open(CHUNKS_FILE) as f:
    for line in f:
        chunks.append(json.loads(line.strip()))
print(f"Loaded {len(chunks)} chunks")

# Set up ChromaDB with a fresh collection
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Delete existing collection if it exists
try:
    chroma_client.delete_collection("acetex_openai")
    print("Deleted existing collection")
except:
    pass

collection = chroma_client.create_collection(
    name="acetex_openai",
    metadata={"hnsw:space": "cosine"}
)

# Embed and insert in batches
print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(chunks))
    batch = chunks[start:end]

    texts = [c["text"] for c in batch]
    ids = [c["id"] for c in batch]
    metadatas = [{"source": c.get("source",""), "title": c.get("title",""), "type": c.get("type",""), "chunk_idx": str(c.get("chunk_idx",0))} for c in batch]

    # Get embeddings from OpenAI
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]

    # Insert into ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"  Batch {batch_idx + 1}/{total_batches} done ({end}/{len(chunks)} chunks)")
    time.sleep(0.2)  # gentle rate limiting

print(f"\nDone! Collection has {collection.count()} vectors.")
print(f"Saved to: {CHROMA_DIR}")

# Quick test query
print("\nRunning test query...")
test_response = client.embeddings.create(
    model=EMBED_MODEL,
    input=["Jaká je dotace na fotovoltaiku v NZÚ Light?"]
)
test_embedding = test_response.data[0].embedding

results = collection.query(
    query_embeddings=[test_embedding],
    n_results=3
)
print("Top 3 results for 'Jaká je dotace na fotovoltaiku v NZÚ Light?':")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"  [{i+1}] [{meta.get('type','?')}] {meta.get('title','?')[:60]}")
    print(f"       {doc[:100]}...")
