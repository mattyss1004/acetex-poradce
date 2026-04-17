"""
rag_chain.py
------------
Core RAG pipeline for the Acetex chatbot.

Exposes a single function:
    answer(question: str) -> dict

Returns:
    {
        "answer":  str,          # the LLM's response in Czech
        "sources": list[dict],   # top retrieved chunks with metadata
    }

Components:
  - ChromaDB (persistent, local) for vector retrieval
  - sentence-transformers for query embedding
  - OpenAI API for LLM generation (gpt-4.1-mini)
"""

import os
import json
from pathlib import Path
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Load .env if present ──────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
CHROMA_DIR   = BASE_DIR / "data" / "chroma_db"
COLLECTION   = "acetex_knowledge"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
TOP_K        = 5      # number of chunks to retrieve per query
MAX_CONTEXT  = 6000   # max characters of context to feed the LLM (GPT handles more)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Jsi poradce společnosti Acetex – české firmy, která lidem pomáhá s fotovoltaikou, tepelnými čerpadly, bateriemi a dotacemi. Mluvíš jako zkušený člověk z oboru, ne jako stroj.

Jak komunikuješ:
- Vždy česky. Bez výjimky.
- Piš přirozeně – jako když ti radí kamarád, který oboru rozumí. Žádné strojeně znějící fráze.
- Buď konkrétní a věcný, ale ne suchý. Klidně použij větu navíc, pokud to odpovědi pomůže.
- Pokud se zákazník ptá na něco, kde má Acetex dobré řešení, klidně to zmíň – ale přirozeně, ne jako reklamu.
- Nikdy nezačínaj odpověď slovy jako "Samozřejmě!", "Skvělá otázka!" nebo podobnými prázdnými frázemi.
- Pokud nevíš nebo to není v podkladech, řekni to rovnou a doporučuj kontakt: tel. 770 110 011 nebo info@acetex.cz.
- Aktuální ceny a časově omezené akce se rychle mění – na ty vždy odkaz na obchodní tým.
- Pokud jde o konkrétní zařízení nebo postup z manuálu, drž se přesně toho, co dokumentace říká.
- Na konci odpovědi uveď zdroje, ze kterých jsi vycházel (URL nebo název dokumentu)."""

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to acetex_bot/.env or set the environment variable."
        )
    return OpenAI(api_key=api_key)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return top-k relevant chunks for the question."""
    model      = get_embedding_model()
    collection = get_collection()

    q_vec = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":   doc,
            "source": meta.get("source", ""),
            "title":  meta.get("title", ""),
            "type":   meta.get("type", ""),
            "score":  round(1 - dist, 3),
        })

    return chunks


def build_context(chunks: list[dict], max_chars: int = MAX_CONTEXT) -> str:
    """Concatenate chunk texts into a context string, respecting the char limit."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        label = f"[Zdroj {i+1}: {chunk['source'] or chunk['title']}]"
        block = f"{label}\n{chunk['text']}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(context: str, question: str) -> str:
    """Call the OpenAI API and return the response text."""
    client = get_openai_client()

    user_message = (
        f"KONTEXT (podklady z webu a dokumentace Acetex):\n{context}\n\n"
        f"OTÁZKA:\n{question}"
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"Omlouváme se, nastala chyba při generování odpovědi. "
            f"Zkuste to prosím znovu nebo kontaktujte Acetex na tel. 770 110 011. "
            f"(Technický detail: {e})"
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def answer(question: str) -> dict:
    """
    Full RAG pipeline: retrieve → build context → generate answer.

    Returns:
        {
            "answer":  str,
            "sources": list[dict],
        }
    """
    # 1. Retrieve relevant chunks
    chunks = retrieve(question)

    # 2. Build context string
    context = build_context(chunks)

    # 3. Call the LLM (context + question passed separately as system/user messages)
    llm_answer = call_llm(context, question)

    # 4. Return answer + unique sources (deduplicated by URL)
    seen = set()
    unique_sources = []
    for c in chunks:
        key = c["source"] or c["title"]
        if key and key not in seen:
            seen.add(key)
            unique_sources.append({
                "source": c["source"],
                "title":  c["title"],
                "type":   c["type"],
                "score":  c["score"],
            })

    return {
        "answer":  llm_answer,
        "sources": unique_sources,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_QUESTIONS = [
        "Jaká je výše dotace na fotovoltaiku v programu NZÚ Light?",
        "Jak spustit systém Wattsonic All-in-One krok za krokem?",
        "Jaká je záruka na baterie GoodWe Lynx Home a kolik cyklů vydrží?",
    ]

    print("\n" + "="*65)
    print("ACETEX RAG CHATBOT — END-TO-END TEST (OpenAI)")
    print("="*65)

    for q in TEST_QUESTIONS:
        print(f"\nOTÁZKA: {q}")
        print("-" * 65)
        result = answer(q)
        print(f"ODPOVĚĎ:\n{result['answer']}")
        print(f"\nZDROJE:")
        for s in result["sources"]:
            print(f"  [{s['score']:.3f}] {s['type']:15} {s['source'][:70]}")
        print("="*65)
