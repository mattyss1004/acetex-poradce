"""
rag_chain.py — Acetex chatbot RAG pipeline
Uses Gemini for both embeddings and chat generation.
Memory footprint: ~80 MB (ChromaDB only, no local ML models).
"""

import os
from pathlib import Path
from functools import lru_cache

import chromadb
from google import genai

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
CHROMA_DIR     = BASE_DIR / "data" / "chroma_db_gemini"
COLLECTION     = "acetex_gemini"
EMBED_MODEL    = "gemini-embedding-001"
CHAT_MODEL     = "gemini-2.5-flash"
TOP_K          = 5
MAX_CONTEXT    = 6000

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

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
- Pokud jde o konkrétní zařízení nebo postup z manuálu, drž se přesně toho, co dokumentace říká."""

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_gemini_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=GEMINI_API_KEY)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question: str, k: int = TOP_K) -> list:
    gemini = get_gemini_client()
    collection = get_collection()

    emb_resp = gemini.models.embed_content(model=EMBED_MODEL, contents=question)
    q_vec = emb_resp.embeddings[0].values

    results = collection.query(
        query_embeddings=[q_vec],
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


def build_context(chunks: list, max_chars: int = MAX_CONTEXT) -> str:
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
    gemini = get_gemini_client()
    user_message = (
        f"KONTEXT (podklady z webu a dokumentace Acetex):\n{context}\n\n"
        f"OTÁZKA:\n{question}"
    )
    try:
        response = gemini.models.generate_content(
            model=CHAT_MODEL,
            contents=[{"role": "user", "parts": [{"text": user_message}]}],
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.3,
                "max_output_tokens": 2048,
            }
        )
        return response.text.strip()
    except Exception as e:
        return (
            f"Omlouváme se, nastala chyba při generování odpovědi. "
            f"Zkuste to prosím znovu nebo kontaktujte Acetex na tel. 770 110 011. "
            f"(Technický detail: {e})"
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def answer(question: str) -> dict:
    chunks = retrieve(question)
    context = build_context(chunks)
    llm_answer = call_llm(context, question)

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

    return {"answer": llm_answer, "sources": unique_sources}


if __name__ == "__main__":
    result = answer("Jaká je výše dotace na fotovoltaiku v NZÚ Light?")
    print("ANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['title']} | {s['source']}")
