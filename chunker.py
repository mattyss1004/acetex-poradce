"""
chunker.py
----------
Context-aware chunker for the acetex.cz knowledge base.

Reads all .txt files from data/html/ and data/pdf/
Applies a different chunking strategy per content type:

  faq            → split by Q&A pair (question + answer = 1 chunk)
  technology     → whole page as 1-2 chunks (they are short)
  article        → paragraph-based with overlap, heading prefix
  subsidy        → paragraph-based with overlap
  pdf_document   → fixed-size with overlap, filename prefix
  general/other  → paragraph-based with overlap

Each chunk is a JSON object written to data/chunks.jsonl:
  {
    "id":        "unique string",
    "text":      "the chunk text",
    "source":    "https://...",
    "title":     "page or document title",
    "type":      "faq|technology|article|...",
    "chunk_idx": 0
  }

Run with:
    python3.11 chunker.py
"""

import json
import re
import hashlib
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_HTML  = BASE_DIR / "data" / "html"
DATA_PDF   = BASE_DIR / "data" / "pdf"
OUTPUT     = BASE_DIR / "data" / "chunks.jsonl"

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE    = 800    # target characters per chunk
CHUNK_OVERLAP = 150   # overlap between consecutive chunks

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_meta(meta_path: Path) -> dict:
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                meta[k.strip()] = v.strip()
    return meta


def make_id(source: str, idx: int) -> str:
    h = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"{h}_{idx:04d}"


def make_chunk(text: str, meta: dict, idx: int) -> dict:
    return {
        "id":        make_id(meta.get("url", meta.get("filename", "unknown")), idx),
        "text":      text.strip(),
        "source":    meta.get("url", meta.get("filename", "")),
        "title":     meta.get("title", meta.get("filename", "")),
        "type":      meta.get("type", "general"),
        "chunk_idx": idx,
    }


def split_by_size(text: str, size: int = CHUNK_SIZE,
                  overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            if current and overlap > 0:
                overlap_text = current[-overlap:]
                current = (overlap_text + "\n\n" + para).strip()
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ── Strategy: FAQ ─────────────────────────────────────────────────────────────

def chunk_faq(text: str, meta: dict) -> list[dict]:
    """
    Split FAQ text into individual Q&A pairs.
    Heuristic: a line that ends with '?' is a question;
    everything until the next question is the answer.
    """
    lines = text.splitlines()
    chunks = []
    current_q = None
    current_a_lines = []
    idx = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect a question line: ends with ? or is short and followed by answer
        # Also handle lines that are clearly category headers (no ?)
        is_question = (
            stripped.endswith("?") and
            len(stripped) < 200 and
            not stripped.startswith("http")
        )

        if is_question:
            # Save previous Q&A pair
            if current_q and current_a_lines:
                answer = " ".join(current_a_lines).strip()
                if len(answer) > 30:  # skip trivial answers
                    chunk_text = f"Otázka: {current_q}\nOdpověď: {answer}"
                    chunks.append(make_chunk(chunk_text, meta, idx))
                    idx += 1
            current_q = stripped
            current_a_lines = []
        else:
            if current_q:
                current_a_lines.append(stripped)

    # Don't forget the last pair
    if current_q and current_a_lines:
        answer = " ".join(current_a_lines).strip()
        if len(answer) > 30:
            chunk_text = f"Otázka: {current_q}\nOdpověď: {answer}"
            chunks.append(make_chunk(chunk_text, meta, idx))

    return chunks


# ── Strategy: Technology page ─────────────────────────────────────────────────

def chunk_technology(text: str, meta: dict) -> list[dict]:
    """
    Technology pages are short (~1200 chars). Keep as one chunk,
    but prepend the product name clearly for retrieval context.
    """
    title = meta.get("title", "").replace(" | Acetex", "").strip()
    # Remove the page title line if it appears at the top of the text
    clean = text.replace(f"{title} | Acetex", "").strip()
    # Prepend product name so retrieval context is always clear
    enriched = f"Produkt: {title}\n\n{clean}"

    chunks = split_by_size(enriched, size=1200, overlap=100)
    return [make_chunk(c, meta, i) for i, c in enumerate(chunks)]


# ── Strategy: Article ─────────────────────────────────────────────────────────

def chunk_article(text: str, meta: dict) -> list[dict]:
    """
    Articles: paragraph-based split with overlap.
    Prepend article title to every chunk for context.
    """
    title = meta.get("title", "").replace(" | Acetex", "").strip()
    # Remove the title line from the top of the text to avoid duplication
    lines = text.splitlines()
    if lines and title.lower() in lines[0].lower():
        text = "\n".join(lines[1:]).strip()

    prefix = f"Článek: {title}\n\n"
    chunks = split_by_size(text)
    result = []
    for i, c in enumerate(chunks):
        enriched = prefix + c if i == 0 else f"(pokračování článku: {title})\n\n{c}"
        result.append(make_chunk(enriched, meta, i))
    return result


# ── Strategy: Subsidy / General page ─────────────────────────────────────────

def chunk_general(text: str, meta: dict) -> list[dict]:
    """Paragraph-based split with title prefix on first chunk."""
    title = meta.get("title", "").replace(" | Acetex", "").strip()
    lines = text.splitlines()
    if lines and title.lower() in lines[0].lower():
        text = "\n".join(lines[1:]).strip()

    prefix = f"Stránka: {title}\n\n"
    chunks = split_by_size(text)
    result = []
    for i, c in enumerate(chunks):
        enriched = prefix + c if i == 0 else f"(pokračování stránky: {title})\n\n{c}"
        result.append(make_chunk(enriched, meta, i))
    return result


# ── Strategy: PDF document ────────────────────────────────────────────────────

def chunk_pdf(text: str, meta: dict) -> list[dict]:
    """
    PDFs can be very long (up to 400k chars for full manuals).
    Use fixed-size chunking with overlap, prefixing the document name.
    """
    filename = meta.get("filename", meta.get("url", "document")).replace(".pdf.txt", "").replace(".pdf", "")
    # Clean up the filename into a readable label
    label = filename.replace("-", " ").replace("_", " ").strip()

    prefix = f"Dokument: {label}\n\n"
    chunks = split_by_size(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    result = []
    for i, c in enumerate(chunks):
        enriched = prefix + c if i == 0 else f"(pokračování dokumentu: {label})\n\n{c}"
        result.append(make_chunk(enriched, meta, i))
    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────

def chunk_document(text: str, meta: dict) -> list[dict]:
    page_type = meta.get("type", "general")

    if page_type == "faq":
        return chunk_faq(text, meta)
    elif page_type == "technology":
        return chunk_technology(text, meta)
    elif page_type == "article":
        return chunk_article(text, meta)
    elif page_type == "pdf_document":
        return chunk_pdf(text, meta)
    else:
        # subsidy, product_service, client_support, about, reference, general
        return chunk_general(text, meta)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_chunks = []
    stats = {}

    # Collect all (text_path, meta_path) pairs
    sources = []
    for txt_file in sorted(DATA_HTML.glob("*.txt")):
        if txt_file.name.endswith(".meta.txt"):
            continue
        meta_file = txt_file.with_suffix("").with_suffix(".meta.txt")
        sources.append((txt_file, meta_file))

    for txt_file in sorted(DATA_PDF.glob("*.txt")):
        if txt_file.name.endswith(".meta.txt"):
            continue
        # PDF text files are named like "Foo.pdf.txt"
        # Their meta files are named like "Foo.pdf.meta.txt"
        # txt_file.stem = "Foo.pdf"  →  meta = "Foo.pdf.meta.txt"
        meta_file = txt_file.parent / (txt_file.stem + ".meta.txt")
        sources.append((txt_file, meta_file))

    print(f"\n{'='*60}")
    print(f"CHUNKER — processing {len(sources)} source files")
    print(f"{'='*60}")

    for txt_path, meta_path in sources:
        text = txt_path.read_text(encoding="utf-8").strip()
        meta = read_meta(meta_path)

        if not text or len(text) < 50:
            continue

        chunks = chunk_document(text, meta)

        # Filter out empty or trivially short chunks
        chunks = [c for c in chunks if len(c["text"]) >= 80]

        page_type = meta.get("type", "general")
        stats[page_type] = stats.get(page_type, 0) + len(chunks)
        all_chunks.extend(chunks)

    # Write output
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nChunks by type:")
    for t, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {t:<20} {count:>5} chunks")

    print(f"\nTotal chunks written : {len(all_chunks)}")
    print(f"Output file          : {OUTPUT}")
    print(f"File size            : {OUTPUT.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")

    # Quick quality spot-check: show 3 sample chunks
    print("SAMPLE CHUNKS (quality check):")
    print("-" * 60)
    import random
    random.seed(42)
    samples = random.sample(all_chunks, min(3, len(all_chunks)))
    for s in samples:
        print(f"\n[{s['type']}] {s['title'][:60]}")
        print(f"Source: {s['source']}")
        print(f"Text preview ({len(s['text'])} chars):")
        print(s['text'][:400])
        print("...")
        print("-" * 60)


if __name__ == "__main__":
    main()
