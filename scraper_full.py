"""
scraper_full.py
---------------
Full knowledge-base harvester for acetex.cz.

Reads two URL lists:
  - html_urls.txt  : all high-priority HTML pages
  - pdf_urls.txt   : all named PDF documents (manuals, tech sheets, warranties)

For each HTML page:
  - Fetches, strips boilerplate, saves clean text to data/html/<slug>.txt
  - Saves a metadata sidecar to data/html/<slug>.meta.txt (url, title, type)

For each PDF:
  - Downloads to data/pdf/<filename>
  - Extracts text with pdfplumber
  - Saves clean text to data/pdf/<filename>.txt
  - Saves a metadata sidecar to data/pdf/<filename>.meta.txt

Progress is printed to stdout. Failed URLs are logged to data/errors.log.
Already-processed files are skipped (safe to re-run).

Run with:
    python3.11 scraper_full.py
"""

import os
import re
import time
import hashlib
import requests
import pdfplumber
import io
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_HTML  = BASE_DIR / "data" / "html"
DATA_PDF   = BASE_DIR / "data" / "pdf"
HTML_URLS  = BASE_DIR / "html_urls.txt"
PDF_URLS   = BASE_DIR / "pdf_urls.txt"
ERROR_LOG  = BASE_DIR / "data" / "errors.log"

DATA_HTML.mkdir(parents=True, exist_ok=True)
DATA_PDF.mkdir(parents=True, exist_ok=True)

# ── HTTP settings ─────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcetexBot/1.0; +https://acetex.cz)"
    )
}
REQUEST_DELAY = 0.8   # seconds between requests — polite crawling
TIMEOUT       = 20

# ── HTML boilerplate stripping ────────────────────────────────────────────────
STRIP_TAGS = [
    "script", "style", "noscript",
    "header", "footer", "nav",
    "form", "button", "input", "textarea", "label", "select",
    "iframe", "svg", "img",
]

STRIP_SELECTORS = [
    "#js-spinoco-desktop",
    ".cookie-bar",
    "[id^='tns']",
    ".footer",
    "[class*='footer']",
    ".site-footer",
    ".bottom-bar",
    ".ratings-bar",
    ".cookie",
    "[class*='cookie']",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def url_to_slug(url: str) -> str:
    """Convert a URL to a safe filename slug."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "__")
    if not path:
        path = "homepage"
    # Truncate very long paths and append a short hash for uniqueness
    if len(path) > 120:
        path = path[:120] + "_" + hashlib.md5(url.encode()).hexdigest()[:6]
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", path)


def clean_text(raw: str) -> str:
    """Collapse excessive blank lines."""
    lines = raw.splitlines()
    cleaned, prev_blank = [], False
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(stripped)
            prev_blank = False
    return "\n".join(cleaned).strip()


def extract_html_text(html: str, url: str) -> tuple[str, str]:
    """Return (title, clean_text) from raw HTML."""
    soup = BeautifulSoup(html, "lxml")

    # Extract title before stripping
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    for tag in STRIP_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    for sel in STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    raw = soup.get_text(separator="\n", strip=True)
    return title, clean_text(raw)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.append(text)
    return clean_text("\n\n".join(lines))


def log_error(url: str, reason: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"ERROR | {url} | {reason}\n")
    print(f"  ✗ FAILED: {reason}")


def write_text_and_meta(text_path: Path, meta_path: Path,
                        text: str, meta: dict):
    text_path.write_text(text, encoding="utf-8")
    with open(meta_path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")


# ── HTML scraper ──────────────────────────────────────────────────────────────

def scrape_html_pages():
    urls = [u.strip() for u in HTML_URLS.read_text().splitlines() if u.strip()]
    print(f"\n{'='*60}")
    print(f"HTML SCRAPER — {len(urls)} pages to process")
    print(f"{'='*60}")

    success, skipped, failed = 0, 0, 0

    for i, url in enumerate(urls, 1):
        slug = url_to_slug(url)
        text_path = DATA_HTML / f"{slug}.txt"
        meta_path = DATA_HTML / f"{slug}.meta.txt"

        if text_path.exists():
            print(f"  [{i:03}/{len(urls)}] SKIP (already done): {url}")
            skipped += 1
            continue

        print(f"  [{i:03}/{len(urls)}] Fetching: {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"

            title, text = extract_html_text(resp.text, url)

            if len(text) < 100:
                log_error(url, f"Too little content ({len(text)} chars)")
                failed += 1
                time.sleep(REQUEST_DELAY)
                continue

            # Classify page type from URL for metadata
            if "/clanky-a-rady/clanek/" in url:
                page_type = "article"
            elif "/technologie/" in url:
                page_type = "technology"
            elif "/dotace" in url:
                page_type = "subsidy"
            elif "/otazky-a-odpovedi" in url:
                page_type = "faq"
            elif "/produkty-a-sluzby" in url:
                page_type = "product_service"
            elif "/pro-klienty" in url:
                page_type = "client_support"
            elif "/reference" in url:
                page_type = "reference"
            elif "/o-nas" in url:
                page_type = "about"
            else:
                page_type = "general"

            meta = {
                "url":   url,
                "title": title,
                "type":  page_type,
                "chars": len(text),
            }
            write_text_and_meta(text_path, meta_path, text, meta)
            print(f"         ✓ {page_type} | {len(text):,} chars | {title[:60]}")
            success += 1

        except Exception as e:
            log_error(url, str(e))
            failed += 1

        time.sleep(REQUEST_DELAY)

    print(f"\nHTML done — ✓ {success} saved | ↷ {skipped} skipped | ✗ {failed} failed\n")


# ── PDF scraper ───────────────────────────────────────────────────────────────

def scrape_pdfs():
    urls = [u.strip() for u in PDF_URLS.read_text().splitlines() if u.strip()]
    print(f"\n{'='*60}")
    print(f"PDF SCRAPER — {len(urls)} documents to process")
    print(f"{'='*60}")

    success, skipped, failed = 0, 0, 0

    for i, url in enumerate(urls, 1):
        filename = Path(urlparse(url).path).name
        # Remove trailing -1 suffix that acetex appends to all uploads
        clean_name = re.sub(r"-1\.pdf$", ".pdf", filename)
        text_path = DATA_PDF / f"{clean_name}.txt"
        meta_path = DATA_PDF / f"{clean_name}.meta.txt"

        if text_path.exists():
            print(f"  [{i:03}/{len(urls)}] SKIP (already done): {clean_name}")
            skipped += 1
            continue

        print(f"  [{i:03}/{len(urls)}] Downloading: {clean_name}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            text = extract_pdf_text(resp.content)

            if len(text) < 50:
                log_error(url, f"PDF yielded too little text ({len(text)} chars) — may be image-only")
                failed += 1
                time.sleep(REQUEST_DELAY)
                continue

            meta = {
                "url":      url,
                "filename": clean_name,
                "type":     "pdf_document",
                "chars":    len(text),
            }
            write_text_and_meta(text_path, meta_path, text, meta)
            print(f"         ✓ {len(text):,} chars extracted")
            success += 1

        except Exception as e:
            log_error(url, str(e))
            failed += 1

        time.sleep(REQUEST_DELAY)

    print(f"\nPDF done — ✓ {success} saved | ↷ {skipped} skipped | ✗ {failed} failed\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_html_pages()
    scrape_pdfs()

    # Final summary
    html_files = list(DATA_HTML.glob("*.txt"))
    pdf_files  = list(DATA_PDF.glob("*.txt"))
    total_chars = sum(p.stat().st_size for p in html_files + pdf_files)

    print(f"\n{'='*60}")
    print(f"HARVEST COMPLETE")
    print(f"  HTML text files : {len(html_files)}")
    print(f"  PDF text files  : {len(pdf_files)}")
    print(f"  Total data size : {total_chars / 1024:.1f} KB")
    print(f"  Errors logged   : {ERROR_LOG}")
    print(f"{'='*60}\n")
