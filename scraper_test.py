"""
scraper_test.py
---------------
Single-page scraper test for acetex.cz.
Fetches one URL, strips boilerplate (nav, footer, forms, scripts, styles),
and prints the clean text content to the console.

Run with:
    python3.11 scraper_test.py
"""

import requests
from bs4 import BeautifulSoup

# --- Configuration ---
TEST_URL = "https://acetex.cz/otazky-a-odpovedi"

# Tags that are pure boilerplate — never contain useful knowledge
STRIP_TAGS = [
    "script", "style", "noscript",
    "header", "footer", "nav",
    "form", "button", "input", "textarea", "label", "select",
    "iframe", "svg", "img",
]

# CSS classes/IDs that are site-specific boilerplate on acetex.cz
STRIP_SELECTORS = [
    "#js-spinoco-desktop",   # live chat widget
    ".cookie-bar",           # cookie consent bar
    "[id^='tns']",           # tiny-slider carousel wrappers (duplicate content)
    ".footer",               # footer block
    "[class*='footer']",     # any footer variant
    ".site-footer",          # alternate footer class
    ".bottom-bar",           # bottom bar
    ".ratings-bar",          # review score bar (4.5, 4.4 etc.)
    ".cookie",               # cookie notice
    "[class*='cookie']",     # any cookie variant
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcetexBot/1.0; "
        "+https://acetex.cz)"
    )
}


def fetch_clean_text(url: str) -> str:
    """Fetch a URL and return clean, stripped plain text."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "lxml")

    # Remove all boilerplate tags
    for tag in STRIP_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    # Remove boilerplate by CSS selector
    for selector in STRIP_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Extract text — separator keeps paragraph breaks readable
    raw_text = soup.get_text(separator="\n", strip=True)

    # Collapse runs of blank lines down to a single blank line
    lines = raw_text.splitlines()
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        if line.strip() == "":
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(line.strip())
            prev_blank = False

    return "\n".join(cleaned_lines)


if __name__ == "__main__":
    print(f"Fetching: {TEST_URL}\n{'='*60}")
    text = fetch_clean_text(TEST_URL)
    print(text)
    print(f"\n{'='*60}")
    print(f"Total characters extracted: {len(text)}")
    print(f"Total lines extracted:      {len(text.splitlines())}")
