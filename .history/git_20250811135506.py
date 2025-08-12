import re

def chunk_text_by_headings(text, headings):
    """
    Split text into chunks based on all occurrences of headings.
    Each heading is matched case-insensitive, optionally followed by colon.
    """
    # Build pattern with optional colon and ignore case, match heading at start of line with optional spaces
    pattern = re.compile(
        r'^\s*(' + '|'.join(re.escape(h) for h in headings) + r')\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )

    # Find all heading matches
    matches = list(pattern.finditer(text))

    if not matches:
        print("No headings found with strict pattern. Trying relaxed pattern.")
        # Relax pattern: heading at start of line followed by anything
        pattern = re.compile(
            r'^\s*(' + '|'.join(re.escape(h) for h in headings) + r')\s*:?.*$',
            re.IGNORECASE | re.MULTILINE
        )
        matches = list(pattern.finditer(text))

    if not matches:
        print("No headings found even with relaxed pattern.")
        return [(None, text.strip())]

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        heading = match.group(1).strip()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk_text = text[start:end].strip()
        chunks.append((heading, chunk_text))

    return chunks

if __name__ == "__main__":
    headings = [
        "OVERVIEW",
        "Features and benefits",
        "How to subscribe",
        "Charges and BIMA Sehat Cover",
        "How do I Claim",
        "Terms & Conditions"
    ]

    with open("insurance_chunk.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text_by_headings(full_text, headings)

    for i, (heading, chunk) in enumerate(chunks, 1):
        print(f"\n=== Chunk {i}: {heading} ===\n")
        print(chunk[:1000])  # print first 1000 chars of each chunk as a preview
