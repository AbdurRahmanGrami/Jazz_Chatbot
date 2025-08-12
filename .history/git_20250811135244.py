import re

def chunk_text_by_headings(text, headings):
    """
    Splits `text` into chunks based on every occurrence of any heading in `headings`.
    Returns a list of tuples (heading, chunk_text).
    """

    pattern = re.compile(
        r'^\s*(' + '|'.join(re.escape(h) for h in headings) + r')\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(pattern.finditer(text))

    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        heading = match.group(1).strip()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk_text = text[start:end].strip()
        chunks.append((heading, chunk_text))

    if not chunks:
        return [(None, text.strip())]

    return chunks


if __name__ == "__main__":
    insurance_headings = [
        "Overview",
        "Features and benefits",
        "How to subscribe",
        "Terms & Conditions",
        "How do I Claim"
    ]

    with open("insurance_chunk.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = chunk_text_by_headings(full_text, insurance_headings)

    for idx, (heading, chunk) in enumerate(chunks, 1):
        print(f"\n=== Chunk {idx}: {heading} ===\n")
        print(chunk[:1000])  # print first 1000 chars for preview
