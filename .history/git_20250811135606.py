import re

def clean_text(text):
    # Replace multiple newlines with one
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with one space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text_by_headings(text, headings):
    # Normalize text first
    text = clean_text(text)

    # Build regex pattern for headings (ignore case), allowing optional colon and ignoring leading/trailing spaces
    # Will match headings anywhere on the line
    headings_pattern = '|'.join([re.escape(h) + r':?' for h in headings])
    
    # Split but keep headings as part of chunks (using positive lookahead)
    pattern = re.compile(r'(?=(' + headings_pattern + r'))', re.IGNORECASE)

    splits = pattern.split(text)
    
    # The split array alternates between content before heading and heading+rest
    # So we need to join heading with its content properly
    
    chunks = []
    # Because split includes separators as separate items, we can pair them
    for i in range(1, len(splits), 2):
        heading_line = splits[i].strip()
        # content is next item or empty string if none
        content = splits[i+1].strip() if i+1 < len(splits) else ''
        chunk_text = heading_line + '\n' + content
        # Extract heading name (without colon) from heading_line for labeling
        heading_name = re.match(r'([^\:]+)', heading_line, re.IGNORECASE).group(1).strip()
        chunks.append((heading_name, chunk_text))

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

    if not chunks:
        print("No chunks found.")
    else:
        for i, (heading, chunk) in enumerate(chunks, 1):
            print(f"\n=== Chunk {i}: {heading} ===\n")
            print(chunk[:1000])  # Preview first 1000 chars
