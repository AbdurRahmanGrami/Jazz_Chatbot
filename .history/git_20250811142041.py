import re
import json

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_text_by_headings(text, headings):
    text = clean_text(text)
    headings_pattern = '|'.join([re.escape(h) + r':?' for h in headings])
    pattern = re.compile(r'(?=(' + headings_pattern + r'))', re.IGNORECASE)
    splits = pattern.split(text)
    
    chunks = []
    for i in range(1, len(splits), 2):
        heading_line = splits[i].strip()
        content = splits[i+1].strip() if i+1 < len(splits) else ''
        chunk_text = heading_line + '\n' + content
        heading_name = re.match(r'([^\:]+)', heading_line, re.IGNORECASE).group(1).strip()
        chunks.append({
            "Heading": heading_name,
            "Content": chunk_text
        })

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
        with open("bima_chunks.json", "w", encoding="utf-8") as f_out:
            json.dump(chunks, f_out, ensure_ascii=False, indent=4)
        print(f"{len(chunks)} chunks saved to bima_chunks.json")
