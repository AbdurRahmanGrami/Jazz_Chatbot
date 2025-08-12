import json
import re

# --- configurable section heading patterns ---
SECTION_PATTERNS = [
    r"features\s+and\s+benefits[:\-]?",
    r"how\s+to\s+subscribe[:\-]?",
    r"definitions",
    r"exclusions",
    r"general\s+provisions",
    r"how\s+do\s+i\s+claim[:\-]?",
    r"insurance\s+benefits?",
    r"disclaimer.*terms\s+of\s+use",
    r"terms\s+of\s+use",
    r"acceptance\s+of\s+these\s+terms",
]

# Compile into a single regex
SECTION_REGEX = re.compile(
    r"(?P<header>" + "|".join(SECTION_PATTERNS) + r")",
    re.IGNORECASE
)

def split_into_sections(text):
    # Find all matches for section headings
    matches = list(SECTION_REGEX.finditer(text))
    if not matches:
        return {"Full Document": text.strip()}

    sections = {}
    for i, match in enumerate(matches):
        start = match.start()
        section_name = match.group("header").strip().title()

        # Determine end index (either next heading or end of text)
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        # Extract section text
        section_text = text[start:end].strip()

        # Store in dict
        sections[section_name] = section_text

    return sections

# --- load your JSON ---
with open("insu.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# If it's a single object
if isinstance(data, dict):
    text = data.get("FullText", "")
    data["FullText"] = split_into_sections(text)

# If it's a list of offers
elif isinstance(data, list):
    for offer in data:
        text = offer.get("FullText", "")
        offer["FullText"] = split_into_sections(text)

# --- save the result ---
with open("insu_structured.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Done. Saved structured file as insu_structured.json")
