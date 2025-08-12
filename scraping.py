import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

# --- Setup Chrome driver ---
options = Options()
options.add_argument("--headless")  # headless mode
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

driver.set_page_load_timeout(12)

# --- Load URLs from prepaid.json ---
with open("prepaid.json", "r", encoding="utf-8") as f:
    prepaid_offers = json.load(f)

output = []

# --- Helper to extract page content (excluding header/footer) ---
def extract_main_content(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Remove header and footer if present
    for tag in soup.find_all(['header', 'footer']):
        tag.decompose()

    # Clean scripts, styles, etc.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Return visible text only
    return soup.get_text(separator="\n", strip=True)

# --- Scrape each page ---
for i, offer in enumerate(prepaid_offers):
    url = offer["URL"]
    print(f"⏳ [{i+1}/{len(prepaid_offers)}] Scraping: {url}")

    try:
        driver.get(url)
        time.sleep(2)  # Wait for content to load
        page_text = extract_main_content(driver)

        output.append({
            "Title": offer.get("Title", ""),
            "URL": url,
            "Description": offer.get("Description", ""),
            "FullText": page_text
        })

    except TimeoutException:
        print(f"⛔ Timeout loading {url}")
    except Exception as e:
        print(f"⛔ Error at {url}: {e}")

# --- Save to file ---
with open("prepaid_full.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

driver.quit()
print(f"\n✅ Done. {len(output)} offers saved to prepaid_full.json")
