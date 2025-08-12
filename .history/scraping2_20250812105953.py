import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

# Replace with your actual list if not loading from file
urls = [
    "https://jazz.com.pk/jazz-insurance"
]

# --- Setup Selenium ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(15)

def extract_main_content():
    soup = BeautifulSoup(driver.page_source, "html.parser")
    for tag in soup.find_all(["header", "footer", "nav", "script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

output = []

# --- Scrape each page ---
for i, url in enumerate(urls):
    print(f"⏳ [{i+1}/{len(urls)}] Scraping: {url}")
    
    if url.lower().endswith(".pdf"):
        print("📄 Skipping PDF link")
        continue

    try:
        driver.get(url)
        time.sleep(2)
        title = driver.title.strip()
        content = extract_main_content()

        output.append({
            "Title": title,
            "URL": url,
            "FullText": content
        })

    except TimeoutException:
        print(f"⛔ Timeout at {url}")
    except Exception as e:
        print(f"⚠️ Error at {url}: {e}")

driver.quit()

# --- Save result ---
with open("device_misc_pages.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done. Scraped {len(output)} pages into device_misc_pages.json")
