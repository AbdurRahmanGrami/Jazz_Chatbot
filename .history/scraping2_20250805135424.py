import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup

# --- Load extracted links ---
with open("jazz_devices_links.json", "r", encoding="utf-8") as f:
    links = json.load(f)

# Optional: Filter only /devices/ pages (exclude homepage, about etc.)
device_links = [url for url in links if "/devices/" in url and len(url.strip("/").split("/")) > 2]

print(f"📦 Scraping {len(device_links)} device pages")

# --- Setup Selenium ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(15)

# --- Extract only main content ---
def extract_main_content(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")

    for tag in soup.find_all(["header", "footer", "nav", "script", "style", "noscript"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)

# --- Visit each page and collect data ---
output = []

for i, url in enumerate(device_links):
    print(f"⏳ [{i+1}/{len(device_links)}] Scraping: {url}")
    try:
        driver.get(url)
        time.sleep(2)  # let content load

        text = extract_main_content(driver)
        title = driver.title.strip()

        output.append({
            "Title": title,
            "URL": url,
            "FullText": text
        })

    except TimeoutException:
        print(f"⛔ Timeout at {url}")
    except Exception as e:
        print(f"⛔ Error at {url}: {e}")

driver.quit()

# --- Save to file ---
with open("devices_full.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done. {len(output)} device pages saved to devices_full.json")
