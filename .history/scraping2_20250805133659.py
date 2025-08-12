import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://jazz.com.pk"
DEVICES_URL = f"{BASE_URL}/devices"

# --- Setup Chrome driver ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)
driver.set_page_load_timeout(12)

# --- Extract all device links from main devices page ---
print(f"📥 Scraping device list from {DEVICES_URL}")
driver.get(DEVICES_URL)
time.sleep(3)  # Let page load

soup = BeautifulSoup(driver.page_source, "html.parser")
device_links = set()

for a in soup.find_all("a", href=True):
    href = a["href"]
    if href.startswith("/devices/") and len(href.strip("/").split("/")) > 1:
        full_url = urljoin(BASE_URL, href)
        device_links.add(full_url)

device_links = sorted(list(device_links))
print(f"🔗 Found {len(device_links)} device pages")

# --- Helper to extract main page content ---
def extract_main_content(driver):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    for tag in soup.find_all(['header', 'footer']):
        tag.decompose()
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

# --- Scrape each device page ---
output = []

for i, url in enumerate(device_links):
    print(f"⏳ [{i+1}/{len(device_links)}] Scraping: {url}")
    try:
        driver.get(url)
        time.sleep(2)
        page_text = extract_main_content(driver)

        title = driver.title.strip()

        output.append({
            "Title": title,
            "URL": url,
            "FullText": page_text
        })

    except TimeoutException:
        print(f"⛔ Timeout loading {url}")
    except Exception as e:
        print(f"⛔ Error at {url}: {e}")

# --- Save to file ---
with open("devices_full.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

driver.quit()
print(f"\n✅ Done. {len(output)} devices saved to devices_full.json")
