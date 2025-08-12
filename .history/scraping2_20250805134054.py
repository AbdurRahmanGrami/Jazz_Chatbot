from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json

BASE_URL = "https://jazz.com.pk/devices"

# --- Setup Selenium ---
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)
driver.get(BASE_URL)

# Wait for JS to load
time.sleep(3)

# --- Extract all <a href="..."> links ---
soup = BeautifulSoup(driver.page_source, "html.parser")
anchors = soup.find_all("a", href=True)

all_links = set()

for a in anchors:
    href = a["href"].strip()
    # Convert relative URLs to full ones
    full_url = urljoin(BASE_URL, href)
    all_links.add(full_url)

driver.quit()

# --- Save to JSON ---
all_links = sorted(list(all_links))
print(f"✅ Found {len(all_links)} links")

with open("jazz_devices_links.json", "w", encoding="utf-8") as f:
    json.dump(all_links, f, indent=2)

# Optional: preview
for link in all_links:
    print(link)
