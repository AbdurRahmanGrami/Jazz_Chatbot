from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, json, re

# Setup
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=chrome_options)

urls = [
    "https://jazz.com.pk/postpaid/postpaid-red",
    "https://jazz.com.pk/postpaid/gold-value",
    "https://jazz.com.pk/postpaid/gold-plus"
]

def clean(text):
    return re.sub(r'\s+', ' ', text).strip()

def scrape_page(url):
    driver.get(url)
    wait = WebDriverWait(driver, 10)
    
    data = {"URL": url}
    
    # Headers
    headers = []
    header_tags = ["h1", "h2", "h4"]
    for tag in header_tags:
        elements = driver.find_elements(By.TAG_NAME, tag)
        for e in elements:
            txt = clean(e.text)
            if txt and txt not in headers:
                headers.append(txt)
    
    # Additional offer details
    offer_details = driver.find_elements(By.CLASS_NAME, "pkg-offer-detail")
    for e in offer_details:
        txt = clean(e.text)
        if txt and txt not in headers:
            headers.append(txt)

    data["Headers"] = headers

    # Rates
    rates = []
    rate_blocks = driver.find_elements(By.CSS_SELECTOR, "#rate-block div.row")
    for row in rate_blocks:
        try:
            title = clean(row.find_element(By.TAG_NAME, "h5").text)
            value = clean(row.find_element(By.TAG_NAME, "p").text)
            if title and value:
                rates.append({"Title": title, "Value": value})
        except:
            continue
    data["Rates"] = rates

    # Overview
    try:
        overview_section = wait.until(EC.presence_of_element_located((By.ID, "overview")))
        paras = overview_section.find_elements(By.TAG_NAME, "p")
        overview = "\n".join(clean(p.text) for p in paras if p.text.strip())
        data["Overview"] = overview
    except:
        data["Overview"] = ""

    # Terms & Conditions
    try:
        tc_section = driver.find_element(By.ID, "terms-and-conditions")
        paras = tc_section.find_elements(By.TAG_NAME, "p")
        terms = "\n\n".join(clean(p.text) for p in paras if p.text.strip())
        data["Terms & Conditions"] = terms
    except:
        data["Terms & Conditions"] = ""

    return data

# Scrape all pages
results = [scrape_page(url) for url in urls]
driver.quit()

# Output
print(json.dumps(results, indent=4, ensure_ascii=False))
