import requests
import os

headers = {
    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}"
}

res = requests.get("https://api.together.ai/models", headers=headers)
models = res.json()

for m in models:
    if "llama" in m["name"].lower():
        print(m["name"])
