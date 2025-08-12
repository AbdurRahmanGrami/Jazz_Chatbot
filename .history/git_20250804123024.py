import requests

API_KEY = "Bearer f84d471630cba764ff7469faa0900231c47bed569ce16ee9e2a0879152bc615d"

headers = {
    "Authorization": API_KEY
}

res = requests.get("https://api.together.ai/models", headers=headers)

print("Status code:", res.status_code)
print("Response text:", res.text[:500])  # print first 500 chars

try:
    models = res.json()
    for m in models:
        if "llama" in m["name"].lower():
            print(m["name"])
except Exception as e:
    print("Failed to decode JSON.")
    raise
