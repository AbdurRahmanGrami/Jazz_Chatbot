import requests

headers = {
    "Authorization": "f84d471630cba764ff7469faa0900231c47bed569ce16ee9e2a0879152bc615d"  # paste it directly for now
}

res = requests.get("https://api.together.ai/models", headers=headers)

try:
    models = res.json()
    for m in models:
        if "llama" in m["name"].lower():
            print(m["name"])
except Exception as e:
    print("Response text:", res.text)
    raise
