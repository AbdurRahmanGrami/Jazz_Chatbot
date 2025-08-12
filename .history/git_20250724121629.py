import openai

openai.api_type = "azure"
openai.api_key = "ELetdaaitvlKBQh1HN9ugysYm7LwzrEuEkykRiEFZ4ms3Jm7sbPMJQQJ99BFACYeBjFXJ3w3AAABACOGb1pi"
openai.api_base = openai.api_base = "https://azure-openai-intern.openai.azure.com"

openai.api_version = "2025-01-01-preview"

response = openai.ChatCompletion.create(
    engine="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response)
