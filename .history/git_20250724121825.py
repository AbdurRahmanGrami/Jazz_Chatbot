from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="ELetdaaitvlKBQh1HN9ugysYm7LwzrEuEkykRiEFZ4ms3Jm7sbPMJQQJ99BFACYeBjFXJ3w3AAABACOGb1pi",
    api_version="2025-01-01-preview",
    azure_endpoint="https://azure-openai-intern.openai.azure.com"
)

response = client.chat.completions.create(
    model="gpt-4o",  # this is your deployment name, not model name
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.choices[0].message.content)
