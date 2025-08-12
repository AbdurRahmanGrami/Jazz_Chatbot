from openai import AzureOpenAI
from openai import APIError, BadRequestError



# Set your values
api_key = "ELetdaaitvlKBQh1HN9ugysYm7LwzrEuEkykRiEFZ4ms3Jm7sbPMJQQJ99BFACYeBjFXJ3w3AAABACOGb1pi"
endpoint = "https://azure-openai-intern.openai.azure.com/"
deployment = "gpt-4o"
api_version = "2025-01-01-preview"

# Init Azure client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

# Create a long prompt to intentionally trigger token overflow
long_prompt = "hello " * 10000

try:
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": long_prompt}],
    )
    print("Response:", response)
except BadRequestError as e:
    print("Token limit exceeded or error:", e)
