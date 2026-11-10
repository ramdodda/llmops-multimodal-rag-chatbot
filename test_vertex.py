
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
)

model = os.getenv(
    "GOOGLE_GENAI_MODEL",
    "gemini-3.1-pro-preview",
)

response = client.models.generate_content(
    model=model,
    contents="Explain LLMOps in three simple sentences.",
)

print("Model:", model)
print("\nAnswer:")
print(response.text)

usage = response.usage_metadata

print("\nFull token usage metadata:")
if usage:
    print(usage.model_dump(exclude_none=True))
else:
    print("Token usage metadata was not returned.")
