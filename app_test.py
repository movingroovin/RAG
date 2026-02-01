import time
import os
from google import genai
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

start = time.time()
print(f"time: {start} Generating content...")
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)
print(f"Time taken: {time.time() - start} seconds")

print(response.text)

print("\nGenerating embeddings...")
texts = [f"text {i}" for i in range(20)]
result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
)

print(result.embeddings)