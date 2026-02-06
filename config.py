import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify that API keys are loaded
google_api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGSMITH_API_KEY")

if google_api_key:
    print("Gemini API Key loaded successfully")
else:
    print("Gemini API Key not found")

if langchain_api_key:
    print("Langsmith API Key loaded successfully")
else:
    print("Langsmith API Key not found")
