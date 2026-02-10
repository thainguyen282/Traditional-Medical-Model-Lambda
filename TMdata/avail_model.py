import json
import time
import uuid
from typing import Dict, List
import os

from google import genai
from google.genai import types

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found in environment")

# ================= INIT CLIENT =================
client = genai.Client(api_key=API_KEY)

def list_available_models():
    """List all available models"""
    try:
        models = client.models.list()
        print("Available models:")
        for model in models:
            print(f"  - {model.name}")
            # Print additional info if available
            if hasattr(model, 'display_name'):
                print(f"    Display name: {model.display_name}")
            if hasattr(model, 'description'):
                print(f"    Description: {model.description[:100]}...")
    except Exception as e:
        print(f"Error listing models: {e}")

# Call this after client is initialized
if __name__ == "__main__":
    print("Checking available models...")
    list_available_models()
    print("\n")