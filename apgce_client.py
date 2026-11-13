"""OpenAI-compatible client for the shared APGCE vLLM server."""
import os
from openai import OpenAI

MODEL = "google/gemma-4-26B-A4B-it"
BASE_URL = os.getenv("APGCE_BASE_URL", "http://172.28.34.1:8000/v1")


def get_client():
    return OpenAI(base_url=BASE_URL, api_key="token-not-needed", timeout=180.0)
