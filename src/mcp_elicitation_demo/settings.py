"""Load environment variables from .env and expose typed settings."""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env from the project root (search upwards from this file's directory).
load_dotenv()


class Settings:
    """Small settings object backed by environment variables."""

    def __init__(self) -> None:
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.path: str = os.getenv("MCP_PATH", "/mcp")

    @property
    def llm_ready(self) -> bool:
        """True when an API key is configured (base URL may use its default)."""
        return bool(self.llm_api_key)


settings = Settings()
