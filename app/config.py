import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _parse_list(val: Optional[str], default: List[str]) -> List[str]:
    if not val:
        return default
    return [x.strip() for x in val.split(",") if x.strip()]


class Settings:
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    PRIMARY_PROVIDER: str = os.getenv("PRIMARY_PROVIDER", "groq")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODELS: List[str] = _parse_list(
        os.getenv("GROQ_MODELS"), ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    )

    FALLBACK_PROVIDER: str = os.getenv("FALLBACK_PROVIDER", "gemini")
    GEMINI_BASE_URL: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_MODELS: List[str] = _parse_list(
        os.getenv("GEMINI_MODELS"), ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
    )

    EVAL_PROVIDER: str = os.getenv("EVAL_PROVIDER", "groq")
    EVAL_MODEL: str = os.getenv("EVAL_MODEL", "openai/gpt-oss-20b")

    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "../Data_files")

    def resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()


settings = Settings()
