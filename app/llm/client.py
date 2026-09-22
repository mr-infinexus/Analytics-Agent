import os
import re
from typing import List, Optional, Tuple
from openai import AsyncOpenAI

from app.config import settings


def clean_sql_output(raw: str) -> str:
    """Strip markdown code blocks and trailing semicolons."""
    text = raw.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().rstrip(";")


class LLMClient:
    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        self.gemini_api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

        self.groq_client = (
            AsyncOpenAI(base_url=settings.GROQ_BASE_URL, api_key=self.groq_api_key, timeout=30.0)
            if self.groq_api_key
            else None
        )
        self.gemini_client = (
            AsyncOpenAI(base_url=settings.GEMINI_BASE_URL, api_key=self.gemini_api_key, timeout=30.0)
            if self.gemini_api_key
            else None
        )

        # Cascade order: Groq models first, then Gemini models
        self.cascade: List[Tuple[str, str]] = []
        for m in settings.GROQ_MODELS:
            self.cascade.append(("groq", m))
        for m in settings.GEMINI_MODELS:
            self.cascade.append(("gemini", m))

    async def generate_sql(self, user_prompt: str, system_prompt: str) -> Tuple[str]:
        errors = []
        for provider, model in self.cascade:
            client = self.groq_client if provider == "groq" else self.gemini_client
            if not client:
                continue

            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=600,
                )
                raw_sql = response.choices[0].message.content or ""
                sql = clean_sql_output(raw_sql)
                if sql:
                    return sql
            except Exception as e:
                errors.append(f"{provider}/{model}: {e}")
                continue

        err_detail = "; ".join(errors) if errors else "No API keys configured"
        raise RuntimeError(f"All LLM models failed: {err_detail}")


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
