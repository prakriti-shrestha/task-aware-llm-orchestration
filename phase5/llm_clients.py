"""
LLM client — single place where all API calls happen.
Supports: Google Gemini (free tier, new google-genai SDK), OpenAI.

Install:
    pip uninstall google-generativeai -y
    pip install google-genai openai python-dotenv

Features:
  - Rate limiting (configurable RPM)
  - Disk cache keyed on (provider, model, temperature, prompt)
    Identical prompts cost 0 API calls on re-runs.
  - Automatic retry with exponential backoff
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Default free-tier model — gemini-2.5-flash-lite has the highest free RPD (1000)
# Override with GEMINI_MODEL env var if you have a paid tier.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Rate limit — 15 RPM for flash-lite free tier = 1 req / 4s. Use 4.2s for safety.
_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "4.2"))

# Cache — disable by setting LLM_CACHE_DISABLE=1
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
_CACHE_DISABLED = os.getenv("LLM_CACHE_DISABLE", "0") == "1"

_lock = threading.Lock()
_last_call = 0.0


def _rate_limit():
    """Block until it is safe to make another API call."""
    global _last_call
    with _lock:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _cache_key(provider: str, model: str, temperature: float, prompt: str) -> str:
    payload = json.dumps(
        {"p": provider, "m": model, "t": round(temperature, 2), "prompt": prompt},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> tuple[str, int] | None:
    if _CACHE_DISABLED:
        return None
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["text"], int(data["tokens"])
    except Exception:
        return None


def _cache_put(key: str, text: str, tokens: int) -> None:
    if _CACHE_DISABLED:
        return
    path = CACHE_DIR / f"{key}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"text": text, "tokens": tokens}, f, ensure_ascii=False)
    except Exception:
        pass  # cache failures should never break the pipeline


# ── Gemini (new google-genai SDK) ─────────────────────────────────────────────
def _call_gemini(prompt: str, temperature: float, model: str) -> tuple[str, int]:
    _rate_limit()
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )

    # The new SDK can return None for .text if the response was blocked.
    text = (response.text or "").strip()
    if not text:
        text = "[empty response]"

    try:
        tokens = int(response.usage_metadata.total_token_count)
    except Exception:
        tokens = len(prompt.split()) + len(text.split())

    return text, tokens


# ── OpenAI ────────────────────────────────────────────────────────────────────
def _call_openai(prompt: str, temperature: float, model: str) -> tuple[str, int]:
    _rate_limit()
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = (response.choices[0].message.content or "").strip()
    tokens = int(response.usage.total_tokens)
    return text, tokens


# ── Public interface ──────────────────────────────────────────────────────────
def call_llm(
    prompt: str,
    temperature: float = 0.0,
    retries: int = 3,
    retry_delay: float = 8.0,
    model: str | None = None,
) -> tuple[str, int]:
    """
    Call the configured LLM provider, with caching and retries.

    Returns:
        (response_text: str, total_tokens: int)
    """
    if model is None:
        model = GEMINI_MODEL if PROVIDER == "gemini" else "gpt-3.5-turbo"

    # Check cache first — this is what makes ablations / baselines affordable
    key = _cache_key(PROVIDER, model, temperature, prompt)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if PROVIDER == "gemini":
                text, tokens = _call_gemini(prompt, temperature, model)
            elif PROVIDER == "openai":
                text, tokens = _call_openai(prompt, temperature, model)
            else:
                raise ValueError(
                    f"Unknown LLM_PROVIDER: {PROVIDER!r}. Use 'gemini' or 'openai'."
                )

            _cache_put(key, text, tokens)
            return text, tokens

        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Longer wait for rate-limit / quota errors
            delay = retry_delay * 4 if ("429" in err_str or "quota" in err_str or "rate" in err_str) else retry_delay
            if attempt == retries:
                break
            print(f"[llm_client] Attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay * attempt)  # exponential backoff

    raise RuntimeError(f"[llm_client] All {retries} attempts failed: {last_err}") from last_err


# Smoke test: python -m phase5.llm_clients
if __name__ == "__main__":
    print(f"Provider: {PROVIDER}")
    print(f"Model:    {GEMINI_MODEL if PROVIDER == 'gemini' else 'openai'}")
    print(f"Min interval: {_MIN_INTERVAL}s")
    print(f"Cache dir: {CACHE_DIR}")
    print()
    text, tokens = call_llm("Say hello in one word.")
    print(f"Response: {text!r}")
    print(f"Tokens:   {tokens}")