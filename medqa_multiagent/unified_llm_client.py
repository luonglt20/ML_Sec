"""Unified Multi-Provider Multi-Tier LLM Client with Automatic Key Rotation & Failover Cascade.

Supports:
- Groq (6 Rotating Keys): llama-3.1-8b-instant (Light) vs llama-3.3-70b-versatile (Strong)
- Gemini (2 Rotating Keys): gemini-2.0-flash-lite (Light) vs gemini-2.5-flash / gemini-2.0-flash (Strong)
- DeepSeek (Backup Provider): deepseek-chat (Light) vs deepseek-reasoner R1 (Strong)

Zero external dependencies: uses 100% pure Python urllib.request.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .llm_client import LLMClient, LLMResponse


class UnifiedLLMClient(LLMClient):
    """Multi-Provider Multi-Tier LLM Client with zero-downtime failover cascade."""

    def __init__(self, timeout: float = 60.0) -> None:
        self._timeout = timeout

        # Load keys from .env / environment
        self._groq_keys = self._load_keys("GROQ_API_KEY", count=6)
        if not self._groq_keys:
            self._groq_keys = [
                "gsk_REDACTED",
                "gsk_REDACTED",
                "gsk_REDACTED",
                "gsk_REDACTED",
                "gsk_REDACTED",
                "gsk_REDACTED",
            ]

        self._gemini_keys = self._load_keys("GEMINI_API_KEY", count=2)
        if not self._gemini_keys:
            self._gemini_keys = [
                "AQ-REDACTED",
                "AQ-REDACTED",
            ]

        self._deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

        # Key Index Pointers
        self._groq_idx = 0
        self._gemini_idx = 0

    def _load_keys(self, prefix: str, count: int) -> List[str]:
        keys = []
        for i in range(1, count + 1):
            val = os.environ.get(f"{prefix}_{i}") or os.environ.get(f"{prefix}{i}")
            if val and val.strip():
                keys.append(val.strip())
        if not keys and os.environ.get(prefix):
            keys.append(os.environ[prefix].strip())

        if not keys:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith(prefix):
                        parts = line.split("=", 1)
                        if len(parts) > 1 and parts[1].strip():
                            k = parts[1].strip()
                            if k not in keys:
                                keys.append(k)
        return keys

    def _determine_tier(self, role: str) -> str:
        """Classify task difficulty: Light (Simple) vs Strong (Complex)."""
        if role in ("router", "memory_specialist", "fast_path", "eval_coverage"):
            return "light"
        return "strong"

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        tier = self._determine_tier(role)

        # ── Cascade Provider 1: Groq API (Rotation through 6 keys) ────────────
        if self._groq_keys:
            groq_model = "llama-3.1-8b-instant" if tier == "light" else "llama-3.3-70b-versatile"
            for _ in range(len(self._groq_keys)):
                key = self._groq_keys[self._groq_idx]
                self._groq_idx = (self._groq_idx + 1) % len(self._groq_keys)
                try:
                    return self._call_groq(key, groq_model, role, prompt, temperature, retrieved_context_id)
                except Exception as exc:
                    print(f"[UnifiedLLM] Groq Key failed: {exc}. Trying next key/provider...", flush=True)
                    continue

        # ── Cascade Provider 2: Gemini API (Rotation through 2 keys) ──────────
        if self._gemini_keys:
            gemini_model = "gemini-2.0-flash-lite" if tier == "light" else "gemini-2.5-flash"
            for _ in range(len(self._gemini_keys)):
                key = self._gemini_keys[self._gemini_idx]
                self._gemini_idx = (self._gemini_idx + 1) % len(self._gemini_keys)
                try:
                    return self._call_gemini(key, gemini_model, role, prompt, temperature, retrieved_context_id)
                except Exception as exc:
                    print(f"[UnifiedLLM] Gemini Key failed: {exc}. Trying next key/provider...", flush=True)
                    continue

        # ── Cascade Provider 3: DeepSeek API (Backup Provider) ────────────────
        if self._deepseek_key:
            ds_model = "deepseek-chat" if tier == "light" else "deepseek-reasoner"
            try:
                return self._call_deepseek(self._deepseek_key, ds_model, role, prompt, temperature, retrieved_context_id)
            except Exception as exc:
                print(f"[UnifiedLLM] DeepSeek failed: {exc}.", flush=True)

        raise RuntimeError("All LLM Providers (Groq, Gemini, DeepSeek) exhausted or rate limited.")

    def _call_groq(
        self, key: str, model_name: str, role: str, prompt: str, temperature: float, context_id: Optional[str]
    ) -> LLMResponse:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        t0 = time.monotonic()
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            elapsed = time.monotonic() - t0
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            text_out = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()) * 4 // 3)
            completion_tokens = usage.get("completion_tokens", len(text_out.split()) * 4 // 3)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return LLMResponse(
                text=text_out,
                model=f"groq/{model_name}",
                temperature=temperature,
                prompt=prompt,
                role=role,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=elapsed,
                retrieved_context_id=context_id,
                cache_hit=False,
            )

    def _call_gemini(
        self, key: str, model_name: str, role: str, prompt: str, temperature: float, context_id: Optional[str]
    ) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
        }

        t0 = time.monotonic()
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            elapsed = time.monotonic() - t0
            body = json.loads(resp.read().decode("utf-8"))
            text_out = ""
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text_out = parts[0].get("text", "")
            usage = body.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", len(prompt.split()) * 4 // 3)
            completion_tokens = usage.get("candidatesTokenCount", len(text_out.split()) * 4 // 3)
            total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)

            return LLMResponse(
                text=text_out,
                model=f"gemini/{model_name}",
                temperature=temperature,
                prompt=prompt,
                role=role,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=elapsed,
                retrieved_context_id=context_id,
                cache_hit=False,
            )

    def _call_deepseek(
        self, key: str, model_name: str, role: str, prompt: str, temperature: float, context_id: Optional[str]
    ) -> LLMResponse:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        t0 = time.monotonic()
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            elapsed = time.monotonic() - t0
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            text_out = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(prompt.split()) * 4 // 3)
            completion_tokens = usage.get("completion_tokens", len(text_out.split()) * 4 // 3)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            return LLMResponse(
                text=text_out,
                model=f"deepseek/{model_name}",
                temperature=temperature,
                prompt=prompt,
                role=role,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=elapsed,
                retrieved_context_id=context_id,
                cache_hit=False,
            )
