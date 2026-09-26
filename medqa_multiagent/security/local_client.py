"""Lazy-loaded Hugging Face client for official undefended/StruQ checkpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..llm_client import LLMResponse
from .structured import StructuredQuery, compose_struq_prompt, recursive_filter


class StruQLocalClient:
    """Causal-LM client supporting both ordinary and StruQ structured prompts.

    Heavy dependencies and model weights are loaded only on construction. Use
    an official ``*_SpclSpclSpcl_None_*`` checkpoint for the matched
    undefended condition and ``*_SpclSpclSpcl_NaiveCompletion_*`` for StruQ.
    """

    def __init__(
        self,
        model_path: str,
        *,
        use_structured_queries: bool,
        quantization: str = "4bit",
        device_map: str = "auto",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "Local StruQ inference requires the 'security' extra: "
                "pip install -e '.[security]'"
            ) from exc

        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError("quantization must be one of: 4bit, 8bit, none")
        self.use_structured_queries = use_structured_queries
        self.model_path = str(model_path)
        self.quantization = quantization
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.sanitization_events = 0
        self.structured_queries = 0
        self.last_peak_vram_bytes = 0
        self._torch = torch

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if quantization in {"4bit", "8bit"}:
            try:
                load_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=quantization == "4bit",
                    load_in_8bit=quantization == "8bit",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{quantization} loading requires a working bitsandbytes installation"
                ) from exc
        else:
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        path = str(Path(model_path).expanduser())
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = transformers.AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        self._model.eval()

    @property
    def model_revision(self) -> str:
        config = getattr(self._model, "config", None)
        revision = getattr(config, "_commit_hash", None)
        return str(revision or "local")

    def _generate(
        self,
        *,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str],
    ) -> LLMResponse:
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])

        started = time.monotonic()
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency = time.monotonic() - started
        generated_ids = output[0][input_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = int(generated_ids.shape[0])
        if torch.cuda.is_available():
            self.last_peak_vram_bytes = int(torch.cuda.max_memory_allocated())

        return LLMResponse(
            text=text,
            model=model or self.model_path,
            temperature=temperature,
            prompt=prompt,
            role=role,
            prompt_tokens=input_length,
            completion_tokens=completion_tokens,
            total_tokens=input_length + completion_tokens,
            latency_seconds=latency,
            retrieved_context_id=retrieved_context_id,
        )

    def complete(
        self,
        role: str,
        prompt: str,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        if self.use_structured_queries:
            prompt = compose_struq_prompt(
                StructuredQuery(instruction=prompt, data=""), apply_filter=True
            )
        return self._generate(
            role=role,
            prompt=prompt,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )

    def complete_structured(
        self,
        role: str,
        query: StructuredQuery,
        model: str,
        temperature: float,
        retrieved_context_id: Optional[str] = None,
    ) -> LLMResponse:
        self.structured_queries += 1
        if recursive_filter(query.data) != query.data:
            self.sanitization_events += 1
        prompt = compose_struq_prompt(query, apply_filter=True)
        return self._generate(
            role=role,
            prompt=prompt,
            model=model,
            temperature=temperature,
            retrieved_context_id=retrieved_context_id,
        )
