from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()

from .settings import DEFAULT_MAX_TOKENS, TEMPERATURE as DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)

PLACEHOLDER_MARKERS = ("replace_with", "your_", "example", "dummy", "changeme")
RATE_LIMIT_MARKERS = ("rate limit", "429", "too many", "quota", "ratelimit", "resource_exhausted")
AUTH_MARKERS = ("401", "403", "authentication", "invalid api key", "unauthorized", "permission", "api_key_invalid", "invalid_argument",)
HARD_QUOTA_MARKERS = ("insufficient_quota", "billing", "payment", "exceeded your current quota")
TIMEOUT_MARKERS = ("timeout", "timed out", "read timeout", "connect timeout", "deadline exceeded", "operation timed out")
SERVER_ERROR_MARKERS = ("500", "502", "503", "504", "bad gateway", "service unavailable", "connection reset", "server error", "internal server error", "overloaded")


class LLMSystemError(RuntimeError):
    """Raised when a call fails due to a system/API/network issue (not model output quality).

    Carries a ``system_status`` string so the runner can map directly to a
    benchmark status code without parsing the error message.
    """
    def __init__(self, message: str, system_status: str):
        super().__init__(message)
        self.system_status = system_status


class EmptyResponseError(RuntimeError):
    """Raised when the API returns a structurally valid response but with empty content.

    Treated as a retriable system failure (SYSTEM_EMPTY_RESPONSE), not a model
    output quality issue.  Never silently swallowed — always surfaces through
    the retry loop so the run is marked as failed rather than writing an empty
    chat log and reporting success.
    """


@dataclass
class ProviderConfig:
    name: str
    provider_type: str
    model_id: str
    keys: list[str]
    display_name: str
    base_url: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    rotate_after_success: bool = True
    temperature: float = DEFAULT_TEMPERATURE
    extra_params: dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class KeyRotator:
    def __init__(self, provider_name: str, keys: list[str]):
        if not keys:
            raise ValueError(f"[{provider_name}] no usable API keys.")
        self.provider_name = provider_name
        self.keys = keys
        self.index = 0
        self.exhausted: set[str] = set()
        self.cooldowns: dict[str, float] = {}

    def current(self) -> str | None:
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.index]
            if key not in self.exhausted and now >= self.cooldowns.get(key, 0):
                return key
            self.index = (self.index + 1) % len(self.keys)
        return None

    def rotate(self, cooldown_sec: float = 0) -> None:
        key = self.keys[self.index]
        if cooldown_sec > 0:
            self.cooldowns[key] = time.time() + cooldown_sec
        self.index = (self.index + 1) % len(self.keys)

    def mark_exhausted(self, key: str) -> None:
        self.exhausted.add(key)

    def all_exhausted(self) -> bool:
        return len(self.exhausted) >= len(self.keys)


class LLMClient:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.rotator = KeyRotator(config.name, config.keys)
        self.call_count = 0

    @property
    def provider(self) -> str:
        return self.config.name

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def display_name(self) -> str:
        return self.config.display_name

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Call the LLM and return the response text.

        Retries on system/API/network errors (rate limits, timeouts, 5xx).
        Raises ``LLMSystemError`` on exhaustion – never retries model-output issues.

        ``extra_params`` is merged on top of ``self.config.extra_params`` for this
        call only; call-level keys take priority over config-level keys.
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        merged_extra_params: dict[str, Any] = {**self.config.extra_params, **(extra_params or {})}
        backoff = 5.0
        max_attempts = max(5, len(self.config.keys) * 2)
        attempt = 0
        last_system_status = "SYSTEM_REQUEST_FAILED"

        while attempt < max_attempts:
            key = self.rotator.current()
            if key is None:
                wait = max(self.rotator.cooldowns.values(), default=time.time()) - time.time() + 1
                time.sleep(max(wait, 1))
                continue

            try:
                started = time.time()
                text = self._dispatch(key, messages, max_tokens, temperature, timeout, merged_extra_params)
                self.call_count += 1
                logger.info(
                    "[%s] OK model=%s key=...%s latency=%.1fs calls=%d",
                    self.provider,
                    self.model_id,
                    key[-6:],
                    time.time() - started,
                    self.call_count,
                )
                if self.config.rotate_after_success and len(self.config.keys) > 1:
                    self.rotator.rotate()
                return text
            except Exception as exc:
                err = str(exc).lower()
                if any(marker in err for marker in AUTH_MARKERS + HARD_QUOTA_MARKERS):
                    logger.warning("[%s] key rejected/exhausted: %s", self.provider, exc)
                    self.rotator.mark_exhausted(key)
                    self.rotator.rotate()
                    if self.rotator.all_exhausted():
                        raise LLMSystemError(
                            f"[{self.provider}] all API keys are exhausted/rejected.",
                            "SYSTEM_ALL_KEYS_EXHAUSTED",
                        ) from exc
                    last_system_status = "SYSTEM_API_ERROR"
                elif any(marker in err for marker in RATE_LIMIT_MARKERS):
                    wait = min(backoff + random.uniform(0, backoff * 0.3), 120)
                    logger.warning("[%s] rate/quota limit; rotate key and wait %.1fs", self.provider, wait)
                    self.rotator.rotate(cooldown_sec=wait)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                    last_system_status = "SYSTEM_RATE_LIMIT"
                elif any(marker in err for marker in TIMEOUT_MARKERS):
                    logger.warning("[%s] request timed out on attempt %d: %s", self.provider, attempt + 1, exc)
                    time.sleep(min(backoff, 60))
                    backoff = min(backoff * 1.5, 120)
                    last_system_status = "SYSTEM_TIMEOUT"
                elif any(marker in err for marker in SERVER_ERROR_MARKERS):
                    logger.warning("[%s] server error on attempt %d: %s", self.provider, attempt + 1, exc)
                    time.sleep(min(backoff, 60))
                    backoff = min(backoff * 1.5, 120)
                    last_system_status = "SYSTEM_API_ERROR"
                elif isinstance(exc, EmptyResponseError):
                    logger.warning(
                        "[%s] empty response on attempt %d (model=%s): %s",
                        self.provider, attempt + 1, self.model_id, exc,
                    )
                    time.sleep(min(backoff, 30))
                    backoff = min(backoff * 1.5, 120)
                    last_system_status = "SYSTEM_EMPTY_RESPONSE"
                else:
                    logger.warning("[%s] request failed on attempt %d: %s", self.provider, attempt + 1, exc)
                    time.sleep(min(backoff, 60))
                    backoff = min(backoff * 1.5, 120)
            attempt += 1

        raise LLMSystemError(
            f"[{self.provider}] failed after {max_attempts} attempts.",
            last_system_status,
        )

    def _dispatch(
        self,
        key: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        timeout: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        if self.config.provider_type == "openai_compatible":
            return _call_openai_compatible(
                key, self.config.base_url, self.model_id, messages,
                max_tokens, temperature, extra_params, timeout=timeout,
            )
        if self.config.provider_type == "anthropic":
            return _call_anthropic(key, self.model_id, messages, max_tokens, temperature, timeout=timeout)
        if self.config.provider_type == "google":
            return _call_google(key, self.model_id, messages, max_tokens, temperature, timeout=timeout)
        raise ValueError(f"Unsupported provider_type: {self.config.provider_type}")


class ClientManager:
    def __init__(self, keys_path: Path):
        self.keys_path = keys_path
        self.clients = self._load_clients(keys_path)

    def get(self, provider: str) -> LLMClient:
        if provider not in self.clients:
            raise KeyError(f"Provider '{provider}' not configured. Available: {list(self.clients)}")
        return self.clients[provider]

    def list_providers(self) -> list[str]:
        return list(self.clients)

    def _load_clients(self, keys_path: Path) -> dict[str, LLMClient]:
        if not keys_path.exists():
            raise FileNotFoundError(f"API key config not found: {keys_path}")

        raw = json.loads(keys_path.read_text(encoding="utf-8"))
        clients: dict[str, LLMClient] = {}
        for name, cfg in raw.items():
            if name.startswith("_") or cfg.get("enabled", True) is False:
                continue
            try:
                provider_cfg = _parse_provider_config(name, cfg)
                clients[name] = LLMClient(provider_cfg)
                logger.info("Loaded provider=%s model=%s keys=%d", name, provider_cfg.model_id, len(provider_cfg.keys))
            except Exception as exc:
                logger.warning("Skip provider '%s': %s", name, exc)

        if not clients:
            raise ValueError(f"No valid providers in {keys_path}. Fill real keys or env:VARIABLE_NAME entries.")
        return clients


def _parse_provider_config(name: str, cfg: dict[str, Any]) -> ProviderConfig:
    provider_type = _required_str(cfg, "provider_type", name)
    model_id = _required_str(cfg, "model_id", name)
    keys = _normalise_keys(cfg.get("keys", []), name)
    return ProviderConfig(
        name=name,
        provider_type=provider_type,
        model_id=model_id,
        keys=keys,
        display_name=str(cfg.get("display_name") or _default_display_name(name, model_id)),
        base_url=cfg.get("base_url"),
        max_tokens=int(cfg.get("max_tokens", DEFAULT_MAX_TOKENS)),
        rotate_after_success=bool(cfg.get("rotate_after_success", True)),
        temperature=float(cfg.get("temperature", DEFAULT_TEMPERATURE)),
        extra_params=dict(cfg.get("extra_params", {})),
    )


def _required_str(cfg: dict[str, Any], field: str, provider_name: str) -> str:
    value = cfg.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"[{provider_name}] missing required field '{field}'.")
    return value.strip()


def _normalise_keys(raw_keys: list[Any], provider_name: str) -> list[str]:
    keys: list[str] = []
    for raw in raw_keys:
        if not isinstance(raw, str):
            continue
        key = raw.strip()
        if key.lower().startswith("env:"):
            env_name = key.split(":", 1)[1].strip()
            env_val = os.getenv(env_name, "").strip()
            if not env_val:
                logger.warning("[%s] env key is not set: %s", provider_name, env_name)
                continue
            # Tach nhieu keys trong 1 bien env
            candidates = [k.strip() for k in env_val.split(",") if k.strip()]
        else:
            candidates = [key]

        for k in candidates:
            if k and not any(marker in k.lower() for marker in PLACEHOLDER_MARKERS):
                keys.append(k)
    return keys


def _default_display_name(provider_name: str, model_id: str) -> str:
    text = "".join(ch for ch in model_id if ch.isalnum())
    return text or provider_name


def _safe_trunc_repr(obj: Any, limit: int = 500) -> str:
    """Return repr(obj) truncated to *limit* characters. Never raises."""
    try:
        r = repr(obj)
    except Exception:
        r = "<repr-failed>"
    return r if len(r) <= limit else r[:limit] + f"…[+{len(r) - limit} chars]"


def _log_empty_response_debug(model: str, response: Any, stage: str) -> None:
    """Log raw response fields to diagnose empty 200-OK responses.

    Called right before raising EmptyResponseError so the retry loop is
    unaffected.  Never raises; safe even when the response is partially
    malformed.  Does NOT log API keys.
    """
    try:
        choice = response.choices[0] if (response and response.choices) else None
        msg = getattr(choice, "message", None) if choice is not None else None
        logger.warning(
            "[%s] empty-response debug  stage=%s  resp_model=%s  "
            "finish_reason=%s  msg_role=%s  content_type=%s  "
            "content_repr=%s  reasoning_content=%s  usage=%s",
            model,
            stage,
            _safe_trunc_repr(getattr(response, "model", None)),
            _safe_trunc_repr(getattr(choice, "finish_reason", None))
            if choice is not None else "no-choice",
            _safe_trunc_repr(getattr(msg, "role", None))
            if msg is not None else "no-message",
            type(getattr(msg, "content", None)).__name__
            if msg is not None else "no-message",
            _safe_trunc_repr(getattr(msg, "content", None))
            if msg is not None else "no-message",
            _safe_trunc_repr(getattr(msg, "reasoning_content", None))
            if msg is not None else "no-message",
            _safe_trunc_repr(getattr(response, "usage", None)),
        )
    except Exception as log_exc:
        logger.warning("[%s] empty-response debug failed to run: %s", model, log_exc)


def _call_openai_compatible(
    key: str,
    base_url: str | None,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    extra_params: dict[str, Any] | None = None,
    timeout: int | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=timeout)
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_params or {},
    )
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    response = client.chat.completions.create(**kwargs)

    if not response or not response.choices:
        raise EmptyResponseError(
            f"[{model}] API returned a response with no choices."
        )
    message = response.choices[0].message
    if message is None:
        _log_empty_response_debug(model, response, "message_is_none")
        raise EmptyResponseError(f"[{model}] API response choice has no message object.")

    content = message.content
    if content is None:
        _log_empty_response_debug(model, response, "content_is_none")
        raise EmptyResponseError(f"[{model}] API response message.content is None.")

    # Some providers return content as a list of typed content blocks.
    if isinstance(content, list):
        logger.debug(
            "[%s] content is a list with %d block(s): types=%s",
            model,
            len(content),
            [_safe_trunc_repr(
                b.get("type", "?") if isinstance(b, dict) else getattr(b, "type", type(b).__name__),
                80,
            ) for b in content],
        )
        parts = [
            block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            for block in content
        ]
        text = "".join(str(p) for p in parts if p)
    else:
        text = str(content)

    text = text.strip()
    if not text:
        _log_empty_response_debug(model, response, "empty_after_join")
        raise EmptyResponseError(
            f"[{model}] API returned an empty or whitespace-only response."
        )
    return text


def _call_anthropic(
    key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int | None = None,
) -> str:
    import anthropic

    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            chat_messages.append(message)

    client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=timeout)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    response = client.messages.create(**kwargs)
    text = "".join(getattr(part, "text", "") for part in response.content).strip()
    if not text:
        raise EmptyResponseError(
            f"[{model}] Anthropic API returned an empty or whitespace-only response."
        )
    return text


def _call_google(
    key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int | None = None,
) -> str:
    # Google generativeai SDK does not expose a simple per-request timeout via
    # this interface; the parameter is accepted to keep the _dispatch signature
    # uniform but is intentionally ignored here.
    del timeout
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    genai.configure(api_key=key)
    gen_model = genai.GenerativeModel(
        model_name=model,
        generation_config=GenerationConfig(temperature=temperature, max_output_tokens=max_tokens),
    )

    pending_system = ""
    history: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "system":
            pending_system = f"{pending_system}\n\n{message['content']}".strip()
            continue
        role = "user" if message["role"] == "user" else "model"
        content = message["content"]
        if role == "user" and pending_system:
            content = f"{pending_system}\n\n{content}"
            pending_system = ""
        history.append({"role": role, "parts": [content]})

    if not history or history[-1]["role"] != "user":
        raise ValueError("Google API requires the final message to be a user message.")

    prior_history, last_turn = history[:-1], history[-1]
    chat = gen_model.start_chat(history=prior_history)
    result = chat.send_message(last_turn["parts"][0])
    text = (result.text or "").strip()
    if not text:
        raise EmptyResponseError(
            f"[{model}] Google API returned an empty or whitespace-only response."
        )
    return text
