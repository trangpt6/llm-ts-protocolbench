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

from .settings import DEFAULT_MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

PLACEHOLDER_MARKERS = ("replace_with", "your_", "example", "dummy", "changeme")
RATE_LIMIT_MARKERS = ("rate limit", "429", "too many", "quota", "ratelimit", "resource_exhausted")
AUTH_MARKERS = ("401", "403", "authentication", "invalid api key", "unauthorized", "permission", "api_key_invalid", "invalid_argument",)
HARD_QUOTA_MARKERS = ("insufficient_quota", "billing", "payment", "exceeded your current quota")


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

    def chat(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        max_tokens = max_tokens or self.config.max_tokens
        backoff = 5.0
        max_attempts = max(5, len(self.config.keys) * 2)
        attempt = 0

        while attempt < max_attempts:
            key = self.rotator.current()
            if key is None:
                wait = max(self.rotator.cooldowns.values(), default=time.time()) - time.time() + 1
                time.sleep(max(wait, 1))
                continue

            try:
                started = time.time()
                text = self._dispatch(key, messages, max_tokens)
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
                        raise RuntimeError(f"[{self.provider}] all API keys are exhausted/rejected.") from exc
                elif any(marker in err for marker in RATE_LIMIT_MARKERS):
                    wait = min(backoff + random.uniform(0, backoff * 0.3), 120)
                    logger.warning("[%s] rate/quota limit; rotate key and wait %.1fs", self.provider, wait)
                    self.rotator.rotate(cooldown_sec=wait)
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                else:
                    logger.warning("[%s] request failed on attempt %d: %s", self.provider, attempt + 1, exc)
                    time.sleep(min(backoff, 60))
                    backoff = min(backoff * 1.5, 120)
            attempt += 1

        raise RuntimeError(f"[{self.provider}] failed after {max_attempts} attempts.")

    def _dispatch(self, key: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        if self.config.provider_type == "openai_compatible":
            return _call_openai_compatible(key, self.config.base_url, self.model_id, messages, max_tokens)
        if self.config.provider_type == "anthropic":
            return _call_anthropic(key, self.model_id, messages, max_tokens)
        if self.config.provider_type == "google":
            return _call_google(key, self.model_id, messages, max_tokens)
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


def _call_openai_compatible(
    key: str,
    base_url: str | None,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    import anthropic

    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            chat_messages.append(message)

    client = anthropic.Anthropic(api_key=key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    response = client.messages.create(**kwargs)
    return "".join(getattr(part, "text", "") for part in response.content)


def _call_google(key: str, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    genai.configure(api_key=key)
    gen_model = genai.GenerativeModel(
        model_name=model,
        generation_config=GenerationConfig(temperature=TEMPERATURE, max_output_tokens=max_tokens),
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
    return result.text or ""
