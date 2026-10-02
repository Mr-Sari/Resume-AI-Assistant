"""Provider-agnostic structured generation for OpenAI and Anthropic Claude.

This replaces the notebook's ``openai_generate`` and ``claude_generate``
helpers. Differences from the notebook:

* Errors raise :class:`LLMError` with a user-safe message instead of being
  returned as ``"Error generating text: ..."`` strings that callers could
  mistake for model output.
* Claude uses the SDK's native structured outputs (``messages.parse``)
  instead of asking for JSON in the prompt and calling ``json.loads`` on the
  raw text, which broke whenever the model wrapped its JSON in code fences.
* API keys are read from the environment (or passed in explicitly) on every
  call and are never logged, stored or included in error messages.
"""

from __future__ import annotations

import os
from typing import TypeVar

import anthropic
import openai
from pydantic import BaseModel, ValidationError

from app.models import PROVIDERS, Provider
from app.utils import redact_secrets

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# System prompt carried over from the notebook's generation helpers.
SYSTEM_PROMPT = "You are a helpful assistant specializing in resume writing and career advice."

DEFAULT_MAX_TOKENS = 8000
REQUEST_TIMEOUT_SECONDS = 180.0


class LLMError(RuntimeError):
    """A generation failure with a message that is safe to show to users."""


class MissingAPIKeyError(LLMError):
    """Raised when no API key is configured for the selected provider."""


def resolve_api_key(provider: Provider, api_key: str | None = None) -> str:
    """Return the explicit key if given, otherwise the provider's env variable."""
    config = PROVIDERS[provider]
    key = (api_key or os.getenv(config.env_var) or "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"No API key found for {config.label}. Set {config.env_var} in your "
            f"environment or .env file, or enter your own key in the sidebar."
        )
    return key


def generate_structured(
    prompt: str,
    schema: type[SchemaT],
    *,
    provider: Provider,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SchemaT:
    """Send ``prompt`` to the selected provider and parse the reply into ``schema``.

    ``temperature`` applies to OpenAI only: the current Anthropic SDK no longer
    accepts sampling parameters, so Claude models use their defaults.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider!r}")
    key = resolve_api_key(provider, api_key)
    model = model or PROVIDERS[provider].default_model

    try:
        if provider == "openai":
            return _openai_generate(prompt, schema, model, key, temperature, max_tokens)
        return _claude_generate(prompt, schema, model, key, max_tokens)
    except LLMError:
        raise
    except (openai.AuthenticationError, anthropic.AuthenticationError):
        # Never echo the provider message here: it can contain a partial key.
        raise LLMError(
            f"Authentication failed for {PROVIDERS[provider].label}. Check that your API key is valid."
        ) from None
    except (openai.PermissionDeniedError, anthropic.PermissionDeniedError):
        raise LLMError(f"Your API key does not have access to the model '{model}'.") from None
    except (openai.NotFoundError, anthropic.NotFoundError):
        raise LLMError(f"The model '{model}' was not found or is not available to your account.") from None
    except (openai.RateLimitError, anthropic.RateLimitError):
        raise LLMError("Rate limit or quota exceeded. Please wait a moment and try again.") from None
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        raise LLMError("Could not reach the AI provider. Check your network connection and try again.") from None
    except (openai.APIStatusError, anthropic.APIStatusError) as exc:
        message = redact_secrets(getattr(exc, "message", str(exc)))
        raise LLMError(f"The AI provider returned an error ({exc.status_code}): {message}") from None
    except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError):
        raise LLMError("The model's response was cut off or filtered. Try shorter inputs.") from None
    except ValidationError:
        raise LLMError("The model's response did not match the expected format. Please try again.") from None


def _openai_generate(
    prompt: str,
    schema: type[SchemaT],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> SchemaT:
    client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format=schema,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise LLMError(f"The model declined the request: {message.refusal}")
    if message.parsed is None:
        raise LLMError("The model returned an empty or invalid response. Please try again.")
    return message.parsed


def _claude_generate(
    prompt: str,
    schema: type[SchemaT],
    model: str,
    api_key: str,
    max_tokens: int,
) -> SchemaT:
    client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS)
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    if response.stop_reason == "refusal":
        raise LLMError("The model declined to process this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("The model's response was cut off. Try a shorter resume or job description.")
    if response.parsed_output is None:
        raise LLMError("The model returned an empty or invalid response. Please try again.")
    return response.parsed_output
