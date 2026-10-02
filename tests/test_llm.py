try:  # openai>=3 uses httpx2; older SDKs use httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx
import openai
import pytest

from app import llm
from app.llm import LLMError, MissingAPIKeyError, generate_structured, resolve_api_key
from app.models import CoverLetter


def test_missing_api_key_raises_clear_error():
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        resolve_api_key("openai")
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        generate_structured("hi", CoverLetter, provider="anthropic")


def test_explicit_key_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve_api_key("openai") == "env-key"
    assert resolve_api_key("openai", "  user-key ") == "user-key"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        generate_structured("hi", CoverLetter, provider="gemini", api_key="x")


def test_authentication_error_never_leaks_the_key(monkeypatch):
    fake_key = "sk-test-SHOULD-NOT-LEAK-1234567890"

    def raise_auth_error(*args, **kwargs):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(401, request=request)
        raise openai.AuthenticationError(f"Incorrect API key provided: {fake_key}", response=response, body=None)

    monkeypatch.setattr(llm, "_openai_generate", raise_auth_error)
    with pytest.raises(LLMError) as excinfo:
        generate_structured("hi", CoverLetter, provider="openai", api_key=fake_key)
    assert fake_key not in str(excinfo.value)
    assert "Authentication failed" in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_generic_api_error_message_is_redacted(monkeypatch):
    def raise_status_error(*args, **kwargs):
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(500, request=request)
        raise openai.InternalServerError("boom sk-FAKE-TEST-VALUE-0000", response=response, body=None)

    monkeypatch.setattr(llm, "_openai_generate", raise_status_error)
    with pytest.raises(LLMError) as excinfo:
        generate_structured("hi", CoverLetter, provider="openai", api_key="k")
    assert "sk-FAKE-TEST-VALUE-0000" not in str(excinfo.value)
    assert "(500)" in str(excinfo.value)


def test_default_model_is_used_when_none_given(monkeypatch):
    seen = {}

    def fake_claude(prompt, schema, model, api_key, max_tokens):
        seen["model"] = model
        return schema(cover_letter="Dear Hiring Manager")

    monkeypatch.setattr(llm, "_claude_generate", fake_claude)
    result = generate_structured("hi", CoverLetter, provider="anthropic", api_key="k")
    assert result.cover_letter == "Dear Hiring Manager"
    assert seen["model"] == "claude-sonnet-4-6"
