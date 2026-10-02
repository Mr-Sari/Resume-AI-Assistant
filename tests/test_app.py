"""Headless UI tests for the Streamlit app (no API keys or network needed)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import analyzer
from app.models import CoverLetter, GapAnalysis, OverallMatch, TailoredResume

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "app.py")


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def fake_llm(monkeypatch):
    def fake_generate(prompt, schema, **kwargs):
        if schema is GapAnalysis:
            return GapAnalysis(
                key_requirements=["FinTech experience"],
                relevant_experience=["KPI dashboards"],
                gaps_and_mismatches=["No FinTech background"],
                potential_strengths=["Sentiment analysis project"],
                overall_match=OverallMatch(
                    match_level="Moderate", summary="Partial fit.", priority_actions=["Quantify impact"]
                ),
            )
        if schema is TailoredResume:
            return TailoredResume(
                updated_resume="**John Doe**\nData Analyst focused on KPI dashboards and campaign analytics.",
                change_notes=["Moved dashboards to the top"],
            )
        return CoverLetter(cover_letter="Dear Hiring Manager, thank you for considering my application.")

    monkeypatch.setattr(analyzer, "generate_structured", fake_generate)


def test_app_renders_main_sections():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "📄 Resume AI Assistant"
    assert {"Resume text", "Job description text"} <= {t.label for t in at.text_area}
    assert at.sidebar.radio[0].options == ["OpenAI", "Anthropic Claude"]
    assert at.sidebar.selectbox[0].value == "gpt-4o"


def test_missing_api_key_shows_error():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    button(at, "Load fictional sample").click().run()
    button(at, "🔍 Analyze resume").click().run()
    assert not at.exception
    assert any("No API key" in e.value for e in at.error)


def test_empty_input_shows_warning(fake_llm):
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.sidebar.text_input[0].input("test-key").run()
    button(at, "🔍 Analyze resume").click().run()
    assert not at.exception
    assert any("provide the resume" in w.value for w in at.warning)


def test_full_flow_with_fake_llm(fake_llm):
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.sidebar.radio[0].set_value("anthropic").run()
    assert at.sidebar.selectbox[0].value == "claude-sonnet-4-6"
    at.sidebar.text_input[0].input("test-key").run()
    button(at, "Load fictional sample").click().run()
    button(at, "🔍 Analyze resume").click().run()
    assert not at.exception and not at.error

    markdown = "\n".join(m.value for m in at.markdown)
    for heading in (
        "1. Key Requirements",
        "2. Relevant Experience",
        "3. Gaps & Mismatches",
        "4. Potential Strengths",
        "5. Overall Match Analysis",
    ):
        assert heading in markdown
    assert "No FinTech background" in markdown

    button(at, "Generate tailored resume").click().run()
    button(at, "Generate cover letter").click().run()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Tailored resume", "Changes vs. original", "Cover letter"]
    # The key lives only in the password widget's server-side state; it is never rendered.
    rendered = [
        e.value for kind in (at.markdown, at.caption, at.info, at.warning, at.error, at.success, at.code) for e in kind
    ]
    assert rendered and not any("test-key" in str(value) for value in rendered)


def test_require_user_api_key_ignores_server_key(monkeypatch, fake_llm):
    monkeypatch.setenv("REQUIRE_USER_API_KEY", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    button(at, "Load fictional sample").click().run()
    button(at, "🔍 Analyze resume").click().run()
    assert any("No API key" in e.value for e in at.error)
