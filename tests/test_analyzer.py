import pytest

from app import analyzer
from app.models import CoverLetter, GapAnalysis, OverallMatch, TailoredResume
from app.utils import InputError, load_example

RESUME = load_example("sample_resume")
JOB = load_example("sample_job_description")

FAKE_ANALYSIS = GapAnalysis(
    key_requirements=["FinTech experience", "KPI dashboards"],
    relevant_experience=["Built 12 Power BI dashboards (Northwind Municipal Services)"],
    gaps_and_mismatches=["No FinTech or financial services experience stated"],
    potential_strengths=["Customer review sentiment project"],
    overall_match=OverallMatch(match_level="Moderate", summary="Solid analytics, no FinTech.", priority_actions=[]),
)


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []

    def fake_generate(prompt, schema, **kwargs):
        calls.append({"prompt": prompt, "schema": schema, **kwargs})
        if schema is GapAnalysis:
            return FAKE_ANALYSIS
        if schema is TailoredResume:
            return TailoredResume(
                updated_resume="**John Doe**\nTailored resume highlighting KPI dashboards and campaign analytics.",
                change_notes=["Reordered skills"],
            )
        return CoverLetter(cover_letter="Dear Hiring Manager, ...")

    monkeypatch.setattr(analyzer, "generate_structured", fake_generate)
    return calls


def test_analyze_resume_builds_prompt_with_both_documents(fake_llm):
    result = analyzer.analyze_resume(RESUME, JOB, provider="anthropic", model="claude-sonnet-4-6", api_key="k")
    assert result is FAKE_ANALYSIS
    call = fake_llm[0]
    assert call["schema"] is GapAnalysis
    assert call["provider"] == "anthropic" and call["model"] == "claude-sonnet-4-6"
    assert "<resume>" in call["prompt"] and "John Doe" in call["prompt"]
    assert "<job_description>" in call["prompt"] and "Example FinTech Corp" in call["prompt"]
    assert "Never invent" in call["prompt"]
    assert "5. Overall Match Analysis" in call["prompt"]


def test_analyze_resume_rejects_empty_inputs_without_calling_llm(fake_llm):
    with pytest.raises(InputError):
        analyzer.analyze_resume("", JOB)
    with pytest.raises(InputError):
        analyzer.analyze_resume(RESUME, "too short")
    assert fake_llm == []


def test_tailored_resume_prompt_includes_analysis_and_honesty_rules(fake_llm):
    analyzer.generate_tailored_resume(RESUME, JOB, FAKE_ANALYSIS, provider="openai", api_key="k")
    prompt = fake_llm[0]["prompt"]
    assert "No FinTech or financial services experience stated" in prompt
    assert "Adding missing skills" not in prompt
    assert "do not create new numbers" in prompt


def test_full_pipeline_chains_tailored_resume_into_cover_letter(fake_llm):
    result = analyzer.run_full_pipeline(RESUME, JOB, provider="openai", api_key="k")
    assert [c["schema"] for c in fake_llm] == [GapAnalysis, TailoredResume, CoverLetter]
    assert "Tailored" in fake_llm[2]["prompt"]
    assert result.cover_letter.cover_letter.startswith("Dear Hiring Manager")


def test_user_text_with_braces_is_not_treated_as_template(fake_llm):
    resume = RESUME + "\nSkills: {python} {sql}"
    analyzer.analyze_resume(resume, JOB, api_key="k")
    assert "{python} {sql}" in fake_llm[0]["prompt"]
