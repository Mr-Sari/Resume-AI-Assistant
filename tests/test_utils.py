import pytest

from app.models import GapAnalysis, OverallMatch
from app.utils import (
    InputError,
    analysis_to_markdown,
    build_diff_html,
    extract_text_from_file,
    load_example,
    redact_secrets,
    validate_text,
)


def test_validate_text_rejects_empty_and_short_input():
    with pytest.raises(InputError, match="provide the resume"):
        validate_text("   ", "resume")
    with pytest.raises(InputError, match="too short"):
        validate_text("Python, SQL", "resume")


def test_validate_text_rejects_overlong_input():
    with pytest.raises(InputError, match="too long"):
        validate_text("x" * 30_001, "job description")


def test_validate_text_strips_whitespace():
    text = "  " + "a" * 60 + "\n"
    assert validate_text(text, "resume") == "a" * 60


def test_extract_text_from_markdown_and_unsupported_types():
    assert extract_text_from_file("cv.md", "# John Doe\nAnalyst".encode()) == "# John Doe\nAnalyst"
    with pytest.raises(InputError, match="Unsupported file type"):
        extract_text_from_file("cv.exe", b"MZ")
    with pytest.raises(InputError, match="No text"):
        extract_text_from_file("cv.txt", b"   ")


def test_extract_text_from_invalid_pdf():
    with pytest.raises(InputError):
        extract_text_from_file("cv.pdf", b"not a real pdf")


def test_redact_secrets_masks_keys():
    text = "key sk-proj-FAKE-TEST-0000 and Bearer FAKE-TEST-TOKEN-0000"
    redacted = redact_secrets(text)
    assert "FAKE-TEST" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_diff_html_marks_changes_and_escapes_html():
    diff = build_diff_html("Built <b>dashboards</b> in Excel", "Built <b>dashboards</b> in Power BI")
    assert '<span class="diff-del">Excel</span>' in diff
    assert '<span class="diff-add">Power BI</span>' in diff
    assert "<b>" not in diff and "&lt;b&gt;" in diff


def test_analysis_to_markdown_has_all_sections():
    analysis = GapAnalysis(
        key_requirements=["SQL"],
        relevant_experience=["Wrote SQL ETL jobs"],
        gaps_and_mismatches=[],
        potential_strengths=["Dashboards"],
        overall_match=OverallMatch(match_level="Moderate", summary="Partial fit.", priority_actions=["Add metrics"]),
    )
    md = analysis_to_markdown(analysis)
    for heading in (
        "Key Requirements",
        "Relevant Experience",
        "Gaps & Mismatches",
        "Potential Strengths",
        "Overall Match",
    ):
        assert heading in md
    assert "- None identified." in md
    assert "**Match level:** Moderate" in md


def test_examples_are_fictional():
    resume = load_example("sample_resume")
    assert "john.doe@example.com" in resume
    assert "Fictional" in resume
    assert load_example("sample_job_description").strip()
