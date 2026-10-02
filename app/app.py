"""Streamlit web interface for the Resume AI Assistant.

Run from the repository root with:  streamlit run app/app.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# `streamlit run app/app.py` puts app/ (not the repo root) on sys.path, where
# this file would shadow the `app` package. Put the repo root first instead.
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if sys.path[0] != ROOT_DIR:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.analyzer import analyze_resume, generate_cover_letter, generate_tailored_resume  # noqa: E402
from app.llm import LLMError, MissingAPIKeyError  # noqa: E402
from app.models import PROVIDERS, GapAnalysis  # noqa: E402
from app.utils import (  # noqa: E402
    MAX_INPUT_CHARS,
    SUPPORTED_UPLOAD_TYPES,
    InputError,
    analysis_to_markdown,
    build_diff_html,
    extract_text_from_file,
    load_example,
    redact_secrets,
)

load_dotenv()

# Set REQUIRE_USER_API_KEY=true on a public deployment so visitors must bring
# their own key and server-side keys (if any) are never used on their behalf.
REQUIRE_USER_API_KEY = os.getenv("REQUIRE_USER_API_KEY", "").strip().lower() in {"1", "true", "yes"}

MATCH_LEVEL_STYLE = {"Strong": st.success, "Moderate": st.warning, "Weak": st.error}
RESULT_KEYS = ("analysis", "analysis_fingerprint", "analysis_run", "tailored", "cover_letter")

st.set_page_config(page_title="Resume AI Assistant", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    for key, default in {"resume_text": "", "job_text": "", "upload_error": None}.items():
        st.session_state.setdefault(key, default)
    for key in RESULT_KEYS:
        st.session_state.setdefault(key, None)


def clear_results() -> None:
    for key in RESULT_KEYS:
        st.session_state[key] = None


def load_sample() -> None:
    st.session_state.resume_text = load_example("sample_resume")
    st.session_state.job_text = load_example("sample_job_description")
    st.session_state.upload_error = None
    clear_results()


def clear_inputs() -> None:
    st.session_state.resume_text = ""
    st.session_state.job_text = ""
    st.session_state.upload_error = None
    clear_results()


def on_resume_upload() -> None:
    uploaded = st.session_state.get("resume_file")
    st.session_state.upload_error = None
    if uploaded is None:
        return
    try:
        text = extract_text_from_file(uploaded.name, uploaded.getvalue())
    except InputError as exc:
        st.session_state.upload_error = str(exc)
        return
    if len(text) > MAX_INPUT_CHARS:
        st.session_state.upload_error = (
            f"The extracted text is too long ({len(text):,} characters; maximum {MAX_INPUT_CHARS:,})."
        )
        return
    st.session_state.resume_text = text


def fingerprint(*texts: str) -> str:
    return hashlib.sha256("\x00".join(t.strip() for t in texts).encode()).hexdigest()


def run_safely(action, spinner_text: str):
    """Run an LLM action, turning known failures into friendly UI messages."""
    try:
        with st.spinner(spinner_text):
            return action()
    except InputError as exc:
        st.warning(str(exc))
    except MissingAPIKeyError as exc:
        st.error(str(exc))
    except LLMError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        st.error(f"Unexpected error: {redact_secrets(type(exc).__name__ + ': ' + str(exc))}")
    return None


# --------------------------------------------------------------------------- #
# Sidebar: provider, model and API key
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")
        provider = st.radio(
            "AI provider",
            options=list(PROVIDERS),
            format_func=lambda p: PROVIDERS[p].label,
        )
        config = PROVIDERS[provider]
        model = st.selectbox("Model", config.models, key=f"model_{provider}")

        user_key = st.text_input(
            f"{config.label} API key",
            type="password",
            key=f"api_key_{provider}",
            placeholder="Paste your own key (optional)" if not REQUIRE_USER_API_KEY else "Required",
            help="Used only for requests in this browser session. It is not stored, logged or shown.",
        ).strip()
        server_key_available = bool(os.getenv(config.env_var)) and not REQUIRE_USER_API_KEY

        if user_key:
            st.caption("🔐 Using the key you entered (this session only).")
        elif server_key_available:
            st.caption(f"🔐 Using the server's `{config.env_var}`.")
        else:
            st.info(
                f"Enter a {config.label} API key above "
                f"([get one here]({config.key_url})) or set `{config.env_var}` on the server."
            )

        st.divider()
        st.markdown(
            "**Privacy:** your resume and job description are sent only to the selected "
            "AI provider for this request and are not saved by this app."
        )

    return {
        "provider": provider,
        "model": model,
        "api_key": user_key or None,
        "has_key": bool(user_key) or server_key_available,
    }


def require_key(settings: dict) -> None:
    if not settings["has_key"]:
        config = PROVIDERS[settings["provider"]]
        hint = "" if REQUIRE_USER_API_KEY else f" or set `{config.env_var}` in your environment / .env file"
        raise MissingAPIKeyError(f"No API key available for {config.label}. Enter one in the sidebar{hint}.")


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def render_list_section(title: str, items: list[str], empty_text: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        if items:
            st.markdown("\n".join(f"- {item}" for item in items))
        else:
            st.caption(empty_text)


def render_analysis(analysis: GapAnalysis, run_info: dict) -> None:
    st.subheader("📊 Analysis Results")
    st.caption(f"Generated with {PROVIDERS[run_info['provider']].label} · `{run_info['model']}`")

    overall = analysis.overall_match
    with st.container(border=True):
        st.markdown("#### 5. Overall Match Analysis")
        MATCH_LEVEL_STYLE.get(overall.match_level, st.info)(f"**Match level: {overall.match_level}**")
        st.markdown(overall.summary)
        if overall.priority_actions:
            st.markdown("**Priority actions**")
            st.markdown("\n".join(f"- {item}" for item in overall.priority_actions))

    left, right = st.columns(2)
    with left:
        render_list_section("1. Key Requirements", analysis.key_requirements, "No requirements identified.")
        render_list_section("3. Gaps & Mismatches", analysis.gaps_and_mismatches, "No gaps identified.")
    with right:
        render_list_section("2. Relevant Experience", analysis.relevant_experience, "No matching experience found.")
        render_list_section("4. Potential Strengths", analysis.potential_strengths, "No strengths identified.")

    st.download_button(
        "⬇️ Download analysis (.md)",
        data=analysis_to_markdown(analysis),
        file_name="resume_gap_analysis.md",
        mime="text/markdown",
    )


def render_generation_tools(settings: dict, resume: str, job_text: str) -> None:
    st.subheader("✍️ Optional: Tailored Resume & Cover Letter")
    st.caption(
        "Rewrites the resume around this job using only facts already in it, then drafts a cover letter. "
        "Always review the output before using it."
    )
    col_resume, col_letter = st.columns(2)
    analysis: GapAnalysis = st.session_state.analysis

    if col_resume.button("Generate tailored resume", width="stretch"):

        def action():
            require_key(settings)
            return generate_tailored_resume(
                resume,
                job_text,
                analysis,
                provider=settings["provider"],
                model=settings["model"],
                api_key=settings["api_key"],
            )

        st.session_state.tailored = run_safely(action, "Rewriting your resume…") or st.session_state.tailored

    if col_letter.button("Generate cover letter", width="stretch"):
        tailored = st.session_state.tailored
        source_resume = tailored.updated_resume if tailored else resume

        def action():
            require_key(settings)
            return generate_cover_letter(
                job_text,
                source_resume,
                provider=settings["provider"],
                model=settings["model"],
                api_key=settings["api_key"],
            )

        st.session_state.cover_letter = (
            run_safely(action, "Writing your cover letter…") or st.session_state.cover_letter
        )

    tailored = st.session_state.tailored
    letter = st.session_state.cover_letter
    if not (tailored or letter):
        return

    tab_names = []
    if tailored:
        tab_names += ["Tailored resume", "Changes vs. original"]
    if letter:
        tab_names.append("Cover letter")
    tabs = dict(zip(tab_names, st.tabs(tab_names), strict=True))

    if tailored:
        with tabs["Tailored resume"]:
            st.markdown(tailored.updated_resume)
            st.download_button(
                "⬇️ Download resume (.md)", tailored.updated_resume, "tailored_resume.md", "text/markdown"
            )
        with tabs["Changes vs. original"]:
            if tailored.change_notes:
                st.markdown("**What changed and why**")
                st.markdown("\n".join(f"- {note}" for note in tailored.change_notes))
            st.markdown("**Word-level diff** — :green[added] · :red[~~removed~~]")
            st.html(build_diff_html(resume, tailored.updated_resume))
    if letter:
        with tabs["Cover letter"]:
            if not tailored:
                st.caption("Based on the original resume. Generate a tailored resume first to base it on that instead.")
            st.markdown(letter.cover_letter)
            st.download_button("⬇️ Download cover letter (.md)", letter.cover_letter, "cover_letter.md", "text/markdown")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    settings = render_sidebar()

    st.title("📄 Resume AI Assistant")
    st.markdown(
        "Compare a resume with a job description using **OpenAI** or **Anthropic Claude**. "
        "Get the key requirements, your relevant experience, gaps, strengths and an overall match "
        "assessment, then optionally generate a tailored resume and cover letter."
    )

    with st.expander("🔑 How to configure API keys"):
        st.markdown(
            "- **Running locally:** copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and/or "
            "`ANTHROPIC_API_KEY`. Never commit your `.env` file.\n"
            "- **Streamlit Community Cloud:** add the keys under *App settings → Secrets*.\n"
            "- **Visitors:** paste your own key in the sidebar. It is kept only in memory for your "
            "session and is never stored or displayed."
        )

    btn_sample, btn_clear, _ = st.columns([1.4, 1, 3.6])
    btn_sample.button("Load fictional sample", on_click=load_sample, width="stretch")
    btn_clear.button("Clear", on_click=clear_inputs, width="stretch")

    col_resume, col_job = st.columns(2)
    with col_resume:
        st.subheader("1️⃣ Resume")
        st.file_uploader(
            "Upload a resume (.txt, .md, .pdf) or paste it below",
            type=list(SUPPORTED_UPLOAD_TYPES),
            key="resume_file",
            on_change=on_resume_upload,
        )
        if st.session_state.upload_error:
            st.warning(st.session_state.upload_error)
        st.text_area(
            "Resume text",
            key="resume_text",
            height=380,
            max_chars=MAX_INPUT_CHARS,
            placeholder="Paste the resume here…",
        )
    with col_job:
        st.subheader("2️⃣ Job Description")
        st.text_area(
            "Job description text",
            key="job_text",
            height=470,
            max_chars=MAX_INPUT_CHARS,
            placeholder="Paste the job description here…",
        )

    resume = st.session_state.resume_text
    job_text = st.session_state.job_text

    if st.button("🔍 Analyze resume", type="primary", width="stretch"):
        clear_results()

        def action():
            require_key(settings)
            return analyze_resume(
                resume,
                job_text,
                provider=settings["provider"],
                model=settings["model"],
                api_key=settings["api_key"],
            )

        analysis = run_safely(action, "Analyzing the resume against the job description…")
        if analysis is not None:
            st.session_state.analysis = analysis
            st.session_state.analysis_fingerprint = fingerprint(resume, job_text)
            st.session_state.analysis_run = {"provider": settings["provider"], "model": settings["model"]}

    if st.session_state.analysis is None:
        return

    st.divider()
    if st.session_state.analysis_fingerprint != fingerprint(resume, job_text):
        st.warning("The resume or job description changed since this analysis. Click **Analyze resume** to refresh it.")
    render_analysis(st.session_state.analysis, st.session_state.analysis_run)
    st.divider()
    render_generation_tools(settings, resume, job_text)


main()
