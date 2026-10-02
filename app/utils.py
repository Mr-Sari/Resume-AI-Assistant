"""Input handling, rendering and safety helpers (no network or LLM calls)."""

from __future__ import annotations

import difflib
import html
import io
import re
from pathlib import Path

from app.models import GapAnalysis

MIN_INPUT_CHARS = 50
MAX_INPUT_CHARS = 30_000
SUPPORTED_UPLOAD_TYPES = ("txt", "md", "pdf")

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
)


class InputError(ValueError):
    """Raised when user-provided input is missing or unusable."""


def redact_secrets(text: str) -> str:
    """Mask anything that looks like an API key or bearer token."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def validate_text(text: str | None, label: str) -> str:
    """Strip and length-check one input field, raising :class:`InputError` if unusable."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise InputError(f"Please provide the {label}.")
    if len(cleaned) < MIN_INPUT_CHARS:
        raise InputError(f"The {label} looks too short (minimum {MIN_INPUT_CHARS} characters).")
    if len(cleaned) > MAX_INPUT_CHARS:
        raise InputError(f"The {label} is too long ({len(cleaned):,} characters; maximum {MAX_INPUT_CHARS:,}).")
    return cleaned


def extract_text_from_file(file_name: str, data: bytes) -> str:
    """Extract plain text from an uploaded .txt, .md or .pdf file."""
    extension = Path(file_name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_UPLOAD_TYPES:
        raise InputError(
            f"Unsupported file type '.{extension}'. Use one of: "
            + ", ".join(f".{ext}" for ext in SUPPORTED_UPLOAD_TYPES)
        )

    if extension == "pdf":
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, ValueError) as exc:
            raise InputError(f"Could not read the PDF file: {exc}") from None
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")

    text = text.strip()
    if not text:
        raise InputError(
            "No text could be extracted from the file. Scanned/image-only PDFs are not "
            "supported; please paste the resume text instead."
        )
    return text


def load_example(name: str) -> str:
    """Load a bundled fictional example (``sample_resume`` or ``sample_job_description``)."""
    return (EXAMPLES_DIR / f"{name}.md").read_text(encoding="utf-8")


def analysis_to_markdown(analysis: GapAnalysis) -> str:
    """Render a gap analysis in the notebook's numbered-section Markdown format."""

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) or "- None identified."

    overall = analysis.overall_match
    return (
        "### 1. Key Requirements from Job Description\n"
        f"{bullets(analysis.key_requirements)}\n\n"
        "### 2. Relevant Experience in Resume\n"
        f"{bullets(analysis.relevant_experience)}\n\n"
        "### 3. Gaps & Mismatches\n"
        f"{bullets(analysis.gaps_and_mismatches)}\n\n"
        "### 4. Potential Strengths\n"
        f"{bullets(analysis.potential_strengths)}\n\n"
        "### 5. Overall Match Analysis\n"
        f"**Match level:** {overall.match_level}\n\n"
        f"{overall.summary}\n\n"
        "**Priority actions:**\n"
        f"{bullets(overall.priority_actions)}\n"
    )


def build_diff_html(original: str, updated: str) -> str:
    """Return a word-level HTML diff: additions in green, removals struck through in red.

    The notebook asked the model to produce this HTML itself. Computing it
    locally is deterministic, always accurate, and (because every token is
    HTML-escaped) safe to render in a public web app.
    """
    old_tokens = re.split(r"(\s+)", original)
    new_tokens = re.split(r"(\s+)", updated)
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)

    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        removed = html.escape("".join(old_tokens[i1:i2]))
        added = html.escape("".join(new_tokens[j1:j2]))
        if tag == "equal":
            parts.append(added)
            continue
        if tag in ("delete", "replace") and removed.strip():
            parts.append(f'<span class="diff-del">{removed}</span>')
        if tag in ("insert", "replace") and added.strip():
            parts.append(f'<span class="diff-add">{added}</span>')
        elif tag == "insert":
            parts.append(added)

    return (
        "<style>"
        ".resume-diff{white-space:pre-wrap;font-family:ui-monospace,monospace;"
        "font-size:0.85rem;line-height:1.5}"
        ".diff-add{color:#15803d;background:rgba(34,197,94,.12)}"
        ".diff-del{color:#b91c1c;background:rgba(239,68,68,.10);text-decoration:line-through}"
        "</style>"
        f'<div class="resume-diff">{"".join(parts)}</div>'
    )
