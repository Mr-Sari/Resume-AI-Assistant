# Resume AI Assistant

An AI-powered assistant that compares a resume with a job description using **OpenAI** or **Anthropic Claude**. It identifies what the employer is looking for, what in the resume already matches, where the gaps are, and how well the candidate fits overall. It can then produce a tailored resume (with a highlighted diff) and a matching cover letter.

The project started as a Jupyter notebook ([`notebook/`](notebook/)) and was refactored into a small, tested Python package with a Streamlit web interface.

## Features

- **Resume vs. job analysis** with five clearly separated sections:
  1. **Key Requirements** – skills, technologies, qualifications, experience and responsibilities from the job description
  2. **Relevant Experience** – resume content that matches those requirements
  3. **Gaps & Mismatches** – requirements that are missing, unclear or underrepresented
  4. **Potential Strengths** – aligned strengths and valuable extras
  5. **Overall Match Analysis** – Strong / Moderate / Weak rating, evidence-based summary and priority actions
- **Tailored resume** rewritten for the job using only facts already in the resume, plus a word-level diff (additions in green, removals struck through in red) and notes explaining the changes
- **Cover letter** generated from the tailored (or original) resume
- **Two providers:** OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`) and Anthropic Claude (`claude-sonnet-4-6`, `claude-opus-5-5`, `claude-haiku-4-5`); `gpt-4o` and `claude-sonnet-4-6` are the defaults used in the original notebook
- **Structured outputs:** every model reply is validated against a Pydantic schema
- **Resume input:** paste text or upload `.txt`, `.md` or text-based `.pdf`
- **Honesty guardrails:** prompts forbid inventing employers, titles, dates, metrics, skills or certifications
- **Downloads** for the analysis, tailored resume and cover letter (Markdown)
- **Fictional sample data** to try the app instantly

## Architecture

```
            ┌─────────────────────────────┐
            │  app/app.py  (Streamlit UI) │  inputs, provider/model/key selection, rendering
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │  app/analyzer.py            │  prompts + analyze / tailor / cover-letter pipeline
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │  app/llm.py                 │  OpenAI & Claude structured generation, key handling,
            └──────────────┬──────────────┘  user-safe error messages
                           │
              OpenAI API ◄─┴─► Anthropic API   (server-side calls only)

 app/models.py  Pydantic schemas (GapAnalysis, TailoredResume, CoverLetter) + provider/model catalog
 app/utils.py   input validation, file text extraction, Markdown export, HTML diff, secret redaction
```

| Module | Responsibility | Notebook origin |
| --- | --- | --- |
| `app/llm.py` | Calls OpenAI (`chat.completions.parse`) or Claude (`messages.parse`) and returns a validated Pydantic object | `openai_generate`, `claude_generate` |
| `app/analyzer.py` | Prompt templates and the analysis → tailored resume → cover letter pipeline | `analyze_resume_against_job_description`, `generate_resume`, `generate_cover_letter`, `run_resume_rocket` |
| `app/models.py` | Output schemas and supported models | `ResumeOutput`, `CoverLetterOutput` |
| `app/utils.py` | Validation, PDF/text extraction, diff rendering, redaction | `print_markdown` display logic |
| `app/app.py` | Streamlit interface | – |

### What changed from the notebook, and why

- **Structured gap analysis.** The notebook returned free-form Markdown with four sections. The app returns a `GapAnalysis` object with a fifth *Overall Match Analysis* section, so each part can be displayed and exported separately.
- **No fabricated experience.** The notebook's resume prompt asked the model to "add missing skills and technologies", which led it to invent responsibilities and change job titles. The prompts now only allow surfacing and reframing facts that are already in the resume; unfillable gaps are reported in the change notes instead.
- **Diff computed locally.** The notebook asked the model to write an HTML diff. The app computes it with Python's `difflib`, which is always accurate and, because every token is HTML-escaped, safe to render in a public web app.
- **Real errors instead of error strings.** The notebook's helpers returned `"Error generating text: …"` as if it were model output. The app raises typed errors with user-friendly messages that never include API keys.
- **Native Claude structured outputs.** The notebook parsed Claude's raw text with `json.loads`, which failed whenever the JSON was wrapped in code fences. The app uses the Anthropic SDK's `messages.parse`.
- **Bug fixes** carried back into the notebook: `cloude_generate2` read a non-existent `response.text`; the Claude branch of `generate_cover_letter` assigned to the wrong variable; the resume prompt asked for `diff_html` while the schema expected `diff_markdown`.

## Tech Stack

- **Python 3.10+**
- **Streamlit** – web interface
- **OpenAI Python SDK** – OpenAI models
- **Anthropic Python SDK** – Claude models
- **Pydantic** – structured output schemas and validation
- **python-dotenv** – loads a local `.env` file
- **pypdf** – text extraction from uploaded PDF resumes
- **pytest** – tests (development only)

## Project Structure

```
Build-a-resume-AI-assistant/
├── app/
│   ├── __init__.py
│   ├── app.py              # Streamlit UI (entry point)
│   ├── analyzer.py         # prompts and analysis pipeline
│   ├── llm.py              # OpenAI / Claude clients and error handling
│   ├── models.py           # Pydantic schemas and model catalog
│   └── utils.py            # validation, file parsing, diff, redaction
├── examples/
│   ├── sample_resume.md            # fictional resume (John Doe)
│   └── sample_job_description.md   # fictional job description
├── notebook/
│   └── Build a resume AI assistant.ipynb   # original exploration notebook (cleaned)
├── tests/                  # unit + headless UI tests (no API keys needed)
├── .github/workflows/tests.yml
├── .streamlit/config.toml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/Mr-Sari/Build-a-resume-AI-assistant.git
cd Build-a-resume-AI-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create your own `.env` file locally from the template:

```bash
cp .env.example .env
```

Then fill in the key(s) for the provider(s) you want to use:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

You only need a key for the provider you select. Alternatively, leave the file empty and paste a key into the app's sidebar at runtime.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Key for OpenAI models |
| `ANTHROPIC_API_KEY` | Key for Claude models |
| `REQUIRE_USER_API_KEY` | Optional. Set to `true` on a public deployment so visitors must enter their own key and server keys are never used for them |

> **Never commit your `.env` file or API keys to GitHub.** `.env` is already listed in `.gitignore`.

## Run the Application

From the repository root:

```bash
streamlit run app/app.py
```

Then open http://localhost:8501.

## Notebook

The original notebook is available under:

```
notebook/
```

It contains the exploration that the app is based on: Pydantic and structured-output warm-ups, the OpenAI/Claude helper functions, the gap analysis, resume tailoring, cover letter generation and the end-to-end `run_resume_rocket` pipeline. For the public version, the personal resume was replaced with the fictional examples in `examples/`, all saved outputs were cleared, and keys are read only from environment variables.

To run it, install the requirements plus Jupyter, configure `.env`, and start Jupyter from the `notebook/` folder.

## Example Workflow

1. **Upload or paste a resume.** Click **Load fictional sample** to try it instantly.
2. **Paste the job description.**
3. **Select the provider and model** in the sidebar, and enter an API key if none is configured.
4. **Run the analysis** with **Analyze resume**.
5. **Review** the key requirements, relevant experience, gaps & mismatches, potential strengths and overall match.
6. *(Optional)* Generate a **tailored resume** (with diff) and a **cover letter**, then download them.

## Deploying to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app → Deploy a public app from GitHub** and choose:
   - Repository: `Mr-Sari/Build-a-resume-AI-assistant`
   - Branch: the branch that contains this code (for example `main`)
   - Main file path: `app/app.py`
3. Under **Advanced settings**, choose Python 3.11 and, if you want, add secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "your-key"
   ANTHROPIC_API_KEY = "your-key"
   ```
   Streamlit exposes these to the app as environment variables; they are never sent to the browser.
4. Click **Deploy**.

> **Cost warning:** if you add your own keys as secrets, every visitor's analysis is billed to your account. For a public demo, consider adding no keys (or setting `REQUIRE_USER_API_KEY = "true"`) so visitors bring their own.

## Security

- API keys are loaded from environment variables, a local `.env` file, Streamlit secrets, or a password field in the sidebar. **No keys are included in this repository.**
- All model calls happen on the server (in Python). Keys are never placed in HTML, JavaScript or client-side configuration.
- Keys entered in the sidebar are kept only in memory for that browser session; they are not stored, logged or displayed.
- Error messages are generated by the app (not echoed from the provider) for authentication failures, and anything that looks like a key is redacted from other error messages.
- Model output is rendered as Markdown; the resume diff is built from HTML-escaped text.
- The app does not store resumes or job descriptions. They are sent only to the provider you select.

## Limitations

- **Output quality depends on the model.** The analysis is an AI opinion, not a hiring decision. Always review generated resumes and cover letters before using them.
- **Plain-text extraction only.** Scanned or image-only PDFs, `.docx` files and complex PDF layouts (tables, columns) are not supported or may extract poorly.
- **Input size limit** of 30,000 characters per field.
- **No sampling control for Claude.** The current Anthropic SDK no longer accepts `temperature`, so Claude runs with default sampling (the notebook used 0.5).
- **Requires a paid API key** for OpenAI or Anthropic; there is no offline mode.
- **English-centric prompts.** Other languages may work but have not been evaluated.
- **No persistence.** Results are lost when the session ends unless downloaded.

## Future Improvements

- `.docx` upload and export of the tailored resume and cover letter as `.docx`/PDF
- Streaming responses so long generations show progress
- A numeric match score with a transparent, requirement-by-requirement breakdown
- Side-by-side comparison of OpenAI and Claude results
- Multiple job descriptions per resume
- An evaluation set of fictional resumes and jobs to measure analysis quality across models
- Prompt caching to reduce cost when re-running on the same resume

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests use fake providers and never call a real API.

## License

Released under the [MIT License](LICENSE).
