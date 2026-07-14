"""
Autonomous Academic Research Agent
------------------------------------
Fetches papers from arXiv, extracts full text from the underlying PDFs,
summarizes them bilingually (English + Marathi) using Google Gemini
structured outputs, and renders a side-by-side dashboard with an
Indian-accent text-to-speech "read aloud" feature.

Run locally:
    streamlit run app.py

Requires a Streamlit secret named GEMINI_API_KEY (see README instructions).
"""

from __future__ import annotations

import asyncio
import io
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import streamlit as st
from bs4 import BeautifulSoup
from gtts import gTTS
from pydantic import BaseModel, Field
from pypdf import PdfReader

from google import genai
from google.genai import types

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ARXIV_API_URL = "http://export.arxiv.org/api/query"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_PDF_PAGES = 10
MAX_EXTRACTED_CHARS = 40_000  # cap the text sent to Gemini for cost/latency control
HTTP_TIMEOUT_SECONDS = 60


# --------------------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------------------

class PaperSummary(BaseModel):
    """Structured bilingual summary returned by Gemini."""

    executive_summary_en: str = Field(
        description="A concise 2-3 sentence executive summary of the paper in English."
    )
    executive_summary_mr: str = Field(
        description="A concise 2-3 sentence executive summary of the paper in fluent, natural Marathi."
    )
    key_highlights_en: List[str] = Field(
        description="A list of 3-6 key bullet-point highlights of the paper in English."
    )
    key_highlights_mr: List[str] = Field(
        description="The same key highlights translated into fluent, natural Marathi, 3-6 bullet points."
    )
    practical_implications_en: str = Field(
        description="2-3 sentences describing the practical/industry implications of this research in English."
    )
    practical_implications_mr: str = Field(
        description="2-3 sentences describing the practical/industry implications of this research in fluent Marathi."
    )


@dataclass
class ArxivPaper:
    title: str
    authors: List[str]
    published: str
    abstract: str
    pdf_link: str
    arxiv_id: str


@dataclass
class ProcessedPaper:
    paper: ArxivPaper
    summary: Optional[PaperSummary] = None
    error: Optional[str] = None
    extracted_chars: int = 0
    audio_bytes: Optional[bytes] = None


# --------------------------------------------------------------------------
# arXiv Ingestion
# --------------------------------------------------------------------------

async def search_arxiv(
    session: aiohttp.ClientSession, query: str, max_results: int
) -> List[ArxivPaper]:
    """Query the official arXiv API and safely parse the Atom/XML response."""
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        async with session.get(
            ARXIV_API_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            raw_xml = await resp.text()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to reach arXiv API: {exc}") from exc

    try:
        soup = BeautifulSoup(raw_xml, "lxml-xml")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse arXiv XML response: {exc}") from exc

    papers: List[ArxivPaper] = []
    for entry in soup.find_all("entry"):
        try:
            title_tag = entry.find("title")
            title = title_tag.get_text(strip=True).replace("\n", " ") if title_tag else "Untitled"

            authors = [
                a.find("name").get_text(strip=True)
                for a in entry.find_all("author")
                if a.find("name")
            ]

            published_tag = entry.find("published")
            published = published_tag.get_text(strip=True) if published_tag else "Unknown"

            summary_tag = entry.find("summary")
            abstract = summary_tag.get_text(strip=True) if summary_tag else ""

            id_tag = entry.find("id")
            arxiv_id = id_tag.get_text(strip=True) if id_tag else ""

            pdf_link = None
            for link in entry.find_all("link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_link = link.get("href")
                    break
            if not pdf_link and arxiv_id:
                # Fallback: derive the PDF URL from the abstract page URL
                pdf_link = arxiv_id.replace("/abs/", "/pdf/")

            if not pdf_link:
                continue

            papers.append(
                ArxivPaper(
                    title=title,
                    authors=authors,
                    published=published,
                    abstract=abstract,
                    pdf_link=pdf_link,
                    arxiv_id=arxiv_id,
                )
            )
        except Exception:  # noqa: BLE001
            # Skip malformed entries but keep processing the rest
            continue

    return papers


# --------------------------------------------------------------------------
# Async PDF Download + CPU-bound Extraction (isolated to a worker thread)
# --------------------------------------------------------------------------

async def download_pdf_bytes(session: aiohttp.ClientSession, url: str) -> bytes:
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        ) as resp:
            resp.raise_for_status()
            return await resp.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to download PDF from {url}: {exc}") from exc


def _extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int) -> str:
    """CPU-bound: run inside asyncio.to_thread to avoid blocking the event loop."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    num_pages = min(len(reader.pages), max_pages)
    text_chunks: List[str] = []
    for i in range(num_pages):
        try:
            page_text = reader.pages[i].extract_text() or ""
            text_chunks.append(page_text)
        except Exception:  # noqa: BLE001
            continue
    full_text = "\n".join(text_chunks).strip()
    return full_text[:MAX_EXTRACTED_CHARS]


async def extract_paper_text(session: aiohttp.ClientSession, pdf_link: str) -> str:
    pdf_bytes = await download_pdf_bytes(session, pdf_link)
    text = await asyncio.to_thread(_extract_text_from_pdf_bytes, pdf_bytes, MAX_PDF_PAGES)
    if not text:
        raise RuntimeError("No extractable text found in the first pages of the PDF.")
    return text


# --------------------------------------------------------------------------
# Gemini Structured Bilingual Summarization
# --------------------------------------------------------------------------

def _build_prompt(title: str, abstract: str, body_text: str) -> str:
    return f"""You are an expert bilingual academic research analyst fluent in English and Marathi.

Analyze the following research paper and produce a structured bilingual summary.

PAPER TITLE:
{title}

ABSTRACT:
{abstract}

EXTRACTED BODY TEXT (first pages):
{body_text}

Instructions:
- Write clear, accurate, and natural English.
- Write fluent, natural, grammatically correct Marathi (not a literal word-for-word translation) that an educated Marathi reader would find natural to read. Use Devanagari script.
- executive_summary: 2-3 sentences capturing the core contribution of the paper.
- key_highlights: 3-6 concise bullet points capturing the most important technical points/findings.
- practical_implications: 2-3 sentences on real-world, industry, or societal impact.
- Keep both language versions semantically equivalent in meaning.
Respond strictly according to the provided JSON schema.
"""


def _call_gemini_sync(client: genai.Client, title: str, abstract: str, body_text: str) -> PaperSummary:
    """Synchronous Gemini call (the SDK is sync); wrapped via asyncio.to_thread by the caller."""
    prompt = _build_prompt(title, abstract, body_text)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PaperSummary,
        temperature=0.3,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, PaperSummary):
        return parsed

    # Fallback: manually validate the raw JSON text if .parsed wasn't populated
    if response.text:
        return PaperSummary.model_validate_json(response.text)

    raise RuntimeError("Gemini returned an empty or unparsable response.")


async def summarize_paper_bilingual(
    client: genai.Client, title: str, abstract: str, body_text: str
) -> PaperSummary:
    return await asyncio.to_thread(_call_gemini_sync, client, title, abstract, body_text)


# --------------------------------------------------------------------------
# Indian-Accent Text-to-Speech
# --------------------------------------------------------------------------

def build_speech_text(summary: PaperSummary) -> str:
    highlights_text = ". ".join(summary.key_highlights_en)
    return (
        f"{summary.executive_summary_en} "
        f"Key highlights: {highlights_text}. "
        f"Practical implications: {summary.practical_implications_en}"
    )


def synthesize_indian_accent_audio(text: str) -> bytes:
    """Generate MP3 audio bytes using an Indian English accent via gTTS."""
    tts = gTTS(text=text, lang="en", tld="co.in")
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()


# --------------------------------------------------------------------------
# Orchestration: process a single paper end-to-end, isolating failures
# --------------------------------------------------------------------------

async def process_single_paper(
    session: aiohttp.ClientSession, client: genai.Client, paper: ArxivPaper
) -> ProcessedPaper:
    result = ProcessedPaper(paper=paper)
    try:
        body_text = await extract_paper_text(session, paper.pdf_link)
        result.extracted_chars = len(body_text)

        summary = await summarize_paper_bilingual(client, paper.title, paper.abstract, body_text)
        result.summary = summary

        speech_text = build_speech_text(summary)
        result.audio_bytes = await asyncio.to_thread(synthesize_indian_accent_audio, speech_text)

    except Exception as exc:  # noqa: BLE001
        result.error = f"{exc}"

    return result


async def run_research_pipeline(
    query: str, max_results: int, api_key: str
) -> List[ProcessedPaper]:
    client = genai.Client(api_key=api_key)

    async with aiohttp.ClientSession() as session:
        papers = await search_arxiv(session, query, max_results)
        if not papers:
            return []

        tasks = [process_single_paper(session, client, p) for p in papers]
        results = await asyncio.gather(*tasks)

    return list(results)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Securely fetch the Gemini API key from Streamlit secrets only.

    No sidebar, no text field — the key is never exposed in the UI.
    """
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:  # noqa: BLE001
        return None


def render_paper_result(index: int, item: ProcessedPaper) -> None:
    paper = item.paper

    with st.container(border=True):
        st.subheader(f"{index}. {paper.title}")
        meta_cols = st.columns([3, 2])
        with meta_cols[0]:
            authors_str = ", ".join(paper.authors) if paper.authors else "Unknown authors"
            st.caption(f"👤 {authors_str}")
        with meta_cols[1]:
            st.caption(f"🗓️ Published: {paper.published[:10]}")
        st.markdown(f"[🔗 View PDF]({paper.pdf_link})")

        if item.error:
            st.error(f"⚠️ Could not fully process this paper: {item.error}")
            with st.expander("Show original abstract"):
                st.write(paper.abstract)
            return

        summary = item.summary
        if summary is None:
            st.warning("No summary available.")
            return

        st.divider()
        col_en, col_mr = st.columns(2, gap="large")

        with col_en:
            st.markdown("### 🇬🇧 English Analysis")
            st.markdown("**Executive Summary**")
            st.write(summary.executive_summary_en)
            st.markdown("**Key Highlights**")
            for point in summary.key_highlights_en:
                st.markdown(f"- {point}")
            st.markdown("**Practical Implications**")
            st.write(summary.practical_implications_en)

        with col_mr:
            st.markdown("### 🇮🇳 मराठी विश्लेषण")
            st.markdown("**कार्यकारी सारांश**")
            st.write(summary.executive_summary_mr)
            st.markdown("**ठळक मुद्दे**")
            for point in summary.key_highlights_mr:
                st.markdown(f"- {point}")
            st.markdown("**व्यावहारिक परिणाम**")
            st.write(summary.practical_implications_mr)

        st.divider()
        st.markdown("### 🎧 Read Aloud (Indian Accent - English Summary)")
        if item.audio_bytes:
            audio_fp = io.BytesIO(item.audio_bytes)
            st.audio(audio_fp, format="audio/mp3")
        else:
            st.info("Audio narration is not available for this paper.")


def main() -> None:
    st.set_page_config(
        page_title="Autonomous Academic Research Agent",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Autonomous Academic Research Agent")
    st.markdown(
        "Fetch, analyze, and synthesize research papers from **arXiv** into a bilingual "
        "**English ↔ Marathi** dashboard, complete with Indian-accent audio narration — "
        "powered by Google Gemini."
    )

    api_key = get_api_key()
    if not api_key:
        st.error(
            "🔒 `GEMINI_API_KEY` was not found in Streamlit secrets. "
            "Please configure it under your app's Settings → Secrets before using this app."
        )
        st.stop()

    st.markdown("---")

    input_col, slider_col = st.columns([3, 1])
    with input_col:
        query = st.text_input(
            "🔎 Research topic",
            placeholder="e.g. retrieval augmented generation, quantum error correction, transformer efficiency",
        )
    with slider_col:
        max_results = st.slider("Paper search limit", min_value=1, max_value=5, value=3)

    run_clicked = st.button("🚀 Run Research Agent", type="primary", use_container_width=False)

    if run_clicked:
        clean_query = query.strip()
        if not clean_query:
            st.warning("Please enter a research topic to search for.")
            st.stop()

        status_placeholder = st.empty()
        start_time = time.time()

        try:
            with status_placeholder:
                with st.spinner(
                    f"Searching arXiv, downloading PDFs, and generating bilingual "
                    f"Gemini summaries for up to {max_results} paper(s)..."
                ):
                    results = asyncio.run(
                        run_research_pipeline(clean_query, max_results, api_key)
                    )
        except Exception as exc:  # noqa: BLE001
            status_placeholder.empty()
            st.error(f"❌ The research pipeline failed: {exc}")
            with st.expander("Show technical details"):
                st.code(traceback.format_exc())
            st.stop()

        status_placeholder.empty()
        elapsed = time.time() - start_time

        if not results:
            st.warning("No papers were found for this query. Try a different or broader topic.")
            st.stop()

        st.success(f"✅ Processed {len(results)} paper(s) in {elapsed:.1f}s.")
        st.markdown("---")

        for idx, item in enumerate(results, start=1):
            render_paper_result(idx, item)
            st.markdown("")


if __name__ == "__main__":
    main()
