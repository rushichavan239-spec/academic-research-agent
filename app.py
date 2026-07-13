import asyncio
import logging
import os
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import aiohttp
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Streamlit पेज कॉन्फिगरेशन
st.set_page_config(page_title="एआय रिसर्च एजंट (Gemini)", page_icon="📄", layout="wide")

# लॉगींग सेटअप
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AcademicAgent")

# --- डेटा साचे (Schemas) ---
class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    university: str
    publication_date: str
    link: str
    pdf_link: str

class PaperSummary(BaseModel):
    executive_summary: str = Field(..., description="2-3 sentences explaining the core breakthrough in English.")
    key_highlights: List[str] = Field(..., description="Bullet points detailing methodology and findings in English.")
    practical_implications: str = Field(..., description="Why this paper matters in English.")

# --- पेपर्स शोधणारे घटक (arXiv Ingestion) ---
class ArxivAPIIngestor:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def fetch_papers(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={limit}"
        try:
            async with self.session.get(url, timeout=15) as response:
                if response.status != 200: return []
                xml_data = await response.text()
                return self._parse_xml(xml_data)
        except Exception:
            return []

    def _parse_xml(self, xml_text: str) -> List[PaperMetadata]:
        soup = BeautifulSoup(xml_text, "xml")
        papers = []
        for entry in soup.find_all("entry"):
            try:
                title = entry.title.text.strip().replace("\n", " ")
                authors = [a.find("name").text.strip() for a in entry.find_all("author")]
                pub_date = entry.published.text.strip()[:10]
                link = entry.id.text.strip()
                pdf_link = link.replace("abs", "pdf") + ".pdf"
                
                papers.append(PaperMetadata(
                    title=title, authors=authors, university="arXiv Repository",
                    publication_date=pub_date, link=link, pdf_link=pdf_link
                ))
            except Exception:
                continue
        return papers

# --- पीडीएफ मधून मजकूर काढणे ---
class DocumentProcessor:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def extract_text_from_pdf(self, pdf_url: str) -> Optional[str]:
        try:
            async with self.session.get(pdf_url, timeout=30) as response:
                if response.status != 200: return None
                pdf_bytes = await response.read()
            return await asyncio.to_thread(self._parse_pdf_bytes, pdf_bytes)
        except Exception:
            return None

    def _parse_pdf_bytes(self, pdf_bytes: bytes) -> str:
        import io
        text_list = []
        with io.BytesIO(pdf_bytes) as f:
            reader = PdfReader(f)
            for page in reader.pages[:10]:
                text = page.extract_text()
                if text: text_list.append(text)
        return "\n".join(text_list)

# --- Gemini सारांश इंजिन ---
class GeminiSummarizationEngine:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash" 

    async def summarize_paper(self, raw_text: str) -> Optional[PaperSummary]:
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PaperSummary,
                temperature=0.3,
                system_instruction="You are a Principal AI Architect. Analyze the research paper text and synthesize it into the requested JSON schema structure perfectly."
            )
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=f"Analyze this research paper text and extract structured summary:\n\n{raw_text}",
                config=config
            )
            
            if response.text:
                return PaperSummary.model_validate_json(response.text)
            return None
        except Exception as e:
            logger.error(f"Gemini एरर: {str(e)}")
            return None

# --- Streamlit UI ---
st.title("📄 स्वायत्त शैक्षणिक संशोधन एजंट (Powered by Gemini)")
st.write("हा एआय एजंट बॅकग्राउंडमध्ये लपवलेल्या सुरक्षित Gemini APIचा वापर करून मोफत रिसर्च पेपर्सचा तांत्रिक सारांश तयार करतो.")

# मुख्य फॉर्म (आता इथे कोणतीही गुपिते उघडी पडणार नाहीत!)
search_query = st.text_input("🔎 संशोधनाचा विषय टाईप करा:")
limit = st.slider("किती पेपर्स शोधायचे आहेत?", min_value=1, max_value=5, value=2)

async def start_pipeline(query: str, paper_limit: int, key: str):
    async with aiohttp.ClientSession() as session:
        arxiv_source = ArxivAPIIngestor(session)
        doc_processor = DocumentProcessor(session)
        llm_engine = GeminiSummarizationEngine(api_key=key)
        
        st.info(f"🔍 '{query}' या विषयावर पेपर्स शोधत आहे...")
        papers = await arxiv_source.fetch_papers(query, limit=paper_limit)
        
        if not papers:
            st.error("एकही paper सापडला नाही.")
            return

        st.success(f"एकूण {len(papers)} पेपर्स सापडले. विश्लेषण सुरू आहे...")
        
        for i, paper in enumerate(papers):
            with st.expander(f"📄 Paper {i+1}: {paper.title}", expanded=True):
                st.markdown(f"**🗓️ तारीख:** {paper.publication_date} | **✍️ लेखक:** {', '.join(paper.authors)}")
                st.markdown(f"🔗 [मूळ लिंक]({paper.link}) | 📥 [पीडीएफ लिंक]({paper.pdf_link})")
                
                with st.spinner("Gemini पेपरचे विश्लेषण करत आहे..."):
                    raw_text = await doc_processor.extract_text_from_pdf(paper.pdf_link)
                    if not raw_text:
                        st.error("पीडीएफ वाचता आली नाही.")
                        continue
                        
                    summary = await llm_engine.summarize_paper(raw_text)
                    
                    if summary:
                        st.markdown("#### 🎯 मुख्य सारांश (Executive Summary)")
                        st.write(summary.executive_summary)
                        st.markdown("#### 💡 महत्त्वाचे मुद्दे (Key Highlights)")
                        for bullet in summary.key_highlights:
                            st.write(f"- {bullet}")
                        st.markdown("#### 🚀 व्यावहारिक उपयोग (Practical Implications)")
                        st.write(summary.practical_implications)
                    else:
                        st.error("सारांश तयार करताना एरर आली. (खर्च किंवा मर्यादा संपली असावी)")

if st.button("🚀 एजंट सुरू करा"):
    # थेट Streamlit Secrets मधून सुरक्षितपणे की शोधणे
    secure_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if not secure_key:
        st.error("त्रुटी: Streamlit Cloud मधील Secrets मध्ये 'GEMINI_API_KEY' सेट केलेली नाही! कृपया ती सेट करा.")
    elif not search_query:
        st.warning("कृपया संशोधनाचा विषय टाईप करा.")
    else:
        asyncio.run(start_pipeline(search_query, limit, secure_key))
