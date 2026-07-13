import asyncio
import logging
import os
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin
import io

import aiohttp
import streamlit as st
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from pypdf import PdfReader
from gtts import gTTS

# Streamlit पेज कॉन्फिगरेशन
st.set_page_config(page_title="एआय रिसर्च एजंट (द्विभाषिक)", page_icon="📄", layout="wide")

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
    # स्क्रीनवर दोन्ही भाषांमध्ये दाखवण्यासाठी स्ट्रक्चर्स
    executive_summary_en: str = Field(..., description="2-3 sentences explaining the core breakthrough in English.")
    executive_summary_mr: str = Field(..., description="Core breakthrough translated/explained in simple Marathi.")
    
    key_highlights_en: List[str] = Field(..., description="Bullet points detailing methodology and findings in English.")
    key_highlights_mr: List[str] = Field(..., description="The same key highlights translated/explained in simple Marathi.")
    
    practical_implications_en: str = Field(..., description="Why this paper matters in English.")
    practical_implications_mr: str = Field(..., description="Why this paper matters explained in simple Marathi.")

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
        except Exception as e:
            logger.error(f"Fetch papers failure: {str(e)}")
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
        except Exception as e:
            logger.error(f"PDF extraction failure: {str(e)}")
            return None

    def _parse_pdf_bytes(self, pdf_bytes: bytes) -> str:
        import io
        text_list = []
        with io.BytesIO(pdf_bytes) as f:
            reader = PdfReader(f)
            for page in reader.pages[:10]:  # पहिल्या १० पानांचे वाचन
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
                system_instruction=(
                    "You are a Principal AI Architect. Analyze the research paper text. "
                    "Provide the summary fields strictly matching the schema. For English fields, write in professional technical English. "
                    "For Marathi fields, translate and explain the technical concepts in fluent, easy-to-understand Marathi."
                )
            )
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=f"Analyze this research paper text and extract the structured bilingual summary:\n\n{raw_text}",
                config=config
            )
            
            if response.text:
                return PaperSummary.model_validate_json(response.text)
            return None
        except Exception as e:
            logger.error(f"Gemini एरर: {str(e)}")
            return None

# --- Streamlit UI ---
st.title("📄 स्वायत्त शैक्षणिक संशोधन एजंट (Bilingual - English & मराठी)")
st.write("हा एआय एजंट सुरक्षित Gemini API चा वापर करून रिसर्च पेपर्सचा इंग्रजी आणि मराठीत तांत्रिक सारांश तयार करतो.")

# मुख्य शोध पट्टी
search_query = st.text_input("🔎 संशोधनाचा विषय टाईप करा (उदा. 'RAG agents', 'Neural Networks'):")
limit = st.slider("किती पेपर्स शोधायचे आहेत?", min_value=1, max_value=5, value=2)

async def start_pipeline(query: str, paper_limit: int, key: str):
    async with aiohttp.ClientSession() as session:
        arxiv_source = ArxivAPIIngestor(session)
        doc_processor = DocumentProcessor(session)
        llm_engine = GeminiSummarizationEngine(api_key=key)
        
        st.info(f"🔍 '{query}' या विषयावर पेपर्स शोधत आहे...")
        papers = await arxiv_source.fetch_papers(query, limit=paper_limit)
        
        if not papers:
            st.error("एकही paper सापडला नाही. कृपया शोधताना सोपे शब्द वापरा.")
            return

        st.success(f"एकूण {len(papers)} पेपर्स सापडले. विश्लेषण सुरू आहे...")
        
        for i, paper in enumerate(papers):
            with st.expander(f"📄 Paper {i+1}: {paper.title}", expanded=True):
                st.markdown(f"**🗓️ तारीख:** {paper.publication_date} | **✍️ लेखक:** {', '.join(paper.authors)}")
                st.markdown(f"🔗 [मूळ लिंक]({paper.link}) | 📥 [पीडीएफ लिंक]({paper.pdf_link})")
                
                with st.spinner("Gemini पेपरचे द्विभाषिक विश्लेषण करत आहे..."):
                    raw_text = await doc_processor.extract_text_from_pdf(paper.pdf_link)
                    if not raw_text:
                        st.error("पीडीएफ वाचता आली नाही.")
                        continue
                        
                    summary = await llm_engine.summarize_paper(raw_text)
                    
                    if summary:
                        # दोन कॉलम्स तयार करणे (डावीकडे इंग्रजी, उजवीकडे मराठी)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🇬🇧 English Analysis")
                            st.markdown("#### 🎯 Executive Summary")
                            st.write(summary.executive_summary_en)
                            st.markdown("#### 💡 Key Highlights")
                            highlights_text_en = ""
                            for bullet in summary.key_highlights_en:
                                st.write(f"- {bullet}")
                                highlights_text_en += f"{bullet}. "
                            st.markdown("#### 🚀 Practical Implications")
                            st.write(summary.practical_implications_en)
                            
                        with col2:
                            st.markdown("### 🇮🇳 मराठी विश्लेषण")
                            st.markdown("#### 🎯 मुख्य सारांश")
                            st.write(summary.executive_summary_mr)
                            st.markdown("#### 💡 महत्त्वाचे मुद्दे")
                            for bullet in summary.key_highlights_mr:
                                st.write(f"- {bullet}")
                            st.markdown("#### 🚀 व्यावहारिक उपयोग")
                            st.write(summary.practical_implications_mr)
                        
                        st.markdown("---")
                        # 🎧 Read Aloud Player (फक्त इंग्रजी मजकुराचा ऑडिओ तयार होईल जेणेकरून उच्चार भारतीय इंग्रजीत स्पष्ट येतील)
                        st.markdown("#### 🎧 Read Aloud (Indian Accent - English Summary)")
                        
                        full_audio_text = (
                            f"Summary for the paper: {paper.title}. "
                            f"Executive Summary: {summary.executive_summary_en} "
                            f"Key Highlights: {highlights_text_en} "
                            f"Practical Implications: {summary.practical_implications_en}"
                        )
                        
                        try:
                            tts = gTTS(text=full_audio_text, lang='en', tld='co.in', slow=False)
                            fp = io.BytesIO()
                            tts.write_to_fp(fp)
                            fp.seek(0)
                            st.audio(fp, format='audio/mp3')
                        except Exception as audio_err:
                            logger.error(f"Audio creation failure: {str(audio_err)}")
                            st.warning("ऑडिओ तयार करता आला नाही.")
                    else:
                        st.error("सारांश तयार करताना एरर आली.")

if st.button("🚀 एजंट सुरू करा"):
    secure_key = st.secrets.get("GEMINI_API_KEY", "")
    if not secure_key:
        st.error("त्रुटी: Secrets मध्ये 'GEMINI_API_KEY' सेट केलेली नाही!")
    elif not search_query:
        st.warning("कृपया संशोधनाचा विषय टाईप करा.")
    else:
        asyncio.run(start_pipeline(search_query, limit, secure_key))
