import asyncio
import logging
import os
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import aiohttp
import instructor
import streamlit as st
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

# Streamlit पेज कॉन्फिगरेशन
st.set_page_config(page_title="एआय रिसर्च एजंट", page_icon="📄", layout="wide")

# लॉगींग सेटअप
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AcademicAgent")

# --- डाटा स्क्रीप्ट / साचे (Schemas) ---
class PaperMetadata(BaseModel):
    title: str = Field(..., description="संशोधन पत्रिकेचे नाव (Title)")
    authors: List[str] = Field(..., description="लेखकांची नावे")
    university: str = Field(..., description="विद्यापीठ किंवा स्त्रोत")
    publication_date: str = Field(..., description="प्रकाशन तारीख")
    link: str = Field(..., description="मूळ लिंक")
    pdf_link: str = Field(..., description="पीडीएफ डाऊनलोड लिंक")

class PaperSummary(BaseModel):
    metadata: PaperMetadata
    executive_summary: str = Field(..., description="२-३ ओळीत मुख्य शोध/ब्रेकथ्रू")
    key_highlights: List[str] = Field(..., description="महत्त्वाचे मुद्दे (पद्धती, निष्कर्ष, आकडेवारी)")
    practical_implications: str = Field(..., description="या संशोधनाचा भविष्यात काय फायदा होईल?")

# --- पेपर्स शोधणारे घटक (Ingestion) ---
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
            for page in reader.pages[:10]: # खूप मोठा पेपर असेल तर सुरुवातीची १० पाने वाचणे (खर्च वाचवण्यासाठी)
                text = page.extract_text()
                if text: text_list.append(text)
        return "\n".join(text_list)

# --- एआय सारांश इंजिन ---
class LLMSummarizationEngine:
    def __init__(self, api_key: str):
        self.client = instructor.patch(AsyncOpenAI(api_key=api_key))
        self.chunk_size = 12000 

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current, length = [], [], 0
        for s in sentences:
            current.append(s)
            length += len(s)
            if length >= self.chunk_size:
                chunks.append(" ".join(current))
                current, length = [], 0
        if current: chunks.append(" ".join(current))
        return chunks

    async def _map_summarize_chunk(self, chunk: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research scientist. Extract key technical findings."},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    async def summarize_paper(self, raw_text: str, metadata: PaperMetadata) -> Optional[PaperSummary]:
        chunks = self._chunk_text(raw_text)
        map_tasks = [self._map_summarize_chunk(c) for c in chunks]
        map_results = await asyncio.gather(*map_tasks)
        
        combined_text = "\n\n".join(map_results)
        try:
            final_summary: PaperSummary = await self.client.chat.completions.create(
                model="gpt-4o",
                response_model=PaperSummary,
                messages=[
                    {"role": "system", "content": "Synthesize the provided text into a unified, high-quality research summary in English."},
                    {"role": "user", "content": combined_text}
                ],
                temperature=0.3
            )
            final_summary.metadata = metadata
            return final_summary
        except Exception:
            return None

# --- Streamlit UI मांडणी ---
st.title("📄 स्वायत्त शैक्षणिक संशोधन एजंट")
st.write("हा एआय एजंट जगभरातील रिसर्च पेपर्स शोधून त्यांचा सखोल तांत्रिक सारांश तयार करतो.")

# डाव्या बाजूची पट्टी (Sidebar)
st.sidebar.header("⚙️ कॉन्फिगरेशन")
api_key = st.sidebar.text_input("तुमची OpenAI API Key टाका:", type="password")
st.sidebar.markdown("[OpenAI API Key कशी मिळवावी?](https://platform.openai.com/api-keys)")

# मुख्य फॉर्म
search_query = st.text_input("🔎 संशोधनाचा विषय टाईप करा (उदा. 'Retrieval-Augmented Generation', 'Quantum Computing'):")
limit = st.slider("किती पेपर्स शोधायचे आहेत?", min_value=1, max_value=5, value=2)

async def start_pipeline(query: str, paper_limit: int, key: str):
    async with aiohttp.ClientSession() as session:
        arxiv_source = ArxivAPIIngestor(session)
        doc_processor = DocumentProcessor(session)
        llm_engine = LLMSummarizationEngine(api_key=key)
        
        st.info(f"🔍 '{query}' या विषयावर पेपर्स शोधत आहे...")
        papers = await arxiv_source.fetch_papers(query, limit=paper_limit)
        
        if not papers:
            st.error("एकही पेपर सापडला नाही. कृपया दुसरा विषय शोधून पहा.")
            return

        st.success(f"कुल {len(papers)} पेपर्स सापडले. आता त्यांचे वाचन आणि विश्लेषण सुरू आहे...")
        
        for i, paper in enumerate(papers):
            with st.expander(f"📄 Paper {i+1}: {paper.title}", expanded=True):
                st.markdown(f"**🗓️ तारीख:** {paper.publication_date} | **✍️ लेखक:** {', '.join(paper.authors)}")
                st.markdown(f"🔗 [मूळ लिंक]({paper.link}) | 📥 [पीडीएफ डाऊनलोड लिंक]({paper.pdf_link})")
                
                with st.spinner("एआय एजंट पीडीएफ वाचत आहे आणि विश्लेषण करत आहे..."):
                    raw_text = await doc_processor.extract_text_from_pdf(paper.pdf_link)
                    
                    if not raw_text:
                        st.error("या पेपरची पीडीएफ फाईल वाचता आली नाही.")
                        continue
                        
                    summary = await llm_engine.summarize_paper(raw_text, paper)
                    
                    if summary:
                        st.markdown("#### 🎯 मुख्य सारांश (Executive Summary)")
                        st.write(summary.executive_summary)
                        
                        st.markdown("#### 💡 महत्त्वाचे मुद्दे (Key Highlights)")
                        for bullet in summary.key_highlights:
                            st.write(f"- {bullet}")
                            
                        st.markdown("#### 🚀 व्यावहारिक उपयोग (Practical Implications)")
                        st.write(summary.practical_implications)
                    else:
                        st.error("एआय सारांश तयार करू शकला नाही.")

if st.button("🚀 एजंट सुरू करा"):
    if not api_key:
        st.error("कृपया डाव्या बाजूच्या पट्टीमध्ये तुमची OpenAI API Key टाका!")
    elif not search_query:
        st.warning("कृपया संशोधनाचा विषय टाईप करा.")
    else:
        asyncio.run(start_pipeline(search_query, limit, api_key))
